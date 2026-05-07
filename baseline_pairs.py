import os
import sys
import csv
from dataclasses import dataclass
from typing import List, Tuple

import torch
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import accuracy_score

from pathlib import Path
ARCFACE_DIR = Path("insightface/recognition/arcface_torch")
sys.path.append(str(ARCFACE_DIR))
from backbones.iresnet import iresnet50  # noqa: E402

def preprocess(path: str) -> torch.Tensor:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.resize(img, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0)

def l2norm(x, eps=1e-12):
    return x / (x.norm(dim=1, keepdim=True) + eps)

def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=1)

def load_model(ckpt_path: str, device: str):
    model = iresnet50(fp16=False).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model

def compute_embeddings(model, device, paths: List[str]) -> torch.Tensor:
    embs = []
    with torch.no_grad():
        for p in paths:
            x = preprocess(p).to(device)
            e = l2norm(model(x))
            embs.append(e.cpu())
    return torch.cat(embs, dim=0)

def find_best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    # simple sweep over candidate thresholds from the scores
    candidates = np.unique(scores)
    best_t, best_acc = 0.0, -1.0
    for t in candidates:
        preds = (scores >= t).astype(np.int32)
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return best_t

def far_frr(scores: np.ndarray, labels: np.ndarray, t: float) -> Tuple[float, float]:
    preds = (scores >= t).astype(np.int32)
    # labels: 1 = same, 0 = different
    same = labels == 1
    diff = labels == 0
    frr = (preds[same] == 0).mean() if same.any() else float("nan")
    far = (preds[diff] == 1).mean() if diff.any() else float("nan")
    return far, frr

def read_pairs_csv(pairs_csv: str) -> List[Tuple[str, str, int]]:
    """
    CSV format: img1,img2,label
    label: 1 same person, 0 different
    """
    pairs = []
    with open(pairs_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pairs.append((row["img1"], row["img2"], int(row["label"])))
    return pairs

def main(pairs_csv: str, ckpt_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    pairs = read_pairs_csv(pairs_csv)
    print("pairs:", len(pairs))

    # collect unique image paths
    all_paths = []
    for a, b, _ in pairs:
        all_paths.append(a); all_paths.append(b)
    uniq_paths = sorted(set(all_paths))
    idx = {p: i for i, p in enumerate(uniq_paths)}

    model = load_model(ckpt_path, device)
    embs = compute_embeddings(model, device, uniq_paths)  # [N, 512]

    scores = []
    labels = []
    for a, b, y in pairs:
        ea = embs[idx[a]].unsqueeze(0)
        eb = embs[idx[b]].unsqueeze(0)
        s = cosine(ea, eb).item()
        scores.append(s)
        labels.append(y)

    scores = np.array(scores, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    # simple split: first 80% threshold selection, last 20% report
    n = len(scores)
    split = int(0.8 * n)
    t = find_best_threshold(scores[:split], labels[:split])

    preds = (scores[split:] >= t).astype(np.int32)
    acc = accuracy_score(labels[split:], preds)
    far, frr = far_frr(scores[split:], labels[split:], t)

    print(f"threshold={t:.4f}")
    print(f"test_acc={acc:.4f}  FAR={far:.4f}  FRR={frr:.4f}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
