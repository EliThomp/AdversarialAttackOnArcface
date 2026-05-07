import os
import sys
import math
import pandas as pd
import numpy as np
import cv2
import torch
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import accuracy_score

ARCFACE_DIR = Path("insightface/recognition/arcface_torch")
sys.path.append(str(ARCFACE_DIR))
from backbones.iresnet import iresnet50

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

def build_image_path(root: str, name: str, num: int) -> str:
    # LFW file naming: Name/Name_0001.jpg
    return os.path.join(root, name, f"{name}_{num:04d}.jpg")

@torch.no_grad()
def embed(model, device, img_path: str) -> torch.Tensor:
    x = preprocess(img_path).to(device)
    e = l2norm(model(x))
    return e.squeeze(0).cpu()

def find_best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    candidates = np.unique(scores)
    best_t, best_acc = 0.0, -1.0
    for t in candidates:
        preds = (scores >= t).astype(np.int32)
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_t = float(t)
    return best_t

def far_frr(scores: np.ndarray, labels: np.ndarray, t: float):
    preds = (scores >= t).astype(np.int32)
    same = labels == 1
    diff = labels == 0
    frr = (preds[same] == 0).mean() if same.any() else float("nan")
    far = (preds[diff] == 1).mean() if diff.any() else float("nan")
    return far, frr

def load_pairs(match_csv: str, mismatch_csv: str):
    m = pd.read_csv(match_csv)
    mm = pd.read_csv(mismatch_csv)

    # normalize columns
    m.columns = [c.lower() for c in m.columns]
    mm.columns = [c.lower() for c in mm.columns]

    pairs = []

    # ----- MATCH (same person) -----
    for _, r in m.iterrows():
        name = r["name"]
        i1 = int(r["imagenum1"])
        i2 = int(r["imagenum2"])
        pairs.append((name, i1, name, i2, 1))

    # ----- MISMATCH (different people) -----
    for _, r in mm.iterrows():
        name1 = r["name"]
        i1 = int(r["imagenum1"])
        name2 = r["name.1"]
        i2 = int(r["imagenum2"])
        pairs.append((name1, i1, name2, i2, 0))

    return pairs


def main(lfw_root: str, match_train: str, mismatch_train: str, match_test: str, mismatch_test: str, ckpt_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    model = load_model(ckpt_path, device)

    train_pairs = load_pairs(match_train, mismatch_train)
    test_pairs = load_pairs(match_test, mismatch_test)

    print("train pairs:", len(train_pairs), "test pairs:", len(test_pairs))

    # cache embeddings to avoid recomputing per pair
    cache = {}

    def get_emb(name: str, idx: int) -> torch.Tensor:
        key = (name, idx)
        if key in cache:
            return cache[key]
        p = build_image_path(lfw_root, name, idx)
        e = embed(model, device, p)
        cache[key] = e
        return e

    def score_pairs(pairs):
        scores = []
        labels = []
        for a, ia, b, ib, y in tqdm(pairs, desc="scoring"):
            ea = get_emb(a, ia)
            eb = get_emb(b, ib)
            s = float((ea * eb).sum().item())
            scores.append(s)
            labels.append(y)
        return np.array(scores, dtype=np.float32), np.array(labels, dtype=np.int32)

    train_scores, train_labels = score_pairs(train_pairs)
    test_scores, test_labels = score_pairs(test_pairs)

   # ---- Threshold selection ----
    if len(sys.argv) > 7:
        t = float(sys.argv[7])
        print("Using manual threshold:", t)
    else:
        t = find_best_threshold(train_scores, train_labels)

    # ---- Evaluation on test set ----
    preds = (test_scores >= t).astype(np.int32)

    acc = accuracy_score(test_labels, preds)
    far, frr = far_frr(test_scores, test_labels, t)

    print(f"threshold={t:.4f}")
    print(f"test_acc={acc:.4f}  FAR={far:.4f}  FRR={frr:.4f}")


    out_dir = "results/baseline"
    os.makedirs(out_dir, exist_ok=True)

    # Save raw scores + labels
    np.save(os.path.join(out_dir, "train_scores.npy"), train_scores)
    np.save(os.path.join(out_dir, "train_labels.npy"), train_labels)
    np.save(os.path.join(out_dir, "test_scores.npy"), test_scores)
    np.save(os.path.join(out_dir, "test_labels.npy"), test_labels)

    # Save threshold + metrics
    summary = {
        "threshold": float(t),
        "test_accuracy": float(acc),
        "FAR": float(far),
        "FRR": float(frr),
        "num_train_pairs": int(len(train_scores)),
        "num_test_pairs": int(len(test_scores)),
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=4)

    # Plot similarity histogram
    same_scores = test_scores[test_labels == 1]
    diff_scores = test_scores[test_labels == 0]

    plt.figure()
    plt.hist(same_scores, bins=50, alpha=0.6, label="Same Person")
    plt.hist(diff_scores, bins=50, alpha=0.6, label="Different Person")
    plt.axvline(t, linestyle="--", label="Threshold")
    plt.legend()
    plt.title("Baseline Similarity Distribution")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(out_dir, "similarity_histogram.png"))
    plt.close()

    print("Artifacts saved to results/baseline/")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
