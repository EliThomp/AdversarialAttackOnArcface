import os
import sys
import json
import numpy as np
import pandas as pd
import cv2
import torch
from tqdm import tqdm
from pathlib import Path

# ArcFace backbone
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
    # normalize to [-1, 1]
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0)

def l2norm(x, eps=1e-12):
    return x / (x.norm(dim=1, keepdim=True) + eps)

def load_model(ckpt_path: str, device: str):
    model = iresnet50(fp16=False).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model

def build_image_path(root: str, name: str, num: int) -> str:
    return os.path.join(root, name, f"{name}_{num:04d}.jpg")

@torch.no_grad()
def embed_nograd(model, x: torch.Tensor) -> torch.Tensor:
    return l2norm(model(x))

def fgsm_false_reject(model, x: torch.Tensor, e_fixed: torch.Tensor, eps: float) -> torch.Tensor:
    x_adv = x.clone().detach().requires_grad_(True)

    e_adv = l2norm(model(x_adv))                 # grad-enabled
    sim = (e_adv * e_fixed).sum(dim=1)           # cosine since both normalized
    loss = (-sim).mean()                         # push similarity down
    loss.backward()

    x_adv = x_adv + eps * x_adv.grad.sign()
    x_adv = x_adv.clamp(-1.0, 1.0).detach()
    return x_adv

def load_same_pairs(match_csv: str):
    m = pd.read_csv(match_csv)
    m.columns = [c.lower() for c in m.columns]   # ['name','imagenum1','imagenum2']
    pairs = []
    for _, r in m.iterrows():
        pairs.append((r["name"], int(r["imagenum1"]), r["name"], int(r["imagenum2"])))
    return pairs

def main(lfw_root: str, match_test_csv: str, ckpt_path: str, baseline_metrics_json: str, eps_list: str, max_pairs: int = 300):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    with open(baseline_metrics_json, "r") as f:
        t = float(json.load(f)["threshold"])
    print("threshold:", t)

    eps_values = [float(x) for x in eps_list.split(",")]
    print("eps values:", eps_values)

    pairs = load_same_pairs(match_test_csv)
    if max_pairs > 0:
        pairs = pairs[:max_pairs]
    print("same-person test pairs used:", len(pairs))

    model = load_model(ckpt_path, device)

    # results per eps
    os.makedirs("results/fgsm_false_reject", exist_ok=True)
    summary = {}

    for eps in eps_values:
        flips = 0
        before_sims = []
        after_sims = []

        for name1, i1, name2, i2 in tqdm(pairs, desc=f"FGSM eps={eps}"):
            p1 = build_image_path(lfw_root, name1, i1)
            p2 = build_image_path(lfw_root, name2, i2)

            x1 = preprocess(p1).to(device)
            x2 = preprocess(p2).to(device)

            with torch.no_grad():
                e2 = embed_nograd(model, x2)
                e1 = embed_nograd(model, x1)
                sim_before = float((e1 * e2).sum(dim=1).item())

            # attack x1 to reduce similarity to x2
            x1_adv = fgsm_false_reject(model, x1, e2.detach(), eps)

            with torch.no_grad():
                e1_adv = embed_nograd(model, x1_adv)
                sim_after = float((e1_adv * e2).sum(dim=1).item())

            before_sims.append(sim_before)
            after_sims.append(sim_after)

            # false reject success: was accepted before (optional), and rejected after
            if sim_after < t:
                flips += 1

        asr = flips / len(pairs)
        summary[str(eps)] = {
            "ASR_false_reject": asr,
            "mean_sim_before": float(np.mean(before_sims)),
            "mean_sim_after": float(np.mean(after_sims)),
        }

        np.save(f"results/fgsm_false_reject/sim_before_eps_{eps}.npy", np.array(before_sims, dtype=np.float32))
        np.save(f"results/fgsm_false_reject/sim_after_eps_{eps}.npy", np.array(after_sims, dtype=np.float32))

        print(f"[eps={eps}] ASR_false_reject={asr:.4f}  mean_sim_before={np.mean(before_sims):.4f}  mean_sim_after={np.mean(after_sims):.4f}")

    with open("results/fgsm_false_reject/summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("Saved results to results/fgsm_false_reject/")

if __name__ == "__main__":
    lfw_root = sys.argv[1]
    match_test = sys.argv[2]
    ckpt = sys.argv[3]
    metrics_json = sys.argv[4]
    eps_list = sys.argv[5]
    max_pairs = int(sys.argv[6]) if len(sys.argv) > 6 else 300
    main(lfw_root, match_test, ckpt, metrics_json, eps_list, max_pairs=max_pairs)
