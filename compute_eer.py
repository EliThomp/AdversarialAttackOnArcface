import sys
from pathlib import Path
from collections import defaultdict
import random

import cv2
import numpy as np
import torch

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


def load_model(ckpt_path: str, device: str):
    model = iresnet50(fp16=False).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    msg = model.load_state_dict(state, strict=False)
    print("Missing keys:", msg.missing_keys)
    print("Unexpected keys:", msg.unexpected_keys)
    return model


@torch.no_grad()
def embed(model, device, img_path: str) -> torch.Tensor:
    x = preprocess(img_path).to(device)
    e = l2norm(model(x))
    return e.squeeze(0).cpu()  


@torch.no_grad()
def create_template(model, device, enrollment_paths: list[str]) -> torch.Tensor:
    if len(enrollment_paths) == 0:
        raise ValueError("enrollment_paths must contain at least one image")

    embs = [embed(model, device, p) for p in enrollment_paths]
    stacked = torch.stack(embs, dim=0)            
    template = stacked.mean(dim=0, keepdim=True)  
    template = l2norm(template).squeeze(0).cpu()  
    return template


def cosine_sim(e1: torch.Tensor, e2: torch.Tensor) -> float:
    return float((e1 * e2).sum().item())


def build_identity_to_paths(root_dir: str) -> dict[str, list[str]]:
    """
    Expects LFW-style structure:
      root_dir/
        Person_A/
          Person_A_0001.jpg
          ...
        Person_B/
          ...
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root_dir}")

    identity_to_paths = defaultdict(list)

    for person_dir in root.iterdir():
        if not person_dir.is_dir():
            continue
        image_paths = sorted(
            [str(p) for p in person_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        if image_paths:
            identity_to_paths[person_dir.name] = image_paths

    return dict(identity_to_paths)


def make_templates_and_scores(
    model,
    device: str,
    identity_to_paths: dict[str, list[str]],
    enroll_count: int,
    impostor_limit_per_identity: int = 10,
    seed: int = 42,
):
    """
    For each identity:
      - first enroll_count images -> enrollment template
      - remaining images -> genuine probes
      - one image from other identities -> impostor probes

    Returns:
      genuine_scores, impostor_scores
    """
    random.seed(seed)

    # Keep only identities with enough images for enrollment + at least 1 genuine probe
    eligible = {
        identity: paths
        for identity, paths in identity_to_paths.items()
        if len(paths) >= enroll_count + 1
    }

    identities = sorted(eligible.keys())

    if len(identities) < 2:
        raise ValueError("Need at least 2 identities with enough images.")

    templates = {}
    genuine_probe_paths = {}

    for identity in identities:
        paths = eligible[identity]
        enrollment_paths = paths[:enroll_count]
        probe_paths = paths[enroll_count:]  # same-person probes

        templates[identity] = create_template(model, device, enrollment_paths)
        genuine_probe_paths[identity] = probe_paths

    genuine_scores = []
    impostor_scores = []

    # Genuine scores
    for identity in identities:
        template = templates[identity]
        for probe_path in genuine_probe_paths[identity]:
            probe_emb = embed(model, device, probe_path)
            score = cosine_sim(probe_emb, template)
            genuine_scores.append(score)

    # Impostor scores
    for target_identity in identities:
        target_template = templates[target_identity]

        other_identities = [i for i in identities if i != target_identity]
        random.shuffle(other_identities)

        used = 0
        for other_identity in other_identities:
            # use first remaining image from the other identity as impostor probe
            impostor_probe_path = eligible[other_identity][enroll_count]  # first non-enrollment image
            probe_emb = embed(model, device, impostor_probe_path)
            score = cosine_sim(probe_emb, target_template)
            impostor_scores.append(score)

            used += 1
            if used >= impostor_limit_per_identity:
                break

    return genuine_scores, impostor_scores


def compute_far_frr(genuine_scores, impostor_scores, thresholds):
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    fars = []
    frrs = []

    for th in thresholds:
        far = np.mean(impostor_scores >= th)  
        frr = np.mean(genuine_scores < th)  
        fars.append(far)
        frrs.append(frr)

    return np.array(fars), np.array(frrs)


def compute_eer(genuine_scores, impostor_scores):
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)

    all_scores = np.concatenate([genuine_scores, impostor_scores])
    thresholds = np.sort(np.unique(all_scores))

    fars, frrs = compute_far_frr(genuine_scores, impostor_scores, thresholds)

    idx = np.argmin(np.abs(fars - frrs))

    eer = 0.5 * (fars[idx] + frrs[idx])
    threshold = thresholds[idx]

    return {
        "eer": float(eer),
        "threshold": float(threshold),
        "far_at_eer": float(fars[idx]),
        "frr_at_eer": float(frrs[idx]),
        "thresholds": thresholds,
        "fars": fars,
        "frrs": frrs,
    }


def main(dataset_root: str, ckpt: str, enroll_count: int, impostor_limit_per_identity: int = 10):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = load_model(ckpt, device)
    identity_to_paths = build_identity_to_paths(dataset_root)

    print(f"Found {len(identity_to_paths)} identities total")

    genuine_scores, impostor_scores = make_templates_and_scores(
        model=model,
        device=device,
        identity_to_paths=identity_to_paths,
        enroll_count=enroll_count,
        impostor_limit_per_identity=impostor_limit_per_identity,
        seed=42,
    )

    print(f"Genuine scores collected : {len(genuine_scores)}")
    print(f"Impostor scores collected: {len(impostor_scores)}")

    result = compute_eer(genuine_scores, impostor_scores)

    print("\n===== EER RESULTS =====")
    print(f"Enrollment count = {enroll_count}")
    print(f"EER              = {result['eer']:.4f}")
    print(f"Threshold        = {result['threshold']:.4f}")
    print(f"FAR @ EER        = {result['far_at_eer']:.4f}")
    print(f"FRR @ EER        = {result['frr_at_eer']:.4f}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python compute_eer.py <dataset_root> <ckpt> <enroll_count> [impostor_limit_per_identity]")
        sys.exit(1)

    dataset_root = sys.argv[1]
    ckpt = sys.argv[2]
    enroll_count = int(sys.argv[3])
    impostor_limit_per_identity = int(sys.argv[4]) if len(sys.argv) > 4 else 10

    main(dataset_root, ckpt, enroll_count, impostor_limit_per_identity)