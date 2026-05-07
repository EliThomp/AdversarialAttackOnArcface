import sys
from pathlib import Path
import cv2
import numpy as np
import torch

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


def load_model(ckpt_path: str, device: str):
    model = iresnet50(fp16=False).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model


@torch.no_grad()
def embed(model, device, img_path: str) -> torch.Tensor:
    x = preprocess(img_path).to(device)
    e = l2norm(model(x))
    return e.squeeze(0)  # shape [512]


@torch.no_grad()
def create_template(model, device, enrollment_paths: list[str]) -> torch.Tensor:
    """
    Build one authentication template from multiple enrollment images
    of the same person.

    Steps:
    1. embed each enrollment image
    2. average embeddings
    3. L2-normalize the average

    Returns:
        template tensor of shape [512]
    """
    if len(enrollment_paths) == 0:
        raise ValueError("enrollment_paths must contain at least one image")

    embs = []
    for path in enrollment_paths:
        embs.append(embed(model, device, path))

    stacked = torch.stack(embs, dim=0)      # [K, 512]
    template = stacked.mean(dim=0, keepdim=True)  # [1, 512]
    template = l2norm(template).squeeze(0)  # [512]
    return template


@torch.no_grad()
def verify_probe_vs_template(model, device, probe_path: str, template: torch.Tensor) -> float:
    """
    Compute cosine similarity between a probe image and an enrollment template.
    """
    probe_emb = embed(model, device, probe_path)   # [512]
    sim = float((probe_emb * template).sum().item())
    return sim


def main(ckpt: str, threshold: float, probe_img: str, enrollment_imgs: list[str]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ckpt, device)

    template = create_template(model, device, enrollment_imgs)
    sim = verify_probe_vs_template(model, device, probe_img, template)
    accept = sim >= threshold

    print("probe image:", probe_img)
    print("enrollment images:")
    for p in enrollment_imgs:
        print("  ", p)

    print(f"cosine_similarity = {sim:.4f}")
    print(f"threshold        = {threshold:.4f}")
    print("decision         =", "ACCEPT (match)" if accept else "REJECT (non-match)")


if __name__ == "__main__":
    """
    Usage:
    python authenticate_with_template.py \
        weights/ms1mv3_arcface_r50_fp16.pth \
        0.3703 \
        probe.jpg \
        enroll1.jpg enroll2.jpg enroll3.jpg
    """
    if len(sys.argv) < 5:
        print(
            "Usage: python authenticate_with_template.py <ckpt> <threshold> <probe_img> <enroll1> [<enroll2> ...]"
        )
        sys.exit(1)

    ckpt = sys.argv[1]
    threshold = float(sys.argv[2])
    probe_img = sys.argv[3]
    enrollment_imgs = sys.argv[4:]

    main(ckpt, threshold, probe_img, enrollment_imgs)
    