from pathlib import Path
import sys

import cv2
import numpy as np
import torch

BASE_DIR = Path(__file__).resolve().parent
ARCFACE_DIR = BASE_DIR / "insightface" / "recognition" / "arcface_torch"
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


def l2norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True) + eps)


def load_model(ckpt_path: str, device: str) -> torch.nn.Module:
    model = iresnet50(fp16=False).to(device).eval()

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}

    msg = model.load_state_dict(state, strict=False)
    print("Missing keys:", msg.missing_keys)
    print("Unexpected keys:", msg.unexpected_keys)

    return model


@torch.no_grad()
def embed(model: torch.nn.Module, device: str, img_path: str) -> torch.Tensor:
    x = preprocess(img_path).to(device)
    e = l2norm(model(x))
    return e.squeeze(0).cpu()


def embed_tensor(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    return l2norm(model(x))


@torch.no_grad()
def create_template(
    model: torch.nn.Module,
    device: str,
    enrollment_paths: list[str],
) -> torch.Tensor:
    if len(enrollment_paths) == 0:
        raise ValueError("enrollment_paths must contain at least one image")

    embs = [embed(model, device, p) for p in enrollment_paths]
    stacked = torch.stack(embs, dim=0)
    template = stacked.mean(dim=0, keepdim=True)
    template = l2norm(template).squeeze(0).cpu()
    return template
