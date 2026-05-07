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

def main(img_path, ckpt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    model = iresnet50(fp16=False).to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)

    x = preprocess(img_path).to(device)

    with torch.no_grad():
        emb = l2norm(model(x))
    print("embedding shape:", tuple(emb.shape), "norm:", float(emb.norm().cpu()))

    x2 = x.clone().requires_grad_(True)
    emb2 = l2norm(model(x2))
    loss = emb2.sum()
    loss.backward()
    print("grad exists?", x2.grad is not None, "grad mean:", float(x2.grad.abs().mean().cpu()))

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
