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
    return e.squeeze(0)

def main(img1: str, img2: str, ckpt: str, threshold: float):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(ckpt, device)

    e1 = embed(model, device, img1)
    e2 = embed(model, device, img2)

    sim = float((e1 * e2).sum().item())
    accept = sim >= threshold

    print("img1:", img1)
    print("img2:", img2)
    print(f"cosine_similarity = {sim:.4f}")
    print(f"threshold        = {threshold:.4f}")
    print("decision         =", "ACCEPT (match)" if accept else "REJECT (non-match)")

if __name__ == "__main__":
    # usage:
    # python verify_two_images.py MyFace.jpg OtherFace.jpg weights/ms1mv3_arcface_r50_fp16.pth 0.3703
    main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]))
