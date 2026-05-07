import base64
import cv2
import numpy as np
import torch

from model_utils import preprocess, embed_tensor, create_template


def tensor_to_base64_data_url(x: torch.Tensor) -> str:
    """
    Convert adversarial tensor in [-1, 1] back to base64 JPEG data URL.
    """
    img = x.detach().cpu().squeeze(0)

    img = img.permute(1, 2, 0).numpy()
    img = (img * 0.5) + 0.5
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    ok, buffer = cv2.imencode(".jpg", img_bgr)
    if not ok:
        raise ValueError("Failed to encode adversarial image")

    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def fgsm_false_reject(
    model: torch.nn.Module,
    x_clean: torch.Tensor,
    template: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    x_adv = x_clean.detach().clone()
    x_adv.requires_grad_(True)

    eps_scaled = eps * 2.0

    emb = embed_tensor(model, x_adv).squeeze(0)
    loss = torch.sum(emb * template)

    grad = torch.autograd.grad(loss, x_adv)[0]

    with torch.no_grad():
        x_adv = x_adv - eps_scaled * grad.sign()
        x_adv = torch.clamp(x_adv, -1.0, 1.0)

    return x_adv.detach()

def pgd_targeted_impersonation(
    model: torch.nn.Module,
    x_clean: torch.Tensor,
    target_template: torch.Tensor,
    eps: float,
    alpha: float,
    steps: int,
) -> torch.Tensor:
    x_orig = x_clean.detach().clone()
    x_adv = x_clean.detach().clone()

    eps_scaled = eps * 2.0
    alpha_scaled = alpha * 2.0

    for _ in range(steps):
        x_adv.requires_grad_(True)

        emb = embed_tensor(model, x_adv).squeeze(0)
        similarity = torch.sum(emb * target_template)
        loss = -similarity

        grad = torch.autograd.grad(loss, x_adv)[0]

        with torch.no_grad():
            x_adv = x_adv - alpha_scaled * grad.sign()
            x_adv = torch.max(torch.min(x_adv, x_orig + eps_scaled), x_orig - eps_scaled)
            x_adv = torch.clamp(x_adv, -1.0, 1.0)

    return x_adv.detach()


def run_recognition_demo(
    model: torch.nn.Module,
    device: str,
    threshold: float,
    probe_img: str,
    enrollment_imgs: list[str],
    matched_profile_id=None,
    matched_name=None,
) -> dict:
    template = create_template(model, device, enrollment_imgs).to(device)
    x_probe = preprocess(probe_img).to(device)

    with torch.no_grad():
        probe_emb = embed_tensor(model, x_probe).squeeze(0)
        score = float(torch.sum(probe_emb * template).item())

    is_match = score >= threshold

    return {
        "isMatch": is_match,
        "confidence": score,
        "matchedProfileId": matched_profile_id,
        "matchedName": matched_name,
        "message": "Recognition complete",
    }


def run_pgd_impersonation_demo(
    model: torch.nn.Module,
    device: str,
    threshold: float,
    eps: float,
    alpha: float,
    steps: int,
    attacker_img: str,
    target_enrollment_imgs: list[str],
    target_profile_id=None,
    target_name=None,
) -> dict:
    target_template = create_template(model, device, target_enrollment_imgs).to(device)
    x_clean = preprocess(attacker_img).to(device)

    with torch.no_grad():
        clean_emb = embed_tensor(model, x_clean).squeeze(0)
        clean_score = float(torch.sum(clean_emb * target_template).item())

    x_adv = pgd_targeted_impersonation(
        model=model,
        x_clean=x_clean,
        target_template=target_template,
        eps=eps,
        alpha=alpha,
        steps=steps,
    )

    with torch.no_grad():
        adv_emb = embed_tensor(model, x_adv).squeeze(0)
        adv_score = float(torch.sum(adv_emb * target_template).item())

    return {
        "attackType": "pgd_targeted_impersonation",
        "targetProfileId": target_profile_id,
        "targetName": target_name,
        "confidence": adv_score,
        "cleanScore": clean_score,
        "scoreIncrease": adv_score - clean_score,
        "success": adv_score >= threshold,
        "epsilon": eps,
        "alpha": alpha,
        "steps": steps,
        "adversarialImage": tensor_to_base64_data_url(x_adv),
        "message": "PGD targeted impersonation complete",
    }

def run_fgsm_false_reject_demo(
    model: torch.nn.Module,
    device: str,
    threshold: float,
    eps: float,
    probe_img: str,
    enrollment_imgs: list[str],
    profile_id=None,
    profile_name=None,
) -> dict:
    template = create_template(model, device, enrollment_imgs).to(device)
    x_clean = preprocess(probe_img).to(device)

    with torch.no_grad():
        clean_emb = embed_tensor(model, x_clean).squeeze(0)
        clean_score = float(torch.sum(clean_emb * template).item())

    x_adv = fgsm_false_reject(
        model=model,
        x_clean=x_clean,
        template=template,
        eps=eps,
    )

    with torch.no_grad():
        adv_emb = embed_tensor(model, x_adv).squeeze(0)
        adv_score = float(torch.sum(adv_emb * template).item())

    return {
        "attackType": "fgsm_false_reject",
        "profileId": profile_id,
        "profileName": profile_name,
        "cleanScore": clean_score,
        "confidence": adv_score,
        "scoreDecrease": clean_score - adv_score,
        "success": adv_score < threshold,
        "epsilon": eps,
        "adversarialImage": tensor_to_base64_data_url(x_adv),
        "message": "FGSM false reject complete",
    }
