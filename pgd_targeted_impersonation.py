import sys
from pathlib import Path
from collections import defaultdict
import random

import cv2
import numpy as np
import torch

ARCFACE_DIR = Path("insightface/recognition/arcface_torch")
sys.path.append(str(ARCFACE_DIR))
from backbones.iresnet import iresnet50  # noqa: E402

def save_adv_tensor(x_adv: torch.Tensor, out_path: str):
    img = x_adv.detach().cpu().squeeze(0)
    img = img.permute(1, 2, 0).numpy()
    img = (img * 0.5) + 0.5
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, img_bgr)

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


def embed_tensor(model, x: torch.Tensor) -> torch.Tensor:
    """
    x is already a tensor shaped [1, 3, 112, 112] on device
    returns normalized embedding [1, 512]
    """
    e = model(x)
    e = l2norm(e)
    return e


@torch.no_grad()
def create_template(model, device, enrollment_paths: list[str]) -> torch.Tensor:
    if len(enrollment_paths) == 0:
        raise ValueError("enrollment_paths must contain at least one image")

    embs = [embed(model, device, p) for p in enrollment_paths]
    stacked = torch.stack(embs, dim=0)            # [K, 512]
    template = stacked.mean(dim=0, keepdim=True)  # [1, 512]
    template = l2norm(template).squeeze(0).cpu()  # [512]
    return template


def cosine_sim(e1: torch.Tensor, e2: torch.Tensor) -> float:
    return float((e1 * e2).sum().item())


def build_identity_to_paths(root_dir: str) -> dict[str, list[str]]:
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


def prepare_templates_and_attacker_probes(model, device, identity_to_paths, enroll_count):
    """
    Build:
      templates[identity] = template
      attacker_probe_paths[identity] = list of same-identity non-enrollment images
    """
    eligible = {
        identity: paths
        for identity, paths in identity_to_paths.items()
        if len(paths) >= enroll_count + 1
    }

    if len(eligible) < 2:
        raise ValueError("Need at least 2 identities with enough images.")

    templates = {}
    attacker_probe_paths = {}

    for identity, paths in eligible.items():
        enrollment_paths = paths[:enroll_count]
        probe_paths = paths[enroll_count:]

        templates[identity] = create_template(model, device, enrollment_paths)
        attacker_probe_paths[identity] = probe_paths

    return templates, attacker_probe_paths


def pgd_targeted_impersonation(
    model,
    x_clean: torch.Tensor,
    target_template: torch.Tensor,
    eps: float = 8 / 255,
    alpha: float = 1 / 255,
    steps: int = 20,
):
    """
    Targeted attack:
    maximize cosine similarity between attacker embedding and target template.

    x_clean: [1, 3, 112, 112] on device
    target_template: [512] on device
    """
    x_orig = x_clean.detach().clone()
    x_adv = x_clean.detach().clone()

    # images are normalized to [-1, 1], so rescale epsilon/alpha from [0,1]
    eps_scaled = eps * 2.0
    alpha_scaled = alpha * 2.0

    for _ in range(steps):
        x_adv.requires_grad_(True)

        emb = embed_tensor(model, x_adv).squeeze(0)     # [512]
        similarity = torch.sum(emb * target_template)   # maximize similarity
        loss = -similarity                              # minimize negative similarity

        grad = torch.autograd.grad(loss, x_adv)[0]

        with torch.no_grad():
            x_adv = x_adv - alpha_scaled * grad.sign()

            # project back to epsilon-ball around original image
            x_adv = torch.max(torch.min(x_adv, x_orig + eps_scaled), x_orig - eps_scaled)

            # clamp to valid normalized range
            x_adv = torch.clamp(x_adv, -1.0, 1.0)

    return x_adv.detach()


def evaluate_impersonation_attack(
    model,
    device,
    templates,
    attacker_probe_paths,
    threshold: float,
    eps: float = 8 / 255,
    alpha: float = 1 / 255,
    steps: int = 20,
    max_pairs: int | None = 200,
    seed: int = 42,
):
    """
    Evaluate targeted impersonation attack.

    For each attacker identity A:
      choose a target identity B != A
      use one or more probe images from A
      try to get accepted as B

    Returns summary dict.
    """
    rng = random.Random(seed)
    identities = sorted(templates.keys())

    clean_scores = []
    adv_scores = []
    total_considered = 0
    attack_successes = 0
    already_accepted_clean = 0

    per_pair_rows = []

    for attacker_identity in identities:
        attacker_paths = attacker_probe_paths[attacker_identity]
        if not attacker_paths:
            continue

        possible_targets = [i for i in identities if i != attacker_identity]
        rng.shuffle(possible_targets)

        for target_identity in possible_targets:
            target_template = templates[target_identity].to(device)

            for probe_path in attacker_paths:
                x_clean = preprocess(probe_path).to(device)

                with torch.no_grad():
                    clean_emb = embed_tensor(model, x_clean).squeeze(0)
                    clean_score = float(torch.sum(clean_emb * target_template).item())

                # only count real attack opportunities:
                # clean sample must be rejected first
                if clean_score >= threshold:
                    already_accepted_clean += 1
                    continue

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

                save_adv_tensor(x_adv, "adv_test.jpg")
                save_adv_tensor(x_adv, "adv_test.png")

                with torch.no_grad():
                    jpg_x = preprocess("adv_test.jpg").to(device)
                    png_x = preprocess("adv_test.png").to(device)

                    jpg_emb = embed_tensor(model, jpg_x).squeeze(0)
                    png_emb = embed_tensor(model, png_x).squeeze(0)

                    jpg_score = float(torch.sum(jpg_emb * target_template).item())
                    png_score = float(torch.sum(png_emb * target_template).item())

                print(f"Raw tensor adv score        = {adv_score:.4f}")
                print(f"Reloaded JPEG adv score     = {jpg_score:.4f}")
                print(f"Reloaded PNG adv score      = {png_score:.4f}")
                print(f"JPEG still succeeds?        = {jpg_score >= threshold}")
                print(f"PNG still succeeds?         = {png_score >= threshold}")

                success = adv_score >= threshold

                clean_scores.append(clean_score)
                adv_scores.append(adv_score)
                total_considered += 1
                attack_successes += int(success)

                per_pair_rows.append({
                    "attacker_identity": attacker_identity,
                    "target_identity": target_identity,
                    "probe_path": probe_path,
                    "clean_score": clean_score,
                    "adv_score": adv_score,
                    "success": success,
                })

                if max_pairs is not None and total_considered >= max_pairs:
                    break

            if max_pairs is not None and total_considered >= max_pairs:
                break

        if max_pairs is not None and total_considered >= max_pairs:
            break

    if total_considered == 0:
        raise ValueError("No clean rejected attacker-target pairs were available for attack.")

    clean_scores = np.array(clean_scores)
    adv_scores = np.array(adv_scores)

    return {
        "num_attacked_pairs": total_considered,
        "num_already_accepted_clean": already_accepted_clean,
        "attack_successes": attack_successes,
        "attack_success_rate": attack_successes / total_considered,
        "avg_clean_score": float(clean_scores.mean()),
        "avg_adv_score": float(adv_scores.mean()),
        "avg_score_increase": float((adv_scores - clean_scores).mean()),
        "per_pair_rows": per_pair_rows,
    }

def main_manual_impersonation(
    ckpt: str,
    threshold: float,
    eps: float,
    alpha: float,
    steps: int,
    attacker_img: str,
    target_enrollment_imgs: list[str],
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = load_model(ckpt, device)

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
    
    save_adv_tensor(x_adv, "adv_test.jpg")
    save_adv_tensor(x_adv, "adv_test.png")

    with torch.no_grad():
        jpg_x = preprocess("adv_test.jpg").to(device)
        png_x = preprocess("adv_test.png").to(device)

        jpg_emb = embed_tensor(model, jpg_x).squeeze(0)
        png_emb = embed_tensor(model, png_x).squeeze(0)

        jpg_score = float(torch.sum(jpg_emb * target_template).item())
        png_score = float(torch.sum(png_emb * target_template).item())

    print(f"Raw tensor adv score        = {adv_score:.4f}")
    print(f"Reloaded JPEG adv score     = {jpg_score:.4f}")
    print(f"Reloaded PNG adv score      = {png_score:.4f}")
    print(f"JPEG still succeeds?        = {jpg_score >= threshold}")
    print(f"PNG still succeeds?         = {png_score >= threshold}")

    success = (clean_score < threshold) and (adv_score >= threshold)

    print("\n===== MANUAL PGD TARGETED IMPERSONATION =====")
    print(f"Attacker image              = {attacker_img}")
    print("Target enrollment images:")
    for p in target_enrollment_imgs:
        print(f"  {p}")
    print(f"Threshold                   = {threshold:.4f}")
    print(f"Epsilon                     = {eps:.6f}")
    print(f"Alpha                       = {alpha:.6f}")
    print(f"Steps                       = {steps}")
    print(f"Clean score                 = {clean_score:.4f}")
    print(f"Adversarial score           = {adv_score:.4f}")
    print(f"Score increase              = {adv_score - clean_score:.4f}")
    print("Attack success              =", success)

def main(
    dataset_root: str,
    ckpt: str,
    enroll_count: int,
    threshold: float,
    eps: float = 8 / 255,
    alpha: float = 1 / 255,
    steps: int = 20,
    max_pairs: int | None = 200,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = load_model(ckpt, device)
    identity_to_paths = build_identity_to_paths(dataset_root)

    print(f"Found {len(identity_to_paths)} identities total")

    templates, attacker_probe_paths = prepare_templates_and_attacker_probes(
        model=model,
        device=device,
        identity_to_paths=identity_to_paths,
        enroll_count=enroll_count,
    )

    result = evaluate_impersonation_attack(
        model=model,
        device=device,
        templates=templates,
        attacker_probe_paths=attacker_probe_paths,
        threshold=threshold,
        eps=eps,
        alpha=alpha,
        steps=steps,
        max_pairs=max_pairs,
        seed=42,
    )

    print("\n===== PGD TARGETED IMPERSONATION RESULTS =====")
    print(f"Enrollment count           = {enroll_count}")
    print(f"Threshold                  = {threshold:.4f}")
    print(f"Epsilon                    = {eps:.6f}")
    print(f"Alpha                      = {alpha:.6f}")
    print(f"Steps                      = {steps}")
    print(f"Attacked attacker-target pairs = {result['num_attacked_pairs']}")
    print(f"Already accepted clean         = {result['num_already_accepted_clean']}")
    print(f"Attack successes               = {result['attack_successes']}")
    print(f"Attack success rate            = {result['attack_success_rate']:.4f}")
    print(f"Average clean score            = {result['avg_clean_score']:.4f}")
    print(f"Average adversarial score      = {result['avg_adv_score']:.4f}")
    print(f"Average score increase         = {result['avg_score_increase']:.4f}")


if __name__ == "__main__":
    """
    Dataset mode:
    python pgd_targeted_impersonation.py <dataset_root> <ckpt> <enroll_count> <threshold> [eps] [alpha] [steps] [max_pairs]

    Manual mode:
    python pgd_targeted_impersonation.py --manual <ckpt> <threshold> <eps> <alpha> <steps> <attacker_img> <target1> [<target2> ...]
    """
    if len(sys.argv) >= 2 and sys.argv[1] == "--manual":
        if len(sys.argv) < 9:
            print("Usage: python pgd_targeted_impersonation.py --manual <ckpt> <threshold> <eps> <alpha> <steps> <attacker_img> <target1> [<target2> ...]")
            sys.exit(1)

        ckpt = sys.argv[2]
        threshold = float(sys.argv[3])
        eps = float(sys.argv[4])
        alpha = float(sys.argv[5])
        steps = int(sys.argv[6])
        attacker_img = sys.argv[7]
        target_enrollment_imgs = sys.argv[8:]

        main_manual_impersonation(
            ckpt=ckpt,
            threshold=threshold,
            eps=eps,
            alpha=alpha,
            steps=steps,
            attacker_img=attacker_img,
            target_enrollment_imgs=target_enrollment_imgs,
        )
    else:
        if len(sys.argv) < 5:
            print("Usage: python pgd_targeted_impersonation.py <dataset_root> <ckpt> <enroll_count> <threshold> [eps] [alpha] [steps] [max_pairs]")
            sys.exit(1)

        dataset_root = sys.argv[1]
        ckpt = sys.argv[2]
        enroll_count = int(sys.argv[3])
        threshold = float(sys.argv[4])
        eps = float(sys.argv[5]) if len(sys.argv) > 5 else 8 / 255
        alpha = float(sys.argv[6]) if len(sys.argv) > 6 else 1 / 255
        steps = int(sys.argv[7]) if len(sys.argv) > 7 else 20
        max_pairs = int(sys.argv[8]) if len(sys.argv) > 8 else 200

        main(dataset_root, ckpt, enroll_count, threshold, eps, alpha, steps, max_pairs)
        