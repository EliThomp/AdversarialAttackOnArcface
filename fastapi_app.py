import base64
import os
import tempfile
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model_utils import load_model
from demo_api_utils import (
    run_recognition_demo,
    run_fgsm_false_reject_demo,
    run_pgd_impersonation_demo,
)

app = FastAPI(title="Face Recognition and Attack Demo API")

DEFAULT_CKPT = "weights/ms1mv3_arcface_r50_fp16.pth"
DEFAULT_THRESHOLD = 0.4520

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = load_model(DEFAULT_CKPT, DEVICE)


class Profile(BaseModel):
    id: Optional[int | str] = None
    name: Optional[str] = None
    image: Optional[str] = None
    photos: Optional[list[str]] = None
    captures: Optional[dict[str, str]] = None


class RecognitionRequest(BaseModel):
    image: str
    selectedProfileId: Optional[str] = None
    referenceProfile: Profile
    threshold: Optional[float] = DEFAULT_THRESHOLD


class FGSMAttackRequest(BaseModel):
    image: str
    targetProfileId: Optional[str] = None
    targetProfile: Profile
    threshold: Optional[float] = DEFAULT_THRESHOLD
    eps: Optional[float] = 8 / 255


class PGDAttackRequest(BaseModel):
    image: str
    targetProfileId: Optional[str] = None
    targetProfile: Profile
    threshold: Optional[float] = DEFAULT_THRESHOLD
    eps: Optional[float] = 8 / 255
    alpha: Optional[float] = 1 / 255
    steps: Optional[int] = 20


def decode_base64_image_to_file(data_url: str, dst_dir: str, filename: str) -> str:
    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url

    try:
        img_bytes = base64.b64decode(encoded)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid base64 image") from exc

    path = os.path.join(dst_dir, filename)

    with open(path, "wb") as f:
        f.write(img_bytes)

    return path


def extract_profile_images(profile: Profile, tmpdir: str, prefix: str) -> list[str]:
    paths = []

    if profile.captures:
        for key, value in profile.captures.items():
            if value and value.startswith("data:image"):
                paths.append(
                    decode_base64_image_to_file(
                        value,
                        tmpdir,
                        f"{prefix}_capture_{key}.jpg",
                    )
                )

    if profile.photos:
        for i, value in enumerate(profile.photos):
            if value and value.startswith("data:image"):
                paths.append(
                    decode_base64_image_to_file(
                        value,
                        tmpdir,
                        f"{prefix}_photo_{i}.jpg",
                    )
                )

    if profile.image and profile.image.startswith("data:image"):
        paths.append(
            decode_base64_image_to_file(
                profile.image,
                tmpdir,
                f"{prefix}_image.jpg",
            )
        )

    if len(paths) == 0:
        raise HTTPException(
            status_code=400,
            detail="Profile must include at least one base64 image in captures, photos, or image.",
        )

    return paths


@app.get("/")
def root():
    return {
        "message": "Face recognition and attack API is running",
        "device": DEVICE,
        "endpoints": [
            "POST /recognition",
            "POST /fgsm",
            "POST /pgd-impersonation",
        ],
    }


@app.post("/recognition")
def recognition(req: RecognitionRequest):
    with tempfile.TemporaryDirectory() as tmpdir:
        probe_path = decode_base64_image_to_file(req.image, tmpdir, "probe.jpg")

        enrollment_paths = extract_profile_images(
            req.referenceProfile,
            tmpdir,
            "reference",
        )

        return run_recognition_demo(
            model=MODEL,
            device=DEVICE,
            threshold=req.threshold,
            probe_img=probe_path,
            enrollment_imgs=enrollment_paths,
            matched_profile_id=req.referenceProfile.id,
            matched_name=req.referenceProfile.name,
        )


@app.post("/fgsm")
def fgsm_attack(req: FGSMAttackRequest):

    print("=== FGSM ENDPOINT HIT ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        attacker_path = decode_base64_image_to_file(req.image, tmpdir, "attacker.jpg")

        target_paths = extract_profile_images(
            req.targetProfile,
            tmpdir,
            "target",
        )

        return run_fgsm_false_reject_demo(
            model=MODEL,
            device=DEVICE,
            threshold=req.threshold,
            eps=req.eps,
            probe_img=attacker_path,
            enrollment_imgs=target_paths,
            profile_id=req.targetProfile.id,
            profile_name=req.targetProfile.name,
)


@app.post("/pgd-impersonation")
def pgd_impersonation(req: PGDAttackRequest):

    print("=== PGD ENDPOINT HIT ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        attacker_path = decode_base64_image_to_file(req.image, tmpdir, "attacker.jpg")

        target_paths = extract_profile_images(
            req.targetProfile,
            tmpdir,
            "target",
        )
        print("PGD API SETTINGS")
        print("threshold:", req.threshold)
        print("eps:", req.eps)
        print("alpha:", req.alpha)
        print("steps:", req.steps)
        print("target name:", req.targetProfile.name)
        print("target id:", req.targetProfile.id)
        print("num target paths:", len(target_paths))
        print("target paths:", target_paths)
        print("attacker path:", attacker_path)

        return run_pgd_impersonation_demo(
            model=MODEL,
            device=DEVICE,
            threshold=req.threshold,
            eps=req.eps,
            alpha=req.alpha,
            steps=req.steps,
            attacker_img=attacker_path,
            target_enrollment_imgs=target_paths,
            target_profile_id=req.targetProfile.id,
            target_name=req.targetProfile.name,
        )
    