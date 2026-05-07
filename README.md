# AdversarialAttackOnArcface
A white-box adversarial attack research project focused on evaluating the robustness of an ArcFace-based face authentication system against FGSM and PGD attacks. This project implements a full biometric verification pipeline using ArcFace embeddings, template-based authentication, Equal Error Rate (EER) thresholding, and adversarial attacks including false reject and targeted impersonation attacks. The project also includes a FastAPI backend for integration with a frontend authentication system. Much of the experimental setup, metrics, and attack evaluation process documented here comes from the project notes and experiments in the capstone design documentation.

Features
ArcFace (iresnet50 backbone) face embeddings
Template-based face authentication
Cosine similarity verification
Equal Error Rate (EER) threshold computation
FGSM false reject attacks
PGD false reject attacks
PGD targeted impersonation attacks
FastAPI backend integration
GPU acceleration with CUDA
Manual attack testing using custom images
System Overview

The authentication pipeline works by:

Preprocessing input face images
Extracting 512-dimensional ArcFace embeddings
L2-normalizing embeddings
Building authentication templates from enrollment images
Computing cosine similarity between probe images and templates
Comparing similarity against a threshold

The ArcFace model used in this project is:

ArcFace iresnet50
MS1M-V3 pretrained weights
PyTorch implementation from InsightFace

The project specifically uses the PyTorch version of ArcFace to allow white-box gradient-based attacks.

Repository Structure
.
├── baseline_pairs.py
├── baseline_lfw_kaggle.py
├── compute_eer.py
├── demo_api_utils.py
├── face_auth.py
├── fastapi_app.py
├── fgsm_false_reject.py
├── pgd_false_reject.py
├── pgd_targeted_impersonation.py
├── smoke_test_torch_arcface.py
├── verify_two_images.py
├── results/
├── insightface/
└── weights/
Environment Setup
Create Conda Environment
conda create -n arcface python=3.11
conda activate arcface
Install Dependencies
pip install torch torchvision torchaudio
pip install opencv-python numpy pandas scikit-learn matplotlib fastapi uvicorn requests
GPU Support

This project was tested using:

NVIDIA RTX 5080
CUDA-enabled PyTorch 2.10
Model Weights

Download ArcFace pretrained weights:

ms1mv3_arcface_r50_fp16.pth

Place the weights file inside:

weights/

The weights file is intentionally excluded from GitHub because it exceeds GitHub’s file size limit.

Smoke Test

Verify the model loads correctly and gradients work:

python smoke_test_torch_arcface.py \
MyFace.jpg \
weights/ms1mv3_arcface_r50_fp16.pth

Expected output:

embedding shape: (1, 512)
grad exists? True
Basic Face Verification

Compare two images directly:

python verify_two_images.py \
MyFace.jpg \
OtherFace.jpg \
weights/ms1mv3_arcface_r50_fp16.pth \
0.3703
Template-Based Authentication

Build an enrollment template from multiple images and authenticate a probe image:

python face_auth.py \
weights/ms1mv3_arcface_r50_fp16.pth \
0.4226 \
probe.jpg \
enroll1.jpg enroll2.jpg enroll3.jpg

This system:

averages multiple enrollment embeddings
normalizes the template
compares probe embeddings using cosine similarity

Compute Equal Error Rate (EER)

Run EER evaluation:

python compute_eer.py \
/home/eet/datasets/lfw-deepfunneled/lfw-deepfunneled \
weights/ms1mv3_arcface_r50_fp16.pth \
5 \
10

Example result:

EER = 0.0926
Threshold = 0.4520

Observed results:

Enrollment Images	EER	Threshold
3	0.1116	0.4226
5	0.0926	0.4520
7	0.0820	0.4597
9	0.0835	0.4691

Increasing enrollment size improved authentication robustness.

FGSM False Reject Attack

Run a single-step FGSM false reject attack:

python fgsm_false_reject.py \
/home/eet/datasets/lfw_df \
/home/eet/datasets/matchpairsDevTest.csv \
weights/ms1mv3_arcface_r50_fp16.pth \
results/baseline/metrics.json \
"0.005,0.01,0.02" \
300

FGSM attempts to lower cosine similarity between a genuine probe and its correct template, causing false rejection.

PGD False Reject Attack

Run iterative PGD false reject attacks:

python pgd_false_reject.py \
/home/eet/datasets/lfw-deepfunneled/lfw-deepfunneled \
weights/ms1mv3_arcface_r50_fp16.pth \
7 \
0.4597 \
0.00196078 \
0.00098039 \
3 \
200

Weak attack results:

Enrollment	Attack Success Rate
3	55.5%
5	52.5%
7	40.0%
9	41.5%

PGD Targeted Impersonation Attack

Run PGD impersonation attacks:

python pgd_targeted_impersonation.py \
/home/eet/datasets/lfw-deepfunneled/lfw-deepfunneled \
weights/ms1mv3_arcface_r50_fp16.pth \
5 \
0.4520 \
0.00196078 \
0.00098039 \
3 \
200

This attack attempts to modify an attacker image so that it matches another user’s template.

Example results:

Enrollment	Attack Success Rate
3	85.0%
5	53.0%
7	52.0%
9	49.5%
Manual Attack Testing

Manual PGD impersonation test:

python pgd_targeted_impersonation.py --manual \
weights/ms1mv3_arcface_r50_fp16.pth \
0.4226 \
0.00196078 \
0.00098039 \
3 \
probe.jpg \
enroll1.jpg enroll2.jpg enroll3.jpg

Example output:

Clean score = 0.2786
Adversarial score = 0.4398
Attack success = True

FastAPI Backend

Run the backend server:

uvicorn fastapi_app:app --reload

Available endpoints include:

/recognition
/fgsm
/pgd-impersonation

The API supports:

face authentication
FGSM false reject attacks
PGD targeted impersonation attacks
Important Notes
Current Limitations
No facial landmark alignment
Simple resize preprocessing only
Pose variation can reduce authentication stability
JPEG compression can weaken weaker PGD attacks

The frontend originally used multiple pose captures (left/right/down/etc.), which reduced template consistency. Using several straight-on captures produced more stable authentication results during testing.

Research Goals

This project explores:

robustness of ArcFace embeddings
biometric authentication vulnerabilities
adversarial machine learning
white-box attack effectiveness
enrollment size vs security tradeoffs
References
ArcFace: Additive Angular Margin Loss for Deep Face Recognition
InsightFace
Labeled Faces in the Wild (LFW)
Goodfellow et al. – Explaining and Harnessing Adversarial Examples
Madry et al. – Towards Deep Learning Models Resistant to Adversarial Attacks
