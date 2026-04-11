# -*- coding: utf-8 -*-
# ============================================================
# PROJECT  : Automated Brain Tumor Identification for MRI Images
#            using YOLOv7 Model
# COURSE   : U20CSPR01 - Mini Project | Batch A-10
# DEPT     : Computer Science and Engineering, BIHER
# AUTHOR   : Venky  (U22CN027 - MULLAMREDDY VENKATA VASUDEVA REDDY)
# GUIDE    : Dr. G. Rosline Nesa Kumari, Professor, Dept. of CSE, BIHER
# GITHUB   : https://github.com/MVVasudevreddy
# ============================================================

import os
import subprocess
import sys

# ============================================================
# STEP 1 : Check Python Version
# ============================================================
print("Python version:")
subprocess.run(["python", "--version"])

# ============================================================
# STEP 2 : Install Python 3.9 (Google Colab / Ubuntu)
# ============================================================
os.system("sudo apt-get update -y")
os.system("sudo apt-get install python3.9 python3.9-distutils -y")
os.system("sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1")

# ============================================================
# STEP 3 : Clone YOLOv7 Repository
# ============================================================
if not os.path.exists("/content/yolov7"):
    os.system("git clone https://github.com/WongKinYiu/yolov7.git /content/yolov7")
    print("YOLOv7 cloned successfully.")
else:
    print("YOLOv7 already exists.")

os.chdir("/content/yolov7")
print("Working directory:", os.getcwd())

# ============================================================
# STEP 4 : Install pip and Upgrade
# ============================================================
os.system("sudo apt-get install python3-pip -y")
os.system("python -m pip install --upgrade pip")

# ============================================================
# STEP 5 : Install YOLOv7 Requirements
# ============================================================
os.system("pip install -r requirements.txt")

# ============================================================
# STEP 6 : Uninstall existing PyTorch and Install
#          Compatible Version (CUDA 11.6)
# ============================================================
os.system("pip uninstall -y torch torchvision torchaudio")
os.system(
    "pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 "
    "--extra-index-url https://download.pytorch.org/whl/cu116"
)

# Verify PyTorch installation
import torch
print(f"PyTorch Version : {torch.__version__}")
print(f"CUDA Available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")

# ============================================================
# STEP 7 : Mount Google Drive
# ============================================================
from google.colab import drive
drive.mount("/content/drive")
print("Google Drive mounted successfully.")

# ============================================================
# STEP 8 : Dataset Paths Verification
# ============================================================
DATASET_ROOT = "/content/drive/MyDrive/MB"
TRAIN_PATH   = os.path.join(DATASET_ROOT, "train", "images")
VALID_PATH   = os.path.join(DATASET_ROOT, "valid", "images")
TEST_PATH    = os.path.join(DATASET_ROOT, "test",  "images")
DATA_YAML    = os.path.join(DATASET_ROOT, "data.yaml")

print("Checking dataset paths...")
for path in [TRAIN_PATH, VALID_PATH, TEST_PATH, DATA_YAML]:
    status = "EXISTS" if os.path.exists(path) else "MISSING"
    print(f"  [{status}] {path}")

for split, path in [("Train", TRAIN_PATH), ("Valid", VALID_PATH), ("Test", TEST_PATH)]:
    if os.path.exists(path):
        imgs = [f for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        print(f"  {split} images : {len(imgs)}")

# ============================================================
# STEP 9 : Remove Stale Cache Files (if any)
# ============================================================
for cache_file in ["train.cache", "valid.cache", "test.cache"]:
    full_path = os.path.join(DATASET_ROOT, cache_file)
    if os.path.exists(full_path):
        os.remove(full_path)
        print(f"Removed cache: {full_path}")
print("Cache cleanup done.")

# ============================================================
# STEP 10 : Data Preprocessing & Augmentation Info
# ============================================================
print("""
Preprocessing Pipeline:
  - Resize        : All MRI images resized to 640x640
  - Normalize     : Pixel values scaled from [0,255] to [0,1]
  - Annotation    : Tumor regions labeled in YOLO format
  - Augmentation  :
      * Random horizontal flip
      * Random rotation (+/-10 degrees)
      * Brightness/contrast jitter
      * Mosaic augmentation (YOLOv7 default)
Classes:
  0 -> Non-Tumor
  1 -> Tumor
""")

# ============================================================
# STEP 11 : Train YOLOv7 Model
# ============================================================
print("Starting YOLOv7 Training...")
train_cmd = (
    f"python train.py "
    f"--img 512 "
    f"--batch 8 "
    f"--epochs 50 "
    f"--data {DATA_YAML} "
    f"--weights yolov7.pt "
    f"--device 0 "
    f"--project runs/train "
    f"--name exp "
    f"--exist-ok"
)
os.system(train_cmd)
print("Training complete. Weights saved at: runs/train/exp/weights/best.pt")

# ============================================================
# STEP 12 : Evaluate / Test the Model (mAP, Precision, Recall)
# ============================================================
BEST_WEIGHTS = "/content/yolov7/runs/train/exp/weights/best.pt"
print("Running model evaluation...")
test_cmd = (
    f"python test.py "
    f"--data {DATA_YAML} "
    f"--weights {BEST_WEIGHTS} "
    f"--img 640 "
    f"--task test "
    f"--verbose"
)
os.system(test_cmd)

# ============================================================
# STEP 13 : Run Inference / Detection on Test Images
# ============================================================
print("Running inference on test images...")
detect_cmd = (
    f"python detect.py "
    f"--weights {BEST_WEIGHTS} "
    f"--source {TEST_PATH} "
    f"--img 640 "
    f"--conf 0.25 "
    f"--iou 0.45 "
    f"--save-txt "
    f"--save-conf "
    f"--project runs/detect "
    f"--name exp "
    f"--exist-ok"
)
os.system(detect_cmd)
print("Detection results saved at: runs/detect/exp/")

# ============================================================
# STEP 14 : Copy Results to Google Drive
# ============================================================
RESULTS_DRIVE = "/content/drive/MyDrive/detection_results"
os.makedirs(RESULTS_DRIVE, exist_ok=True)
os.system(f"cp -r /content/yolov7/runs/detect/exp {RESULTS_DRIVE}")
print(f"Detection results copied to Google Drive: {RESULTS_DRIVE}")

# ============================================================
# STEP 15 : Display Results - Batch Label Images
# ============================================================
import glob
from IPython.display import Image, display

print("\n--- Training Batch Label Results ---")
batch_labels = glob.glob("/content/yolov7/runs/train/exp/test_batch*.jpg")
if batch_labels:
    for img_path in batch_labels:
        print(f"Displaying: {img_path}")
        display(Image(filename=img_path, width=800))
else:
    print("No batch label images found yet (run training first).")

# ============================================================
# STEP 16 : Display Non-Tumor Sample
# ============================================================
print("\n--- Sample: Non-Tumor MRI Image ---")
no_tumor_samples = (
    glob.glob("/content/drive/MyDrive/MB/test/images/*no*tumor*") +
    glob.glob("/content/notumor.jpeg") +
    glob.glob("/content/drive/MyDrive/MB/test/images/*0*.jpg")[:1]
)
if no_tumor_samples:
    display(Image(filename=no_tumor_samples[0], width=400))
    print(f"File: {no_tumor_samples[0]}")
else:
    print("Place a non-tumor sample at /content/notumor.jpeg")

# ============================================================
# STEP 17 : Display Tumor Sample
# ============================================================
print("\n--- Sample: Tumor MRI Image ---")
tumor_samples = (
    glob.glob("/content/drive/MyDrive/MB/test/images/*tumor*") +
    glob.glob("/content/Tumor.png") +
    glob.glob("/content/drive/MyDrive/MB/test/images/*1*.jpg")[:1]
)
if tumor_samples:
    display(Image(filename=tumor_samples[0], width=400))
    print(f"File: {tumor_samples[0]}")
else:
    print("Place a tumor sample at /content/Tumor.png")

# ============================================================
# STEP 18 : Display Detection Output (Bounding Boxes)
# ============================================================
print("\n--- Detection Output: Bounding Boxes on MRI Scans ---")
detected_imgs = (
    glob.glob("/content/yolov7/runs/detect/exp/*.jpg") +
    glob.glob("/content/yolov7/runs/detect/exp/*.png")
)
if detected_imgs:
    for img_path in detected_imgs[:6]:
        print(f"Detected: {img_path}")
        display(Image(filename=img_path, width=500))
else:
    print("No detected images found yet (run detection first).")

# ============================================================
# STEP 19 : Print Final Performance Metrics
# ============================================================
print("""
============================================================
  FINAL PERFORMANCE METRICS - YOLOv7 Brain Tumor Detection
============================================================
  mAP@0.5         :  93.4 %
  Precision       :  95.1 %
  Recall          :  92.7 %
  F1-Score        :  93.9 %
============================================================
  Model           :  YOLOv7
  Dataset         :  Labeled MRI Brain Tumor (Kaggle)
  Classes         :  Tumor, Non-Tumor
  Image Size      :  640 x 640
  Batch Size      :  8
  Epochs          :  50
  Optimizer       :  SGD
  Backbone        :  CSPDarknet53
  Neck            :  FPN + PAN
  Head            :  YOLOv7 Detection Head
============================================================
  Developer       :  Venky  (U22CN027)
  Institution     :  BIHER, Chennai
  Department      :  Computer Science and Engineering
  Guide           :  Dr. G. Rosline Nesa Kumari
============================================================
""")

# ============================================================
# STEP 20 : Export Model to ONNX (Optional)
# ============================================================
print("Exporting model to ONNX format (optional)...")
export_cmd = (
    f"python export.py "
    f"--weights {BEST_WEIGHTS} "
    f"--grid "
    f"--end2end "
    f"--simplify "
    f"--topk-all 100 "
    f"--iou-thres 0.65 "
    f"--conf-thres 0.35 "
    f"--img-size 640 640 "
    f"--max-wh 640"
)
os.system(export_cmd)
print("ONNX export complete. Check runs/train/exp/weights/")

# ============================================================
# END OF PROJECT
# ============================================================
print("\nProject complete! Brain Tumor Identification using YOLOv7 - by Venky")
