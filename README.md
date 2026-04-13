<div align="center">

# 🧠 Brain Tumor Identification from MRI Images using YOLOv7

[![Python](https://img.shields.io/badge/Python-3.9-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![YOLOv7](https://img.shields.io/badge/Model-YOLOv7-FF0000?style=for-the-badge)](https://github.com/WongKinYiu/yolov7)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Deep Learning](https://img.shields.io/badge/Domain-Deep%20Learning-00B4D8?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)]()
[![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)]()

**An end-to-end, real-time brain tumor detection system using YOLOv7 on MRI scans.**

</div>

---

## 📌 Overview

Brain tumors are among the most critical neurological conditions where early and accurate diagnosis can significantly improve patient outcomes. Manual interpretation of MRI scans is time-consuming, prone to human error, and depends heavily on radiologist expertise.

This project presents a **deep learning-powered, automated detection pipeline** that uses **YOLOv7** — a state-of-the-art real-time object detection model — to **localize and classify brain tumor regions** in MRI images. The system delivers high accuracy at fast inference speeds, making it suitable for clinical decision support.

> **mAP@0.5: 93.4% | Precision: 95.1% | Recall: 92.7% | F1-Score: 93.9%**

---

## 🚀 Key Features

- **Real-time Detection** — Fast YOLOv7 inference suitable for immediate clinical use
- **High Accuracy** — 93.4% mAP on MRI brain tumor benchmark dataset
- **Binary Classification** — Tumor / Non-Tumor detection with precise bounding boxes
- **End-to-End Pipeline** — From raw MRI input to annotated output in one script (`vasu.py`)
- **Google Colab Ready** — Designed to run on free GPU cloud environments
- **CUDA Accelerated** — Optimized for GPU training and inference with CUDA 11.6
- **Reproducible** — Complete training, evaluation, and inference commands provided

---

## 🛠️ Tech Stack

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.9 | Core programming language |
| PyTorch | 1.13.0 | Deep learning framework |
| YOLOv7 | Latest | Object detection model |
| OpenCV | 4.x | Image loading and preprocessing |
| Google Colab | — | Cloud GPU training environment |
| CUDA | 11.6 | GPU acceleration |

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| **mAP@0.5** | **93.4%** |
| **Precision** | **95.1%** |
| **Recall** | **92.7%** |
| **F1-Score** | **93.9%** |

- Effectively detects small and irregularly shaped tumor regions
- Minimal false positives and false negatives on real-world MRI scans
- Real-time inference speed suitable for clinical decision support tools

---

## 🧬 Model Architecture — YOLOv7

```
Input MRI Image (640x640)
        |
        v
┌─────────────────────────┐
│  Backbone: CSPDarknet53 │  ← Spatial + contextual feature extraction
└─────────────────────────┘
        |
        v
┌─────────────────────────┐
│   Neck: FPN + PAN       │  ← Multi-scale feature aggregation
└─────────────────────────┘
        |
        v
┌─────────────────────────┐
│  Head: Detection Head   │  ← Bounding boxes + confidence + class scores
└─────────────────────────┘
        |
        v
Output: Tumor / Non-Tumor (with bounding box + confidence score)
```

**Additional YOLOv7 Modules:**
- E-ELAN (Extended Efficient Layer Aggregation Network)
- Re-parameterized convolutional layers for a strong speed-accuracy tradeoff

---

## 📁 Project Structure

```
Braintumouridentification/
├── vasu.py               # Complete end-to-end pipeline (20 steps: setup → train → evaluate → infer)
├── data.yaml             # YOLOv7 dataset configuration (paths, class names)
├── requirements.txt      # Python package dependencies
├── README.md             # Full project documentation
└── .gitignore            # Git ignore rules
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Google Colab (recommended) or Ubuntu/Linux system
- GPU with CUDA 11.6 support (T4, A100, or local NVIDIA GPU)
- Google Drive with dataset uploaded at the expected path

### Step 1 — Clone YOLOv7

```bash
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
```

### Step 2 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3 — Install PyTorch with CUDA 11.6

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 \
  --extra-index-url https://download.pytorch.org/whl/cu116
```

---

## 🏋️ Training

```bash
python train.py \
  --img 512 \
  --batch 8 \
  --epochs 50 \
  --data /content/drive/MyDrive/MB/data.yaml \
  --weights yolov7.pt \
  --device 0
```

---

## 📐 Evaluation

```bash
python test.py \
  --data /content/drive/MyDrive/MB/data.yaml \
  --weights runs/train/exp/weights/best.pt \
  --img 640 \
  --task test \
  --verbose
```

---

## 🔍 Inference

```bash
python detect.py \
  --weights runs/train/exp/weights/best.pt \
  --source /content/drive/MyDrive/MB/test/images \
  --img 640 \
  --conf 0.25 \
  --iou 0.45 \
  --save-txt
```

---

## 📦 Dataset

| Property | Details |
|---|---|
| **Source** | Kaggle — Labeled MRI Brain Tumor Dataset |
| **Link** | [ammarahmed310/labeled-mri-brain-tumor-dataset](https://www.kaggle.com/datasets/ammarahmed310/labeled-mri-brain-tumor-dataset) |
| **Classes** | Tumor, Non-Tumor |
| **Annotation Format** | YOLO format (normalized bounding boxes) |
| **Preprocessing** | Resize to 640×640, normalize, augment |

---

## 🔮 Future Scope

- [ ] Extend to multi-class tumor type classification (glioma, meningioma, pituitary)
- [ ] Integrate 3D MRI volumetric analysis for depth-aware detection
- [ ] Deploy as a web or mobile application for remote clinical access
- [ ] Upgrade to YOLOv8 or hybrid transformer architectures for improved precision
- [ ] Add real-time DICOM file support for direct hospital system integration

---

## 📚 References

1. W. K. Yiu et al., "YOLOv7: Trainable Bag-of-Freebies", arXiv:2207.02696, 2022
2. A. Ahmed, "Labeled MRI Brain Tumor Dataset", Kaggle, 2021
3. S. Deepak & P. M. Ameer, "Brain Tumor Classification Using Deep CNN", Biomedical Signal Processing, 2022
4. B. H. Menze et al., "The Multimodal Brain Tumor Segmentation Benchmark (BRATS)", IEEE Trans. Medical Imaging, 2015
5. G. Litjens et al., "A Survey on Deep Learning in Medical Image Analysis", Medical Image Analysis, 2017

---

## 👨‍💻 Developer

| Field | Details |
|---|---|
| **Name** | Mulamreddy Venkata Vasu Deva Reddy |
| **Reg. No** | U22CN027 |
| **GitHub** | [@MVVasudevreddy](https://github.com/MVVasudevreddy) |
| **LinkedIn** | [venkata-vasu-deva-reddy-mulamreddy-6666vdr](https://www.linkedin.com/in/venkata-vasu-deva-reddy-mulamreddy-6666vdr) |
| **Institution** | Bharath Institute of Higher Education and Research (BIHER), Chennai |
| **Department** | Computer Science and Engineering |
| **Guide** | Dr. G. Rosline Nesa Kumari, Professor, Dept. of CSE, BIHER |
| **Project Type** | Mini Project — B.Tech CSE (2022–2026) |

---

<div align="center">

**⭐ If this project was helpful, consider giving it a star — it keeps the motivation going!**

Department of Computer Science and Engineering | BIHER, Chennai

</div>
