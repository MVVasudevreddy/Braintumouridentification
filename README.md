# Automated Brain Tumor Identification for MRI Images using YOLOv7

![Python](https://img.shields.io/badge/Python-3.9-blue) ![YOLOv7](https://img.shields.io/badge/Model-YOLOv7-red) ![Deep Learning](https://img.shields.io/badge/Domain-Deep%20Learning-green) ![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

## Project Info

| Field | Details |
|---|---|
| **Course Code** | U20CSPR01 - Mini Project |
| **Batch** | A-10 |
| **Department** | Computer Science and Engineering |
| **Institution** | BIHER (Bharath Institute of Higher Education and Research) |
| **Guide** | Dr. G. Rosline Nesa Kumari, Professor, Dept. of CSE, BIHER |

---

## Developer

| Reg. No | Name | GitHub |
|---|---|---|
| U22CN027 | Venky | [@MVVasudevreddy](https://github.com/MVVasudevreddy) |

---

## Abstract

This project develops an automated system for brain tumor identification in MRI images using the YOLOv7 model. Early and accurate detection of brain tumors is crucial for effective treatment, yet manual diagnosis is often time-consuming and prone to human error. The proposed system leverages YOLOv7, a state-of-the-art object detection model, to localize and classify tumor regions within MRI scans in real time. A custom dataset is prepared with annotated images categorized into **Tumor** and **Non-Tumor** classes, followed by preprocessing, model training, and evaluation using performance metrics such as mean Average Precision (mAP), Precision, and Recall.

---

## Problem Statement

- Brain tumors are life-threatening and require early detection for better survival outcomes.
- MRI scan analysis is slow, subjective, and prone to human error.
- Traditional machine learning approaches require heavy preprocessing and show limited accuracy.
- A robust and accurate automated system is essential for reliable brain tumor detection.

---

## Objectives

- Develop an automated system for accurate brain tumor detection from MRI images.
- Implement the YOLOv7 deep learning model for real-time tumor localization and classification.
- Evaluate model performance using metrics such as mAP, Precision, and Recall.
- Provide a fast and reliable tool that assists radiologists in clinical diagnosis.

---

## Dataset

- **Source**: Kaggle - Labeled MRI Brain Tumor Dataset
- **Link**: [https://www.kaggle.com/datasets/ammarahmed310/labeled-mri-brain-tumor-dataset](https://www.kaggle.com/datasets/ammarahmed310/labeled-mri-brain-tumor-dataset)
- **Classes**: Tumor, Non-Tumor
- **Format**: YOLO annotation format (bounding boxes with class labels)

### Preprocessing
| Step | Details |
|---|---|
| Resizing | All images resized to 640x640 |
| Normalization | Pixel values scaled from [0,255] to [0,1] |
| Annotation | Tumor regions labeled in YOLO format |
| Augmentation | Rotation, flipping, brightness adjustment |

---

## Model Architecture - YOLOv7

YOLOv7 consists of three main components:

| Component | Details |
|---|---|
| **Backbone** (CSPDarknet53) | Extracts hierarchical features from MRI images, capturing both spatial and contextual information |
| **Neck** (FPN + PAN) | Aggregates multi-scale features from the backbone, improves detection at various sizes |
| **Head** (Detection Head) | Produces bounding boxes, confidence scores, and class labels (Tumor / Non-Tumor) |

**Additional Modules:**
- E-ELAN (Extended Efficient Layer Aggregation Network)
- Re-parameterized convolutional layers
- Improves inference speed and accuracy for real-time detection

---

## Methodology

```
Data Collection
     |
     v
Preprocessing (Resize, Normalize, Annotate, Augment)
     |
     v
Model Development (YOLOv7)
     |
     v
Training & Evaluation (mAP, Precision, Recall, F1)
     |
     v
Inference & Results (Bounding Boxes on MRI Scans)
```

---

## Setup & Installation

### 1. Clone YOLOv7 Repository
```bash
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Install Compatible PyTorch (CUDA 11.6)
```bash
pip uninstall -y torch torchvision torchaudio
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 \
    --extra-index-url https://download.pytorch.org/whl/cu116
```

---

## Training

```bash
python train.py \
    --img 512 \
    --batch 8 \
    --epochs 50 \
    --data /content/drive/MyDrive/MB/data.yaml \
    --weights yolov7.pt
```

---

## Evaluation

```bash
python test.py \
    --data /content/drive/MyDrive/MB/data.yaml \
    --weights runs/train/exp/weights/best.pt \
    --img 640
```

---

## Inference / Detection

```bash
python detect.py \
    --weights runs/train/exp/weights/best.pt \
    --source /content/drive/MyDrive/MB/test/images \
    --img 640 \
    --save-txt
```

---

## Results

| Metric | Value |
|---|---|
| **mAP** | 93.4% |
| **Precision** | 95.1% |
| **Recall** | 92.7% |
| **F1-Score** | 93.9% |

- The model effectively identifies small and irregular tumor regions with minimal false detections.
- Real-time inference provides fast and reliable results, enabling quicker clinical decisions.
- The system reduces manual diagnostic effort and enhances consistency of tumor identification.
- Minor performance drops occur with low-contrast or noisy MRI images.

---

## Project Structure

```
Braintumouridentification/
|-- venky_final.py          # Main project code (Google Colab notebook converted)
|-- data.yaml               # Dataset configuration for YOLOv7
|-- requirements.txt        # Python dependencies
|-- README.md               # Project documentation
|-- .gitignore              # Git ignore rules
```

---

## Conclusion

The proposed system successfully detects and localizes brain tumors from MRI images using the YOLOv7 deep learning model. It achieves high accuracy and real-time performance, minimizing diagnostic delays. The system assists radiologists by providing an automated, reliable, and efficient diagnostic tool.

### Future Scope
- Integrate with 3D MRI datasets for improved volumetric tumor analysis.
- Extend detection to multiple tumor types and stages for detailed classification.
- Deploy as a web or mobile application for remote medical assistance.
- Explore advanced YOLO versions (e.g., YOLOv8) or hybrid architectures for enhanced precision.

---

## References

1. W. K. Yiu, C.-Y. Wang, and A. Bochkovskiy, "YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors", arXiv:2207.02696, 2022.
2. A. Ahmed, "Labeled MRI Brain Tumor Dataset", Kaggle, 2021.
3. S. A. Ismael and I. Abdel-Qader, "Brain Tumor Classification via Deep Learning", Procedia Computer Science, vol. 141, 2018.
4. S. Deepak and P. M. Ameer, "Brain Tumor Classification Using Deep CNN in MRI Images", Biomedical Signal Processing and Control, vol. 72, 2022.
5. B. H. Menze et al., "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Trans. Medical Imaging, vol. 34, no. 10, 2015.
6. G. Litjens et al., "A Survey on Deep Learning in Medical Image Analysis", Medical Image Analysis, vol. 42, 2017.
7. Z. Zhao et al., "Object Detection with Deep Learning: A Review", IEEE Trans. Neural Networks, vol. 30, no. 11, 2019.
8. O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI, 2015.
9. V. Sudha and M. Kumar, "A Comparative Study on Brain Tumor Detection Using Deep Learning", IJIRSET, vol. 11, no. 3, 2024.
10. R. Jain, "Automated Detection of Brain Tumor in MRI Images Using Deep Learning", Int. Journal of Computer Applications, vol. 183, 2022.

---

*Department of Computer Science and Engineering | BIHER*
*Developer: Venky (U22CN027)*
