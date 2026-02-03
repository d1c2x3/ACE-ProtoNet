# ACE-ProtoNet: Adaptive Covariance Eigen-Gate and Uncertainty-Aware Prototype Learning for Coronary Artery Segmentation

This repository contains the **official PyTorch implementation** of the paper: **"ACE-ProtoNet: Adaptive Covariance Eigen-Gate and Uncertainty-Aware Prototype Learning for Coronary Artery Segmentation"**.

## 📌 Overview

Accurate segmentation of coronary arteries from Coronary CT Angiography (CCTA) is essential for quantitative stenosis evaluation, plaque characterization, and surgical planning. However, automated segmentation faces significant challenges due to low vessel-to-background contrast, high anatomical variability, and complex, tree-like vascular morphology.

To address these challenges, we propose **ACE-ProtoNet**, a unified framework that couples an **Adaptive Covariance Eigen-Gate (ACE-Gate)** with an **Uncertainty-aware Prototype Learning Head (UPL-Head)** to achieve robust and accurate coronary artery segmentation.

This repository provides:
* Training and inference code for ACE-ProtoNet
* Data organization guidelines
* Reproducible experimental setup

---

## 🛠️ Requirements

The code is implemented in Python using PyTorch. We recommend using a virtual environment.

### Install Dependencies
```bash
# Install PyTorch (adjust cuda version if necessary)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other necessary libraries
pip install numpy SimpleITK tqdm scikit-learn
```

---

## 📂 Data Preparation

As used in `train.py` and `test.py`, the preprocessed CCTA data should be organized as follows. The data format is expected to be NumPy arrays (`.npy`).

```text
Data/
└── npy/
    ├── img/
    │   ├── 1.npy
    │   ├── 2.npy
    │   └── ...
    └── mask/
        ├── 1.npy
        ├── 2.npy
        └── ...
```

* **`img/`**: Preprocessed CCTA volumes saved as NumPy arrays.
* **`mask/`**: Corresponding ground-truth coronary artery masks.
* **Note**: Each image–mask pair **must share the same filename** (e.g., `1.npy` in `img/` corresponds to `1.npy` in `mask/`).

---

## 🚀 Usage

### 1. Training
To train the ACE-ProtoNet model on your prepared dataset, run the following command:

```bash
python train.py
```

### 2. Inference / Testing
To evaluate the trained model on the test set, run:

```bash
python test.py
```

---


## 📧 Contact
For any questions or issues, please open an issue in this repository.
