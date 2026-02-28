# ACE-ProtoNet

**Adaptive Covariance Eigen-Gate and Uncertainty-Aware Prototype Learning for Coronary Artery Segmentation**

This repository contains the **official PyTorch implementation** of our paper:

> **ACE-ProtoNet: Adaptive Covariance Eigen-Gate and Uncertainty-Aware Prototype Learning for Coronary Artery Segmentation**

---


## ✨ Key Features

* End-to-end 3D segmentation framework
* Covariance-driven structural gating mechanism
* Uncertainty-aware prototype learning
* Fully reproducible training and evaluation pipeline

---

## 🛠️ Requirements

The codebase is implemented in **Python (≥3.8)** using **PyTorch**.

We strongly recommend creating a virtual environment.

### Install Dependencies

```bash
# Install PyTorch (adjust CUDA version if necessary)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install additional dependencies
pip install numpy SimpleITK tqdm scikit-learn
```

---

## 📂 Data Preparation

As used in `train.py` and `test.py`, the preprocessed CCTA data should be organized in the following structure.

All volumes are expected in **NumPy (`.npy`) format**.

```
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

### Directory Description

* `img/` — Preprocessed CCTA volumes
* `mask/` — Corresponding ground-truth coronary artery masks

⚠️ **Important:**
Each image–mask pair **must share the same filename**.
For example:

```
img/1.npy  ↔  mask/1.npy
```

---

### 🔀 Dataset Splitting

A dataset splitting utility is provided:

```
datasets/create_folder.py
```

This script:

* Automatically generates the required folder structure
* Splits the dataset into training / validation / testing subsets
* Ensures reproducibility of experimental results

---

## 🚀 Usage

### 1️⃣ Training

To train ACE-ProtoNet on your prepared dataset:

```bash
python train.py
```

---

### 2️⃣ Evaluation / Testing

To evaluate a trained model on the test set:

```bash
python test.py
```

---
## 🔎 Post-processing

To further improve segmentation quality and suppress small false-positive regions, we provide simple yet effective post-processing utilities in:

```
postprocess/
├── get_patch.py
└── keep_the_largest_area.py
```

### 1️⃣ `keep_the_largest_area.py`

Removes small disconnected components from the predicted segmentation and retains only the largest connected region.

This is particularly useful for coronary artery segmentation, where small isolated predictions may appear due to noise or uncertainty in low-contrast regions.

**Purpose:**

* Eliminate false-positive fragments
* Improve structural consistency
* Enhance quantitative evaluation stability

---

### 2️⃣ `get_patch.py`

Extracts local patches from volumetric predictions for further refinement or analysis.

---

## 📦 Pre-trained Weights

Due to GitHub file size limitations, pretrained checkpoints are hosted externally.

| Dataset   | Checkpoint             | Download                                                                                              |
| --------- | ---------------------- | ----------------------------------------------------------------------------------------------------- |
| **ASOCA** | `model_best_model.ptk` | [🔗 Google Drive](https://drive.google.com/file/d/1unZwue8W2pGoleUawu-85CrCMlljtw7T/view?usp=sharing) |

### Usage Instructions

1. Download the checkpoint file
2. Place it in:

```
./checkpoints/
```

3. Run `test.py` for evaluation

---

## 📊 Reproducibility

To ensure reproducibility:

* Use identical data preprocessing
* Maintain consistent file naming conventions
* Verify CUDA / PyTorch compatibility

For exact experimental settings, please refer to the paper.


---

## 📧 Contact

If you encounter any issues or have questions:

* Please open an issue in this repository
* Or contact the authors via  caixia_dong@xjtu.edu.cn

We sincerely appreciate your interest in our work.
If this repository is helpful, a ⭐ star would be greatly appreciated!
