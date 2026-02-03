# ACE-ProtoNet

Official implementation of **ACE-ProtoNet**, a prototype-driven framework for coronary artery segmentation from coronary computed tomography angiography (CCTA).

---

## 📌 Overview

Accurate coronary artery segmentation is challenging due to complex vessel topology, small vessel branches, and ambiguous boundaries.  
ACE-ProtoNet addresses these challenges by introducing uncertainty-aware prototype learning into a deep encoder–decoder architecture, enabling robust representation of tubular structures in CCTA volumes.

This repository provides:
- Training and inference code for ACE-ProtoNet
- Data organization guidelines
- Reproducible experimental setup
- 
# Example dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy SimpleITK tqdm scikit-learn

## 📁 Data Preparation
As used in train.py and test.py, the preprocessed CCTA data should be organized as follows:
data
└── npy
    ├── img
    │   ├── 1.npy
    │   ├── 2.npy
    │   └── ...
    └── mask
        ├── 1.npy
        ├── 2.npy
        └── ...
img/: preprocessed CCTA volumes saved as NumPy arrays
mask/: corresponding ground-truth coronary artery masks
Each image–mask pair must share the same filename

##Training
python train.py

##Inference / Testing
python test.py

