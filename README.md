# ACE-ProtoNet
We recommend using conda to create a virtual environment:

conda create -n casnet python=3.8
conda activate casnet
Install the required dependencies (we recommend creating a requirements.txt file based on your environment):

# Example dependencies, please modify according to your project
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy SimpleITK tqdm scikit-learn
Data Preparation
As seen in test.py or train.py, the preprocessed CCTA data should be organized with the following structure:

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
Usage
Training
Use the following command to start training the CAS-Net model. Please modify the parameters according to your setup.

python train.py 
Inference / Testing
Use the trained model weights to perform inference on the test set.

python test.py 
Acknowledgements
We would like to extend our special thanks to the authors of CS²-Net (DOI: 10.1016/j.media.2020.101874). Their work served as an important backbone for our study, and their public code repository is exceptionally well-structured.
