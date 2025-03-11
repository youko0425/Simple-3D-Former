# Simple-3D-Former
## Environment Setup
  1. Use anaconda to create a VM environment
     ```bash
         conda create -n Simple-3D-Former Python==3.7.0
         conda activate Simple-3D-Former
  3. Install the following packages
     ```bash
        pip install torch
        pip install transformers
        pip install matplotlib
        pip install pillow
        pip install mpl_toolkits
     
## Dataset Preparation
ModelNet40: Download it from here: https://modelnet.cs.princeton.edu/ModelNet40.zip

## How to run (Change the route for accessing dataset if needed)
```bash
    python preprocess.py
    python train.py
    python predict.py
    python test.py
