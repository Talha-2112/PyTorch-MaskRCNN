# PyTorch-MaskRCNN

An easy-to-use **custom Mask R-CNN model** implemented in PyTorch.  
This repo contains a single Python file `Model.py` for training and validating Mask R-CNN on your own dataset.

---

## Features
- Hyperparameters are defined at the top of the code for easy customization
- Custom backbone support (default: ConvNeXt-base with FPN)
- Single-file, easy to import and use
- Custom dataset handling via class `CustomDataset`
- Supports checkpointing, logging, and mixed-precision training
- Calculates **training loss**, **validation loss**, **mAP**, **mAP@50**, and **IoU** metrics
- Required libraries are provided in the requirements.txt file

---

## IMPORTANT NOTE
Before using the model, the user **MUST** redefine the `CustomDataset` class to match their own annotation format.  
Additionally, the user may modify the model or backbone according to their requirements.  
For usage examples and hyperparameter settings, see `Model.py` under the `main()` function.
