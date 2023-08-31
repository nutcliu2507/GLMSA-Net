# GLMSA-Net
This repo implements GLMSA-net methods for semantic segmentation of remote sensing images on ISPRS datasets.
# Requirements
python=3.x   
tensorflow=2.5   
scikit-image   
scikit-learn   
tifffile    
tqdm   
# Usage
1. split_img.py -- Split images
2. train.py -- For model training
3. constants.py -- Train data & Test data setting
4. model.py -- Setting which Encoder & Decoder using
5. resnet_def.py -- Resnet
6. unet_def.py -- our decoder
7. Attention.py -- our Global Local Attention & Multi-Scale Attention module
8. loss.py -- Focal Loss
