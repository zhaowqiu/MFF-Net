# MFF-Net：Semantic Segmentation of Buildings in Remote Sensing Images
Semantic Segmentation Code for Buildings in Remote Sensing Images,This repository does not contain comparison test code.
## Datasets
We use the ISPRS Potsdam and Vaihingen [28] remote sensing datasets for experiments,Dataset download address:   
https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx
### Data description
The data Vaihingen set contains 33 patches (of different sizes), each consisting of a true orthophoto (TOP) extracted from a larger TOP mosaic, see Figure below and a DSM.  For further information about the original input data, please refer to the data description of the object detection and 3d reconstruction benchmark.  The data Potsdam set contains 38 patches (of the same size), each consisting of a true orthophoto (TOP) extracted from a larger TOP mosaic, see Figure below and a DSM. 
### Data preparation
After downloading the data set from the official website, split it according to the size of 520*520, and divide it into training set, verification set and test set according to 8:1:1.
Data image             |  Data label
:-------------------------:|:-------------------------:
<img src="https://github.com/zhaowqiu/MFF-Net/blob/main/pic/1.png" width="300" height="300" alt="数据集"/><br/>  |  <img src="https://github.com/zhaowqiu/MFF-Net/blob/main/pic/2.png" width="300" height="300" alt="数据集"/><br/>
## Environment
``torch==1.10.0  torchvision==0.11.1``  
Environment configuration reference [requirements.txt](https://github.com/zhaowqiu/MFF-Net/blob/main/requirements.txt) file.
## Training
Download pretrained weights [resnet50.pt]()  Run [train.py](https://github.com/zhaowqiu/MFF-Net/blob/main/train.py) after modifying the pre-training weight path
## Verification
Modify the test set path of the [vaildation.py](https://github.com/zhaowqiu/MFF-Net/blob/main/validation.py) file to test the performance of the trained model.  Modify the trained model file path
