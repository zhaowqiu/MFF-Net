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
Download pretrained weights [resnet50.pt]()  Run [train.py](https://github.com/zhaowqiu/MFF-Net/blob/main/train.py) after modifying the pre-training weight path.
  Log files are saved in the log folder
## Verification
Modify the test set path of the [vaildation.py](https://github.com/zhaowqiu/MFF-Net/blob/main/validation.py) file to test the performance of the trained model.  Modify the trained model file path
## src
This folder is used to store the network structure files used in the project. These files contain code that defines a machine learning or deep learning model, such as the architecture of a neural network, layer definitions, model configuration, and more. You can put these network structure files in the src directory for easy project use and management.  [backbone.py](https://github.com/zhaowqiu/MFF-Net/blob/main/src/backbone.py)resnet backbone network.  [mobilenet-backbone.py](https://github.com/zhaowqiu/MFF-Net/blob/main/src/mobilenet-backbone.py)mobilenetv2 backbone network.  [cbamblock.py](https://github.com/zhaowqiu/MFF-Net/blob/main/src/cbamblock.py)attention mechanism module.  .................
## save_weights
This folder is used to hold the model files used in the project. These files contain trained machine learning models, pretrained models, or other related model files. You can put these model files in the src directory for easy project use and management.
## train_utils
The `train_utils` folder contains utility functions and modules related to training machine learning models. It includes evaluation and validation functions along with commonly used evaluation metrics for model performance analysis.

### Evaluation and Validation Functions
The `train_utils` provides a set of functions for evaluating and validating machine learning models. These functions help in assessing the performance of the models on unseen data and verifying their generalization capabilities.

### Evaluation Metrics
In addition to evaluation functions, `train_utils` also offers a range of evaluation metrics commonly used in machine learning tasks. These metrics provide quantitative measures to assess the model's performance in various domains such as classification, regression, and clustering.

Some of the evaluation metrics included in `train_utils` are:
- Accuracy
- Precision
- Recall
- F1-score
- Mean Absolute Error (MAE)

Feel free to explore the files in the `train_utils` folder to utilize these evaluation and validation functions as well as the evaluation metrics in your machine learning projects.
## Heat map visualization

