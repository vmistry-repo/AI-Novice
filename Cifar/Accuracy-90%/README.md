# PyTorch CNN Model for Image Classification - CIFAR-10 Dataset
## Target - Achieve 90% Test Validation accuracy under 24 EPOCH - Use OneCycleLR Policy and Residual Connections

This directory is organised as:
- model.py
- transform.py
- utils.py
- dataset.py
- Cifar10.ipynb

## Description

### About Model

This model follows Resnet alike architecture. It is defined as below
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
              ReLU-3           [-1, 64, 32, 32]               0
           Dropout-4           [-1, 64, 32, 32]               0
            Conv2d-5          [-1, 128, 32, 32]          73,728
         MaxPool2d-6          [-1, 128, 16, 16]               0
       BatchNorm2d-7          [-1, 128, 16, 16]             256
              ReLU-8          [-1, 128, 16, 16]               0
           Dropout-9          [-1, 128, 16, 16]               0
           Conv2d-10          [-1, 128, 16, 16]         147,456
      BatchNorm2d-11          [-1, 128, 16, 16]             256
             ReLU-12          [-1, 128, 16, 16]               0
          Dropout-13          [-1, 128, 16, 16]               0
           Conv2d-14          [-1, 128, 16, 16]         147,456
      BatchNorm2d-15          [-1, 128, 16, 16]             256
             ReLU-16          [-1, 128, 16, 16]               0
          Dropout-17          [-1, 128, 16, 16]               0
           Conv2d-18          [-1, 256, 16, 16]         294,912
        MaxPool2d-19            [-1, 256, 8, 8]               0
      BatchNorm2d-20            [-1, 256, 8, 8]             512
             ReLU-21            [-1, 256, 8, 8]               0
          Dropout-22            [-1, 256, 8, 8]               0
           Conv2d-23            [-1, 512, 8, 8]       1,179,648
        MaxPool2d-24            [-1, 512, 4, 4]               0
      BatchNorm2d-25            [-1, 512, 4, 4]           1,024
             ReLU-26            [-1, 512, 4, 4]               0
          Dropout-27            [-1, 512, 4, 4]               0
           Conv2d-28            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-29            [-1, 512, 4, 4]           1,024
             ReLU-30            [-1, 512, 4, 4]               0
          Dropout-31            [-1, 512, 4, 4]               0
           Conv2d-32            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-33            [-1, 512, 4, 4]           1,024
             ReLU-34            [-1, 512, 4, 4]               0
          Dropout-35            [-1, 512, 4, 4]               0
        MaxPool2d-36            [-1, 512, 1, 1]               0
           Linear-37                   [-1, 10]           5,120
================================================================
Total params: 6,573,120
Trainable params: 6,573,120
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 8.00
Params size (MB): 25.07
Estimated Total Size (MB): 33.09
----------------------------------------------------------------
```

## Targets
To achieve ~90% test accuracy under 24 EPOCHS, using the above defined architecture.
To use OneCycleLR Policy with following attributes and values:
```
Total Epochs = 24
Max at Epoch = 5
LRMIN = FIND
LRMAX = FIND
NO Annihilation
```
To use following transformations:
- RandomCrop 32, 32 (after padding of 4)
- FlipLR
- CutOut(8, 8)

## Results

- _Best Train Accuracy_: 98.01 <br>
- _Best Test Accuracy_:  92.72 <br>

Accuracy/Loss Graph
![image](https://github.com/vmistry-repo/AI-Novice/assets/12965753/2ff6033e-aa85-479e-bfc5-d885d048d006)

## Analysis

OneCycleLR is a learning rate scheduling technique used in deep learning that can help improve the performance and speed of model training.<br>
Here are some benefits of using OneCycleLR:

_Faster convergence_: OneCycleLR helps to speed up the convergence of deep learning models by allowing them to quickly find a good set of weights during training.

_Improved accuracy_: OneCycleLR can help improve the accuracy of deep learning models by allowing them to explore the parameter space more efficiently, leading to better generalization performance.

_Reduced overfitting_: OneCycleLR can help reduce overfitting of deep learning models by preventing the learning rate from becoming too large, which can cause the model to overfit to the training data.

Though we can see signs of overfitting which can be reduced with modifications in the Netowrk or Transformations.

## Misclassified Images

We have details of the failed validation test cases shown below. Following is the snapshot from the model training itself

![image](https://github.com/vmistry-repo/AI-Novice/assets/12965753/20481b14-b1e6-4b27-b0f3-d119bb7a7f61)

## Usage

To run this notebook, you will need Jupyter Notebook or JupyterLab installed on your computer.
You can download and install Jupyter Notebook from the official website, or install it using a package manager like pip or conda.

Once you have Jupyter Notebook installed, you can open this notebook in the Jupyter Notebook interface by navigating to the directory where the notebook is saved and running the command:

Pull the repo into any jupiter notebook enabled environment, and within specific directory execute the 
```
juniper notebook <ipynb Notebook>
```

This will open the notebook in your web browser, where you can run each code cell by clicking on the "Run" button or pressing Shift+Enter.

**Note:** The current code is not completely modular and is specifically designed to work with the MNIST dataset. However, future updates will decouple the code and make it more generic, so that it can be used with any dataset and model.

## Acknowledgements
This project was inspired by the SchoolofAI tutorial on Image Classification and adapted to a simpler architecture for demonstration purposes.
