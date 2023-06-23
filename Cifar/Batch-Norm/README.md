## PyTorch CNN Model for Image Classification - CIFAR-10 dataset
# Using Batch Norm

Implementation has been divided into following files
- _model.py_
- _utils.py_
- _Cifar_10_BN.ipynb_

## Requirements
- Python 3.x
- PyTorch
- torchvision
- Jupyter Notebook or JupyterLab

## Targets
To achieve ~70 % training and test accuracy under 50k params

## Model Structure and Params

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 30, 30]             432
              ReLU-2           [-1, 16, 30, 30]               0
       BatchNorm2d-3           [-1, 16, 30, 30]              32
           Dropout-4           [-1, 16, 30, 30]               0
            Conv2d-5           [-1, 32, 28, 28]           4,608
              ReLU-6           [-1, 32, 28, 28]               0
       BatchNorm2d-7           [-1, 32, 28, 28]              64
           Dropout-8           [-1, 32, 28, 28]               0
            Conv2d-9           [-1, 10, 28, 28]             320
        MaxPool2d-10           [-1, 10, 14, 14]               0
           Conv2d-11           [-1, 16, 14, 14]           1,440
             ReLU-12           [-1, 16, 14, 14]               0
      BatchNorm2d-13           [-1, 16, 14, 14]              32
          Dropout-14           [-1, 16, 14, 14]               0
           Conv2d-15           [-1, 16, 12, 12]           2,304
             ReLU-16           [-1, 16, 12, 12]               0
      BatchNorm2d-17           [-1, 16, 12, 12]              32
          Dropout-18           [-1, 16, 12, 12]               0
           Conv2d-19           [-1, 16, 12, 12]           2,304
             ReLU-20           [-1, 16, 12, 12]               0
      BatchNorm2d-21           [-1, 16, 12, 12]              32
          Dropout-22           [-1, 16, 12, 12]               0
           Conv2d-23           [-1, 10, 12, 12]             160
        MaxPool2d-24             [-1, 10, 6, 6]               0
           Conv2d-25             [-1, 16, 6, 6]           1,440
             ReLU-26             [-1, 16, 6, 6]               0
      BatchNorm2d-27             [-1, 16, 6, 6]              32
          Dropout-28             [-1, 16, 6, 6]               0
           Conv2d-29             [-1, 16, 6, 6]           2,304
             ReLU-30             [-1, 16, 6, 6]               0
      BatchNorm2d-31             [-1, 16, 6, 6]              32
          Dropout-32             [-1, 16, 6, 6]               0
           Conv2d-33             [-1, 16, 6, 6]           2,304
             ReLU-34             [-1, 16, 6, 6]               0
      BatchNorm2d-35             [-1, 16, 6, 6]              32
          Dropout-36             [-1, 16, 6, 6]               0
        AvgPool2d-37             [-1, 16, 1, 1]               0
           Conv2d-38             [-1, 10, 1, 1]             160
================================================================
Total params: 18,064
Trainable params: 18,064
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 1.58
Params size (MB): 0.07
Estimated Total Size (MB): 1.66
----------------------------------------------------------------

## Analysis

_Faster training_: Batch normalization speeds up the training of neural networks by reducing the internal covariate shift. The distribution of inputs to each layer of the network remains more stable during training, which can lead to faster and more stable convergence.

_Improved generalization_: Batch normalization also improves the generalization performance of neural networks by reducing overfitting. This is because it acts as a form of regularization by adding noise to the input of each layer, which can prevent the network from memorizing the training data too closely.

_Reduced sensitivity to initialization_: Batch normalization make neural networks less sensitive to the choice of initialization parameters. This means that the same network architecture can be trained successfully with different initialization parameters, which can make it easier to train complex models.

_Better gradient flow_: Batch normalization improve the flow of gradients through the network during backpropagation. This is because it normalizes the inputs to each layer, which can reduce the magnitude of the gradients and prevent them from vanishing or exploding.

_Increased stability_: Batch normalization increases the stability of the training process by reducing the likelihood of numerical instability. This is because it normalizes the inputs to each layer, which can prevent the values from becoming too large or too small and causing numerical issues.

## Usage

To run this notebook, you will need Jupyter Notebook or JupyterLab installed on your computer.
You can download and install Jupyter Notebook from the official website, or install it using a package manager like pip or conda.

Once you have Jupyter Notebook installed, you can open this notebook in the Jupyter Notebook interface by navigating to the directory where the notebook is saved and running the command:

Pull the repo into any jupiter notebook enabled environment, and execute the 
```
juniper notebook Cifar_10_BN.ipynb
```

This will open the notebook in your web browser, where you can run each code cell by clicking on the "Run" button or pressing Shift+Enter.

**Note:** The current code is not completely modular and is specifically designed to work with the CIFAR dataset.

## Acknowledgements
This project was inspired by the SchoolofAI tutorial on Image Classification and adapted to a simpler architecture for demonstration purposes.
