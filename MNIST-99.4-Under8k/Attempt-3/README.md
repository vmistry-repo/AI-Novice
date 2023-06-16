# Attempt-3
Achiving the target 
- 99.4% Test accuracy with consistency
- Less than 8k Model Parameters
- Used BN, Dropout at each ConvBlock
- Well Under 15 EPOCH achived the target 

## Requirements
- Python 3.x
- PyTorch
- torchvision
- Jupyter Notebook or JupyterLab

## PyTorch CNN Model for Image Classification - MNIST-Dataset

Implementation has been divided into following files
- _model.py_
- _utils.py_
- _s5.ipynb_

## Targets
To achieve 99.4 % test accuracy under 8k params

## Results
Best Accuracy: 99.42

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 10, 26, 26]              90
              ReLU-2           [-1, 10, 26, 26]               0
       BatchNorm2d-3           [-1, 10, 26, 26]              20
           Dropout-4           [-1, 10, 26, 26]               0
            Conv2d-5           [-1, 16, 24, 24]           1,440
              ReLU-6           [-1, 16, 24, 24]               0
       BatchNorm2d-7           [-1, 16, 24, 24]              32
           Dropout-8           [-1, 16, 24, 24]               0
            Conv2d-9           [-1, 10, 24, 24]             160
        MaxPool2d-10           [-1, 10, 12, 12]               0
           Conv2d-11           [-1, 10, 10, 10]             900
             ReLU-12           [-1, 10, 10, 10]               0
      BatchNorm2d-13           [-1, 10, 10, 10]              20
          Dropout-14           [-1, 10, 10, 10]               0
           Conv2d-15             [-1, 10, 8, 8]             900
             ReLU-16             [-1, 10, 8, 8]               0
      BatchNorm2d-17             [-1, 10, 8, 8]              20
          Dropout-18             [-1, 10, 8, 8]               0
           Conv2d-19             [-1, 13, 6, 6]           1,170
             ReLU-20             [-1, 13, 6, 6]               0
      BatchNorm2d-21             [-1, 13, 6, 6]              26
          Dropout-22             [-1, 13, 6, 6]               0
           Conv2d-23             [-1, 16, 4, 4]           1,872
             ReLU-24             [-1, 16, 4, 4]               0
      BatchNorm2d-25             [-1, 16, 4, 4]              32
          Dropout-26             [-1, 16, 4, 4]               0
        AvgPool2d-27             [-1, 16, 1, 1]               0
           Conv2d-28             [-1, 10, 1, 1]             160
================================================================
Total params: 6,842
Trainable params: 6,842
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.61
Params size (MB): 0.03
Estimated Total Size (MB): 0.64
----------------------------------------------------------------

## Analysis
Using ADAM with ReduceLROnPlateau, does help the model training atleast for MNIST compared to SGD on same model, further we have used squeez and expand as required. Also RF of the model above 20 is required to achieve such accuracy 

## Usage

To run this notebook, you will need Jupyter Notebook or JupyterLab installed on your computer.
You can download and install Jupyter Notebook from the official website, or install it using a package manager like pip or conda.

Once you have Jupyter Notebook installed, you can open this notebook in the Jupyter Notebook interface by navigating to the directory where the notebook is saved and running the command:

Pull the repo into any jupiter notebook enabled environment, and execute the 
```
juniper notebook s7.ipynb
```

This will open the notebook in your web browser, where you can run each code cell by clicking on the "Run" button or pressing Shift+Enter.

**Note:** The current code is not completely modular and is specifically designed to work with the MNIST dataset. However, future updates will decouple the code and make it more generic, so that it can be used with any dataset and model.

## Acknowledgements
This project was inspired by the SchoolofAI tutorial on Image Classification and adapted to a simpler architecture for demonstration purposes.
