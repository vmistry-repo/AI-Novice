# Attempt-1
First attempt on the way to reach the goal to reduce the Model Parameters under 8k, and achieve minimum of 99.4% test accuracy, Under 15 EPOCH. Not adding the training dataset but can augment the existing ones from 60k MNIST Dataset.

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
To reduce the number of parameters under 8k and achieve 99% test accuracy under 10 EPOCH

## Results
Best Accuracy: 99.15
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 10, 26, 26]              90
              ReLU-2           [-1, 10, 26, 26]               0
       BatchNorm2d-3           [-1, 10, 26, 26]              20
           Dropout-4           [-1, 10, 26, 26]               0
            Conv2d-5           [-1, 10, 24, 24]             900
              ReLU-6           [-1, 10, 24, 24]               0
       BatchNorm2d-7           [-1, 10, 24, 24]              20
           Dropout-8           [-1, 10, 24, 24]               0
         MaxPool2d-9           [-1, 10, 12, 12]               0
           Conv2d-10           [-1, 10, 10, 10]             900
             ReLU-11           [-1, 10, 10, 10]               0
      BatchNorm2d-12           [-1, 10, 10, 10]              20
          Dropout-13           [-1, 10, 10, 10]               0
           Conv2d-14             [-1, 10, 8, 8]             900
             ReLU-15             [-1, 10, 8, 8]               0
      BatchNorm2d-16             [-1, 10, 8, 8]              20
          Dropout-17             [-1, 10, 8, 8]               0
           Conv2d-18             [-1, 10, 6, 6]             900
             ReLU-19             [-1, 10, 6, 6]               0
      BatchNorm2d-20             [-1, 10, 6, 6]              20
          Dropout-21             [-1, 10, 6, 6]               0
           Conv2d-22             [-1, 30, 4, 4]           2,700
             ReLU-23             [-1, 30, 4, 4]               0
      BatchNorm2d-24             [-1, 30, 4, 4]              60
          Dropout-25             [-1, 30, 4, 4]               0
        AvgPool2d-26             [-1, 30, 1, 1]               0
           Conv2d-27             [-1, 10, 1, 1]             300
================================================================
Total params: 6,850
Trainable params: 6,850
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.47
Params size (MB): 0.03
Estimated Total Size (MB): 0.50
----------------------------------------------------------------
```
## Analysis
- High number of Channels is not a way to increase MNIST accuracy for platform/parameter constraint environment.
- Need to have BN, Dropouts after each convolution block.
- AntMan(1x1) is our Channel DJ(mix and reducing channel by merging similar feature channels).

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
