# Attempt-2
Second attempt on the way to reach the goal to reduce the Model Parameters under 8k, and achieve minimum of 99.4% test accuracy, Under 15 EPOCH. Not adding the training dataset but can augment the existing ones from 60k MNIST Dataset.

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
To add the image transformation of removing RandomErasing, making training hard, to reduce overfitting

## Results
Best Accuracy: 99.07

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
            Conv2d-9            [-1, 5, 24, 24]              50
        MaxPool2d-10            [-1, 5, 12, 12]               0
           Conv2d-11           [-1, 10, 10, 10]             450
             ReLU-12           [-1, 10, 10, 10]               0
      BatchNorm2d-13           [-1, 10, 10, 10]              20
          Dropout-14           [-1, 10, 10, 10]               0
           Conv2d-15             [-1, 10, 8, 8]             900
             ReLU-16             [-1, 10, 8, 8]               0
      BatchNorm2d-17             [-1, 10, 8, 8]              20
          Dropout-18             [-1, 10, 8, 8]               0
           Conv2d-19             [-1, 20, 6, 6]           1,800
             ReLU-20             [-1, 20, 6, 6]               0
      BatchNorm2d-21             [-1, 20, 6, 6]              40
          Dropout-22             [-1, 20, 6, 6]               0
           Conv2d-23              [-1, 5, 6, 6]             100
           Conv2d-24             [-1, 20, 4, 4]             900
             ReLU-25             [-1, 20, 4, 4]               0
      BatchNorm2d-26             [-1, 20, 4, 4]              40
          Dropout-27             [-1, 20, 4, 4]               0
        AvgPool2d-28             [-1, 20, 1, 1]               0
           Conv2d-29             [-1, 10, 1, 1]             200
================================================================
Total params: 5,550
Trainable params: 5,550
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.49
Params size (MB): 0.02
Estimated Total Size (MB): 0.52
----------------------------------------------------------------

## Analysis
Image transformation does help reducing overfitting, but this doesn't always results in better model training, need to observe data type and then add from several transformations which suits the best

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
