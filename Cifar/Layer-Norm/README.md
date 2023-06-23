## PyTorch CNN Model for Image Classification - CIFAR-10 dataset
# Using Layer Norm

Implementation has been divided into following files
- _model.py_
- _utils.py_
- _Cifar_10_LN.ipynb_

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
            Conv2d-1            [-1, 4, 30, 30]             108
              ReLU-2            [-1, 4, 30, 30]               0
         LayerNorm-3            [-1, 4, 30, 30]           7,200
           Dropout-4            [-1, 4, 30, 30]               0
            Conv2d-5            [-1, 4, 28, 28]             144
              ReLU-6            [-1, 4, 28, 28]               0
         LayerNorm-7            [-1, 4, 28, 28]           6,272
           Dropout-8            [-1, 4, 28, 28]               0
            Conv2d-9            [-1, 4, 28, 28]              16
        MaxPool2d-10            [-1, 4, 14, 14]               0
           Conv2d-11           [-1, 16, 14, 14]             576
             ReLU-12           [-1, 16, 14, 14]               0
        LayerNorm-13           [-1, 16, 14, 14]           6,272
          Dropout-14           [-1, 16, 14, 14]               0
           Conv2d-15           [-1, 16, 12, 12]           2,304
             ReLU-16           [-1, 16, 12, 12]               0
        LayerNorm-17           [-1, 16, 12, 12]           4,608
          Dropout-18           [-1, 16, 12, 12]               0
           Conv2d-19           [-1, 16, 12, 12]           2,304
             ReLU-20           [-1, 16, 12, 12]               0
        LayerNorm-21           [-1, 16, 12, 12]           4,608
          Dropout-22           [-1, 16, 12, 12]               0
           Conv2d-23           [-1, 10, 12, 12]             160
        MaxPool2d-24             [-1, 10, 6, 6]               0
           Conv2d-25             [-1, 16, 6, 6]           1,440
             ReLU-26             [-1, 16, 6, 6]               0
        LayerNorm-27             [-1, 16, 6, 6]           1,152
          Dropout-28             [-1, 16, 6, 6]               0
           Conv2d-29             [-1, 16, 6, 6]           2,304
             ReLU-30             [-1, 16, 6, 6]               0
        LayerNorm-31             [-1, 16, 6, 6]           1,152
          Dropout-32             [-1, 16, 6, 6]               0
           Conv2d-33             [-1, 16, 6, 6]           2,304
             ReLU-34             [-1, 16, 6, 6]               0
        LayerNorm-35             [-1, 16, 6, 6]           1,152
          Dropout-36             [-1, 16, 6, 6]               0
        AvgPool2d-37             [-1, 16, 1, 1]               0
           Conv2d-38             [-1, 10, 1, 1]             160
================================================================
Total params: 44,236
Trainable params: 44,236
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 0.54
Params size (MB): 0.17
Estimated Total Size (MB): 0.72
----------------------------------------------------------------

## Analysis

_More stable on small batches_: LN is more stable than BN and GN on small batches. BN and GN compute batch statistics, which can produce unstable results on small batches. LN normalizes the inputs for each individual sample, which can reduce the noise introduced by small batch sizes.

_Less sensitive to batch size and distribution_: LN is less sensitive to batch size and distribution than BN and GN. BN requires a large batch size to compute accurate batch statistics, while GN requires the inputs to be divided into groups. LN does not require an assumption about the distribution of the inputs, making it more flexible and robust to different types of data.

_Better suited to sequential data_: LN is better suited to sequential data than BN and GN. BN assumes that the inputs to each layer are independent and identically distributed, which may not hold true for sequential data. GN normalizes the inputs based on the statistics of each group, which can be problematic for variable-length sequences. LN normalizes the inputs for each individual sample, making it more suited to sequential data.

_Better performance on small models_: LN can achieve better performance than BN and GN on small models. For models with a small number of parameters, BN and GN can introduce too much regularization, which can lead to underfitting. LN can provide a more appropriate amount of regularization, leading to better performance on small models.

## Usage

To run this notebook, you will need Jupyter Notebook or JupyterLab installed on your computer.
You can download and install Jupyter Notebook from the official website, or install it using a package manager like pip or conda.

Once you have Jupyter Notebook installed, you can open this notebook in the Jupyter Notebook interface by navigating to the directory where the notebook is saved and running the command:

Pull the repo into any jupiter notebook enabled environment, and execute the 
```
juniper notebook Cifar_10_GN.ipynb
```

This will open the notebook in your web browser, where you can run each code cell by clicking on the "Run" button or pressing Shift+Enter.

**Note:** The current code is not completely modular and is specifically designed to work with the CIFAR dataset.

## Acknowledgements
This project was inspired by the SchoolofAI tutorial on Image Classification and adapted to a simpler architecture for demonstration purposes.
