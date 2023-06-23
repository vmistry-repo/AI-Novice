## PyTorch CNN Model for Image Classification - CIFAR-10 dataset
# Using Group Norm

Implementation has been divided into following files
- _model.py_
- _utils.py_
- _Cifar_10_GN.ipynb_

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
            Conv2d-5            [-1, 8, 28, 28]             288
              ReLU-6            [-1, 8, 28, 28]               0
         LayerNorm-7            [-1, 8, 28, 28]          12,544
           Dropout-8            [-1, 8, 28, 28]               0
            Conv2d-9            [-1, 4, 28, 28]              32
        MaxPool2d-10            [-1, 4, 14, 14]               0
           Conv2d-11           [-1, 10, 14, 14]             360
             ReLU-12           [-1, 10, 14, 14]               0
        LayerNorm-13           [-1, 10, 14, 14]           3,920
          Dropout-14           [-1, 10, 14, 14]               0
           Conv2d-15           [-1, 16, 12, 12]           1,440
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
Total params: 47,236
Trainable params: 47,236
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 0.60
Params size (MB): 0.18
Estimated Total Size (MB): 0.79
----------------------------------------------------------------

## Analysis

_Less sensitive to batch size_: GN is less sensitive to batch size than BN. BN requires a large batch size to compute accurate batch statistics, while GN can work well with smaller batch sizes. This makes GN more suitable for training models on limited computational resources or with small datasets.

_More stable on small batches_: GN is more stable than BN on small batches. BN may produce unstable statistics on small batches, which can negatively affect the training of the model. GN, on the other hand, normalizes each group of inputs separately, which can reduce the noise introduced by small batch sizes.

_Better generalization_: GN may lead to better generalization than BN. GN normalizes the inputs to each layer based on the statistics of each group, which can reduce the dependence of the network on the specific batch of inputs used during training. This can improve the generalization performance of the model.

_Lower memory requirements_: GN requires less memory than BN. BN requires storing the batch statistics for each layer during training, which can be memory-intensive for large models with many layers. GN, on the other hand, only requires storing the mean and variance of each group, which can reduce the memory requirements.

_Requires_higher_params_:In BN, the number of parameters required for each layer depends on the number of features or channels in the input, which is typically denoted as C. For each feature, BN requires two additional parameters: a scaling factor and a shift factor, which are learned during training. Therefore, the total number of parameters for a BN layer is 2C.

In GN, the number of parameters required for each layer depends on the number of groups of inputs, which is typically denoted as G. For each group, GN requires two additional parameters: a scaling factor and a shift factor, which are learned during training. Therefore, the total number of parameters for a GN layer is 2G.

The number of parameters in GN is higher than BN if the number of groups is larger than the number of features, i.e., G > C. This is because GN requires a separate set of parameters for each group, while BN only requires a single set of parameters for each feature.

_training_accuracy_: As compared to BN it decreases even though the Params count is increased.

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
