# PyTorch CNN Model for Image Classification - CIFAR-10 Dataset
## Comparing Batch Norm vs Group Norm vs Layer Norm

This directory is organised as:
- Batch-Norm
- Group-Norm
- Layer-Norm

Each of the directory has the _ipynb_ along with _model.py_ and _utils.py_ which is the working implementation of the specific Norm on CIFAR-10 dataset

## Comparision

BN vs GN vs LN

![image](https://github.com/vmistry-repo/AI-Novice/assets/12965753/ee62d63e-5099-4d23-81d3-57871d69a920)

Batch normalization is a method that normalizes activations in a network across the mini-batch of definite size. For each feature, batch normalization computes the mean and variance of that feature in the mini-batch. It then subtracts the mean and divides the feature by its mini-batch standard deviation.

Layer normalization normalizes input across the features instead of normalizing input features across the batch dimension in batch normalization.
A mini-batch consists of multiple examples with the same number of features. Mini-batches are matrices(or tensors) where one axis corresponds to the batch and the other axis(or axes) correspond to the feature dimensions.

Group Normalization normalizes over group of channels for each training examples. When we put all the channels into a single group, group normalization becomes Layer normalization.

## Targets
To achieve ~70% test accuracy under 50k params, while comparing BN/GN/LN

## Results

### Using BN:
_Best Train Accuracy_: 68.80
_Best Test Accuracy_:  70.75
```
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
```
Accuracy/Loss Graph
![image](https://github.com/vmistry-repo/AI-Novice/assets/12965753/f2feb571-a148-40fe-8995-bd988c6bf348)

### Using GN:
_Best Train Accuracy_: 60.18
_Best Test Accuracy_:  63.97%
```
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
```
Accuracy/Loss Graph
![image](https://github.com/vmistry-repo/AI-Novice/assets/12965753/5da863a2-be0f-4b6c-83df-f2a88ffaaba8)

### Using LN:
_Best Train Accuracy_: 53
_Best Test Accuracy_:  57.04%
```
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
           Conv2d-11            [-1, 8, 14, 14]             288
             ReLU-12            [-1, 8, 14, 14]               0
        LayerNorm-13            [-1, 8, 14, 14]           3,136
          Dropout-14            [-1, 8, 14, 14]               0
           Conv2d-15            [-1, 8, 12, 12]             576
             ReLU-16            [-1, 8, 12, 12]               0
        LayerNorm-17            [-1, 8, 12, 12]           2,304
          Dropout-18            [-1, 8, 12, 12]               0
           Conv2d-19            [-1, 8, 12, 12]             576
             ReLU-20            [-1, 8, 12, 12]               0
        LayerNorm-21            [-1, 8, 12, 12]           2,304
          Dropout-22            [-1, 8, 12, 12]               0
           Conv2d-23            [-1, 4, 12, 12]              32
        MaxPool2d-24              [-1, 4, 6, 6]               0
           Conv2d-25             [-1, 10, 6, 6]             360
             ReLU-26             [-1, 10, 6, 6]               0
        LayerNorm-27             [-1, 10, 6, 6]             720
          Dropout-28             [-1, 10, 6, 6]               0
           Conv2d-29             [-1, 10, 6, 6]             900
             ReLU-30             [-1, 10, 6, 6]               0
        LayerNorm-31             [-1, 10, 6, 6]             720
          Dropout-32             [-1, 10, 6, 6]               0
           Conv2d-33             [-1, 10, 6, 6]             900
             ReLU-34             [-1, 10, 6, 6]               0
        LayerNorm-35             [-1, 10, 6, 6]             720
          Dropout-36             [-1, 10, 6, 6]               0
        AvgPool2d-37             [-1, 10, 1, 1]               0
           Conv2d-38             [-1, 10, 1, 1]             100
================================================================
Total params: 27,376
Trainable params: 27,376
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 0.39
Params size (MB): 0.10
Estimated Total Size (MB): 0.51
----------------------------------------------------------------
```
Accuracy/Loss Graph
![image](https://github.com/vmistry-repo/AI-Novice/assets/12965753/5da863a2-be0f-4b6c-83df-f2a88ffaaba8)

## Analysis

_Batch Normalization (BN)_: BN has been shown to be effective in improving the performance and stability of network. It normalized the activations of each layer over the entire batch, which  reduces the effect of internal covariate shift and improve generalization performance. In CIFAR-10, BN can help to reduce overfitting and improve the accuracy of the model, especially on larger batch sizes.

_Group Normalization (GN)_: GN is an alternative to BN that performs normalization across groups of channels instead of the entire batch. It has been shown to be effective on smaller batch sizes and when the data has strong spatial or channel-wise correlations. For our purposes we have kept the batch size same to see the diffrence and reduced the Channels to adhere 50k params constraint.

_Layer Normalization (LN)_: LN normalizes the activations of each layer over the features dimension (i.e., the channels), which can make it well-suited for sequential data or tasks where the data has a strong temporal or spatial structure. In CIFAR-10, as of now we don't have similar kind of data sorted out, and we observed the performance. Also, LN did not perform as well as BN or GN on larger batch sizes or when the data is less structured, hence has lowest accuracy too in our case

## Misclassification Images

We have misclassification of images in each case. Following is the snapshot from the model using BN

![image](https://github.com/vmistry-repo/AI-Novice/assets/12965753/8129594b-b4d9-4bc5-a425-8bd3264a3068)

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
