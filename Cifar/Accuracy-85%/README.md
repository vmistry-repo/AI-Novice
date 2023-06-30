# PyTorch CNN Model for Image Classification - CIFAR-10 Dataset
## Target - Achieve 85% Test Validation accuracy under 200k parameters

This directory is organised as:
- model.py
- transform.py
- utils.py
- Cifar10.ipynb

## Description

The model consists of several convolutional blocks, each of which contains multiple convolutional layers with ReLU activation, batch normalization, and dropout. The convolutional blocks are connected in a residual fashion, where the output of each block is added to the output of the previous block. This helps to mitigate the vanishing gradient problem and enables the model to learn more complex features.

The output block of the model consists of a global average pooling layer, followed by a 1x1 convolutional layer that reduces the number of channels to the number of classes (10 in this case). The output is then passed through a log softmax activation function to produce the final class probabilities.

The code above modularizes the creation of each convolutional block and the output block, making it easier to modify the architecture by changing the parameters for each block or adding/removing blocks altogether.

Here model is using
- Regular Conv Block
- Dialted Conv Block
- Depthwise Searable Conv Block

### Benifits of using _Dialated Convolutions_

_Increased receptive field_: Dilated convolutions can increase the receptive field (i.e., the area of the input that influences a given output) without increasing the number of parameters or the computation required. This is because dilated convolutions use a sparse kernel with gaps (dilation) between the kernel elements, effectively increasing the size of the kernel.

![conv-dial](https://github.com/vmistry-repo/AI-Novice/assets/12965753/4c1f0f45-573a-493e-a53d-63ab33207eb9)

_Improved multiscale processing_: Dilated convolutions can be used to process an image at multiple scales, by varying the dilation rate. This allows the network to capture features at different scales, which can be useful for tasks such as object detection and segmentation.

_Reduced spatial resolution loss_: In traditional convolutional networks, the spatial resolution of the feature maps decreases as the layers become deeper. Dilated convolutions can reduce this spatial resolution loss by allowing the network to process larger receptive fields while preserving the spatial resolution.

_Better parameter efficiency_: Dilated convolutions can be used to increase the effective size of the convolutional kernel without increasing the number of parameters, which can improve the parameter efficiency of the network. Observe the Parameters below.

### Benifits of using _Depthwise Separable Convolution_

_Lower computational cost_: Depthwise Separable Convolution separates the spatial and channel-wise convolution operations into two separate layers. This reduces the number of parameters and computations required compared to traditional convolutional layers. As a result, models that use Depthwise Separable Convolution are faster and more memory-efficient.

_Better accuracy on small datasets_: Depthwise Separable Convolution can help improve the accuracy of models trained on small datasets. This is because it reduces the risk of overfitting by reducing the number of model parameters.

_Improved generalization_: Depthwise Separable Convolution can help improve the generalization performance of models by enabling them to learn more robust and discriminative features. This is because it allows the model to learn the spatial patterns and channel-wise interactions separately.

_Faster training_: Depthwise Separable Convolution can speed up training time by reducing the number of computations required. This allows models to be trained faster and with fewer resources.

## Targets
To achieve ~85% test accuracy under 200K params, while using _Dialated Convolutions_ and _Depthwise Separable Convolution_

## Results

- _Best Train Accuracy_: 79.29 <br>
- _Best Test Accuracy_:  85.84 <br>

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 30, 30]           1,728
              ReLU-2           [-1, 64, 30, 30]               0
       BatchNorm2d-3           [-1, 64, 30, 30]             128
           Dropout-4           [-1, 64, 30, 30]               0
            Conv2d-5          [-1, 128, 28, 28]           2,304
              ReLU-6          [-1, 128, 28, 28]               0
       BatchNorm2d-7          [-1, 128, 28, 28]             256
           Dropout-8          [-1, 128, 28, 28]               0
            Conv2d-9           [-1, 64, 28, 28]           8,192
             ReLU-10           [-1, 64, 28, 28]               0
      BatchNorm2d-11           [-1, 64, 28, 28]             128
          Dropout-12           [-1, 64, 28, 28]               0
           Conv2d-13          [-1, 128, 26, 26]           2,304
             ReLU-14          [-1, 128, 26, 26]               0
      BatchNorm2d-15          [-1, 128, 26, 26]             256
          Dropout-16          [-1, 128, 26, 26]               0
           Conv2d-17           [-1, 64, 26, 26]           8,192
             ReLU-18           [-1, 64, 26, 26]               0
      BatchNorm2d-19           [-1, 64, 26, 26]             128
          Dropout-20           [-1, 64, 26, 26]               0
           Conv2d-21          [-1, 128, 26, 26]           2,304
             ReLU-22          [-1, 128, 26, 26]               0
      BatchNorm2d-23          [-1, 128, 26, 26]             256
          Dropout-24          [-1, 128, 26, 26]               0
           Conv2d-25           [-1, 64, 26, 26]           8,192
             ReLU-26           [-1, 64, 26, 26]               0
      BatchNorm2d-27           [-1, 64, 26, 26]             128
          Dropout-28           [-1, 64, 26, 26]               0
           Conv2d-29          [-1, 128, 26, 26]           2,304
             ReLU-30          [-1, 128, 26, 26]               0
      BatchNorm2d-31          [-1, 128, 26, 26]             256
          Dropout-32          [-1, 128, 26, 26]               0
           Conv2d-33           [-1, 64, 26, 26]           8,192
             ReLU-34           [-1, 64, 26, 26]               0
      BatchNorm2d-35           [-1, 64, 26, 26]             128
          Dropout-36           [-1, 64, 26, 26]               0
           Conv2d-37          [-1, 128, 26, 26]           2,304
             ReLU-38          [-1, 128, 26, 26]               0
      BatchNorm2d-39          [-1, 128, 26, 26]             256
          Dropout-40          [-1, 128, 26, 26]               0
           Conv2d-41           [-1, 64, 26, 26]           8,192
             ReLU-42           [-1, 64, 26, 26]               0
      BatchNorm2d-43           [-1, 64, 26, 26]             128
          Dropout-44           [-1, 64, 26, 26]               0
           Conv2d-45          [-1, 128, 26, 26]           2,304
             ReLU-46          [-1, 128, 26, 26]               0
      BatchNorm2d-47          [-1, 128, 26, 26]             256
          Dropout-48          [-1, 128, 26, 26]               0
           Conv2d-49           [-1, 64, 26, 26]           8,192
             ReLU-50           [-1, 64, 26, 26]               0
      BatchNorm2d-51           [-1, 64, 26, 26]             128
          Dropout-52           [-1, 64, 26, 26]               0
           Conv2d-53          [-1, 128, 26, 26]           2,304
             ReLU-54          [-1, 128, 26, 26]               0
      BatchNorm2d-55          [-1, 128, 26, 26]             256
          Dropout-56          [-1, 128, 26, 26]               0
           Conv2d-57           [-1, 64, 26, 26]           8,192
             ReLU-58           [-1, 64, 26, 26]               0
      BatchNorm2d-59           [-1, 64, 26, 26]             128
          Dropout-60           [-1, 64, 26, 26]               0
           Conv2d-61          [-1, 128, 26, 26]           2,304
             ReLU-62          [-1, 128, 26, 26]               0
      BatchNorm2d-63          [-1, 128, 26, 26]             256
          Dropout-64          [-1, 128, 26, 26]               0
           Conv2d-65           [-1, 64, 26, 26]           8,192
             ReLU-66           [-1, 64, 26, 26]               0
      BatchNorm2d-67           [-1, 64, 26, 26]             128
          Dropout-68           [-1, 64, 26, 26]               0
           Conv2d-69          [-1, 128, 26, 26]           2,304
             ReLU-70          [-1, 128, 26, 26]               0
      BatchNorm2d-71          [-1, 128, 26, 26]             256
          Dropout-72          [-1, 128, 26, 26]               0
           Conv2d-73           [-1, 64, 26, 26]           8,192
             ReLU-74           [-1, 64, 26, 26]               0
      BatchNorm2d-75           [-1, 64, 26, 26]             128
          Dropout-76           [-1, 64, 26, 26]               0
        AvgPool2d-77             [-1, 64, 1, 1]               0
           Conv2d-78             [-1, 10, 1, 1]             640
================================================================
Total params: 100,416
Trainable params: 100,416
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 38.04
Params size (MB): 0.38
Estimated Total Size (MB): 38.43
----------------------------------------------------------------
```
### Accuracy/Loss Graph

![image](https://github.com/vmistry-repo/AI-Novice/assets/12965753/04c7577e-d2f0-426c-8b11-aed398d23e12)

## Analysis

If we were not to use _Depthwise Separable Convolution_ we would not be able to achieve the target due to parameter violation.

_Dialated Convolutions_ on the other hand helps to increase the spatial feature extraction, and increase the Receptive field rapidly.

If we have resource constraint environment, _Depthwise Separable Convolution_ are very useful.

## Misclassified Images

We have details of the failed validation test cases shown below. Following is the snapshot from the model training itself

![image](https://github.com/vmistry-repo/AI-Novice/assets/12965753/324a48ec-26ae-46ca-9c2c-9c8dcb1bb56e)

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
