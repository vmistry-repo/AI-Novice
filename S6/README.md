# Part 1 - Neural Network README
This is a simple neural network with an input layer, one hidden layers, and an output layer. The network is designed to predict two target variables (t1 and t2) based on two input variables (i1 and i2). Please refer to the excel sheet - BackPropogation.xlsx

## Architecture
The architecture of the network is as follows:
- Input layer: Two input neurons (i1 and i2)
- Hidden layer : Two neurons (h1 and h2), each with two input connections (w1, w2, w3, w4)
- Output layer: Two neurons (o1 and o2), each with two input connections (w5, w6, w7, w8)

Sigmoid activation function (σ) appied to h1, h2, o1, o2 to make them a_h1, a_h2, a_o1, a_o2 (a signifies activation function)

## Training
The network is trained using backpropagation with gradient descent. <br>
The loss function used is the `½ * (expected_output - received_output)²` <br>
The training process involves adjusting the weights `(w1-w8)` to minimize the loss.<br> 
This is done by computing the gradient of the loss function with respect to each weight using the chain rule, and then using the gradients to update the weights in the direction that reduces the loss.

## Loss Function
`E_total`: This is the total loss of the network, which is the sum of the losses for both output neurons. The purpose of this loss function is to provide a measure of how well the network is performing overall on the given dataset. It is calculated as the sum of E1 and E2.

`E1`: This is the loss function used for the first output neuron. It measures the difference between the predicted value a_o1 and the target value t1. The purpose of this loss function is to provide a measure of how well the network is able to predict the first target variable based on the given input variables. It is calculated as half of the squared difference between a_o1 and t1.

`E2`: This is the loss function used for the second output neuron. It measures the difference between the predicted value a_o2 and the target value t2. The purpose of this loss function is to provide a measure of how well the network is able to predict the second target variable based on the given input variables. It is calculated as half of the squared difference between a_o2 and t2.

Note that all the loss functions are calculated using the squared difference between the predicted and target values. This is a common choice for regression problems, where the goal is to predict continuous variables. The use of the squared difference ensures that larger errors are penalized more heavily than smaller errors, which helps the network converge to a solution that minimizes the overall loss.

## Basic Formulas
```
h1 = w1*i1 + w2*i2
h2 = w3*i1 + w4*i2
a_h1 = σ(h1) = 1/(1 + exp(-h1))
a_h2 = σ(h2)
o1 = w5*a_h1 + w6*a_h2
o2 = w7*a_h1 + w8*a_h2
a_o1 = σ(o1)
a_o2 = σ(o2)
E_total = E1 + E2
E1 = ½ * (t1 - a_o1)²
E2 = ½ * (t2 - a_o2)²
```

## Propogating Loss - Formula
This is the chain explaination w.r.t w5 (same will be followed for w6, w7 and w8)

`∂E_total/∂w5 = ∂(E1 + E2)/∂w5`: This equation represents the gradient of the total loss with respect to the weight `w5`. The total loss `E_total` is the sum of `E1` and `E2`, so the gradient of E_total with respect to `w5` is the same as the gradient of `E1` with respect to `w5`. Because `E2` is not dependent on `w5`.

`∂E1/∂a_o1 = (a_o1 - t1)`: This equation represents the gradient of `E1` with respect to the output activation `a_o1`. It measures how much the loss changes as the output of the first output neuron changes. The gradient is simply the difference between the predicted value `a_o1` and the target value `t1`.

`∂a_o1/∂o1 = a_o1 * (1 - a_o1)`: This equation represents the gradient of the output activation `a_o1` with respect to the input `o1`. It measures how much the output activation changes as the input to the activation function changes. For the sigmoid activation function used in this network, the gradient is simply the output activation `a_o1` multiplied by `(1 - a_o1)`.

`∂o1/∂w5 = a_h1`: This equation represents the gradient of the input to the second output neuron o1 with respect to the weight `w5`. It measures how much the input to the second output neuron changes as the weight `w5` changes. The gradient is simply the output activation of the first hidden neuron `a_h1`.

Summing it all up w.r.t `w5` we will get: 
```
∂E_total/∂w5 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h1
```

Same goes for w6, w7, w8
```
∂E_total/∂w6 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h2
∂E_total/∂w7 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h1
∂E_total/∂w8 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h2
```
Extending the same for `w1` to `w4` we will get 
```
∂E_total/∂w1 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i1
∂E_total/∂w2 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i2
∂E_total/∂w3 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i1
∂E_total/∂w4 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i2
```
These equations are used in the backpropagation algorithm to update the weights of the network during training. The gradient of the loss with respect to each weight is computed using these equations, and then the weights are updated in the direction that reduces the loss. The process is repeated iteratively until the loss converges to a minimum value.


## Excel Calculations
![image](https://github.com/vmistry-repo/AI-Novice/assets/12965753/2e3fec89-1665-44b4-8a9f-6d3d91480b73)

The snapshot above is from the excel sheet - BackPropogation.xlsx
This is an Interactive excel sheet, which shows the effects of LR on the Loss function. 
The snapshot above is with
```
- LR = 1   - Learning Rate
- t1 = 0.5 - Expected target 1
- t2 = 0.5 - Expected target 2
```

`w1 - w8` are weights which gets updated with each iteration
`i1, i2, h1, h2, o1, o2` are layer neurons, `a_h1, a_h2, a_o1, a_o2` are neurons after applying activation function sigmoid

### LR effect on Loss and a_o1 and a_o2
![ezgif-5-6c37a715ce](https://github.com/vmistry-repo/AI-Novice/assets/12965753/2d543988-3709-422f-8a19-610b40007800)

The learning rate is a hyperparameter that controls the step size of the weight updates during training. A high learning rate can cause the weights to update quickly, but it can also cause the loss to fluctuate and potentially diverge. A low learning rate can result in slow convergence and may require more epochs to reach an acceptable level of accuracy.

Here is what happens when the LR is set to different values:

`LR = 0.1`: This is a relatively low learning rate, which can result in slow convergence but can also help prevent the loss from fluctuating too much. The network may require more epochs to reach an acceptable level of accuracy, but once it does, it is likely to produce stable and reliable results.

`LR = 0.2`: This is a slightly higher learning rate, which can help the network converge more quickly than a LR of 0.1. However, if the LR is too high, the loss may still fluctuate and the network may not converge to an optimal solution.

`LR = 0.5`: This is a moderate learning rate, which can help the network converge relatively quickly without causing the loss to fluctuate too much. This is a good starting point for many neural networks, but the optimal LR may vary depending on the specific problem and dataset.

`LR = 0.8`: This is a higher learning rate, which can help the network converge more quickly than a LR of 0.5. However, if the LR is too high, the loss may start to fluctuate and the network may not converge to an optimal solution.

`LR = 1`: This is a very high learning rate, which can cause the loss to fluctuate rapidly and potentially diverge. This is generally not recommended, as it can prevent the network from converging to an optimal solution.

`LR = 2`: This is an extremely high learning rate, which is likely to cause the loss to diverge and prevent the network from learning anything useful. This is not recommended and should be avoided.

In general, the optimal learning rate for a neural network depends on the specific problem and dataset, and it may require some experimentation to find the best value. A good approach is to start with a moderate LR and adjust it based on the observed convergence and fluctuations in the loss.

# Part 2 - S6 model README

This is a PyTorch implementation of a Convolutional Neural Network (CNN) for image classification. The network architecture consists of multiple convolutional layers followed by fully connected layers. The network is trained to classify images from the MNIST dataset, which contains handwritten digits.

## Architecture
The network consists of the following layers:

- Convolutional layer with 32 filters, kernel size of 3x3, and ReLU activation function.
- Batch normalization layer to normalize the output of the convolutional layer.
- Dropout layer to prevent overfitting.
- Convolutional layer with 16 filters, kernel size of 1x1, and ReLU activation function.
- Batch normalization layer to normalize the output of the convolutional layer.
- Dropout layer to prevent overfitting.
- Convolutional layer with 32 filters, kernel size of 3x3, and ReLU activation function.
- Batch normalization layer to normalize the output of the convolutional layer.
- Dropout layer to prevent overfitting.
- Convolutional layer with 16 filters, kernel size of 1x1, and ReLU activation function.
- Batch normalization layer to normalize the output of the convolutional layer.
- Dropout layer to prevent overfitting.
- Max pooling layer with a kernel size of 2x2.
- ReLU activation function.
- Batch normalization layer to normalize the output of the max pooling layer.
- Convolutional layer with 16 filters, kernel size of 3x3, and ReLU activation function.
- Batch normalization layer to normalize the output of the convolutional layer.
- Dropout layer to prevent overfitting.
- Convolutional layer with 32 filters, kernel size of 3x3, and ReLU activation function.
- Batch normalization layer to normalize the output of the convolutional layer.
- Dropout layer to prevent overfitting.
- Convolutional layer with 16 filters, kernel size of 1x1, and ReLU activation function.
- Batch normalization layer to normalize the output of the convolutional layer.
- Dropout layer to prevent overfitting.
- Max pooling layer with a kernel size of 2x2.
- ReLU activation function.
- Batch normalization layer to normalize the output of the max pooling layer.
- Fully connected layer with 256 neurons and ReLU activation function.
- Fully connected layer with 10 neurons and log softmax activation function.

## Training
The network is trained using the cross-entropy loss function and the SGD optimizer. The learning rate is set to 0.01 and the batch size is set to 64. The network is trained for 20 epochs.
Total parameters used in the model is as below. Dimension of 28x28 convolution keeps updaing as per the Conv2d below.
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             288
              ReLU-2           [-1, 32, 26, 26]               0
       BatchNorm2d-3           [-1, 32, 26, 26]              64
           Dropout-4           [-1, 32, 26, 26]               0
            Conv2d-5           [-1, 16, 26, 26]             512
              ReLU-6           [-1, 16, 26, 26]               0
       BatchNorm2d-7           [-1, 16, 26, 26]              32
           Dropout-8           [-1, 16, 26, 26]               0
            Conv2d-9           [-1, 32, 24, 24]           4,608
             ReLU-10           [-1, 32, 24, 24]               0
      BatchNorm2d-11           [-1, 32, 24, 24]              64
          Dropout-12           [-1, 32, 24, 24]               0
           Conv2d-13           [-1, 16, 24, 24]             512
             ReLU-14           [-1, 16, 24, 24]               0
      BatchNorm2d-15           [-1, 16, 24, 24]              32
          Dropout-16           [-1, 16, 24, 24]               0
        MaxPool2d-17           [-1, 16, 12, 12]               0
             ReLU-18           [-1, 16, 12, 12]               0
      BatchNorm2d-19           [-1, 16, 12, 12]              32
           Conv2d-20           [-1, 16, 10, 10]           2,304
             ReLU-21           [-1, 16, 10, 10]               0
      BatchNorm2d-22           [-1, 16, 10, 10]              32
          Dropout-23           [-1, 16, 10, 10]               0
           Conv2d-24             [-1, 32, 8, 8]           4,608
             ReLU-25             [-1, 32, 8, 8]               0
      BatchNorm2d-26             [-1, 32, 8, 8]              64
          Dropout-27             [-1, 32, 8, 8]               0
           Conv2d-28             [-1, 16, 8, 8]             512
             ReLU-29             [-1, 16, 8, 8]               0
      BatchNorm2d-30             [-1, 16, 8, 8]              32
          Dropout-31             [-1, 16, 8, 8]               0
        MaxPool2d-32             [-1, 16, 4, 4]               0
             ReLU-33             [-1, 16, 4, 4]               0
      BatchNorm2d-34             [-1, 16, 4, 4]              32
           Linear-35                   [-1, 10]           2,560
           Linear-36                   [-1, 10]             100
================================================================
Total params: 16,388                                                            <<<< This shows total parameters used in this model
Trainable params: 16,388
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 2.04
Params size (MB): 0.06
Estimated Total Size (MB): 2.10
----------------------------------------------------------------
```

## Evaluation
The performance of the network is evaluated using the test set from the MNIST dataset. The accuracy of the network on the test set is reported as below per epoch.

```
Epoch 1
Train: Loss=0.1756 Batch_id=234 Accuracy=80.53: 100%|██████████| 235/235 [01:41<00:00,  2.33it/s]
Test set: Average loss: 0.0634, Accuracy: 9808/10000 (98.08%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 2
Train: Loss=0.1393 Batch_id=234 Accuracy=97.31: 100%|██████████| 235/235 [01:41<00:00,  2.32it/s]
Test set: Average loss: 0.0432, Accuracy: 9867/10000 (98.67%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 3
Train: Loss=0.1740 Batch_id=234 Accuracy=98.08: 100%|██████████| 235/235 [01:41<00:00,  2.31it/s]
Test set: Average loss: 0.0369, Accuracy: 9886/10000 (98.86%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 4
Train: Loss=0.1632 Batch_id=234 Accuracy=98.25: 100%|██████████| 235/235 [01:43<00:00,  2.27it/s]
Test set: Average loss: 0.0309, Accuracy: 9902/10000 (99.02%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 5
Train: Loss=0.2257 Batch_id=234 Accuracy=98.46: 100%|██████████| 235/235 [01:40<00:00,  2.35it/s]
Test set: Average loss: 0.0278, Accuracy: 9915/10000 (99.15%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 6
Train: Loss=0.1446 Batch_id=234 Accuracy=98.60: 100%|██████████| 235/235 [01:41<00:00,  2.32it/s]
Test set: Average loss: 0.0239, Accuracy: 9930/10000 (99.30%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 7
Train: Loss=0.1484 Batch_id=234 Accuracy=98.64: 100%|██████████| 235/235 [01:41<00:00,  2.31it/s]
Test set: Average loss: 0.0240, Accuracy: 9927/10000 (99.27%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 8
Train: Loss=0.1209 Batch_id=234 Accuracy=98.81: 100%|██████████| 235/235 [01:40<00:00,  2.34it/s]
Test set: Average loss: 0.0225, Accuracy: 9928/10000 (99.28%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 9
Train: Loss=0.1443 Batch_id=234 Accuracy=98.89: 100%|██████████| 235/235 [01:41<00:00,  2.31it/s]
Test set: Average loss: 0.0204, Accuracy: 9939/10000 (99.39%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 10
Train: Loss=0.1083 Batch_id=234 Accuracy=98.91: 100%|██████████| 235/235 [01:40<00:00,  2.34it/s]
Test set: Average loss: 0.0206, Accuracy: 9936/10000 (99.36%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 11
Train: Loss=0.1494 Batch_id=234 Accuracy=98.92: 100%|██████████| 235/235 [01:40<00:00,  2.34it/s]
Test set: Average loss: 0.0210, Accuracy: 9938/10000 (99.38%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 12
Train: Loss=0.1389 Batch_id=234 Accuracy=98.97: 100%|██████████| 235/235 [01:41<00:00,  2.30it/s]
Test set: Average loss: 0.0203, Accuracy: 9943/10000 (99.43%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 13
Train: Loss=0.1148 Batch_id=234 Accuracy=98.99: 100%|██████████| 235/235 [01:40<00:00,  2.33it/s]
Test set: Average loss: 0.0205, Accuracy: 9937/10000 (99.37%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 14
Train: Loss=0.0958 Batch_id=234 Accuracy=99.01: 100%|██████████| 235/235 [01:40<00:00,  2.33it/s]
Test set: Average loss: 0.0176, Accuracy: 9943/10000 (99.43%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 15
Train: Loss=0.1203 Batch_id=234 Accuracy=99.01: 100%|██████████| 235/235 [01:42<00:00,  2.30it/s]
Test set: Average loss: 0.0191, Accuracy: 9942/10000 (99.42%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 16
Train: Loss=0.1427 Batch_id=234 Accuracy=99.21: 100%|██████████| 235/235 [01:42<00:00,  2.29it/s]
Test set: Average loss: 0.0171, Accuracy: 9940/10000 (99.40%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 17
Train: Loss=0.1767 Batch_id=234 Accuracy=99.21: 100%|██████████| 235/235 [01:41<00:00,  2.30it/s]
Test set: Average loss: 0.0168, Accuracy: 9938/10000 (99.38%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 18
Train: Loss=0.1130 Batch_id=234 Accuracy=99.31: 100%|██████████| 235/235 [01:43<00:00,  2.27it/s]
Test set: Average loss: 0.0164, Accuracy: 9942/10000 (99.42%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 19
Train: Loss=0.1045 Batch_id=234 Accuracy=99.29: 100%|██████████| 235/235 [01:42<00:00,  2.29it/s]
Test set: Average loss: 0.0164, Accuracy: 9940/10000 (99.40%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 20
Train: Loss=0.1631 Batch_id=234 Accuracy=99.28: 100%|██████████| 235/235 [01:42<00:00,  2.29it/s]
Test set: Average loss: 0.0161, Accuracy: 9945/10000 (99.45%)                                           <<<<<< This shows the test accuracy at the End of 20th Epoch

Adjusting learning rate of group 0 to 1.0000e-03.
```

## Usage
Same details as above can be verified by loading S6.ipynb along with utils.py and model.py, and executing all the cells of the notebook.

## Requirements
The following packages are required to run the code:

- PyTorch
- NumPy
- Matplotlib
- Python 3.x
- torchvision
- Jupyter Notebook or JupyterLab
