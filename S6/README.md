# Neural Network README
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
