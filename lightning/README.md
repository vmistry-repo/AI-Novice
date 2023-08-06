# PyTorch CNN Model for Image Classification - CIFAR-10 Dataset
## Target - To use Pytorch Lightning to train CustomResnet Model

## Description

This is the same model that we had used in Cifar/Accuracy-85%. 
For details about the model please refer [README](https://github.com/vmistry-repo/AI-Novice/blob/main/Cifar/Accuracy-85%25/README.md)


### Benifits of using _Pytorch Lightning_

_Code Organization and Readability_: PyTorch Lightning enforces a clear and organized structure for your training code by separating out different components like data loading, model definition, training loop, and evaluation. This leads to cleaner and more readable code, making it easier to collaborate with others and maintain your projects.

_Automatic Gradient Accumulation and Optimization_: PyTorch Lightning handles gradient accumulation and optimization automatically. It takes care of accumulating gradients over multiple batches if you need to use larger batch sizes for memory reasons, simplifying the process of gradient accumulation.

_Standardized Training Loop_: With PyTorch Lightning, you don't need to write explicit training and validation loops. The framework provides a standardized training loop that includes batching, forward and backward passes, loss computation, and optimization. This reduces boilerplate code and lets you focus on model architecture and hyperparameters.

_Integration with Experiment Tracking Tools_: PyTorch Lightning integrates well with popular experiment tracking tools like TensorBoard and Neptune, allowing you to visualize and analyze training metrics, losses, and other relevant information. This makes it easier to monitor model performance and make informed decisions during training.

_Automatic Distributed Training and Multi-GPU Support_: Distributed training across multiple GPUs or machines is simplified in PyTorch Lightning. The framework handles details like data parallelism and synchronization across devices, making it easier to scale up your training to leverage more computational resources.

_Support for Advanced Training Techniques_: PyTorch Lightning provides built-in support for advanced training techniques like mixed-precision training, early stopping, learning rate scheduling, and more. These features help improve training efficiency and convergence.

## Targets
To achieve ~85% test accuracy under 200K params, while using _Pytorch Lightning_
## Results

- _Best Train Accuracy_: 89.31 <br>
```
   | Name           | Type               | Params
-------------------------------------------------------
0  | preplayer      | Sequential         | 1.9 K 
1  | preplayer.0    | Conv2d             | 1.7 K 
2  | preplayer.1    | BatchNorm2d        | 128   
3  | preplayer.2    | ReLU               | 0     
4  | preplayer.3    | Dropout            | 0     
5  | conv1_layer1   | Sequential         | 74.0 K
6  | conv1_layer1.0 | Conv2d             | 73.7 K
7  | conv1_layer1.1 | MaxPool2d          | 0     
8  | conv1_layer1.2 | BatchNorm2d        | 256   
9  | conv1_layer1.3 | ReLU               | 0     
10 | conv1_layer1.4 | Dropout            | 0     
11 | resblock1      | Sequential         | 295 K 
12 | resblock1.0    | Sequential         | 147 K 
13 | resblock1.0.0  | Conv2d             | 147 K 
14 | resblock1.0.1  | BatchNorm2d        | 256   
15 | resblock1.0.2  | ReLU               | 0     
16 | resblock1.0.3  | Dropout            | 0     
17 | resblock1.1    | Sequential         | 147 K 
18 | resblock1.1.0  | Conv2d             | 147 K 
19 | resblock1.1.1  | BatchNorm2d        | 256   
20 | resblock1.1.2  | ReLU               | 0     
21 | resblock1.1.3  | Dropout            | 0     
22 | conv2_layer2   | Sequential         | 295 K 
23 | conv2_layer2.0 | Conv2d             | 294 K 
24 | conv2_layer2.1 | MaxPool2d          | 0     
25 | conv2_layer2.2 | BatchNorm2d        | 512   
26 | conv2_layer2.3 | ReLU               | 0     
27 | conv2_layer2.4 | Dropout            | 0     
28 | conv3_layer3   | Sequential         | 1.2 M 
29 | conv3_layer3.0 | Conv2d             | 1.2 M 
30 | conv3_layer3.1 | MaxPool2d          | 0     
31 | conv3_layer3.2 | BatchNorm2d        | 1.0 K 
32 | conv3_layer3.3 | ReLU               | 0     
33 | conv3_layer3.4 | Dropout            | 0     
34 | resblock2      | Sequential         | 4.7 M 
35 | resblock2.0    | Sequential         | 2.4 M 
36 | resblock2.0.0  | Conv2d             | 2.4 M 
37 | resblock2.0.1  | BatchNorm2d        | 1.0 K 
38 | resblock2.0.2  | ReLU               | 0     
39 | resblock2.0.3  | Dropout            | 0     
40 | resblock2.1    | Sequential         | 2.4 M 
41 | resblock2.1.0  | Conv2d             | 2.4 M 
42 | resblock2.1.1  | BatchNorm2d        | 1.0 K 
43 | resblock2.1.2  | ReLU               | 0     
44 | resblock2.1.3  | Dropout            | 0     
45 | maxpool        | MaxPool2d          | 0     
46 | fc             | Linear             | 5.1 K 
47 | model          | Sequential         | 6.6 M 
48 | accuracy       | MulticlassAccuracy | 0     
-------------------------------------------------------
6.6 M     Trainable params
0         Non-trainable params
6.6 M     Total params
26.292    Total estimated model params size (MB)
```

## Accuracy/Loss Graph

_Note_: This won;t be visible in .ipynb because it is an JavaScript object as now we have used the Tensorboard to plot them, please see image below for reference.

![image](https://github.com/vmistry-repo/AI-Novice/assets/12965753/2dc045f3-e432-4c62-beb8-fba88ec29ce9)

![image](https://github.com/vmistry-repo/AI-Novice/assets/12965753/c531af40-4c9f-43a9-8429-dc5e62d1997e)

## Analysis

Pytorch lightning allows to focus more on the Data Augmentation and Model Creation, and takes care of the Training and validation logic, which speeds up the development.

## Misclassified Images

We have details of the failed validation test cases shown below. Following is the snapshot from the model training itself

![image](https://github.com/vmistry-repo/AI-Novice/assets/12965753/5510b1ef-3a9c-4c1c-a3a9-c0790443745e)

## HuggingFace Space 

Link: https://huggingface.co/spaces/VMistry/S12

You can play around with this app, which uses the same Model and its weights saved as part of this training.

It does following
- Upload and classify Images into classes (max upto: 10 as Cifar10 dataset)
- See x number of Misclassified Images (Max: 50)
- See x number of GradCAM Images (Max: 50)

_Note_: These Max limits are set to get the outputs faster and play around with multiple options available

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
