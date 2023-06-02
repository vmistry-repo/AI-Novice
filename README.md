# AI-Novice
This repository is a collection of projects and code samples for beginners who are just starting to learn about artificial intelligence. As an AI novice and a pupil of School of AI, I created this repository to share my learning journey with others who are interested in AI. The code samples are written in Python uisng PyTorch framework. 

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

The CNN model is defined in the _model.py_ file and uses the following architecture:
- 4 convolutional layers with ReLU activation and max pooling
- 2 fully connected layers with ReLU activation and a log-softmax output

**Note:** The model takes in 28x28 grayscale images as input and outputs a probability distribution over 10 classes.

Utility functions are defined in _utils.py_
Description of functions is as
- **plot_loss_accuracy_graph**: A function that plots the training and testing loss and accuracy curves on a 2x2 grid.
- **get_model_summary**: A function that prints a summary of the given PyTorch model's architecture and parameters based on the specified input size.
- **test**: A function that evaluates a trained PyTorch model on a test dataset and reports the average loss and accuracy.
            It also appends the test accuracy and loss to the given lists of test accuracy and loss.
- **send_model_to_device**: A function that returns a new instance of the given PyTorch model that is moved to the specified device.
- **run_for_epoch**: A function that trains a PyTorch model for one epoch on a training dataset and evaluates its performance on a test dataset.
                     It also updates the learning rate using a scheduler. The training and testing accuracy and loss curves are appended to the given lists of accuracy and loss.
- **check_cuda**: A function that checks if a CUDA-enabled GPU is available on the system and returns a boolean value.
- **get_device**: A function that returns the device (CPU or CUDA) to be used for running PyTorch code based on the availability of a CUDA-enabled GPU.
- **GetCorrectPredCount**: A function that takes in predicted and target labels and returns the number of correct predictions.
- **train**: A function that trains a PyTorch model on a given training dataset and reports the training loss and accuracy for each batch.
             It also appends the training accuracy and loss to the given lists of training accuracy and loss.

_s5.ipynb_ is a jupiter notebook, perforing following steps
- Imports the required PyTorch libraries and custom modules such as model, utils, optim, datasets, and transforms.
- Defines the device (CPU or GPU) to run the model on using the utils.get_device() function.
- Defines the image dimensions x and y, the mean and standard deviation of the dataset (mean and std), and the batch size (batch_size).
- Defines the data transformations for the training and test datasets using the train_transforms and test_transforms Compose objects from the transforms module.
- Loads the MNIST training and test datasets using the datasets.MNIST() function and applies the respective data transformations.
- Defines the data loaders for the training and test datasets using the torch.utils.data.DataLoader() function.
- Plots a sample of 12 images from the training dataset using the matplotlib.pyplot library.
- Defines a convolutional neural network (CNN) model using the model.Net class.
- Sends the CNN model to the device defined earlier using the utils.send_model_to_device() function.
- Defines the optimizer and scheduler for training the CNN model using the optim.SGD() and optim.lr_scheduler.StepLR() functions.
- Executes the training and testing loop for 10 epochs using the utils.run_for_epoch() function, which trains the model on the training data, computes the accuracy on the test data,
  and logs the loss and accuracy for each epoch.
- Prints the summary of the CNN model using the utils.get_model_summary() function, which displays the number of parameters and the output shape of each layer in the model.
- Plots the loss and accuracy graphs using the utils.plot_loss_accuracy_graph() function.

## Usage

To run this notebook, you will need Jupyter Notebook or JupyterLab installed on your computer.
You can download and install Jupyter Notebook from the official website, or install it using a package manager like pip or conda.

Once you have Jupyter Notebook installed, you can open this notebook in the Jupyter Notebook interface by navigating to the directory where the notebook is saved and running the command:

Pull the repo into any jupiter notebook enabled environment, and execute the 
```
juniper notebook s5.ipynb
```

This will open the notebook in your web browser, where you can run each code cell by clicking on the "Run" button or pressing Shift+Enter.

**Note:** The current code is not completely modular and is specifically designed to work with the MNIST dataset. However, future updates will decouple the code and make it more generic, so that it can be used with any dataset and model.

## Acknowledgements
This project was inspired by the SchoolofAI tutorial on Image Classification and adapted to a simpler architecture for demonstration purposes.
