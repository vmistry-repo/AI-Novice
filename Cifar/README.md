## PyTorch CNN Model for Image Classification - CIFAR-10 Dataset
# Comparing Batch Norm vs Group Norm vs Layer Norm

This directory is organised as:
- Batch-Norm
- Group-Norm
- Layer-Norm

## Comparision

BN vs GN vs LN


## Targets
To achieve ~70% test accuracy under 50k params, while comparing BN/GN/LN

## Results

Using BN:

Using GN:

Using LN:

## Analysis



## Misclassification Images

We have misclassification of images in each case. Following is the snapshot from the model using BN

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
