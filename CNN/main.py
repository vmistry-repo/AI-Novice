####################################################
############# IMPORTS #############

import cv2
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets
from torchvision import transforms
from torchsummary import summary
from torch_lr_finder import LRFinder
from torch.optim.lr_scheduler import OneCycleLR

import numpy as np

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Local Imports
import models.model as model 
import data.dataset as dataset
import utils.utils as utils
import trainers.transform as trans

####################################################
############# VARS #############

SEED = 42
'''
pixel_means
pixel_stds
train_transforms
test_transforms
trainset
train_loader
testset
test_loader
classes
criterion
scheduler
lr_finder
optimizer
'''
####################################################
############# CUDA #############

device = utils.get_device()
cuda = torch.cuda.is_available()

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)
    
####################################################
############# MEAN and STD #############

def set_pixel_mean_std(dataset):
    # Define the transform to normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    global pixel_means, pixel_stds

    # Load the CIFAR-10 dataset
    #cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    pixel_means = dataset.data.mean(axis=(0,1,2)) / 255.0
    pixel_stds = dataset.data.std(axis=(0,1,2)) / 255.0

    print('CIFAR-10 pixel means:', pixel_means)
    print('CIFAR-10 pixel stds:', pixel_stds)
    print(cifar10_train.data.shape)

####################################################
############# INIT ARGS #############

class args():
    def __init__(self, batch_size=512, device = 'cpu' ,use_cuda = False) -> None:
        self.batch_size = batch_size
        self.device = device
        self.use_cuda = use_cuda
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if self.use_cuda else {}
        
####################################################
############# SET TRANSFORMS #############

def set_transforms():
    global train_transforms, test_transforms
    train_transforms = trans.get_train_transforms(pixel_means, pixel_stds)
    test_transforms = trans.get_test_transforms(pixel_means, pixel_stds)
    return train_transforms, test_transforms

####################################################
############# SET LOADER #############

def set_loaders(trainset, testset):
    global train_loader, test_loader
    #trainset = dataset.Cifar10SearchDataset(root='./data', train=True,
    #                                download=True, transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args().batch_size,
                                              shuffle=True, **args().kwargs)

    #testset = dataset.Cifar10SearchDataset(root='./data', train=False,
    #                               download=True, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args().batch_size,
                                             shuffle=True, **args().kwargs)
                                         
####################################################
############# CLASSES #############

def set_classes(_classes):
    global classes
    classes = _classes
#classes = ('plane', 'car', 'bird', 'cat',
#           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

####################################################
############# VISUALIZE move to ipynb #############
#utils.visualise_dataset(train_loader, classes)        

####################################################
############# SUMMARY #############

def get_model_summary(model):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    cnnmodel = model.Net().to(device)
    summary(cnnmodel, input_size=(3, 32, 32))

####################################################
############# LOSS Fnq #############

def set_lossFn(loss):
    global criterion
    criterion = loss() #nn.CrossEntropyLoss()

####################################################
############# OPTIMIZER #############

def set_optim(optimizerFnq, model, lr=0.03, weight_decay=1e-5):
    global optimizer
    optimizer = optimizerFnq(model.parameters(), lr, weight_decay)
    #optim.Adam(cnnmodel.parameters(), lr=0.03, weight_decay=1e-5)

####################################################
############# LRFinder #############

def set_lrfinder(model):
    global lr_finder
    lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
    lr_finder.range_test(train_loader, end_lr=10, num_iter=200, step_mode="exp")
    lr_finder.plot()
    lr_finder.reset()

####################################################
############# SCHEDULER #############

EPOCHS = 24
misclassified = []
actual_labels = []
predicted_labels = []

def set_scheduler():
    global scheduler
    scheduler = OneCycleLR(
        optimizer=optimizer,
        max_lr=4.93E-02,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS,
        pct_start=5/EPOCHS,
        div_factor=100,
        final_div_factor=100,
        three_phase=False,
        anneal_strategy="linear"
    )

####################################################
############# TRAIN-TEST #############

def test_and_train(model, EPOCHS=24):
    for epoch in range(1, EPOCHS+1):
        print(f'Epoch {epoch}')
        model.train(model, device, train_loader, criterion, optimizer, epoch, scheduler)
        model.test(model, device, test_loader)

####
# To show the misclassified Images
# model.plot_misclassifeid_images()


####
# Graph
# model.plot_loss_accuracy_graph()
