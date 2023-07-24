import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
import numpy as np

import cv2
import torchvision
import torch
import torchvision.transforms as transforms
# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Imports
import models.model as model 
import data.dataset as dataset
import utils.utils as utils
import trainers.transform as trans
####

device = utils.get_device()
cuda = torch.cuda.is_available()
SEED = 42

# For reproducibility
torch.manual_seed(SEED)

if cuda:
    torch.cuda.manual_seed(SEED)
    
####

## To get the std and mean value

import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Define the transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load the CIFAR-10 dataset
cifar10_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
pixel_means = cifar10_train.data.mean(axis=(0,1,2)) / 255.0
pixel_stds = cifar10_train.data.std(axis=(0,1,2)) / 255.0

print('CIFAR-10 pixel means:', pixel_means)
print('CIFAR-10 pixel stds:', pixel_stds)
print(cifar10_train.data.shape)

####

class args():
    def __init__(self,device = 'cpu' ,use_cuda = False) -> None:
        self.batch_size = 512
        self.device = device
        self.use_cuda = use_cuda
        self.kwargs = {'num_workers': 1, 'pin_memory': True} if self.use_cuda else {}
        
####

train_transforms = trans.get_train_transforms(pixel_means, pixel_stds)
test_transforms = trans.get_test_transforms(pixel_means, pixel_stds)

trainset = dataset.Cifar10SearchDataset(root='./data', train=True,
                                download=True, transform=train_transforms)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args().batch_size,
                                          shuffle=True, **args().kwargs)

testset = dataset.Cifar10SearchDataset(root='./data', train=False,
                               download=True, transform=test_transforms)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args().batch_size,
                                         shuffle=True, **args().kwargs)
                                         
####

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

utils.visualise_dataset(train_loader, classes)        

####

from torchsummary import summary
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
cnnmodel = model.Net().to(device)
summary(cnnmodel, input_size=(3, 32, 32))

####

from torch_lr_finder import LRFinder

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnnmodel.parameters(), lr=0.03, weight_decay=1e-5)
lr_finder = LRFinder(cnnmodel, optimizer, criterion, device="cuda")
lr_finder.range_test(train_loader, end_lr=10, num_iter=200, step_mode="exp")
lr_finder.plot()
lr_finder.reset()

####

from torch.optim.lr_scheduler import OneCycleLR

EPOCHS = 24
misclassified = []
actual_labels = []
predicted_labels = []

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

####

for epoch in range(1, EPOCHS+1):
  print(f'Epoch {epoch}')
  model.train(cnnmodel, device, train_loader, criterion, optimizer, epoch, scheduler)
  model.test(cnnmodel, device, test_loader)
  
  
####
# To show the misclassified Images
model.plot_misclassifeid_images()


####
# Graph
model.plot_loss_accuracy_graph()
