####################################################
############# IMPORTS #############

import cv2
import math
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

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import matplotlib.pyplot as plt

import numpy as np

# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Local Imports
import models.model as model 
import data.dataset as dataset
import utils.utils as utils
import trainers.transform as trans

from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
####################################################
############# VARS #############

SEED = 42

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
    print(dataset.data.shape)

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

def get_transforms():
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
def visualise_dataset():
    utils.visualise_dataset(train_loader, classes)        

####################################################
############# SUMMARY #############

def get_model_summary(mymodel):
    global _model, device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    _model = mymodel.Net().to(device)
    summary(_model, input_size=(3, 32, 32))

####################################################
############# LOSS Fnq #############

def set_lossFn(loss):
    global criterion
    criterion = loss() #nn.CrossEntropyLoss()

####################################################
############# OPTIMIZER #############

def set_optim(optimizerFnq, lr=0.03, weight_decay=1e-5):
    global optimizer
    #optimizer = optimizerFnq(_model.parameters(), lr, weight_decay)
    optimizer = optim.Adam(_model.parameters(), lr=0.03, weight_decay=1e-5)

####################################################
############# LRFinder #############

def set_lrfinder():
    global lr_finder
    lr_finder = LRFinder(_model, optimizer, criterion, device="cuda")
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

def test_and_train(EPOCHS=24):
    for epoch in range(1, EPOCHS+1):
        print(f'Epoch {epoch}')
        model.train(_model, device, train_loader, criterion, optimizer, epoch, scheduler)
        model.test(_model, device, test_loader)

####################################################
# To show the misclassified Images
def display_misclassified_images():
	model.plot_misclassifeid_images()


####################################################
# Graph
def display_loss_accuracy_graph():
	model.plot_loss_accuracy_graph()


####################################################
# -------------------- GradCam --------------------
def display_gradcam_output(data: list,
                           classes: list[str],
                           inv_normalize: transforms.Normalize,
                           model: 'DL Model',
                           target_layers: list['model_layer'],
                           targets=None,
                           number_of_samples: int = 10,
                           transparency: float = 0.60):
    """
    Function to visualize GradCam output on the data
    :param data: List[Tuple(image, label)]
    :param classes: Name of classes in the dataset
    :param inv_normalize: Mean and Standard deviation values of the dataset
    :param model: Model architecture
    :param target_layers: Layers on which GradCam should be executed
    :param targets: Classes to be focused on for GradCam
    :param number_of_samples: Number of images to print
    :param transparency: Weight of Normal image when mixed with activations
    """
    # Plot configuration
    fig = plt.figure(figsize=(10, 10))
    x_count = 5
    y_count = 1 if number_of_samples <= 5 else math.floor(number_of_samples / x_count)

    # Create an object for GradCam
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    # Iterate over number of specified images
    for i in range(number_of_samples):
        plt.subplot(y_count, x_count, i + 1)
        input_tensor = data[i][0]

        # Get the activations of the layer for the images
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        # Get back the original image
        img = input_tensor.squeeze(0).to('cpu')
        img = inv_normalize(img)
        rgb_img = np.transpose(img, (1, 2, 0))
        rgb_img = rgb_img.numpy()

        # Mix the activations on the original image
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight=transparency)

        # Display the images on the plot
        plt.imshow(visualization)
        plt.title(r"Correct: " + classes[data[i][1].item()] + '\n' + 'Output: ' + classes[data[i][2].item()])
        plt.xticks([])
        plt.yticks([])
        
def get_misclassified_imagNlable(misclassified_data):
    """
    Function to extract misclassified images, actual labels and predicted labels from the misclassified_data list
    :param misclassified_data: list of misclassified images, labels and predictions
    """
    # Lists to store misclassified Images, actual labels and predicted labels
    misclassified_images = []
    actual_labels = []
    predicted_labels = []
    
    for data in misclassified_data:
        # Extract image, actual label and predicted label from the misclassified data
        image, label, pred = data
        
        # Append the data to the corresponding lists
        misclassified_images.append(image)
        actual_labels.append(label)
        predicted_labels.append(pred)

    return misclassified_images, actual_labels, predicted_labels

def get_misclassified_data(model, device, test_loader):
    """
    Function to run the model on test set and return misclassified images
    :param model: Network Architecture
    :param device: CPU/GPU
    :param test_loader: DataLoader for test set
    """
    # Prepare the model for evaluation i.e. drop the dropout layer
    model.eval()

    # List to store misclassified Images
    misclassified_data = []

    # Reset the gradients
    with torch.no_grad():
        # Extract images, labels in a batch
        for data, target in test_loader:

            # Migrate the data to the device
            data, target = data.to(device), target.to(device)

            # Extract single image, label from the batch
            for image, label in zip(data, target):

                # Add batch dimension to the image
                _image = image.unsqueeze(0)

                # Get the model prediction on the image
                output = model(_image)

                # Convert the output from one-hot encoding to a value
                pred = output.argmax(dim=1, keepdim=True)

                # If prediction is incorrect, append the data
                if pred != label:
                    misclassified_data.append((image, label, pred))
    return misclassified_data

class CustomResnet(LightningModule):
    def __init__(self, train_loader, test_loader, val_loader, hidden_size=16, learning_rate=2e-4, dropout_value=0.05):
        super().__init__()

        # Set our init args as class attributes
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_value
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader

        # Hardcode some dataset specific attributes
        self.num_classes = 10

        # PrepLayer
        self.preplayer    = self.conv_block(in_channels=3, out_channels=64, kernel_size=(3,3), stride=1, padding=1)
        # Layer 1
        self.conv1_layer1 = self.conv_block(in_channels=64, out_channels=128, kernel_size=(3,3),
                                            stride=1, padding=1, max_pool=True)
        self.resblock1    = self.res_block(in_channels=128, out_channels=128, kernel_size=(3,3))
        # Layer 2
        self.conv2_layer2 = self.conv_block(in_channels=128, out_channels=256, kernel_size=(3,3),
                                            stride=1, padding=1, max_pool=True)
        # Layer 3
        self.conv3_layer3 = self.conv_block(in_channels=256, out_channels=512, kernel_size=(3,3),
                                            stride=1, padding=1, max_pool=True)
        self.resblock2    = self.res_block(in_channels=512, out_channels=512, kernel_size=(3,3))
        # Max Pool
        self.maxpool      = nn.MaxPool2d(kernel_size=(4,4))
        # FC Layer
        self.fc           = nn.Linear(512, 10, bias=False)

        # Define PyTorch model
        self.model = nn.Sequential(
            self.preplayer,
            self.conv1_layer1,
            self.resblock1,
            self.conv2_layer2,
            self.conv3_layer3,
            self.resblock2,
            self.maxpool,
            self.fc
        )

        self.accuracy = Accuracy(task="MULTICLASS", num_classes=10)

    def forward(self, x):
        x  = self.preplayer(x)
        x  = self.conv1_layer1(x)
        r1 = self.resblock1(x)
        x  = x + r1
        x  = self.conv2_layer2(x)
        x  = self.conv3_layer3(x)
        r2 = self.resblock2(x)
        x  = x + r2
        x  = self.maxpool(x)
        x  = x.view(-1, 512)
        x  = self.fc(x)

        return F.log_softmax(x, dim=-1)

    def conv_block(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                   dropout_value=0, groups=1, dilation=1, max_pool=False):
        x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                       kernel_size=kernel_size, padding=padding, dilation=dilation,
                       groups=groups, bias=False)
        x2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        x3 = nn.BatchNorm2d(out_channels)
        x4 = nn.ReLU()
        x5 = nn.Dropout(dropout_value)
        if max_pool == True:
            return nn.Sequential(x1, x2, x3, x4, x5)
        return nn.Sequential(x1, x3, x4, x5)

    def res_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
                self.conv_block(in_channels, out_channels, kernel_size, stride=1, padding=1),
                self.conv_block(in_channels, out_channels, kernel_size, stride=1, padding=1)
        )

    def dial_conv_block(self, in_channels, out_channels, kernel_size, padding, dropout_value, groups=1, dilation=1):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=groups, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout_value),
            nn.Conv2d(in_channels=out_channels, out_channels=in_channels, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels),
            nn.Dropout(dropout_value),
        )

    def output_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.AvgPool2d(kernel_size=26),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), padding=0, bias=False)
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.logger.experiment.add_scalar("Loss/Train",
                                           loss,
                                           self.current_epoch)

        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)
        self.logger.experiment.add_scalar("Accuracy/Train",
                                           self.accuracy.compute().item(),
                                           self.current_epoch)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)

        self.logger.experiment.add_scalar("Loss/Test",
                                           loss,
                                           self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/Test",
                                           self.accuracy.compute().item(),
                                           self.current_epoch)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

