import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torchsummary import summary
from torchvision.transforms import ToTensor

train_losses = []
test_losses = []
train_acc = []
test_acc = []
g_misclassified = []
g_actual_labels = []
g_predicted_labels = [] 

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class Net(nn.Module):
    def __init__(self, dropout_value=0.05):
        super(Net, self).__init__()

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
        x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=groups, bias=False)
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
        
def train(model, device, train_loader, criterion, optimizer, epoch, scheduler):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    # get samples
    data, target = data.to(device), target.to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)

    # Calculate loss
    loss = criterion(y_pred, target)
    train_losses.append(loss)

    # Backpropagation
    loss.backward()
    optimizer.step()
    scheduler.step()
    # Update pbar-tqdm

    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)

    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    train_acc.append(100*correct/processed)
  return loss


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    misclassified = []  # create a list to store misclassified images
    actual_labels = []  # create a list to store actual labels of misclassified images
    predicted_labels = []  # create a list to store predicted labels of misclassified images
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            # check for misclassified images
            misclassified_mask = (pred != target.view_as(pred)).squeeze()
            misclassified.append(data[misclassified_mask])
            actual_labels.append(target.view_as(pred)[misclassified_mask])
            predicted_labels.append(pred[misclassified_mask])

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))

    misclassified = torch.cat(misclassified, dim=0)
    actual_labels = torch.cat(actual_labels, dim=0)
    predicted_labels = torch.cat(predicted_labels, dim=0)
    
    global g_misclassified
    global g_actual_labels
    global g_predicted_labels
    g_misclassified = misclassified
    g_actual_labels = actual_labels
    g_predicted_labels = predicted_labels
    # return the misclassified images tensor
    return
    
    
def plot_loss_accuracy_graph():
    t = [t_items.item() for t_items in train_losses]
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(t)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")
