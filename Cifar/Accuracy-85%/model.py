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

        self.convblock1 = self.conv_block(3, 64, (3, 3), 0, dropout_value)
        self.convblock2 = self.dial_conv_block(64, 128, (3, 3), 0, dropout_value, groups=32)
        self.convblock3 = self.dial_conv_block(64, 128, (3, 3), 0, dropout_value, groups=32)
        self.convblock4 = self.dial_conv_block(64, 128, (3, 3), 2, dropout_value, groups=32, dilation=2)
        self.convblock5 = self.dial_conv_block(64, 128, (3, 3), 4, dropout_value, groups=32, dilation=4)
        self.convblock6 = self.dial_conv_block(64, 128, (3, 3), 8, dropout_value, groups=32, dilation=8)
        self.convblock7 = self.dial_conv_block(64, 128, (3, 3), 12, dropout_value, groups=32, dilation=12)
        self.convblock8 = self.dial_conv_block(64, 128, (3, 3), 16, dropout_value, groups=32, dilation=16)
        self.convblock9 = self.dial_conv_block(64, 128, (3, 3), 20, dropout_value, groups=32, dilation=20)
        self.convblock10 = self.dial_conv_block(64, 128, (3, 3), 22, dropout_value, groups=32, dilation=22)
        self.convblock11 = self.output_block(64, 10)

        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = x + self.convblock4(x)
        x = x + self.convblock5(x)
        x = x + self.convblock6(x)
        x = x + self.convblock7(x)
        x = x + self.convblock8(x)
        x = x + self.convblock9(x)
        x = x + self.convblock10(x)
        x = self.convblock11(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

    def conv_block(self, in_channels, out_channels, kernel_size, padding, dropout_value, groups=1, dilation=1):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, groups=groups, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout_value)
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
        
def train(model, device, train_loader, criterion, optimizer, epoch):
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

def plot_misclassifeid_images():
    images=g_misclassified
    actual_labels=g_actual_labels
    predicted_labels=g_predicted_labels
    n_rows=5
    n_cols=2
    n_images=10
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        if i < len(images) and i < n_images:
            image = images[i].cpu().numpy().transpose((1, 2, 0))
            image = (image * 0.5) + 0.5
            actual = actual_labels[i].item()
            predicted = predicted_labels[i].item()
            ax.imshow(image)
            ax.set_title(f"Actual: {classes[actual]}\nPredicted: {classes[predicted]}")
            ax.axis('off')
        else:
            ax.axis('off')

    fig.subplots_adjust(hspace=1)
    plt.show()