import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchsummary import summary
from torchvision.transforms import ToTensor
import numpy as np
import torchvision
import torchvision.transforms as transforms

# Data to plot accuracy and loss graphs

def check_cuda():
  cuda_available = torch.cuda.is_available()
  print("CUDA Available: ", cuda_available)
  return cuda_available

def get_device():
  device = torch.device("cuda" if check_cuda() else "cpu")

def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()
    

def send_model_to_device(model, device):
  return model().to(device)

def get_model_summary(model, device, x, y):
  #model = model().to(device)
  summary(model, input_size=(1, x, y))

def visualise_dataset(trainloader, classes):
  # Get a batch of images from the dataloader
  dataiter = iter(trainloader)
  images, labels = next(dataiter)

  # Plot the images using Matplotlib
  fig, axs = plt.subplots(4, 4, figsize=(12, 6))
  for i in range(16):
      # Calculate the subplot index
      row = i // 4
      col = i % 4
      # Unnormalize the image
      img = images[i] / 2 + 0.5
      # Convert the image to a NumPy array
      npimg = img.numpy()
      # Transpose the image from (C, H, W) to (H, W, C)
      npimg = np.transpose(npimg, (1, 2, 0))
      # Plot the image
      axs[row, col].imshow(npimg)
      axs[row, col].set_title(classes[labels[i]])
      axs[row, col].axis('off')

  fig.subplots_adjust(hspace=1)
  plt.show()
