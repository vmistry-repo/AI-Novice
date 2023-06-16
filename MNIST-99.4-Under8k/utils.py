import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchsummary import summary
from torchvision.transforms import ToTensor

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

