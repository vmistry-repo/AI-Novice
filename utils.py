import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchsummary import summary
from torchvision import transforms

def check_cuda():
  cuda_available = torch.cuda.is_available()
  print("CUDA Available?", cuda_available)
  return cuda_available

def get_device():
  device = torch.device("cuda" if check_cuda() else "cpu")

def get_transforms_for_train_data(x, y, mean, std):
  return transforms.Compose([
    # Instead of 22 it would be random later on
    transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
    transforms.Resize((x,  y)),
    transforms.RandomRotation((-15., 15.), fill=0),
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,)),
    ])

def get_transforms_for_test_data(mean, std):
  return transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((mean,), (std,))
    ])

def get_loader(data, batch_size):
  kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': False}
  return torch.utils.data.DataLoader(data, **kwargs)

def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def train(model, device, train_loader, optimizer, train_acc, train_losses):
  model.train()
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()

    # Predict
    pred = model(data)

    # Calculate loss
    loss = F.nll_loss(pred, target)
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    optimizer.step()
    
    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  train_acc.append(100*correct/processed)
  train_losses.append(train_loss/len(train_loader))

def test(model, device, test_loader, test_acc, test_losses):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)


    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def send_model_to_device(model, device):
  return model().to(device)

def run_for_epoch(num_epochs, model, device, train_loader, test_loader, optimizer, scheduler, train_acc, train_losses, test_acc, test_losses):
  for epoch in range(1, num_epochs+1):
    print(f'Epoch {epoch}')
    train(model, device, train_loader, optimizer, train_acc, train_losses)
    scheduler.step()
    test(model, device, test_loader, test_acc, test_losses)
  
def plot_dataset_images(batch_data, batch_label):
  fig = plt.figure()
  for i in range(12):
    plt.subplot(3,4,i+1)
    plt.tight_layout()
    plt.imshow(batch_data[i].squeeze(0), cmap='gray')
    plt.title(batch_label[i].item())
    plt.xticks([])
    plt.yticks([])

def plot_loss_accuracy_graph(train_losses, train_acc, test_losses, test_acc):
  fig, axs = plt.subplots(2,2,figsize=(15,10))
  axs[0, 0].plot(train_losses)
  axs[0, 0].set_title("Training Loss")
  axs[1, 0].plot(train_acc)
  axs[1, 0].set_title("Training Accuracy")
  axs[0, 1].plot(test_losses)
  axs[0, 1].set_title("Test Loss")
  axs[1, 1].plot(test_acc)
  axs[1, 1].set_title("Test Accuracy")

def get_model_summary(model, device, x, y):
  #model = model().to(device)
  summary(model, input_size=(1, x, y))
