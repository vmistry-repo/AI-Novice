import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, bias=False),            # Conv 3 - 32 x 26 x 26
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.05),
            
            nn.Conv2d(32, 16, 1, bias=False),           # Conv 1 - 16 x 26 x 26
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
            
            nn.Conv2d(16, 32, 3, bias=False),            # Conv 3 - 32 x 24 x 24
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.05),
            
            nn.Conv2d(32, 16, 1, bias=False),           # Conv 1 - 16 x 24 x 24
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
            
            nn.MaxPool2d(2, 2),                         # MAX - 16 x 12 x 12
            nn.ReLU(),
            nn.BatchNorm2d(16),
            
            nn.Conv2d(16, 16, 3, bias=False),           # Conv 3 - 16 x 10 x 10 
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
            
            nn.Conv2d(16, 32, 3, bias=False),           # Conv 3 - 32 x 8 x 8
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.05),

            nn.Conv2d(32, 16, 1, bias=False),           # Conv 1 - 16 x 8 x 8
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
            
            nn.MaxPool2d(2, 2),                         # MAX - 16 x 4 x 4
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )
        
        self.fc1 = nn.Linear(256, 10, bias=False)       # Linear - Input: (16x4x4) Output: 10
        self.fc2 = nn.Linear(10, 10, bias=False)        # Linear - Input: 10  Output: 10

    def forward(self, x): 
        x = F.relu(self.conv(x))
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
