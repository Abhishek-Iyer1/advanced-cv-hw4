import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()

        # Initialize Layers
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Sigmoid(),
            )
        self.output = nn.Sequential(
            nn.Linear(64, output_size),
            # Not using softmax as Cross Entropy applies that internally
        )
    
    def forward(self, x):
        # Forward pass
        h1 = self.layer1(x)
        preds = self.output(h1)

        return preds
    
class CNN(nn.Module):
    def __init__(self, channels, classes):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 128, 5)
        self.fc1 = nn.Linear(128 * 5 * 5, 512) # 16, 5, 5, 120 / 16, 53, 53, 120
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x