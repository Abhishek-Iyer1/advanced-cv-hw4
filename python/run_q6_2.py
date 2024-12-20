import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from custom_datasets import NIST36
from custom_models import MLP, CNN

# Q6_3 (CNN Model on NIST36)

# Fix seed
np.random.seed(0)
torch.manual_seed(0)

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 16
learning_rate = 1e-3
epochs = 25
num_classes = 17

transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

train_dataset = torchvision.datasets.ImageFolder("/home/lonewolf/dev_ws/advanced_cv/advanced-cv-hw4/data/oxford-flowers17/train", transform=transform)
test_dataset = torchvision.datasets.ImageFolder("/home/lonewolf/dev_ws/advanced_cv/advanced-cv-hw4/data/oxford-flowers17/test", transform=transform)

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Define/Load Model
# model = MLP(input_size = 1024, output_size=36).to(device)
model_CNN = CNN(channels=3, classes=17)
model_squeezenet = torchvision.models.squeezenet1_1(pretrained=True)
final_conv = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
model_squeezenet.classifier = nn.Sequential(
    nn.Dropout(p=0.5),
    final_conv,
    nn.ReLU(inplace=True),
    nn.AdaptiveAvgPool2d((1, 1))
)

model = model_CNN
model = model.to(device)
# Define Hyperparameters (Optimization, Loss, etc.)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in tqdm(range(epochs)):
    # Training Loop
    model.train()
    train_epoch_loss = 0
    train_correct = 0
    train_total = 0
    for batch, (x,y) in tqdm(enumerate(trainloader)):
        x, y = x.to(device), y.to(device)

        # Zero Gradients
        optimizer.zero_grad()

        # Forward Pass
        preds = model.forward(x)

        # Loss
        batch_train_loss = loss_fn(preds, y)
        train_epoch_loss += batch_train_loss.item()

        # Backward
        batch_train_loss.backward()

        # Update Gradients and then reset for next batch
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(preds.data, 1)
        # y_labels = torch.argmax(y, dim=1)  # Convert one-hot encoded y to class indices
        train_total += y.size(0)
        train_correct += (predicted == y).sum().item()
    
    train_accuracy = 100 * train_correct / train_total
    print(f"Epoch: {epoch + 1}, Training Loss: {train_epoch_loss / len(trainloader)}, Training Accuracy: {train_accuracy:.2f}%")
    
model.eval()
test_epoch_loss = 0
test_correct = 0
test_total = 0
with torch.no_grad():
    for test_batch, (test_x, test_y) in enumerate(testloader):
        test_x, test_y = test_x.to(device), test_y.to(device)

        # Forward Pass
        test_preds = model.forward(test_x)

        # Loss
        batch_test_loss = loss_fn(test_preds, test_y)
        test_epoch_loss += batch_test_loss

        # Calculate accuracy
        _, test_predicted = torch.max(test_preds.data, 1)
        test_total += test_y.size(0)
        test_correct += (test_predicted == test_y).sum().item()
        
    test_accuracy = 100 * test_correct / test_total
    print(f"Test Loss: {test_epoch_loss / len(testloader)}, Test Accuracy: {test_accuracy:.2f}%")