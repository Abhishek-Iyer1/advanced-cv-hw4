import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
from custom_models import MLP, CNN

# Q6_3 (CNN Model on NIST36)

# Fix seed
np.random.seed(0)
torch.manual_seed(0)

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
learning_rate = 1e-1
epochs = 25

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define/Load Model
# model = MLP(input_size = 1024, output_size=36).to(device)
model = CNN(channels=3, classes=10).to(device)

# Define Hyperparameters (Optimization, Loss, etc.)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Initialize History
history = {
    "train_loss" : [],
    "train_acc" : [],
    "val_loss" : [],
    "val_acc" : [],
}

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

        # x = torch.reshape(x, (-1, 1, 32, 32))
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
    history["train_loss"].append(train_epoch_loss)
    history["train_acc"].append(train_accuracy)
    
model.eval()
test_epoch_loss = 0
test_correct = 0
test_total = 0

with torch.no_grad():
    for test_batch, (test_x, test_y) in enumerate(testloader):
        test_x, test_y = test_x.to(device).float(), test_y.to(device)
        # test_x = torch.reshape(test_x, (-1, 1, 32, 32))

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
        
# Plot Loss
plt.figure(figsize=(10, 5))
plt.plot(history["train_loss"], label="Train Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()

# Plot Accuracy
plt.figure(figsize=(10, 5))
plt.plot(history["train_acc"], label="Train Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.show()