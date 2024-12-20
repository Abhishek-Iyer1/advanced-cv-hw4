import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from custom_datasets import NIST36
from custom_models import MLP, CNN

# Q6_1 (Fully Connected Model on NIST36)

# Fix seed
np.random.seed(0)
torch.manual_seed(0)

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define Data Paths
train_path = "/home/lonewolf/dev_ws/advanced_cv/advanced-cv-hw4/data/nist36_train.mat"
val_path = "/home/lonewolf/dev_ws/advanced_cv/advanced-cv-hw4/data/nist36_valid.mat"
test_path = "/home/lonewolf/dev_ws/advanced_cv/advanced-cv-hw4/data/nist36_test.mat"

# Instantiate Dataset Classes
train_dataset = NIST36(train_path, "train")
val_dataset = NIST36(val_path, "valid")
test_dataset = NIST36(test_path, "test")

batch_size = 64
learning_rate = 1e-1
epochs = 100

# Create Data Loaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Define/Load Model
# model = MLP(input_size = 1024, output_size=36).to(device)
model = CNN(channels=1, classes=36).to(device)

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
    for batch, (x,y) in enumerate(train_dataloader):
        x, y = x.to(device), y.to(device)

        # Zero Gradients
        optimizer.zero_grad()

        # Uncomment for CNN
        x = torch.reshape(x, (-1, 1, 32, 32))

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
    print(f"Epoch: {epoch + 1}, Training Loss: {train_epoch_loss / len(train_dataloader)}, Training Accuracy: {train_accuracy:.2f}%")
    history["train_loss"].append(train_epoch_loss)
    history["train_acc"].append(train_accuracy)

    # Validation loop
    model.eval()
    val_epoch_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for val_batch, (val_x,val_y) in enumerate(val_dataloader):
            val_x, val_y = val_x.to(device), val_y.to(device)

            # Uncomment for CNN
            val_x = torch.reshape(val_x, (-1, 1, 32, 32))

            # Forward Pass
            val_preds = model.forward(val_x)

            # Loss
            batch_val_loss = loss_fn(val_preds, val_y)
            val_epoch_loss += batch_val_loss

            # Calculate accuracy
            _, val_predicted = torch.max(val_preds.data, 1)
            val_total += val_y.size(0)
            val_correct += (val_predicted == val_y).sum().item()

    val_accuracy = 100 * val_correct / val_total
    print(f"Epoch: {epoch + 1}, Validation Loss: {val_epoch_loss / len(val_dataloader)}, Validation Accuracy: {val_accuracy:.2f}%")
    history["val_loss"].append(val_epoch_loss)
    history["val_acc"].append(val_accuracy)
    
model.eval()
test_epoch_loss = 0
test_correct = 0
test_total = 0
for test_batch, (test_x, test_y) in enumerate(test_dataloader):
    test_x, test_y = test_x.to(device).float(), test_y.to(device)

    # Uncomment for CNN
    test_x = torch.reshape(test_x, (-1, 1, 32, 32))

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
print(f"Test Loss: {test_epoch_loss / len(test_dataloader)}, Test Accuracy: {test_accuracy:.2f}%")

# Plot Loss
plt.figure(figsize=(10, 5))
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()

# Plot Accuracy
plt.figure(figsize=(10, 5))
plt.plot(history["train_acc"], label="Train Accuracy")
plt.plot(history["val_acc"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.show()