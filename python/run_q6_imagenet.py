import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
# import cv2


# Q6_3 (CNN Model on ImageNet)

# Fix seed
np.random.seed(0)
torch.manual_seed(0)

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

batch_size = 16
num_classes = 1000

# Load ImageNet validation data and labels
data = np.load('/home/lonewolf/dev_ws/advanced_cv/advanced-cv-hw4/data/data.npy')
labels = np.load('/home/lonewolf/dev_ws/advanced_cv/advanced-cv-hw4/data/labels.npy')

# Convert data and labels to PyTorch tensors
data = torch.tensor(data, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.long)

# Reshape the data to [B, 3, 64, 64]
data = data.view(-1, 3, 64, 64)

# Data is in range 7 to 207?
data = data / 255.0

# Visualize the downsampled images to see if they make sense
data_for_specific_class = data[labels == 2] # Dog
labels_for_specific_class = labels[labels==2]
# for i in range(len(data_for_specific_class)):
#     plt.imshow(data_for_specific_class[i].permute(1, 2, 0))
#     plt.title(f"Label: {labels_for_specific_class[i]}")
#     plt.show()

# Create a TensorDataset and DataLoader
val_dataset = TensorDataset(data_for_specific_class, labels_for_specific_class)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load the pretrained ResNet-50 model
model_resnet50 = torchvision.models.resnet50(pretrained=True).to(device)

# Define Hyperparameters (Optimization, Loss, etc.)
loss_fn = nn.CrossEntropyLoss()

# Perform the validation loop
model_resnet50.eval()
val_epoch_loss = 0
val_correct = 0
val_total = 0
with torch.no_grad():
    for val_batch, (val_x, val_y) in enumerate(tqdm(val_dataloader)):
        val_x, val_y = val_x.to(device), val_y.to(device)

        # Forward Pass
        val_preds = model_resnet50(val_x)

        # Loss
        batch_val_loss = loss_fn(val_preds, val_y)
        val_epoch_loss += batch_val_loss.item()

        # Calculate accuracy
        _, val_predicted = torch.max(val_preds.data, 1)
        val_total += val_y.size(0)
        val_correct += (val_predicted == val_y).sum().item()

val_accuracy = 100 * val_correct / val_total
print(f"Validation Loss: {val_epoch_loss / len(val_dataloader)}, Validation Accuracy: {val_accuracy:.2f}%")

# Second part of the question
# Process video and run inference on frames
vid_path = '/home/lonewolf/dev_ws/advanced_cv/advanced-cv-hw4/data/dog.mp4'
cap = cv2.VideoCapture(vid_path)

# Resize to match model input                                                                 
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

frame_count = 0
correct_frames = 0
total_frames = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to Tensor and move to GPU
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = transform(frame)
    frame = frame.unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        preds = model_resnet50(frame)
        _, predicted = torch.max(preds.data, 1) # Reverse one hot encoding
        print(f"Frame {frame_count}: Predicted class {predicted.item()}")

    # Ground truth is index 2 for dog
    ground_truth_label = 2
    if predicted.item() == ground_truth_label:
        correct_frames += 1
    total_frames += 1
    frame_count += 1

cap.release()
cv2.destroyAllWindows()

vid_accuracy = 100 * correct_frames / total_frames
print(f"Video Accuracy: {vid_accuracy:.2f}%")