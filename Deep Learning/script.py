
import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import s3fs
from sagemaker import get_execution_role
from PIL import Image

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 1

# Setting up data transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

# Custom dataset class to directly read images from S3
class S3Dataset(torch.utils.data.Dataset):
    def __init__(self, s3_path, transform = None):
        self.s3_path = s3_path
        self.transform = transform
        self.s3fs = s3fs.S3FileSystem()
        self.image_paths = [path for path in self.s3fs.glob(f"{s3_path}/**/*.jpg")]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        with self.s3fs.open(image_path, "rb") as f:
            image = Image.open(f)
            if self.transform:
                image = self.transform(image)
        label = 0 if "cat" in image_path else 1
        return image, label

train_path = "s3://aws-sagemaker-project-2/sagemaker/DL/torch_container/train"
test_path = "s3://aws-sagemaker-project-2/sagemaker/DL/torch_container/test"

# Load training data from S3 using the custom dataset
train_dataset = S3Dataset(train_path, transform = transform)
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)

# Load test data from S3 using the custom dataset
test_dataset = S3Dataset(test_path, transform = transform)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False)

# Initialize pre-trained model
model = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False

# Modify the final fully connected layer for binary classification (2 classes: Cats, Dogs)
model.fc = nn.Linear(model.fc.in_features, 2)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr = LEARNING_RATE)

# Device configuration (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

# Save the model to S3
fs = s3fs.S3FileSystem()

s3_model_path = "s3://aws-sagemaker-project-2/sagemaker/DL/torch_container/model.pth"
with fs.open(s3_model_path, "wb") as f:
    torch.save(model.state_dict(), f)

print(f"Model saved to {s3_model_path}")

# Evaluation loop
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")