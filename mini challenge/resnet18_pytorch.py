import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from torchsummary import summary
# Set random seed for numpy and PyTorch
np.random.seed(42)
torch.manual_seed(42)

def train(train_loader, val_loader, num_classes):
    print("Training model...")

    # Load ResNet18
    model = models.resnet18(pretrained=True)

    # Don't Freeze parameters finetune it
    for param in model.parameters():
        param.requires_grad = True

    # Replace the final fully connected layer with a new one
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)


    print("Created new model.")

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True)

    # Early stopping
    early_stop = False
    best_val_loss = float('inf')
    patience = 10
    counter = 0

    # Train the model
    num_epochs = 1000
    for epoch in range(num_epochs):
        if not early_stop:
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            print("start training")
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss = running_loss / len(train_loader.dataset)
            train_acc = correct / total

            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            val_loss = val_loss / len(val_loader.dataset)
            val_acc = correct / total

            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    early_stop = True
                    print("Early stopping")

            # Update learning rate
            scheduler.step(val_loss)

    # Save the model
    torch.save(model.state_dict(), 'train_RES_main_data50.pth')
    print("Training completed.")

    return model

def main():
    # Setup directories and labels file
    labels_file = 'purdue-face-recognition-challenge-2024/train.csv'
    data_dir = 'main_data_50'  # Change this to your main data directory
    val_dir = 'validation'  # Change this to your validation directory

    # Read labels file
    labels_df = pd.read_csv(labels_file)
    num_classes = len(labels_df['Category'].unique())

    # Define data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Add color jitter
            transforms.RandomRotation(degrees=10),  # Randomly rotate the image by up to 15 degrees
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ]),
    }

    # Load datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'validation']}

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=256, shuffle=True)
    val_loader = torch.utils.data.DataLoader(image_datasets['validation'], batch_size=256, shuffle=False)

    # Train the model
    model = train(train_loader, val_loader, num_classes)


if __name__ == "__main__":
    main()
