# This script trains an autoencoder on binary images from a specified folder.

from autoencoder import Autoencoder, BinaryImageDataset
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
import numpy as np
from torchvision import transforms
from datetime import datetime

def train_autoencoder(model, dataloader, criterion, optimizer, num_epochs=10):

    model.train()  # Set the model to training mode
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images in dataloader:
            images = images.to(device)  # Move images to the device
            optimizer.zero_grad()  # Zero the gradients

            outputs = model(images)  # Forward pass

            loss = criterion(outputs, images)  # Compute the loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    print("Training complete.")
    # Save the trained model
    if not os.path.exists("/autoencoders"):
        os.makedirs("autoencoders")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = f"autoencoders/autoencoder_{timestamp}.pth"
    print(f"Saving model to {model_save_path}")
    torch.save(model.state_dict(), model_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an autoencoder on binary images.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train the autoencoder.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--images_folder", type=str, default='./map_binary_images', help="Folder containing binary images.")
    args = parser.parse_args()

    images_folder = args.images_folder  # Folder containing binary images

    # Transformations include resizing, flipping, cropping, deforming, squeezing, and rotating
    
    augmentation_pool = [
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.RandomAffine(degrees=180),
        transforms.RandomRotation(degrees=180),
    ]

    transform = transforms.Compose([
        # transforms.Resize((28, 28)),
        transforms.RandomChoice(augmentation_pool),
        transforms.ToTensor()
    ])

    dataset = BinaryImageDataset(images_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = Autoencoder(latent_dim=64)  # Initialize the autoencoder with a latent dimension of 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = torch.nn.MSELoss()  # Loss function for reconstruction
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)  # Optimizer for training

    train_autoencoder(model, dataloader, criterion, optimizer, num_epochs=args.num_epochs)

# Execution example:
# python train_autoencoder.py --num_epochs 20 --batch_size 64 --learning_rate 0.0001 --images_folder ./map_binary_images