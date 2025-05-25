import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from PIL import Image, ImageOps

# Dataset class for loading binary images
class BinaryImageDataset(Dataset):
    def __init__(self, images_folder, transform=None):
        self.images = []
        self.image_names = []
        for img_name in os.listdir(images_folder):
            img_path = os.path.join(images_folder, img_name)

            # Open and convert to grayscale
            img = Image.open(img_path).convert("L")

            # Resize to 64x64
            img = img.resize((64, 64), Image.Resampling.NEAREST)

            # Convert to numpy and binarize with reasonable threshold
            img = np.array(img)
            img = (img > 127).astype(np.float32)

            self.images.append(img)
            self.image_names.append(img_name)

            # Save processed image (for debugging)
            # Image.fromarray((img * 255).astype(np.uint8)).save(f"processed_{img_name}")
            # exit()  # Only process one image for debugging

        self.images = np.array(self.images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sample = self.images[idx]
        sample = Image.fromarray((sample * 255).astype(np.uint8))

        # Save before transform (for debugging)
        # sample.save(f"sample_{idx}.png")

        if self.transform:
            sample = self.transform(sample)

            # Save transformed image (for debugging)
            # transformed_image = (sample.squeeze().numpy() * 255).astype(np.uint8)
            # Image.fromarray(transformed_image).save(f"transformed_sample_{idx}.png")
        else:
            sample = torch.tensor(np.array(sample), dtype=torch.float32).unsqueeze(0)

        return sample

# Encoder (mapping from image to latent space)
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        # First: pass image through convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # -> 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> 7x7
            nn.ReLU()
        )

        # Second: pass through fully connected layers
        self.linear_layers = nn.Sequential(
            nn.Linear(64 * 16 * 16, 512),  # Assuming input image size is 64x64
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the feature maps
        latent_features = self.linear_layers(x)
        return latent_features


# Decoder (mapping from latent space to image)
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        # First: pass the latent features through fully connected layers
        self.linear_layers = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 64 * 16 * 16),
            nn.ReLU()
        )

        # Second: pass through deconvolution layers to reconstruct the image 
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 16 → 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32 → 64
            nn.Sigmoid()
        )

    def forward(self, z):
        z = self.linear_layers(z)
        z = z.view(z.size(0), 64, 16, 16)  # Reshape into feature map
        z = self.deconv_layers(z)
        return z

# Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        latent = self.encoder(x)
        x_recon = self.decoder(latent)
        return x_recon

    def loss_function(self, x, x_recon):
        # Reconstruction loss (binary cross-entropy for binary images)
        recon_loss = nn.BCELoss(reduction='sum')(x_recon.view(-1), x.view(-1))
        return recon_loss