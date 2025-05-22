import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Encoder (mapping from image to latent space)
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()

        # First: pass image through convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        # Second: pass through fully connected layers
        self.linear_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),  # Assuming input image size is 28x28
            nn.ReLU(),
            nn.Linear(256, latent_dim)
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
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64 * 7 * 7),
            nn.ReLU()
        )

        # Second: pass through deconvolution layers to reconstruct the image 
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, z):
        z = self.linear_layers(z)
        z = z.view(z.size(0), 64, 7, 7)  # Reshape into feature map
        z = self.deconv_layers(z)
        x_recon = torch.sigmoid(z)  # Sigmoid for binary output (0 or 1)
        return x_recon

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