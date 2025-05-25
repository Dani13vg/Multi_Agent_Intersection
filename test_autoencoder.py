from autoencoder import Autoencoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms
import os
from datetime import datetime
from PIL import Image

def dummy_img():
    # Create a dummy image for testing with a circle
    dummy_image = np.zeros((64, 64), dtype=np.float32)
    rr, cc = np.ogrid[:64, :64]
    circle = (rr - 32) ** 2 + (cc - 32) ** 2 < 16 ** 2
    dummy_image[circle] = 1.0

    # Convert dummy image to PIL format
    dummy_image_pil = Image.fromarray((dummy_image * 255).astype(np.uint8), mode='L')

    # Apply the same transform to the dummy image
    dummy_image_tensor = transform(dummy_image_pil).unsqueeze(0).to(device)
    # Save dummy image for debugging
    dummy_image_path = "dummy_image.png"
    Image.fromarray((dummy_image * 255).astype(np.uint8)).save(dummy_image_path)
    print(f"Dummy image saved to {dummy_image_path}")

    # Use the dummy image for testing
    return dummy_image_tensor

model_path = "autoencoders/autoencoder_20250525_201748.pth"

# Load the model
model = Autoencoder(latent_dim=64)
model.load_state_dict(torch.load(model_path))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

input_image = 'map_binary_images/simple_separate_10m_binary.png'

# Load and preprocess the input image
input_img_pil = Image.open(input_image).convert("L")
input_img_pil = input_img_pil.resize((64, 64), Image.Resampling.NEAREST)

# Define augmentation pool and transform
augmentation_pool = [
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomVerticalFlip(p=1.0),
    transforms.RandomAffine(degrees=180),
    transforms.RandomRotation(degrees=180),
]
transform = transforms.Compose([
    transforms.RandomChoice(augmentation_pool),
    transforms.ToTensor()
])

# Apply transform to input image
input_image_tensor = transform(input_img_pil).unsqueeze(0).to(device)

# Save input image for debugging
input_image_path = "input_image.png"
input_img_np = (input_image_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
Image.fromarray(input_img_np).save(input_image_path)
print(f"Input image saved to {input_image_path}")

# Create a dummy image for testing (uncomment to use)
# input_image_tensor = dummy_img()

# Forward pass through the model
with torch.no_grad():
    output_image = model(input_image_tensor)
output_image = output_image.squeeze().cpu().numpy()

# Convert output to binary image
output_image = (output_image > 0.5).astype(np.float32) * 255
output_image = Image.fromarray(output_image.astype(np.uint8))

# Save the output image
output_image_path = "output_image.png"
output_image.save(output_image_path)
print(f"Output image saved to {output_image_path}")

