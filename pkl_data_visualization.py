import argparse
import os
import numpy as np
import torch
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from dataset import CarDataset  # Your dataset script
from utils.config import PRED_LEN      # Your config script

def wrap_angle(angle_rad):
    """Wraps an angle in radians to the range [-pi, pi]."""
    return (angle_rad + np.pi) % (2 * np.pi) - np.pi

def visualize_raw_displacements(data_folder, batch_size, output_file):
    """
    Loads a batch and creates a scatter plot of the raw, relative (dx, dy)
    displacements from the ground truth data.
    """
    # --- Data Loading ---
    device = torch.device("cpu") # No GPU needed for this
    try:
        dataset = CarDataset(preprocess_folder=data_folder)
        # Use shuffle=True to see a random, representative batch
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        batch = next(iter(data_loader))
        batch = batch.to(device)
    except (StopIteration, FileNotFoundError):
        print(f"Error: Could not load data from '{data_folder}'. Is the path correct and the dataset non-empty?")
        return

    print(f"Loaded a batch of size {batch.num_graphs} from '{data_folder}'. Processing raw displacements...")

    # --- Data Processing ---
    # `batch.y` is [N, 180], needs to be reshaped to [N, 30, 6]
    gt_data = batch.y.reshape(-1, PRED_LEN, 6)
    
    # Extract the raw relative (dx, dy) displacements and global angles
    gt_pos_relative = gt_data[:, :, [0, 1]]  # Shape: [N, 30, 2]
    gt_angles_global = gt_data[:, :, 3]      # Shape: [N, 30]

    # Calculate the change in yaw (turn rate) between steps to use for coloring
    # We compare step t with step t-1, so we'll have 29 change values per car
    angle_diff = gt_angles_global[:, 1:] - gt_angles_global[:, :-1]
    d_yaw = wrap_angle(angle_diff) # Wrap to [-pi, pi] to handle angle jumps

    # We will plot the displacements from step 1 onwards, as d_yaw is defined for them
    displacements_to_plot = gt_pos_relative[:, 1:, :].reshape(-1, 2).numpy()
    d_yaw_flat = d_yaw.flatten().numpy()

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 10))

    # Create a scatter plot of the (dx, dy) points
    # Color each point by its corresponding change in yaw
    sc = ax.scatter(
        displacements_to_plot[:, 0], # dx (forward displacement)
        displacements_to_plot[:, 1], # dy (leftward displacement)
        c=d_yaw,                     # Color by turn rate
        cmap='RdBu_r',               # Red=Right Turn, Blue=Left Turn
        alpha=0.6,
        s=15,                        # Marker size
        vmin=-0.2,                   # Set color limits for consistency
        vmax=0.2
    )
    
    # --- Figure Formatting ---
    ax.set_aspect('equal', 'box')
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Add reference lines
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    
    ax.set_title(f'Distribution of Single-Step Relative Displacements (Batch Size: {batch.num_graphs})', fontsize=14)
    ax.set_xlabel('Forward Displacement (dx) [m per step]', fontsize=12)
    ax.set_ylabel('Leftward Displacement (dy) [m per step]', fontsize=12)
    
    # Add a colorbar to explain the colors
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('Change in Yaw (Turn Rate) [radians per step]', fontsize=12)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\nSuccessfully saved raw data visualization to: {output_file}")
    print("\nInterpreting the plot:")
    print(" - Each point is one car's movement over a single timestep.")
    print(" - Points on the x-axis represent going perfectly straight.")
    print(" - Points in the upper half are left turns (positive dy).")
    print(" - Points in the lower half are right turns (negative dy).")
    print(" - Blue points are sharp left turns; Red points are sharp right turns.")


# --- Script Execution ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize raw (dx, dy) ground truth data from a dataset batch.")
    parser.add_argument('--data_folder', type=str, required=True, help='Path to the preprocessed data folder (e.g., csv/train_pre).')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of cars to sample for the visualization.')
    parser.add_argument('--output_file', type=str, default='output/raw_data_distribution.png', help='Path to save the output image.')
    args = parser.parse_args()

    visualize_raw_displacements(
        data_folder=args.data_folder,
        batch_size=args.batch_size,
        output_file=args.output_file
    )