# train_transformer.py

import argparse
import os
import pickle
import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader
import matplotlib
matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt
import imageio.v3 as iio
from transformer_model import GraphTransformer # Import the new model
from tqdm import tqdm
import wandb
from datetime import datetime
import random
import math
import pdb

from dataset import CarDataset
from utils.config import DT, OBS_LEN, PRED_LEN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
obs_len, pred_len, dt = OBS_LEN, PRED_LEN, DT

parser = argparse.ArgumentParser(description="")
parser.add_argument('--train_folder', type=str, help='path to the training set', default='csv/train_pre')
parser.add_argument('--val_folder', type=str, help='path to the validation set', default='csv/val_pre')
parser.add_argument('--epoch', type=int, help='number of total training epochs', default=20)
parser.add_argument('--exp_id', type=str, help='experiment ID', default='sumo_transformer')
parser.add_argument('--batch_size', type=int, help='batch size', default=64)
parser.add_argument('--map', type=str, help='map name', default='None')
args = parser.parse_args()

batch_size = args.batch_size

if args.map == "None":
    train_folder = args.train_folder
    val_folder = args.val_folder
else:
    train_folder = f"csv/train_pre_1k_{args.map}"
    val_folder = f"csv/val_pre_1k_{args.map}"

exp_id = args.exp_id
model_path = f"trained_params/{exp_id}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.map}"
os.makedirs(model_path, exist_ok=True)

# --- Model Configuration ---
INPUT_DIM = 6         # x, y, yaw, intention (3-bit)
HIDDEN_CHANNELS = 128
OUTPUT_DIM = pred_len * 3 # 30 steps * (x, y, angle)
NUM_HEADS = 4
ANGLE_WEIGHT = 15    # Hyperparameter to balance position and angle loss
COLLISION_WEIGHT = 150.0
collision_penalty = True

# --- Dataset and DataLoader ---
train_dataset = CarDataset(preprocess_folder=train_folder, mpc_aug=True)
val_dataset = CarDataset(preprocess_folder=val_folder, mpc_aug=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

# --- Model Initialization ---
model = GraphTransformer(
    input_dim=INPUT_DIM,
    hidden_channels=HIDDEN_CHANNELS,
    output_dim=OUTPUT_DIM,
    num_heads=NUM_HEADS
).to(device)
print(model)

 # --- WandB setup ---
wandb_on = False  # Set to True to enable Weights & Biases logging
if wandb_on:
    run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_GraphTransformer_{exp_id}_{args.map}"
    wandb.init(
        project="MTP",
        entity="danivg",
        name=run_name,
        config={
            "epoch": args.epoch,
            "batch_size": batch_size,
            "collision_penalty": collision_penalty,
            "exp_id": exp_id,
            "map": args.map,
            "obs_len": obs_len,
            "pred_len": pred_len,
            "dt": dt
        },
        sync_tensorboard=True,
        save_code=True,
    )

def get_graph_slices(batch):
    """Return index ranges for each graph in a batched PyG data."""
    # batch.batch is a vector of length N_nodes with graph ids
    gids = batch.batch.detach().cpu().numpy()
    # find boundaries
    starts = [0]
    for i in range(1, len(gids)):
        if gids[i] != gids[i-1]:
            starts.append(i)
    starts.append(len(gids))
    return [(starts[k], starts[k+1]) for k in range(len(starts)-1)]

def render_scene_video(pred_pos, gt_pos, save_path, fps=5, title=""):
    """
    pred_pos, gt_pos: [N_agents, T, 2] on CPU numpy or torch
    Writes MP4 at save_path. Falls back to GIF if MP4 writer not available.
    """
    if torch.is_tensor(pred_pos): pred_pos = pred_pos.detach().cpu().numpy()
    if torch.is_tensor(gt_pos):   gt_pos   = gt_pos.detach().cpu().numpy()

    N, T, _ = pred_pos.shape
    # axis limits with margin
    all_xy = np.concatenate([pred_pos.reshape(-1,2), gt_pos.reshape(-1,2)], axis=0)
    xmin, ymin = all_xy.min(axis=0)
    xmax, ymax = all_xy.max(axis=0)
    pad = 5.0
    xmin, xmax = xmin - pad, xmax + pad
    ymin, ymax = ymin - pad, ymax + pad

    frames = []
    fig, ax = plt.subplots(figsize=(6, 6))
    for t in range(T):
        ax.clear()
        # GT in green, predictions in red, many cars at once
        for n in range(N):
            # past path up to t
            ax.plot(gt_pos[n, :t+1, 0], gt_pos[n, :t+1, 1], 'g-', linewidth=1.8, alpha=0.8)
            ax.plot(pred_pos[n, :t+1, 0], pred_pos[n, :t+1, 1], 'r-', linewidth=1.4, alpha=0.8)
            # current points
            ax.scatter(gt_pos[n, t, 0], gt_pos[n, t, 1], c='g', s=16)
            ax.scatter(pred_pos[n, t, 0], pred_pos[n, t, 1], c='r', s=16)
        ax.set_xlim([xmin, xmax]); ax.set_ylim([ymin, ymax]); ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{title}  t = {t+1}/{T}")
        ax.set_xlabel("X"); ax.set_ylabel("Y")

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
    plt.close(fig)

    try:
        iio.imwrite(save_path, frames, fps=fps, codec="h264", quality=8)
    except Exception:
        # fallback to gif if mp4 writer is unavailable
        gif_path = os.path.splitext(save_path)[0] + ".gif"
        iio.imwrite(gif_path, frames, fps=fps)
        print(f"MP4 writer unavailable, saved GIF instead at {gif_path}")

def save_static_grid(pred_pos, gt_pos, out_png, max_samples=16):
    if torch.is_tensor(pred_pos): pred_pos = pred_pos.detach().cpu().numpy()
    if torch.is_tensor(gt_pos):   gt_pos   = gt_pos.detach().cpu().numpy()
    K = min(max_samples, pred_pos.shape[0])
    rows, cols = 4, 4
    plt.figure(figsize=(12, 10))
    for i in range(K):
        ax = plt.subplot(rows, cols, i+1)
        ax.plot(gt_pos[i, :, 0], gt_pos[i, :, 1], 'g-', linewidth=1.5, alpha=0.8, label='GT' if i==0 else None)
        ax.plot(pred_pos[i, :, 0], pred_pos[i, :, 1], 'r-', linewidth=1.2, alpha=0.8, label='Pred' if i==0 else None)
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3); ax.set_title(f"#{i}")
    if K > 0: plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()


def get_angle_diff(a, b):
    """Calculates the shortest angle difference between two angles in radians."""
    diff = a - b
    return torch.atan2(torch.sin(diff), torch.cos(diff))

def train(model, device, data_loader, optimizer, collision_penalty=False):
    model.train()
    total_loss = 0

    step_weights = torch.ones(pred_len, device=device)
    step_weights[:5] *= 5
    step_weights[0] *= 5
    dist_threshold = 4

    for batch in data_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # Prepare input features: [x, y, yaw, intention(3-bit)]
        # Assuming batch.x columns are: 0:x, 1:y, 2:v, 3:yaw, 4:int1, 5:int2, 6:int3
        # pdb.set_trace()
        input_features = batch.x[:, [0, 1, 3, 4, 5, 6]] # [N_nodes, 6]

        # Get model output
        out = model(input_features, batch.edge_index) # [N_nodes, 6] -> [N_nodes, 90]
        out = out.reshape(-1, pred_len, 3)  # [N_nodes, 90] -> [N_nodes, 30, 3 (x,y,angle)]
        
        pred_pos = out[:, :, :2] # Predicted positions [N_nodes, 30, 2] (x, y)
        pred_angle = out[:, :, 2] # Predicted angles [N_nodes, 30]
        # pdb.set_trace()

        # Prepare ground truth
        gt = batch.y.reshape(-1, pred_len, 6) # y is [N_nodes, 180] -> [N_nodes, 30, 6 (x,y,v,yaw,acc,steering)]
        gt_pos = gt[:, :, [0, 1]]
        gt_angle = gt[:, :, 3]
        # pdb.set_trace()

        # --- Calculate Loss Components ---
        # 1. Position Loss (MSE)
        loss_pos = ((gt_pos - pred_pos).square().sum(-1) * step_weights).sum(-1)

        # 2. Angle Loss (Handles wrapping)
        angle_diff = get_angle_diff(pred_angle, gt_angle)
        loss_angle = (angle_diff.square() * step_weights).sum(-1)

        # Combine losses with weights
        combined_error = loss_pos + ANGLE_WEIGHT * loss_angle
        loss = (batch.weights * combined_error).nanmean()

        if epoch % 20 == 0:  # Save plots
            train_plot_path = os.path.join(model_path, f'train_plots/')
            os.makedirs(train_plot_path, exist_ok=True)
            plot_trajectory(pred_pos.cpu(), gt_pos.cpu(), train_plot_path, epoch)

        # 3. Collision Penalty
        if collision_penalty:
            mask = (batch.edge_index[0,:] < batch.edge_index[1,:])
            _edge = batch.edge_index[:, mask].T
            dist = torch.linalg.norm(pred_pos[_edge[:,0]] - pred_pos[_edge[:,1]], dim=-1)
            dist = dist_threshold - dist[dist < dist_threshold] # Only consider edges within the threshold
            loss_collision_penalty = dist.square().mean()
            loss += loss_collision_penalty * COLLISION_WEIGHT

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(data_loader), loss_angle * ANGLE_WEIGHT, loss_collision_penalty if collision_penalty else 0

def evaluate(model, device, data_loader, epoch=None, viz_root=None, render_video=False):
    """
    Evaluate the model on the validation set.
    Metrics:
    - ADE: Average Displacement Error
    - FDE: Final Displacement Error
    - MR: Miss Rate (percentage of trajectories with FDE > 4)
    - MAE: Mean Absolute Angle Error
    - FAE: Final Angle Error
    - Collision Rate: Percentage of edges with distance < 2
    """
    model.eval()
    ade_list, fde_list, angle_err_list, final_angle_err_list = [], [], [], []
    n_edge, n_collision = [], []
    dist_threshold = 4
    mr_threshold = 4

    with torch.no_grad():
        first_render_done = False
        for b_idx, batch in enumerate(data_loader):
            batch = batch.to(device)

            input_features = batch.x[:, [0, 1, 3, 4, 5, 6]]
            out = model(input_features, batch.edge_index).reshape(-1, pred_len, 3)
            pred_pos = out[:, :, :2]
            pred_angle = out[:, :, 2]

            gt = batch.y.reshape(-1, pred_len, 6)
            gt_pos = gt[:, :, [0, 1]]
            gt_angle = gt[:, :, 3]

            pos_error_per_step = torch.linalg.norm(gt_pos - pred_pos, dim=-1)
            ade_list.append(pos_error_per_step.mean(dim=-1))
            fde_list.append(pos_error_per_step[:, -1])

            angle_error_per_step = torch.abs(get_angle_diff(pred_angle, gt_angle)) % (2 * math.pi)
            angle_err_list.append(angle_error_per_step.mean(dim=-1))
            final_angle_err_list.append(angle_error_per_step[:, -1])

            # Collision metrics
            mask = (batch.edge_index[0, :] < batch.edge_index[1, :])
            _edge = batch.edge_index[:, mask].T
            if _edge.shape[0] > 0:
                # min distance over time for each pair
                d_all = torch.linalg.norm(pred_pos[_edge[:,0]] - pred_pos[_edge[:,1]], dim=-1)  # [E, T]
                min_dist = d_all.min(dim=-1)[0]  # [E]
                n_edge.append(len(min_dist))
                n_collision.append((min_dist < 2).sum().item())

            # Render only once per eval if requested
            if render_video and not first_render_done and viz_root is not None:
                os.makedirs(viz_root, exist_ok=True)
                # pick the first graph in this batch
                slices = get_graph_slices(batch)
                g0 = slices[0]
                p0 = pred_pos[g0[0]:g0[1]].cpu()
                g0pos = gt_pos[g0[0]:g0[1]].cpu()
                # static grid
                save_static_grid(p0, g0pos, os.path.join(viz_root, f"val_grid_epoch_{epoch}.png"))
                # animated video
                render_scene_video(
                    p0, g0pos,
                    save_path=os.path.join(viz_root, f"val_scene_epoch_{epoch}.mp4"),
                    fps=5,
                    title=f"Epoch {epoch}  Validation"
                )
                first_render_done = True

    ade = torch.cat(ade_list).mean().item()
    fde_all = torch.cat(fde_list)
    fde = fde_all.mean().item()
    mr = ((fde_all > mr_threshold).sum() / len(fde_all)).item()
    mae = torch.cat(angle_err_list).mean().item()
    fae = torch.cat(final_angle_err_list).mean().item()
    collision_rate = (sum(n_collision) / sum(n_edge)) if sum(n_edge) > 0 else 0
    return ade, fde, mr, mae, fae, collision_rate

# --- Training Loop ---
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # Transformers often prefer smaller LRs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',      # Minimizar la métrica (FDE)
    factor=0.1,      # Reducir LR por 10 (nuevo_lr = lr * 0.1)
    patience=10,     # Esperar 10 evaluaciones sin mejora
    min_lr=1e-7,     # LR mínimo
    verbose=True     # Imprimir cuando se reduzca el LR
)

min_fde = 1e6
best_epoch = 0
record = []

for epoch in tqdm(range(0, args.epoch)):
    loss, loss_angle, loss_collision = train(model, device, train_loader, optimizer, collision_penalty=collision_penalty)
    scheduler.step(loss)

    if epoch % 5 == 0:
        ade, fde, mr, mae, fae, cr = evaluate(model, device, val_loader)
        # ade = Average Displacement Error. This is the mean distance between predicted and ground truth positions.
        # fde = Final Displacement Error. This is the distance between the final predicted and ground truth positions.
        # mr = Miss Rate (percentage of trajectories with FDE > 4). This indicates how many trajectories were significantly off.
        # mae = Mean Absolute Angle Error. This is the mean absolute difference between predicted and ground truth angles.
        # fae = Final Angle Error. This is the difference between the final predicted and ground truth angles.
        # cr = Collision Rate (percentage of edges with distance < 2). This indicates how many edges are in collision.
        record.append([ade, fde, mr, mae, fae, cr])
        print(f"\nEpoch {epoch}: Train Loss: {loss:.4f}, ADE: {ade:.4f}, FDE: {fde:.4f}, MR: {mr:.4f}, "
              f"MAE(angle): {mae:.4f}, FAE(angle): {fae:.4f}, CR: {cr:.4f}, "
              f"lr: {optimizer.param_groups[0]['lr']:.6f}.")
        
        if wandb_on:
            wandb.log({
                "epoch": epoch, "train_loss": loss,
                "train_loss_angle": loss_angle,
                "train_loss_collision": loss_collision,
                "val_ade": ade, "val_fde": fde, "val_mr": mr,
                "val_mae_angle": mae, "val_fae_angle": fae,
                "val_collision_rate": cr,
                "lr": optimizer.param_groups[0]['lr'],
            }, step=epoch)
            
        if fde < min_fde:
            min_fde = fde
            best_epoch = epoch
            print("!!! New smallest FDE, saving model !!!")
            torch.save(model.state_dict(), model_path + f"/model_transformer_best.pth")
            
# Save final model and records
torch.save(model.state_dict(), model_path + f"/model_transformer_final.pth")
pkl_file = f"transformer_records_{exp_id}.pkl"
with open(f'{model_path}/{pkl_file}', 'wb') as handle:
    pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)