# train_transformer.py

import argparse
import os
import pickle
import torch
from torch import nn
from torch_geometric.loader import DataLoader
from transformer_model import GraphTransformer # Import the new model
from tqdm import tqdm
import wandb
from datetime import datetime
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
parser.add_argument('--batch_size', type=int, help='batch size', default=10)
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
ANGLE_WEIGHT = 0.5    # Hyperparameter to balance position and angle loss
COLLISION_WEIGHT = 20.0
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
wandb_on = True  # Set to True to enable Weights & Biases logging
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
        input_features = batch.x[:, [0, 1, 3, 4, 5, 6]] # [N_nodes, 6]

        # Get model output
        out = model(input_features, batch.edge_index)
        out = out.reshape(-1, pred_len, 3)  # [N_nodes, 90] -> [N_nodes, 30, 3 (x,y,angle)]
        
        pred_pos = out[:, :, :2]
        pred_angle = out[:, :, 2]

        # Prepare ground truth
        gt = batch.y.reshape(-1, pred_len, 6) # y is [N_nodes, 180] -> [N_nodes, 30, 6 (x,y,v,yaw,acc,steering)]
        gt_pos = gt[:, :, [0, 1]]
        gt_angle = gt[:, :, 3]

        # --- Calculate Loss Components ---
        # 1. Position Loss (MSE)
        loss_pos = ((gt_pos - pred_pos).square().sum(-1) * step_weights).sum(-1)
        
        # 2. Angle Loss (Handles wrapping)
        angle_diff = get_angle_diff(pred_angle, gt_angle)
        loss_angle = (angle_diff.square() * step_weights).sum(-1)
        
        # Combine losses with weights
        combined_error = loss_pos + ANGLE_WEIGHT * loss_angle
        loss = (batch.weights * combined_error).nanmean()

        # 3. Collision Penalty
        if collision_penalty:
            mask = (batch.edge_index[0,:] < batch.edge_index[1,:])
            _edge = batch.edge_index[:, mask].T
            dist = torch.linalg.norm(pred_pos[_edge[:,0]] - pred_pos[_edge[:,1]], dim=-1)
            dist = dist_threshold - dist[dist < dist_threshold]
            _collision_penalty = dist.square().mean()
            loss += _collision_penalty * COLLISION_WEIGHT

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(data_loader)

def evaluate(model, device, data_loader):
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
    val_losses, collision_penalties = [], []
    
    dist_threshold = 4
    mr_threshold = 4

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            
            input_features = batch.x[:, [0, 1, 3, 4, 5, 6]]
            out = model(input_features, batch.edge_index)
            out = out.reshape(-1, pred_len, 3)

            pred_pos = out[:, :, :2]
            pred_angle = out[:, :, 2]
            
            gt = batch.y.reshape(-1, pred_len, 6)
            gt_pos = gt[:, :, [0, 1]]
            gt_angle = gt[:, :, 3]

            # Position error metrics
            pos_error_per_step = torch.linalg.norm(gt_pos - pred_pos, dim=-1) # [N_nodes, 30]
            ade_list.append(pos_error_per_step.mean(dim=-1)) # ADE per trajectory
            fde_list.append(pos_error_per_step[:, -1])       # FDE per trajectory
            
            # Angle error metrics
            angle_error_per_step = torch.abs(get_angle_diff(pred_angle, gt_angle))
            angle_err_list.append(angle_error_per_step.mean(dim=-1)) # Mean angle error
            final_angle_err_list.append(angle_error_per_step[:, -1])   # Final angle error

            # Collision metrics
            mask = (batch.edge_index[0,:] < batch.edge_index[1,:])
            _edge = batch.edge_index[:, mask].T
            if _edge.shape[0] > 0:
                dist = torch.linalg.norm(pred_pos[_edge[:,0]] - pred_pos[_edge[:,1]], dim=-1)
                min_dist_per_pair = torch.min(dist, dim=-1)[0]
                n_edge.append(len(min_dist_per_pair))
                n_collision.append((min_dist_per_pair < 2).sum().item())


    # Aggregate metrics
    ade = torch.cat(ade_list).mean().item()
    fde_all = torch.cat(fde_list)
    fde = fde_all.mean().item()
    mr = ((fde_all > mr_threshold).sum() / len(fde_all)).item()
    
    mae = torch.cat(angle_err_list).mean().item() # Mean Absolute Angle Error
    fae = torch.cat(final_angle_err_list).mean().item() # Final Angle Error

    collision_rate = sum(n_collision) / sum(n_edge) if sum(n_edge) > 0 else 0
    
    return ade, fde, mr, mae, fae, collision_rate

# --- Training Loop ---
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # Transformers often prefer smaller LRs
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

min_fde = 1e6
best_epoch = 0
record = []

for epoch in tqdm(range(0, args.epoch)):
    loss = train(model, device, train_loader, optimizer, collision_penalty=collision_penalty)
    scheduler.step()

    if epoch % 5 == 0:
        ade, fde, mr, mae, fae, cr = evaluate(model, device, val_loader)
        record.append([ade, fde, mr, mae, fae, cr])
        print(f"\nEpoch {epoch}: Train Loss: {loss:.4f}, ADE: {ade:.4f}, FDE: {fde:.4f}, MR: {mr:.4f}, "
              f"MAE(angle): {mae:.4f}, FAE(angle): {fae:.4f}, CR: {cr:.4f}, "
              f"lr: {optimizer.param_groups[0]['lr']:.6f}.")
        
        if wandb_on:
            wandb.log({
                "epoch": epoch, "train_loss": loss,
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