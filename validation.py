#!/usr/bin/env python3
# validate.py
import argparse
import os
from datetime import datetime

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from dataset import CarDataset
from model import GNN_mtl_gnn, GNN_mtl_mlp
from utils.config import DT, OBS_LEN, PRED_LEN           # keep these in sync with training

# ──────────────────────────────
# helpers
# ──────────────────────────────
def rotation_matrix_back(yaw: float) -> torch.Tensor:
    """Undo the ego-centric 90° rotation applied in preprocessing."""
    m = np.array([
        [np.cos(-np.pi/2 + yaw), -np.sin(-np.pi/2 + yaw)],
        [np.sin(-np.pi/2 + yaw),  np.cos(-np.pi/2 + yaw)],
    ])
    return torch.tensor(m, dtype=torch.float32)

@torch.no_grad()
def evaluate(model: torch.nn.Module,
             loader: DataLoader,
             device: torch.device):
    """Return ADE, FDE, miss-rate, collision-rate, val-loss, collision-penalty."""
    step_weights = torch.ones(30, device=device)
    step_weights[:5] *= 5
    step_weights[0]  *= 5
    dist_thr, mr_thr = 4, 4

    ade, fde, n_edge, n_col = [], [], [], []
    val_losses, col_penalties = [], []

    model.eval()
    for batch in loader:
        batch = batch.to(device)
        # ── forward ──────────────────────────────────
        out = model(batch.x[:, [0, 1, 4, 5, 6]], batch.edge_index) \
                .reshape(-1, 30, 2)                     # [N, 30, 2]
        # bring to world frame
        rot = torch.stack([rotation_matrix_back(y) for y in batch.x[:, 3].cpu()]).to(device)
        out = torch.bmm(rot, out.permute(0, 2, 1)).permute(0, 2, 1)
        out += batch.x[:, [0, 1]].unsqueeze(1)

        gt = batch.y.reshape(-1, 30, 6)[:, :, :2]        # target xy
        error = (gt - out).pow(2).sum(-1).sqrt()         # [N, 30]

        ade.append(error.mean(1))                        # mean over time
        fde.append(error[:, -1])                         # final step

        # weighted loss (same as train)
        w_err = ((gt - out).pow(2).sum(-1) * step_weights).sum(-1)
        val_losses.append((batch.weights * w_err).nanmean())

        # collision penalty
        mask  = batch.edge_index[0] < batch.edge_index[1]
        edges = batch.edge_index[:, mask].T              # [E, 2]
        dist  = torch.linalg.norm(out[edges[:, 0]] - out[edges[:, 1]], dim=-1)
        cp    = (dist_thr - dist[dist < dist_thr]).pow(2).mean() * 20
        col_penalties.append(cp)

        # collision rate
        min_dist = torch.min(dist, dim=-1).values
        n_edge.append(len(min_dist))
        n_col.append((min_dist < 2).sum().item())

    ade = torch.cat(ade).mean().item()
    fde = torch.cat(fde)
    mr  = (fde > mr_thr).float().mean().item()
    fde = fde.mean().item()
    cr  = sum(n_col) / sum(n_edge)
    val_loss = torch.tensor(val_losses).mean().item()
    cp  = torch.tensor(col_penalties).mean().item()
    return ade, fde, mr, cr, val_loss, cp

# ──────────────────────────────
# CLI
# ──────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True,
                        help="path to .pth checkpoint")
    parser.add_argument("--data_folder", default="csv/val_pre",
                        help="folder with pre-processed csv files")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--mlp", action="store_true",
                        help="load the MLP variant instead of the GNN")
    parser.add_argument("--hidden_channels", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] device: {device}")

    # dataset & loader
    ds = CarDataset(preprocess_folder=args.data_folder,
                    mlp=args.mlp,
                    mpc_aug=True)
    loader = DataLoader(ds,
                        batch_size=args.batch_size,
                        shuffle=False,
                        num_workers=0,
                        drop_last=False)

    # model
    Model = GNN_mtl_mlp if args.mlp else GNN_mtl_gnn
    model = Model(hidden_channels=args.hidden_channels).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    print(f"Loaded weights from {args.weights}")

    # run
    metrics = evaluate(model, loader, device)
    names = ("ADE", "FDE", "Miss-Rate", "Collision-Rate",
             "Val-Loss", "Collision-Penalty")
    for k, v in zip(names, metrics):
        print(f"{k:17s}: {v:.6f}")

if __name__ == "__main__":
    main()
