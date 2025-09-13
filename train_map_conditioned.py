# train_map_conditioned.py
import os
import glob
import argparse
import pickle
from datetime import datetime

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader

from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

import wandb

# ----------------------------
# Utils
# ----------------------------

def key_from_filename(p: str) -> str:
    base = os.path.basename(p)
    name, _ = os.path.splitext(base)
    return name.replace("_binary", "")


def rotation_matrix_back(yaw):
    R = np.array([[np.cos(-np.pi / 2 + yaw), -np.sin(-np.pi / 2 + yaw)],
                  [np.sin(-np.pi / 2 + yaw),  np.cos(-np.pi / 2 + yaw)]], dtype=np.float32)
    return torch.from_numpy(R)


def build_image_cache(image_dir: str, keys: list, img_size: int = 224):
    """Return dict: key -> [3,H,W] float tensor on CPU."""
    tfm = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])
    # map file base to path
    name_to_path = {}
    for p in glob.glob(os.path.join(image_dir, "*.png")):
        base = os.path.basename(p)
        k = os.path.splitext(base)[0].replace("_binary", "")
        name_to_path[k] = p

    cache = {}
    missing = []
    for k in sorted(set(keys)):
        p = name_to_path.get(k)
        if p is None:
            missing.append(k)
            cache[k] = torch.zeros(3, img_size, img_size)
        else:
            img = Image.open(p).convert("RGB")
            cache[k] = tfm(img)
    if missing:
        print(f"[warn] {len(missing)} maps missing images, using zeros. Examples: {missing[:5]}")
    return cache


@torch.no_grad()
def precompute_z_if_frozen(encoder: nn.Module, image_cache: dict, device: torch.device):
    """Return dict key -> z tensor on CPU. Encoder not updated."""
    encoder.eval()
    key2z = {}
    keys = list(image_cache.keys())
    B = 64
    for i in range(0, len(keys), B):
        ks = keys[i:i + B]
        imgs = torch.stack([image_cache[k] for k in ks]).to(device)  # [B,3,H,W]
        z = encoder(imgs).cpu()
        for k, zi in zip(ks, z):
            key2z[k] = zi
    return key2z

# ----------------------------
# Dataset
# ----------------------------

class CarMultiMapImageDataset(InMemoryDataset):
    """
    Loads all PKL graphs from subfolders under a root.
    Stores only tensors needed for training plus a string 'map_key' per graph.
    """
    def __init__(self,
                 pkls_root: str,
                 image_dir: str,
                 img_size: int = 224,
                 mpc_aug: bool = True):
        self.pkls_root = pkls_root
        self.image_dir = image_dir
        self.img_size = img_size
        self.mpc_aug = mpc_aug
        super().__init__(root=".")
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        tag = f"cache_{os.path.basename(self.pkls_root)}_{self.img_size}{'_aug' if self.mpc_aug else ''}.pt"
        return [tag]

    def process(self):
        graphs = []
        subdirs = [os.path.join(self.pkls_root, d) for d in os.listdir(self.pkls_root)
                   if os.path.isdir(os.path.join(self.pkls_root, d))]
        subdirs.sort()

        print("Processing subdirectories:")
        for folder in subdirs:
            print(f"Extracting data from {folder}")
            files = sorted(glob.glob(os.path.join(folder, "*.pkl")))
            if not files:
                continue

            # folder name encodes the map key
            key = os.path.basename(folder)
            key = key.split('_', 3)[-1]  # drop train/val_pre_1k_

            for pkl_path in files:
                if not self.mpc_aug and os.path.splitext(pkl_path)[0].split('-')[-1] != '0':
                    continue

                data = pickle.load(open(pkl_path, "rb"))
                x_np, y_np, edge_index, t = data[0], data[1], data[2], data[3]

                x = torch.as_tensor(x_np, dtype=torch.float32)
                y = torch.as_tensor(y_np, dtype=torch.float32)
                ei = torch.as_tensor(edge_index, dtype=torch.long)

                n_v = x.size(0)
                weights = torch.ones(n_v, dtype=torch.float32)
                turn_index = (x[:, 4] + x[:, 6]).bool()
                center1 = (x[:, 0].abs() < 30) & (x[:, 1].abs() < 30)
                center2 = (x[:, 0].abs() < 40) & (x[:, 1].abs() < 40)
                weights[turn_index] *= 1.5
                weights[center1] *= 4
                weights[center2] *= 4

                g = Data(x=x, y=y, edge_index=ei, t=t, weights=weights)
                g.map_key = key  # string, collated to list in batches
                graphs.append(g)

            break 
        
        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])


# ----------------------------
# Map encoder (ResNet18)
# ----------------------------
class ResNetMapEncoder(nn.Module):
    """
    ResNet18 encoder with ImageNet weights.
    Outputs a vector (z_dim). If z_dim is None, uses backbone dim (512).
    """
    def __init__(self, z_dim: int = 128, finetune: bool = False):
        super().__init__()
        # Handle both older and newer torchvision versions
        try:
            # For newer torchvision versions
            m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except (AttributeError, TypeError):
            # For older torchvision versions
            m = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # [B,512,1,1]
        self.out_dim = 512
        self.proj = nn.Identity() if z_dim is None or z_dim == self.out_dim else nn.Linear(self.out_dim, z_dim)
        self.freeze = not finetune
        if self.freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, img_bchw: torch.Tensor) -> torch.Tensor:
        if self.freeze:
            self.backbone.eval()
            with torch.no_grad():
                f = self.backbone(img_bchw)
        else:
            f = self.backbone(img_bchw)
        f = f.flatten(1)  # [B,512]
        z = self.proj(f)  # [B,z_dim]
        return z


# ----------------------------
# Node models
# ----------------------------
class GNN_mtl_gnn_map(nn.Module):
    def __init__(self, hidden_channels: int, in_dim: int):
        super().__init__()
        from torch_geometric.nn import GraphConv
        self.conv1 = GraphConv(hidden_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.linear1 = nn.Linear(in_dim, 64)
        self.linear2 = nn.Linear(64, hidden_channels)
        self.linear3 = nn.Linear(hidden_channels, hidden_channels)
        self.linear4 = nn.Linear(hidden_channels, hidden_channels)
        self.linear5 = nn.Linear(hidden_channels, 30 * 2)

    def forward(self, x, edge_index):
        x = self.linear1(x).relu()
        x = self.linear2(x).relu()
        x = self.linear3(x).relu() + x
        x = self.linear4(x).relu() + x
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.linear5(x)
        return x


class GNN_mtl_mlp_map(nn.Module):
    def __init__(self, hidden_channels: int, in_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, hidden_channels)
        self.fc3 = nn.Linear(hidden_channels, hidden_channels)
        self.fc4 = nn.Linear(hidden_channels, hidden_channels)
        self.out = nn.Linear(hidden_channels, 30 * 2)

    def forward(self, x, edge_index=None):
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        x = self.fc3(x).relu() + x
        x = self.fc4(x).relu() + x
        x = self.out(x)
        return x


# ----------------------------
# Wrapper that holds the encoder and core
# ----------------------------
class MapConditionedModel(nn.Module):
    def __init__(self, base_in_dim: int, z_dim: int, hidden: int, use_gnn: bool, finetune_encoder: bool):
        super().__init__()
        self.map_encoder = ResNetMapEncoder(z_dim=z_dim, finetune=finetune_encoder)
        in_dim = base_in_dim + (z_dim if z_dim is not None else self.map_encoder.out_dim)
        self.core = GNN_mtl_gnn_map(hidden, in_dim) if use_gnn else GNN_mtl_mlp_map(hidden, in_dim)


# ----------------------------
# Train / Eval
# ----------------------------
def forward_core_with_maps(model, base_x, edge_index, batch_vec, map_keys, image_cache, key2z_frozen, device):
    """Build per-graph z, broadcast to nodes, and run the core."""
    # map_keys is a list[str] of length num_graphs in this batch
    if key2z_frozen is not None:
        z_graph = torch.stack([key2z_frozen[k] for k in map_keys]).to(device)  # [G,z]
    else:
        imgs = torch.stack([image_cache[k] for k in map_keys]).to(device)      # [G,3,H,W]
        z_graph = model.map_encoder(imgs)                                      # [G,z]
    z_nodes = z_graph[batch_vec]                                               # [N,z]
    x_in = torch.cat([base_x, z_nodes], dim=1)
    out = model.core(x_in, edge_index)
    return out


def train_epoch(model, loader, optimizer, device, image_cache, key2z_frozen=None):
    model.train()
    total = 0.0
    step_weights = torch.ones(30, device=device)
    step_weights[:5] *= 5
    step_weights[0] *= 5

    for batch in loader:
        batch = batch.to(device)
        base_x = batch.x[:, [0, 1, 4, 5, 6]]  # 5 dims
        keys = batch.map_key  # list[str], one per graph

        out = forward_core_with_maps(model, base_x, batch.edge_index, batch.batch,
                                     keys, image_cache, key2z_frozen, device)

        # back to global XY
        out = out.reshape(-1, 30, 2).permute(0, 2, 1)
        yaw = batch.x[:, 3].detach().cpu().numpy()
        R = torch.stack([rotation_matrix_back(y) for y in yaw]).to(out.device)
        out = torch.bmm(R, out).permute(0, 2, 1)
        out += batch.x[:, [0, 1]].unsqueeze(1)

        gt = batch.y.reshape(-1, 30, 6)[:, :, [0, 1]]
        err = ((gt - out).square().sum(-1) * step_weights).sum(-1)

        w = getattr(batch, "weights", None)
        if w is None:
            w = torch.ones_like(err)
        loss = (w * err).nanmean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()

    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, image_cache, key2z_frozen=None):
    model.eval()
    step_weights = torch.ones(30, device=device)
    step_weights[:5] *= 5
    step_weights[0] *= 5
    mr_threshold = 4.0
    ade_list, fde_list, n_edge, n_collision, val_losses = [], [], [], [], []

    for batch in loader:
        batch = batch.to(device)
        base_x = batch.x[:, [0, 1, 4, 5, 6]]
        keys = batch.map_key

        out = forward_core_with_maps(model, base_x, batch.edge_index, batch.batch,
                                     keys, image_cache, key2z_frozen, device)

        out = out.reshape(-1, 30, 2).permute(0, 2, 1)
        yaw = batch.x[:, 3].detach().cpu().numpy()
        R = torch.stack([rotation_matrix_back(y) for y in yaw]).to(out.device)
        out = torch.bmm(R, out).permute(0, 2, 1)
        out += batch.x[:, [0, 1]].unsqueeze(1)

        gt = batch.y.reshape(-1, 30, 6)[:, :, [0, 1]]
        _err = (gt - out).square().sum(-1)
        d = _err.sqrt()
        ade_list.append(d.mean(dim=-1))
        fde_list.append(d[:, -1])

        _err_w = (_err * step_weights).sum(-1)
        w = getattr(batch, "weights", None)
        if w is None:
            w = torch.ones_like(_err_w)
        val_losses.append((w * _err_w).nanmean())

        # collisions
        mask = (batch.edge_index[0, :] < batch.edge_index[1, :])
        ei = batch.edge_index[:, mask].T
        if ei.numel() > 0:
            dist = torch.linalg.norm(out[ei[:, 0]] - out[ei[:, 1]], dim=-1)
            m = dist.min(dim=-1)[0]
            n_edge.append(len(m))
            n_collision.append((m < 2.0).sum().item())

    ade = torch.cat(ade_list).mean().item()
    fde_all = torch.cat(fde_list)
    fde = fde_all.mean().item()
    mr = ((fde_all > mr_threshold).sum() / fde_all.numel()).item()
    cr = (sum(n_collision) / sum(n_edge)) if sum(n_edge) > 0 else 0.0
    vloss = torch.stack(val_losses).mean().item()
    return ade, fde, mr, cr, vloss


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser("Train with map-conditioned ResNet encoder")
    ap.add_argument("--train_root", type=str, required=True, help="folder containing many PKL folders for training")
    ap.add_argument("--val_root",   type=str, required=True, help="folder containing many PKL folders for validation")
    ap.add_argument("--image_dir",  type=str, required=True, help="folder with *_binary.png images")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--z_dim", type=int, default=128, help="projection dim from ResNet encoder")
    ap.add_argument("--img_size", type=int, default=224, help="resize for the map images")
    ap.add_argument("--use_gnn", action="store_true", help="use GNN core (default MLP if not set)")
    ap.add_argument("--finetune_encoder", action="store_true", help="update ResNet weights")
    ap.add_argument("--out", type=str, default="trained_params/with_map_resnet")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # datasets
    train_ds = CarMultiMapImageDataset(args.train_root, args.image_dir, img_size=args.img_size, mpc_aug=True)
    val_ds   = CarMultiMapImageDataset(args.val_root,   args.image_dir, img_size=args.img_size, mpc_aug=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    # model
    model = MapConditionedModel(
        base_in_dim=5,
        z_dim=args.z_dim,
        hidden=args.hidden,
        use_gnn=args.use_gnn,
        finetune_encoder=args.finetune_encoder
    ).to(device)
    print(model)

    #wandb
    wandb_on = False
    if wandb_on:
        run_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.map}_Multimap"
        wandb.init(
            project="MTP",
            entity="danivg",
            name=run_name,
            config={
                "epoch": args.epochs,
                "batch_size": args.batch_size,
                "mlp": not args.use_gnn,
                "collision_penalty": True,
                "exp_id": args.out,
                "pred_len": 30,
                "dt": 0.1
            },
            sync_tensorboard=True,
            save_code=True,
        )

    # collect unique map keys
    def all_keys(ds):
        ks = set()
        for i in range(len(ds)):
            ks.add(ds.get(i).map_key)
        return sorted(ks)

    keys = sorted(set(all_keys(train_ds) + all_keys(val_ds)))

    # image cache once on CPU
    image_cache = build_image_cache(args.image_dir, keys, img_size=args.img_size)

    # if encoder is frozen, precompute z for speed
    key2z_frozen = None
    if not args.finetune_encoder:
        key2z_frozen = precompute_z_if_frozen(model.map_encoder, image_cache, device)

    # train
    os.makedirs(args.out, exist_ok=True)
    run_dir = os.path.join(args.out, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    best_fde = 1e9

    for epoch in range(args.epochs):
        tr = train_epoch(model, train_loader, opt, device, image_cache, key2z_frozen)
        if epoch % 5 == 0:
            ade, fde, mr, cr, vloss = evaluate(model, val_loader, device, image_cache, key2z_frozen)
            print(f"epoch {epoch} | train {tr:.4f} | ADE {ade:.3f} FDE {fde:.3f} MR {mr:.3f} CR {cr:.3f} | val {vloss:.3f}")
            torch.save(model.state_dict(), os.path.join(run_dir, f"model_{epoch:04d}.pth"))
            if wandb_on:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": tr,
                    "val_loss": vloss,
                    "ADE": ade,
                    "FDE": fde,
                    "MR": mr,
                    "CR": cr
                })
            if fde < best_fde:
                best_fde = fde
                torch.save(model.state_dict(), os.path.join(run_dir, "model_best.pth"))

    torch.save(model.state_dict(), os.path.join(run_dir, "model_final.pth"))


if __name__ == "__main__":
    main()
