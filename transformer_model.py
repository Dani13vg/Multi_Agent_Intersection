# transformer_model.py

import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv

class GraphTransformer(nn.Module):
    """
    Graph Transformer Network for trajectory prediction.
    - It uses TransformerConv to apply attention over the graph defined by edge_index.
    - It predicts global coordinates and angles directly.
    """
    def __init__(self, input_dim, hidden_channels, output_dim, num_heads=4):
        """
        Args:
            input_dim (int): Dimensionality of the input features per node (e.g., 6 for x, y, yaw, intention).
            hidden_channels (int): Dimensionality of the hidden layers.
            output_dim (int): Total dimensionality of the output prediction per node (e.g., PRED_LEN * 3 for x, y, angle).
            num_heads (int): Number of attention heads in the TransformerConv layers.
        """
        super().__init__()
        torch.manual_seed(21)
        
        # Input MLP to project raw features into the hidden dimension
        self.input_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )

        # Graph Transformer layers
        # TransformerConv applies self-attention based on the graph structure (edge_index)
        self.conv1 = TransformerConv(hidden_channels, hidden_channels, heads=num_heads, concat=False, dropout=0.1)
        self.conv2 = TransformerConv(hidden_channels, hidden_channels, heads=num_heads, concat=False, dropout=0.1)
        
        # Output MLP to project the final node embeddings to the desired trajectory dimensions
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, output_dim)
        )

    def forward(self, x, edge_index):
        """
        Forward pass of the Graph Transformer model.
        Args:
            x (torch.Tensor): Input node features of shape [num_nodes, input_dim].
            edge_index (torch.Tensor): Graph connectivity in COO format with shape [2, num_edges].
        Returns:
            torch.Tensor: Output tensor of shape [num_nodes, output_dim].
        """
        # 1. Embed input features
        h = self.input_mlp(x).relu() # x.shape: [num_nodes, input_dim] to [num_nodes, hidden_channels]
        
        # 2. Apply graph transformer layers with residual connections
        h_res = self.conv1(h, edge_index).relu() # h.shape: [num_nodes, hidden_channels] to [num_nodes, hidden_channels]
        h = h + h_res # Residual connection

        h_res = self.conv2(h, edge_index).relu() # h.shape: [num_nodes, hidden_channels] to [num_nodes, hidden_channels]
        h = h + h_res # Residual connection
        
        # 3. Predict the output trajectory
        out = self.output_mlp(h) # h.shape: [num_nodes, hidden_channels] to [num_nodes, output_dim]
        
        return out