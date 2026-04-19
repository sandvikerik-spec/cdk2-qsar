"""
model.py - AttentiveFP graph neural network for CDK2 pIC50 prediction.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import AttentiveFP


class CDK2GNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        hidden_channels = config["model"]["hidden_channels"]
        num_layers      = config["model"]["num_layers"]
        num_timesteps   = config["model"]["num_timesteps"]
        dropout         = config["model"]["dropout"]

        self.gnn = AttentiveFP(
            in_channels     = 150,
            hidden_channels = hidden_channels,
            out_channels    = 1,
            edge_dim        = 8,
            num_layers      = num_layers,
            num_timesteps   = num_timesteps,
            dropout         = dropout,
        )

    def forward(self, data):
        return self.gnn(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch
        ).squeeze(-1)
