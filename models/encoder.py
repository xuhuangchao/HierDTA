"""Drug, residue-graph, and protein-surface encoders for DTAModel."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.utils import scatter

from .egnn_clean import EGNN


# ── 1. 分子 GNN (通用: atom / motif) ──
class AtomGNN(nn.Module):
    """Molecular graph encoder — works for both atom-bond and motif-connect graphs.

    node_key / edge_key determine which HeteroData fields are consumed,
    making the same class reusable for atom-level and motif-level encoding.
    """
    def __init__(self, in_dim=37, hidden_dim=128, edge_dim=13, num_layers=2, dropout=0.2,
                 node_key="atom", edge_key=("atom", "bond", "atom")):
        super().__init__()
        self.node_key = node_key
        self.edge_key = edge_key
        self.conv1 = GATv2Conv(in_dim, hidden_dim, edge_dim=edge_dim, dropout=dropout)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, edge_dim=edge_dim, dropout=dropout)

    def forward(self, data):
        x = data[self.node_key].x
        ei = data[self.edge_key].edge_index
        ea = data[self.edge_key].edge_attr

        x = F.relu(self.conv1(x, ei, edge_attr=ea))
        x = F.relu(self.conv2(x, ei, edge_attr=ea))

        return x, data[self.node_key].batch


# ── 2. 口袋残基编码器 (EGNN) ──
class PocketGraphEncoder(nn.Module):
    """Pocket residue encoder with E(n) Equivariant GNN.

    EGNN updates both residue features and Cα coordinates through
    message passing.  The coordinate-update mechanism preserves 3D
    spatial structure; edges are distance-filtered (5A cutoff from
    data pipeline) to keep computation tractable.
    """
    def __init__(self, pocket_in_dim=649, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.egnn = EGNN(
            in_node_nf=pocket_in_dim,
            hidden_nf=hidden_dim,
            out_node_nf=hidden_dim,
            in_edge_nf=2,           # [min_dist*0.1, max_dist*0.1]
            n_layers=num_layers,
            residual=True,
            normalize=True,
            tanh=False,
        )
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, coords, batch):
        # x = self.dropout(x)
        h, _ = self.egnn(x, coords, edge_index, edge_attr=edge_attr)
        return h, batch
    
