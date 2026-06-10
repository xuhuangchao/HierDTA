"""Drug and pocket-residue encoders for DTAModel."""

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

from .egnn_clean import EGNN


class AtomGNN(nn.Module):
    """Reusable homogeneous branch over one node/edge type in a HeteroData."""

    def __init__(
        self,
        in_dim=37,
        hidden_dim=128,
        edge_dim=13,
        num_layers=2,
        dropout=0.2,
        node_key="atom",
        edge_key=("atom", "bond", "atom"),
    ):
        super().__init__()
        self.node_key = node_key
        self.edge_key = edge_key
        self.convs = nn.ModuleList(
            [
                GATv2Conv(
                    in_dim if i == 0 else hidden_dim,
                    hidden_dim,
                    edge_dim=edge_dim,
                    dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )

    def forward(self, data):
        x = data[self.node_key].x
        edge_index = data[self.edge_key].edge_index
        edge_attr = data[self.edge_key].edge_attr
        for conv in self.convs:
            x = F.elu(conv(x, edge_index, edge_attr=edge_attr))

        return x, data[self.node_key].batch


class PocketGraphEncoder(nn.Module):
    """Pocket residue encoder with E(n) equivariant message passing."""

    def __init__(self, pocket_in_dim=608, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.egnn = EGNN(
            in_node_nf=pocket_in_dim,
            hidden_nf=hidden_dim,
            out_node_nf=hidden_dim,
            in_edge_nf=2,
            n_layers=num_layers,
            residual=True,
            normalize=True,
            tanh=False,
        )

    def forward(self, x, edge_index, edge_attr, coords, batch):
        h, _ = self.egnn(x, coords, edge_index, edge_attr=edge_attr)
        return h, batch
