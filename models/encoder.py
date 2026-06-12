"""Drug and pocket-residue encoders for DTAModel."""

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, GINConv, global_add_pool, global_max_pool, global_mean_pool


class AtomGNN(nn.Module):
    """Reusable homogeneous graph branch over one node/edge type in a HeteroData."""

    def __init__(
        self,
        in_dim=37,
        edge_dim=13,
        num_layers=1,
        heads=2,
        gat_dropout=0.0,
        graph_pool_type="mean_add_max",
        graph_out_dim=1024,
        post_dropout=0.3,
        node_key="atom",
        edge_key=("atom", "bond", "atom"),
    ):
        super().__init__()
        if num_layers != 1:
            raise ValueError("AtomGNN is configured as a single GATv2Conv layer.")
        self.node_key = node_key
        self.edge_key = edge_key
        self.graph_pool_type = graph_pool_type
        self.conv = GATv2Conv(
            in_dim,
            in_dim,
            heads=heads,
            concat=True,
            edge_dim=edge_dim,
            dropout=gat_dropout,
        )
        self.activation = nn.ReLU()

        conv_out_dim = in_dim * heads
        pooled_dim = self._pooled_dim(conv_out_dim, graph_pool_type)
        self.graph_proj = nn.Sequential(
            nn.Linear(pooled_dim, graph_out_dim),
            nn.ReLU(),
            nn.Dropout(post_dropout),
        )

    @staticmethod
    def _pooled_dim(hidden_dim, pool_type):
        parts = pool_type.split("_")
        return hidden_dim * len(parts)

    @staticmethod
    def _pool(x, batch, pool_type):
        pooled = []
        for pool_name in pool_type.split("_"):
            if pool_name == "mean":
                pooled.append(global_mean_pool(x, batch))
            elif pool_name == "add":
                pooled.append(global_add_pool(x, batch))
            elif pool_name == "max":
                pooled.append(global_max_pool(x, batch))
            else:
                raise ValueError(f"Unsupported graph_pool_type component: {pool_name}")
        return pooled[0] if len(pooled) == 1 else torch.cat(pooled, dim=-1)

    def forward(self, data):
        x = data[self.node_key].x
        batch = data[self.node_key].batch
        edge_index = data[self.edge_key].edge_index
        edge_attr = data[self.edge_key].edge_attr
        x = self.activation(self.conv(x, edge_index, edge_attr=edge_attr))

        return self.graph_proj(self._pool(x, batch, self.graph_pool_type))


class PocketGraphEncoder(nn.Module):
    """Protein graph encoder with GIN message passing."""

    def __init__(
        self,
        pocket_in_dim=41,
        hidden_dim=256,
        num_layers=1,
        graph_pool_type="mean_add_max",
        graph_out_dim=1024,
        post_dropout=0.3,
    ):
        super().__init__()
        self.graph_pool_type = graph_pool_type
        self.convs = nn.ModuleList()
        for layer_idx in range(num_layers):
            in_dim = pocket_in_dim if layer_idx == 0 else hidden_dim
            self.convs.append(
                GINConv(
                    nn.Sequential(
                        nn.Linear(in_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                )
            )
        self.activation = nn.ReLU()
        self.graph_proj = nn.Sequential(
            nn.Linear(AtomGNN._pooled_dim(hidden_dim, graph_pool_type), graph_out_dim),
            nn.BatchNorm1d(graph_out_dim),
            nn.ReLU(),
            nn.Dropout(post_dropout),
        )

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = self.activation(conv(x, edge_index))
        return self.graph_proj(AtomGNN._pool(x, batch, self.graph_pool_type))
