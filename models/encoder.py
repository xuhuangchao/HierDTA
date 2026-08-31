"""Drug and protein-residue encoders for DTAModel."""

import torch
import torch.nn as nn
from torch_geometric.nn import (
    GATv2Conv,
    GCNConv,
    GINConv,
    GINEConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

_DEFAULT_POOL_TYPE = "mean_add_max"


class DrugGraphEncoder(nn.Module):
    """Reusable homogeneous graph branch over one node/edge type in a HeteroData."""

    def __init__(
        self,
        in_dim=37,
        edge_dim=13,
        hidden_dim=256,
        num_layers=1,
        heads=2,
        input_dropout=0.1,
        graph_out_dim=1024,
        post_dropout=0.3,
        build_graph_head=True,
        gnn_type="gat",
        node_key="atom",
        edge_key=("atom", "bond", "atom"),
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gnn_type = gnn_type
        self.node_key = node_key
        self.edge_key = edge_key

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(input_dropout),
        )
        self.convs = nn.ModuleList()
        layer_in = hidden_dim
        for _ in range(num_layers):
            if gnn_type == "gat":
                conv = GATv2Conv(
                    layer_in,
                    hidden_dim // heads,
                    heads=heads,
                    concat=True,
                    edge_dim=edge_dim,
                )
            elif gnn_type == "gin":
                update_mlp = nn.Sequential(
                    nn.Linear(layer_in, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                conv = GINConv(update_mlp)
            else:
                conv = GCNConv(layer_in, hidden_dim)
            self.convs.append(conv)
            layer_in = hidden_dim
        self.activation = nn.ReLU()

        self.graph_proj = None
        if build_graph_head:
            pooled_dim = self._pooled_dim(hidden_dim, _DEFAULT_POOL_TYPE)
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
                raise ValueError(f"Unsupported pool_type component: {pool_name}")
        return pooled[0] if len(pooled) == 1 else torch.cat(pooled, dim=-1)

    @property
    def node_out_dim(self):
        """Width of the node embeddings produced by the final GNN layer."""
        return self.hidden_dim

    def encode_nodes(self, data, x_override=None):
        """Encode nodes without graph-level pooling.

        ``x_override`` can replace the raw node features before projection.
        """
        x = data[self.node_key].x if x_override is None else x_override
        x = self.input_proj(x)
        edge_index = data[self.edge_key].edge_index
        edge_attr = data[self.edge_key].edge_attr
        for conv in self.convs:
            if self.gnn_type == "gat":
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)
            x = self.activation(x)
        return x

    def readout(self, node_h, batch, pool_type=_DEFAULT_POOL_TYPE):
        """Pool node embeddings and project them to a graph representation."""
        if self.graph_proj is None:
            raise RuntimeError("Graph readout was disabled with build_graph_head=False")
        return self.graph_proj(self._pool(node_h, batch, pool_type))

    def forward(self, data, x_override=None):
        node_h = self.encode_nodes(data, x_override=x_override)
        return self.readout(node_h, data[self.node_key].batch)


class ProteinGraphEncoder(nn.Module):
    """Encode a residue graph with GINE over all 10-dimensional edge features."""

    def __init__(
        self,
        protein_in_dim=41,
        hidden_dim=256,
        num_layers=2,
        input_dropout=0.1,
        graph_out_dim=1024,
        post_dropout=0.3,
        protein_edge_dim=10,
        build_graph_head=True,
    ):
        super().__init__()

        self.protein_edge_dim = protein_edge_dim
        self.input_proj = nn.Sequential(
            nn.Linear(protein_in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(input_dropout),
        )
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            update_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(
                GINEConv(update_mlp, edge_dim=protein_edge_dim)
            )

        self.activation = nn.ReLU()
        self.graph_proj = None
        if build_graph_head:
            self.graph_proj = nn.Sequential(
                nn.Linear(
                    DrugGraphEncoder._pooled_dim(hidden_dim, _DEFAULT_POOL_TYPE),
                    graph_out_dim,
                ),
                nn.BatchNorm1d(graph_out_dim),
                nn.ReLU(),
                nn.Dropout(post_dropout),
            )

    def encode_nodes(self, x, edge_index, edge_attr=None):
        """Return residue embeddings after full-edge GINE message passing."""
        x = self.input_proj(x)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = self.activation(x)
        return x

    def readout(self, node_h, batch, pool_type=_DEFAULT_POOL_TYPE):
        """Pool residue embeddings and project them to a graph representation."""
        if self.graph_proj is None:
            raise RuntimeError("Graph readout was disabled with build_graph_head=False")
        pooled = DrugGraphEncoder._pool(node_h, batch, pool_type)
        return self.graph_proj(pooled)

    def forward(self, x, edge_index, batch, edge_attr=None):
        node_h = self.encode_nodes(x, edge_index, edge_attr=edge_attr)
        return self.readout(node_h, batch)
