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
        build_graph_head=True,
        gnn_type="gat",
        node_key="atom",
        edge_key=("atom", "bond", "atom"),
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("AtomGNN requires num_layers >= 1")
        if gnn_type not in {"gat", "gin", "gcn"}:
            raise ValueError("gnn_type must be 'gat', 'gin', or 'gcn'")
        self.in_dim = in_dim
        self.heads = heads
        self.gnn_type = gnn_type
        self.node_key = node_key
        self.edge_key = edge_key
        self.graph_pool_type = graph_pool_type

        self.convs = nn.ModuleList()
        layer_in = in_dim
        node_out_dim = in_dim * heads
        for _ in range(num_layers):
            if gnn_type == "gat":
                conv = GATv2Conv(
                    layer_in,
                    in_dim,
                    heads=heads,
                    concat=True,
                    edge_dim=edge_dim,
                    dropout=gat_dropout,
                )
            elif gnn_type == "gin":
                update_mlp = nn.Sequential(
                    nn.Linear(layer_in, node_out_dim),
                    nn.ReLU(),
                    nn.Linear(node_out_dim, node_out_dim),
                )
                conv = GINConv(update_mlp)
            else:
                conv = GCNConv(layer_in, node_out_dim)
            self.convs.append(conv)
            layer_in = node_out_dim
        self.activation = nn.ReLU()

        self.graph_proj = None
        if build_graph_head:
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

    @property
    def node_out_dim(self):
        """Width of the node embeddings produced by the final GATv2 layer."""
        return self.in_dim * self.heads

    def encode_nodes(self, data, x_override=None):
        """Encode nodes without graph-level pooling.

        ``x_override`` allows the motif branch to consume node features updated
        by an explicit atom-to-motif message-passing module.
        """
        x = data[self.node_key].x if x_override is None else x_override
        edge_index = data[self.edge_key].edge_index
        edge_attr = data[self.edge_key].edge_attr
        for conv in self.convs:
            if self.gnn_type == "gat":
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)
            x = self.activation(x)
        return x

    def readout(self, node_h, batch):
        """Pool node embeddings and project them to a graph representation."""
        if self.graph_proj is None:
            raise RuntimeError("Graph readout was disabled with build_graph_head=False")
        return self.graph_proj(self._pool(node_h, batch, self.graph_pool_type))

    def forward(self, data, x_override=None):
        node_h = self.encode_nodes(data, x_override=x_override)
        return self.readout(node_h, data[self.node_key].batch)


class ProteinGraphEncoder(nn.Module):
    """Encode a residue graph with GINE over all 10-dimensional edge features."""

    def __init__(
        self,
        protein_in_dim=41,
        hidden_dim=256,
        num_layers=1,
        graph_pool_type="mean_add_max",
        graph_out_dim=1024,
        post_dropout=0.3,
        protein_edge_dim=10,
        build_graph_head=True,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("ProteinGraphEncoder requires num_layers >= 1")
        if protein_edge_dim < 1:
            raise ValueError("protein_edge_dim must be positive")

        self.graph_pool_type = graph_pool_type
        self.protein_edge_dim = protein_edge_dim
        self.convs = nn.ModuleList()
        for layer_idx in range(num_layers):
            in_dim = protein_in_dim if layer_idx == 0 else hidden_dim
            update_mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
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
                    AtomGNN._pooled_dim(hidden_dim, graph_pool_type),
                    graph_out_dim,
                ),
                nn.BatchNorm1d(graph_out_dim),
                nn.ReLU(),
                nn.Dropout(post_dropout),
            )

    def encode_nodes(self, x, edge_index, edge_attr=None):
        """Return residue embeddings after full-edge GINE message passing."""
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = self.activation(x)
        return x

    def readout(self, node_h, batch):
        """Pool residue embeddings and project them to a graph representation."""
        if self.graph_proj is None:
            raise RuntimeError("Graph readout was disabled with build_graph_head=False")
        pooled = AtomGNN._pool(node_h, batch, self.graph_pool_type)
        return self.graph_proj(pooled)

    def forward(self, x, edge_index, batch, edge_attr=None):
        node_h = self.encode_nodes(x, edge_index, edge_attr=edge_attr)
        return self.readout(node_h, batch)
