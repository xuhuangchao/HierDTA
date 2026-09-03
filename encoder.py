"""Drug and protein-residue encoders for DTAModel."""

import torch
import torch.nn as nn
from torch_geometric.nn import (
    GATv2Conv,
    GCNConv,
    GINConv,
    GINEConv,
)


class DrugGraphEncoder(nn.Module):
    """Reusable homogeneous graph branch over one node/edge type in a HeteroData."""

    def __init__(
        self,
        in_dim=37,
        edge_dim=13,
        num_layers=1,
        heads=2,
        gnn_type="gat",
        node_key="atom",
        edge_key=("atom", "bond", "atom"),
    ):
        super().__init__()
        self.in_dim = in_dim
        self.heads = heads
        self.gnn_type = gnn_type
        self.node_key = node_key
        self.edge_key = edge_key

        self.convs = nn.ModuleList()
        layer_in = in_dim
        node_out_dim = in_dim
        for _ in range(num_layers):
            if gnn_type == "gat":
                conv = GATv2Conv(
                    layer_in,
                    node_out_dim,
                    heads=heads,
                    concat=False,
                    edge_dim=edge_dim,
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

    @property
    def node_out_dim(self):
        """Width of the node embeddings produced by the final GNN layer."""
        return self.in_dim

    def encode_layer(self, data, x, layer_idx):
        """Run one GNN layer so hierarchical branches can be interleaved."""
        conv = self.convs[layer_idx]
        edge_index = data[self.edge_key].edge_index
        if self.gnn_type == "gat":
            edge_attr = data[self.edge_key].edge_attr
            x = conv(x, edge_index, edge_attr=edge_attr)
        else:
            x = conv(x, edge_index)
        return self.activation(x)

    def encode_nodes(self, data, x_override=None):
        """Encode nodes without graph-level pooling.

        ``x_override`` can replace the raw node features before message passing.
        """
        x = data[self.node_key].x if x_override is None else x_override
        for layer_idx in range(len(self.convs)):
            x = self.encode_layer(data, x, layer_idx)
        return x


class ProteinGraphEncoder(nn.Module):
    """Encode a residue graph with GINE over all 10-dimensional edge features."""

    def __init__(
        self,
        protein_in_dim=41,
        hidden_dim=256,
        num_layers=2,
        protein_edge_dim=10,
    ):
        super().__init__()

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

    def encode_nodes(self, x, edge_index, edge_attr=None):
        """Return residue embeddings after full-edge GINE message passing."""
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr=edge_attr)
            x = self.activation(x)
        return x
