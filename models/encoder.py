"""Drug and protein-residue encoders for DTAModel."""

import torch
import torch.nn as nn
from torch_geometric.nn import (
    GATv2Conv,
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
        node_key="atom",
        edge_key=("atom", "bond", "atom"),
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("AtomGNN requires num_layers >= 1")
        self.in_dim = in_dim
        self.heads = heads
        self.node_key = node_key
        self.edge_key = edge_key
        self.graph_pool_type = graph_pool_type

        self.convs = nn.ModuleList()
        layer_in = in_dim
        for _ in range(num_layers):
            self.convs.append(
                GATv2Conv(
                    layer_in,
                    in_dim,
                    heads=heads,
                    concat=True,
                    edge_dim=edge_dim,
                    dropout=gat_dropout,
                )
            )
            layer_in = in_dim * heads
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
            x = self.activation(conv(x, edge_index, edge_attr=edge_attr))
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
    """Encode mutually exclusive covalent and noncovalent residue views.

    ``cov`` uses a GIN over peptide/covalent edges. ``noncov`` uses a GINE
    over the remaining residue pairs and their nine physicochemical
    interaction channels. ``dual_view`` encodes both views independently,
    concatenates corresponding residue embeddings, and fuses them with an MLP
    before graph-level pooling.
    """

    def __init__(
        self,
        protein_in_dim=41,
        hidden_dim=256,
        num_layers=1,
        graph_pool_type="mean_add_max",
        graph_out_dim=1024,
        post_dropout=0.3,
        protein_edge_dim=10,
        protein_graph_mode="cov",
        build_graph_head=True,
    ):
        super().__init__()
        valid_modes = {"cov", "noncov", "dual_view"}
        if protein_graph_mode not in valid_modes:
            raise ValueError(
                f"protein_graph_mode must be one of {sorted(valid_modes)}"
            )
        if num_layers < 1:
            raise ValueError("ProteinGraphEncoder requires num_layers >= 1")
        if protein_edge_dim < 2:
            raise ValueError(
                "protein_edge_dim must include covalent and noncovalent channels"
            )

        self.graph_pool_type = graph_pool_type
        self.protein_edge_dim = protein_edge_dim
        self.protein_graph_mode = protein_graph_mode

        if protein_graph_mode == "cov":
            self.convs = self._build_convs(
                conv_type="gin",
                protein_in_dim=protein_in_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                noncov_edge_dim=protein_edge_dim - 1,
            )
        elif protein_graph_mode == "noncov":
            self.convs = self._build_convs(
                conv_type="gine",
                protein_in_dim=protein_in_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                noncov_edge_dim=protein_edge_dim - 1,
            )
        else:
            self.cov_convs = self._build_convs(
                conv_type="gin",
                protein_in_dim=protein_in_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                noncov_edge_dim=protein_edge_dim - 1,
            )
            self.noncov_convs = self._build_convs(
                conv_type="gine",
                protein_in_dim=protein_in_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                noncov_edge_dim=protein_edge_dim - 1,
            )
            self.node_fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(post_dropout),
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

    @staticmethod
    def _build_convs(
        conv_type,
        protein_in_dim,
        hidden_dim,
        num_layers,
        noncov_edge_dim,
    ):
        convs = nn.ModuleList()
        for layer_idx in range(num_layers):
            in_dim = protein_in_dim if layer_idx == 0 else hidden_dim
            update_mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if conv_type == "gin":
                convs.append(GINConv(update_mlp))
            elif conv_type == "gine":
                convs.append(
                    GINEConv(update_mlp, edge_dim=noncov_edge_dim)
                )
            else:
                raise ValueError(f"Unsupported protein convolution type: {conv_type}")
        return convs

    def _split_edges(self, edge_index, edge_attr):
        if edge_attr is None:
            raise ValueError(
                f"edge_attr is required for protein_graph_mode='{self.protein_graph_mode}'"
            )
        if edge_attr.ndim != 2 or edge_attr.size(0) != edge_index.size(1):
            raise ValueError(
                "Protein edge_attr must have shape [num_edges, protein_edge_dim]"
            )
        if edge_attr.size(1) != self.protein_edge_dim:
            raise ValueError(
                f"Expected {self.protein_edge_dim} protein edge features, "
                f"got {edge_attr.size(1)}"
            )

        cov_mask = edge_attr[:, 0] > 0
        noncov_mask = (
            (edge_attr[:, 0] == 0)
            & (edge_attr[:, 1:].sum(dim=-1) > 0)
        )
        cov_edge_index = edge_index[:, cov_mask]
        noncov_edge_index = edge_index[:, noncov_mask]
        noncov_edge_attr = edge_attr[noncov_mask, 1:]
        return cov_edge_index, noncov_edge_index, noncov_edge_attr

    def _encode_gin(self, x, edge_index, convs):
        for conv in convs:
            x = self.activation(conv(x, edge_index))
        return x

    def _encode_gine(self, x, edge_index, edge_attr, convs):
        for conv in convs:
            x = self.activation(conv(x, edge_index, edge_attr=edge_attr))
        return x

    def encode_nodes(self, x, edge_index, edge_attr=None):
        """Return edge-type-aware residue embeddings before graph pooling."""
        cov_edge_index, noncov_edge_index, noncov_edge_attr = (
            self._split_edges(edge_index, edge_attr)
        )

        if self.protein_graph_mode == "cov":
            x = self._encode_gin(x, cov_edge_index, self.convs)
        elif self.protein_graph_mode == "noncov":
            x = self._encode_gine(
                x,
                noncov_edge_index,
                noncov_edge_attr,
                self.convs,
            )
        else:
            cov_x = self._encode_gin(
                x,
                cov_edge_index,
                self.cov_convs,
            )
            noncov_x = self._encode_gine(
                x,
                noncov_edge_index,
                noncov_edge_attr,
                self.noncov_convs,
            )
            x = self.node_fusion(torch.cat([cov_x, noncov_x], dim=-1))

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
