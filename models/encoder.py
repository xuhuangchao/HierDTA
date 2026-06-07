"""Drug, residue-graph, and protein-surface encoders for DTAModel."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch_geometric.utils import scatter


class HierMolGNN(nn.Module):
    """Hierarchical molecular GNN with per-layer atom→motif cross-level aggregation.

    Mirrors ProteinGraphEncoder's GCNConv→GATConv pattern. Edge features are
    mean-reduced to scalar weights (zero parameters). After each message-passing
    step, atom features are scatter-mean-aggregated into motif features via the
    atom→motif edge index, creating a proper hierarchical information cascade.
    """

    def __init__(
        self,
        atom_in_dim: int = 37,
        motif_in_dim: int = 50,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        # ── Layer 1: weighted GCNConv (projection + convolution, mirrors protein encoder) ──
        self.atom_gcn = GCNConv(atom_in_dim, hidden_dim)
        self.motif_gcn = GCNConv(motif_in_dim, hidden_dim)

        # ── Layers 2..N: GATConv (mirrors protein encoder) ──
        self.atom_gats = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        self.motif_gats = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # ── Cross-level projection (atom → motif, shared across layers) ──
        self.cross = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: HeteroData) -> tuple[Tensor, Tensor]:
        aa_ei = data["atom", "bond", "atom"].edge_index
        aa_ea = data["atom", "bond", "atom"].edge_attr
        mm_ei = data["motif", "connects", "motif"].edge_index
        mm_ea = data["motif", "connects", "motif"].edge_attr
        am_ei = data["atom", "in", "motif"].edge_index

        # Edge features → scalar weights (zero-parameter mean)
        aa_w = aa_ea.mean(dim=-1)
        mm_w = mm_ea.mean(dim=-1)

        # ── GCN layer ──
        h_a = self.dropout(F.relu(self.atom_gcn(data["atom"].x, aa_ei, edge_weight=aa_w)))
        h_m = self.dropout(F.relu(self.motif_gcn(data["motif"].x, mm_ei, edge_weight=mm_w)))
        h_m = h_m + self.cross(
            scatter(h_a[am_ei[0]], am_ei[1], dim=0,
                    dim_size=h_m.size(0), reduce="mean"))

        # ── GAT layers ──
        for gat_a, gat_m in zip(self.atom_gats, self.motif_gats):
            h_a = self.dropout(F.relu(gat_a(h_a, aa_ei)))
            h_m = self.dropout(F.relu(gat_m(h_m, mm_ei)))
            h_m = h_m + self.cross(
                scatter(h_a[am_ei[0]], am_ei[1], dim=0,
                        dim_size=h_m.size(0), reduce="mean"))

        return (global_mean_pool(h_a, data["atom"].batch),
                global_mean_pool(h_m, data["motif"].batch))


class ProteinGraphEncoder(nn.Module):
    """Encode full-length ESMC residue nodes with GCNConv→GATConv stack.

    Simplified from the original ProteinGraphNet: no BatchNorm, only ReLU+Drop
    after each layer, matching the HierMolGNN and MLPDecoder conventions.
    """

    def __init__(
        self,
        prot_in_dim: int = 1152,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.gcn = GCNConv(in_channels=prot_in_dim, out_channels=hidden_dim)
        self.gat_layers = nn.ModuleList([
            GATConv(in_channels=hidden_dim, out_channels=hidden_dim)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        batch: Tensor,
    ) -> Tensor:
        h = self.dropout(F.relu(self.gcn(x, edge_index, edge_weight=edge_weight)))
        for gat in self.gat_layers:
            h = self.dropout(F.relu(gat(h, edge_index)))
        return global_mean_pool(h, batch)


class SurfaceEncoder(nn.Module):
    """Mean-pool valid dMaSIF surface points. Zero learnable parameters."""

    def forward(self, surface_x: Tensor, surface_mask: Tensor) -> Tensor:
        surface_x = surface_x.masked_fill(~surface_mask.unsqueeze(-1), 0.0)
        valid_count = surface_mask.sum(dim=1, keepdim=True).float().clamp(min=1)
        return surface_x.sum(dim=1) / valid_count
