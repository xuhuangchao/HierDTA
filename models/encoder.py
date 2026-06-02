"""Drug, residue-graph, and protein-surface encoders for DTAModel."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, GCNConv, global_mean_pool
from torch_geometric.nn.models import AttentiveFP


class HeteroMolGNN(nn.Module):
    """Encode fine-grained atom tokens and coarse-grained motif tokens."""

    def __init__(
        self,
        atom_in_dim: int = 37,
        motif_in_dim: int = 50,
        hidden_dim: int = 256,
        atom_num_layers: int = 3,
        motif_num_layers: int = 2,
        aa_edge_dim: int = 13,
        mm_edge_dim: int = 37,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.atom_encoder = AttentiveFP(
            in_channels=atom_in_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            edge_dim=aa_edge_dim,
            num_layers=atom_num_layers,
            num_timesteps=2,
            dropout=dropout,
        )
        self.motif_encoder = AttentiveFP(
            in_channels=motif_in_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            edge_dim=mm_edge_dim,
            num_layers=motif_num_layers,
            num_timesteps=2,
            dropout=dropout,
        )

    def forward(self, data: HeteroData) -> tuple[Tensor, Tensor]:
        aa_ei = data["atom", "bond", "atom"].edge_index
        aa_ea = data["atom", "bond", "atom"].edge_attr
        mm_ei = data["motif", "connects", "motif"].edge_index
        mm_ea = data["motif", "connects", "motif"].edge_attr
        atom_mol_out = self.atom_encoder(
            data["atom"].x, aa_ei, aa_ea, data["atom"].batch
        )
        motif_mol_out = self.motif_encoder(
            data["motif"].x, mm_ei, mm_ea, data["motif"].batch
        )
        return atom_mol_out, motif_mol_out


class ProteinGraphEncoder(nn.Module):
    """Encode full-length ESMC residue nodes following ProteinGraphNet."""

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
        self.gcn_bn = nn.BatchNorm1d(hidden_dim)
        self.gat_layers = nn.ModuleList([
            GATConv(in_channels=hidden_dim, out_channels=hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.gat_bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        batch: Tensor,
    ) -> Tensor:
        # Keep ProteinGraphNet ordering: weighted GCN, then unweighted GAT blocks.
        h = F.relu(self.gcn(x, edge_index, edge_weight=edge_weight))
        h = self.gcn_bn(h)
        for gat, batch_norm in zip(self.gat_layers, self.gat_bns):
            h = F.relu(gat(h, edge_index))
            h = batch_norm(h)
        # Keep ProteinGraphNet's post-pooling MLP while returning a fusion embedding.
        h = global_mean_pool(h, batch)
        return h


class SurfaceEncoder(nn.Module):
    """Attention-pool valid dMaSIF surface points into one local vector."""

    def __init__(self, in_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, surface_x: Tensor, surface_mask: Tensor) -> Tensor:
        h = self.proj(surface_x)
        scores = self.score(h).squeeze(-1)
        scores = scores.masked_fill(~surface_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return torch.sum(h * weights.unsqueeze(-1), dim=1)
