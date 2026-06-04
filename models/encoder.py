"""Drug, residue-graph, and protein-surface encoders for DTAModel."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import AttentiveFP, GATConv, GCNConv, global_add_pool


class HeteroMolGNN(nn.Module):
    """Read atom and motif molecular graphs with AttentiveFP."""

    def __init__(
        self,
        atom_in_dim: int = 37,
        motif_in_dim: int = 50,
        atom_edge_dim: int = 13,
        motif_edge_dim: int = 37,
        hidden_dim: int = 256,
        atom_num_layers: int = 2,
        motif_num_layers: int = 2,
        atom_num_timesteps: int = 2,
        motif_num_timesteps: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.atom_encoder = AttentiveFP(
            in_channels=atom_in_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            edge_dim=atom_edge_dim,
            num_layers=atom_num_layers,
            num_timesteps=atom_num_timesteps,
            dropout=dropout,
        )
        self.motif_encoder = AttentiveFP(
            in_channels=motif_in_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            edge_dim=motif_edge_dim,
            num_layers=motif_num_layers,
            num_timesteps=motif_num_timesteps,
            dropout=dropout,
        )

    def forward(
        self,
        data: HeteroData,
    ) -> tuple[Tensor, Tensor]:
        atom_out = self.atom_encoder(
            x=data["atom"].x,
            edge_index=data["atom", "bond", "atom"].edge_index,
            edge_attr=data["atom", "bond", "atom"].edge_attr,
            batch=data["atom"].batch,
        )
        motif_out = self.motif_encoder(
            x=data["motif"].x,
            edge_index=data["motif", "connects", "motif"].edge_index,
            edge_attr=data["motif", "connects", "motif"].edge_attr,
            batch=data["motif"].batch,
        )
        return atom_out, motif_out


class GraphAddPoolEncoder(nn.Module):
    """GCN-GAT graph encoder with global_add_pool, matching backupcode style."""

    def __init__(
        self,
        in_dim: int,
        conv_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.gcn = GCNConv(in_channels=in_dim, out_channels=conv_dim)
        self.gcn_bn = nn.BatchNorm1d(conv_dim)
        self.gat_layers = nn.ModuleList([
            GATConv(in_channels=conv_dim, out_channels=conv_dim)
            for _ in range(num_layers)
        ])
        self.gat_bns = nn.ModuleList([
            nn.BatchNorm1d(conv_dim) for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(conv_dim, out_dim)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch: Tensor,
    ) -> Tensor:
        edge_weight = edge_attr.mean(dim=1)
        h = self.relu(self.gcn(x, edge_index, edge_weight=edge_weight))
        h = self.gcn_bn(h)
        for gat, batch_norm in zip(self.gat_layers, self.gat_bns):
            h = F.relu(gat(h, edge_index))
            h = batch_norm(h)
        h = global_add_pool(h, batch)
        h = F.relu(self.out_proj(h))
        return self.dropout(h)


class ProteinGraphEncoder(nn.Module):
    """Read full-length ESMC residue graphs following backup ProteinGraphNet."""

    def __init__(
        self,
        prot_in_dim: int = 1152,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoder = GraphAddPoolEncoder(
            in_dim=prot_in_dim,
            conv_dim=2 * hidden_dim,
            out_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Tensor,
        batch: Tensor,
    ) -> Tensor:
        if edge_weight.dim() == 1:
            edge_weight = edge_weight.unsqueeze(-1)
        return self.encoder(x, edge_index, edge_weight, batch)


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

    def forward(
        self,
        surface_x: Tensor,
        surface_mask: Tensor,
        return_tokens: bool = False,
    ):
        h = self.proj(surface_x)
        scores = self.score(h).squeeze(-1)
        scores = scores.masked_fill(~surface_mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(h * weights.unsqueeze(-1), dim=1)
        if return_tokens:
            return pooled, h
        return pooled
