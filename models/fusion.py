"""Hierarchical local and global fusion modules for DTAModel."""

import math

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import scatter


class Aggregation(nn.Module):
    """Inject mean-pooled atom context into each motif representation."""

    def __init__(self, hidden_dim: int):
        super().__init__()

    def forward(
        self,
        atom_tokens: Tensor,
        motif_tokens: Tensor,
        atom_to_motif_edge_index: Tensor,
    ) -> Tensor:
        atom_index, motif_index = atom_to_motif_edge_index
        atom_context = scatter(
            atom_tokens[atom_index],
            motif_index,
            dim=0,
            dim_size=motif_tokens.size(0),
            reduce="mean",
        )
        motif_tokens_add = motif_tokens + atom_context
        return motif_tokens_add


class Interaction(nn.Module):
    """Summarize ligand-conditioned motif-residue interaction weights."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.interaction_proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.motif_gate = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        motif_tokens: Tensor,
        residue_tokens: Tensor,
        motif_mask: Tensor,
        residue_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        query = self.query_proj(motif_tokens)
        key = self.key_proj(residue_tokens)
        value = self.value_proj(residue_tokens)

        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = scores / math.sqrt(query.size(-1))
        scores = scores.masked_fill(~residue_mask[:, None, :], -1e4)
        attention = torch.softmax(scores, dim=-1)

        residue_context = torch.matmul(attention, value)
        interaction_tokens = self.interaction_proj(torch.cat([
            motif_tokens,
            residue_context,
        ], dim=-1))

        motif_scores = self.motif_gate(interaction_tokens)
        motif_scores = motif_scores.masked_fill(~motif_mask[:, :, None], -1e4)
        motif_weight = torch.softmax(motif_scores, dim=1)
        local_interaction = torch.sum(motif_weight * interaction_tokens, dim=1)
        return local_interaction, attention, motif_weight
