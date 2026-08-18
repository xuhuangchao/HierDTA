"""Cross-scale interaction modules used by the hierarchical drug encoder."""

import torch
import torch.nn as nn


class BottomUpAtomMotifFusion(nn.Module):
    """Update motif features with context-enriched member-atom embeddings.

    Messages are propagated only from atoms to motifs along explicit
    ``atom-in-motif`` membership edges. Mean aggregation prevents large or
    overlapping motifs from receiving systematically larger message norms.
    """

    def __init__(self, atom_dim=74, motif_dim=50, dropout=0.1):
        super().__init__()
        self.atom_dim = atom_dim
        self.motif_dim = motif_dim
        self.atom_proj = nn.Sequential(
            nn.Linear(atom_dim, motif_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(
            nn.Linear(motif_dim * 2, motif_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(motif_dim)

    def forward(self, atom_h, motif_x, membership_edge_index):
        if atom_h.dim() != 2 or atom_h.size(-1) != self.atom_dim:
            raise ValueError(
                f"Expected atom_h [N, {self.atom_dim}], got {tuple(atom_h.shape)}"
            )
        if motif_x.dim() != 2 or motif_x.size(-1) != self.motif_dim:
            raise ValueError(
                f"Expected motif_x [M, {self.motif_dim}], got {tuple(motif_x.shape)}"
            )
        if membership_edge_index.dim() != 2 or membership_edge_index.size(0) != 2:
            raise ValueError(
                "membership_edge_index must have shape [2, E], got "
                f"{tuple(membership_edge_index.shape)}"
            )
        if membership_edge_index.size(1) == 0:
            return motif_x

        atom_index = membership_edge_index[0].long()
        motif_index = membership_edge_index[1].long()
        if atom_index.min() < 0 or atom_index.max() >= atom_h.size(0):
            raise IndexError("Atom index in membership_edge_index is out of bounds")
        if motif_index.min() < 0 or motif_index.max() >= motif_x.size(0):
            raise IndexError("Motif index in membership_edge_index is out of bounds")

        atom_messages = self.atom_proj(atom_h[atom_index])
        motif_context = atom_messages.new_zeros((motif_x.size(0), self.motif_dim))
        motif_context.index_add_(0, motif_index, atom_messages)
        motif_counts = atom_messages.new_zeros(motif_x.size(0))
        motif_counts.index_add_(
            0,
            motif_index,
            torch.ones_like(motif_index, dtype=atom_messages.dtype),
        )
        motif_context = motif_context / motif_counts.clamp_min(1.0).unsqueeze(-1)
        gate = self.gate(torch.cat([motif_x, motif_context], dim=-1))
        return self.norm(motif_x + gate * motif_context)
