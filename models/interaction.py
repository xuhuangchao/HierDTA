"""Cross-scale interaction modules used by the hierarchical drug encoder."""

import torch
import torch.nn as nn
from torch_geometric.utils import scatter


# Retired: pre-GNN bottom-up injection. It mixed learned atom embeddings
# into hand-crafted motif *input* features (mismatched spaces) and only
# flowed atom -> motif. Superseded by CrossLevelExchange, which exchanges
# learned post-GNN embeddings. Retained for ablation reference.
#
# class BottomUpAtomMotifFusion(nn.Module):
#     """Update motif features with context-enriched member-atom embeddings.
#
#     Messages are propagated only from atoms to motifs along explicit
#     ``atom-in-motif`` membership edges. Mean aggregation prevents large or
#     overlapping motifs from receiving systematically larger message norms.
#     """
#
#     def __init__(self, atom_dim=74, motif_dim=50, dropout=0.1):
#         super().__init__()
#         self.atom_dim = atom_dim
#         self.motif_dim = motif_dim
#         self.atom_proj = nn.Sequential(
#             nn.Linear(atom_dim, motif_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#         )
#         self.gate = nn.Sequential(
#             nn.Linear(motif_dim * 2, motif_dim),
#             nn.Sigmoid(),
#         )
#
#     def forward(self, atom_h, motif_x, membership_edge_index):
#         if atom_h.dim() != 2 or atom_h.size(-1) != self.atom_dim:
#             raise ValueError(
#                 f"Expected atom_h [N, {self.atom_dim}], got {tuple(atom_h.shape)}"
#             )
#         if motif_x.dim() != 2 or motif_x.size(-1) != self.motif_dim:
#             raise ValueError(
#                 f"Expected motif_x [M, {self.motif_dim}], got {tuple(motif_x.shape)}"
#             )
#         if membership_edge_index.dim() != 2 or membership_edge_index.size(0) != 2:
#             raise ValueError(
#                 "membership_edge_index must have shape [2, E], got "
#                 f"{tuple(membership_edge_index.shape)}"
#             )
#         if membership_edge_index.size(1) == 0:
#             return motif_x
#
#         atom_index = membership_edge_index[0].long()
#         motif_index = membership_edge_index[1].long()
#         if atom_index.min() < 0 or atom_index.max() >= atom_h.size(0):
#             raise IndexError("Atom index in membership_edge_index is out of bounds")
#         if motif_index.min() < 0 or motif_index.max() >= motif_x.size(0):
#             raise IndexError("Motif index in membership_edge_index is out of bounds")
#
#         atom_messages = self.atom_proj(atom_h[atom_index])
#         motif_context = atom_messages.new_zeros((motif_x.size(0), self.motif_dim))
#         motif_context.index_add_(0, motif_index, atom_messages)
#         motif_counts = atom_messages.new_zeros(motif_x.size(0))
#         motif_counts.index_add_(
#             0,
#             motif_index,
#             torch.ones_like(motif_index, dtype=atom_messages.dtype),
#         )
#         motif_context = motif_context / motif_counts.clamp_min(1.0).unsqueeze(-1)
#         gate = self.gate(torch.cat([motif_x, motif_context], dim=-1))
#         return motif_x + gate * motif_context


class CrossLevelExchange(nn.Module):
    """Exchange atom/motif information on post-GNN node embeddings."""

    def __init__(self, atom_dim=256, motif_dim=256, dropout=0.1,
                 direction="up"):
        super().__init__()
        if direction not in {"up", "down", "bidir"}:
            raise ValueError(
                f"direction must be 'up', 'down', or 'bidir', got {direction!r}"
            )
        self.direction = direction
        # motif_update: input = atom_dim (atom message), hidden = motif_dim.
        # atom_update: input = motif_dim (motif message), hidden = atom_dim.
        self.motif_update = None
        if direction in {"up", "bidir"}:
            self.motif_update = nn.GRUCell(atom_dim, motif_dim)
        self.atom_update = None
        if direction in {"down", "bidir"}:
            self.atom_update = nn.GRUCell(motif_dim, atom_dim)

    def forward(self, atom_h, motif_h, membership_edge_index):
        atom_index, motif_index = membership_edge_index[0], membership_edge_index[1]
        atom_out, motif_out = atom_h, motif_h

        if self.direction in {"up", "bidir"}:
            # atom -> motif: message content comes from member atoms.
            atom_message = scatter(
                atom_h[atom_index],
                motif_index,
                dim=0,
                dim_size=motif_h.size(0),
                reduce="mean",
            )
            m_mask = torch.zeros(
                motif_h.size(0), dtype=torch.bool, device=motif_h.device
            )
            m_mask[motif_index] = True
            motif_out = motif_h.clone()
            motif_out[m_mask] = self.motif_update(
                atom_message[m_mask], motif_h[m_mask]
            )

        if self.direction in {"down", "bidir"}:
            # motif -> atom: message content comes from owning motifs.
            motif_message = scatter(
                motif_h[motif_index],
                atom_index,
                dim=0,
                dim_size=atom_h.size(0),
                reduce="mean",
            )
            a_mask = torch.zeros(
                atom_h.size(0), dtype=torch.bool, device=atom_h.device
            )
            a_mask[atom_index] = True
            atom_out = atom_h.clone()
            atom_out[a_mask] = self.atom_update(
                motif_message[a_mask], atom_h[a_mask]
            )

        return atom_out, motif_out
