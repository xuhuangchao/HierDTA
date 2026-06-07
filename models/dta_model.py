"""Top-level DTA model and custom batching for nested PyG graph objects."""

from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.utils import to_dense_batch

from .encoder import HierMolGNN


class DTABatch:
    """Container for drug graphs, surface points, globals, and labels."""

    __slots__ = (
        "hetero",
        "fingerprint",
        "esm_global",
        "surface_embedding",
        "surface_mask",
        "y",
        "smiles",
        "key",
    )

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to(self, device: torch.device) -> "DTABatch":
        for attr in self.__slots__:
            value = getattr(self, attr, None)
            if isinstance(value, (Tensor, Data, HeteroData, Batch)):
                setattr(self, attr, value.to(device))
        return self

    def __repr__(self) -> str:
        batch_size = self.y.size(0) if self.y is not None else "?"
        return f"DTABatch(B={batch_size})"


def dta_collate_fn(data_list: List[Any]) -> DTABatch:
    """Batch nested heterogeneous drug graphs and surface point clouds."""

    max_surface_points = max(data.surface_embedding.size(0) for data in data_list)
    surface_dim = data_list[0].surface_embedding.size(1)
    surface_embedding = data_list[0].surface_embedding.new_zeros(
        len(data_list), max_surface_points, surface_dim
    )
    surface_mask = torch.zeros(
        len(data_list), max_surface_points, dtype=torch.bool
    )
    for index, data in enumerate(data_list):
        point_count = data.surface_embedding.size(0)
        surface_embedding[index, :point_count] = data.surface_embedding
        surface_mask[index, :point_count] = True

    return DTABatch(
        hetero=Batch.from_data_list([data.hetero for data in data_list]),
        fingerprint=torch.cat([data.fingerprint for data in data_list], dim=0),
        esm_global=torch.cat([data.esm_global for data in data_list], dim=0),
        surface_embedding=surface_embedding,
        surface_mask=surface_mask,
        y=torch.cat([data.y for data in data_list], dim=0),
        smiles=[getattr(data, "smiles", "") for data in data_list],
        key=[getattr(data, "key", "") for data in data_list],
    )


class MLPDecoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim1: int,
        hidden_dim2: int,
        binary: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, binary)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)


class DTASurfaceCrossHead(nn.Module):
    """Cross-attention between drug atoms/motifs and dMaSIF surface points.

    A learnable point sampler selects top-k surface points first, reducing
    the softmax competition from 512-way to k-way. Drug substructures then
    cross-attend to these focused points.
    drug_repr already encodes drug-surface interaction, so surface_mean is
    excluded from the final MLP.
    Final: [drug_repr, fp, esm].
    """

    def __init__(
        self,
        d: int = 128,
        num_tasks: int = 1,
        dropout: float = 0.2,
        fp_in_dim: int = 1024,
        esm_dim: int = 1152,
    ):
        super().__init__()
        self.d = d
        self.num_sampled_points = 128

        # ── Learnable point sampler (512 → k, drug-independent) ──
        self.point_sampler = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.ReLU(),
            nn.Linear(d // 4, 1),
        )

        # ── Cross-attention: drug substructures → sampled surface points ──
        self.atom_cross_attn = nn.MultiheadAttention(d, num_heads=1, batch_first=True, dropout=dropout)
        self.motif_cross_attn = nn.MultiheadAttention(d, num_heads=1, batch_first=True, dropout=dropout)

        # ── Learned pooling: which drug substructures drive binding? ──
        self.atom_pool = nn.Sequential(nn.Linear(d, d // 4), nn.ReLU(), nn.Linear(d // 4, 1))
        self.motif_pool = nn.Sequential(nn.Linear(d, d // 4), nn.ReLU(), nn.Linear(d // 4, 1))

        # ── Precomputed feature projections ──
        self.fp_proj = nn.Sequential(
            nn.Linear(fp_in_dim, d),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.esm_proj = nn.Sequential(
            nn.Linear(esm_dim, d),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Final MLP: 3d input (drug_repr + fp + esm), surface_mean removed ──
        self.mlp = MLPDecoder(
            in_dim=3 * d,
            hidden_dim1=2 * d,
            hidden_dim2=d,
            binary=num_tasks,
            dropout=dropout,
        )

    def forward(
        self,
        atom_nodes: Tensor,       # [total_atoms, d]
        atom_batch: Tensor,       # [total_atoms]
        motif_nodes: Tensor,      # [total_motifs, d]
        motif_batch: Tensor,      # [total_motifs]
        surface_embedding: Tensor,  # [B, N_s, surface_dim]
        surface_mask: Tensor,     # [B, N_s]
        fingerprint: Tensor,      # [B, fp_in_dim]
        esm_global: Tensor,       # [B, esm_dim]
    ) -> tuple[Tensor, dict]:
        if fingerprint.dim() == 3:
            fingerprint = fingerprint.squeeze(1)

        # ── Dense batch conversion ──
        atom_d, atom_mask = to_dense_batch(atom_nodes, atom_batch)      # [B, N_a, d]
        motif_d, motif_mask = to_dense_batch(motif_nodes, motif_batch)  # [B, N_m, d]

        # ── Surface points as key/value for cross-attention ──
        # surface_embedding is already dense [B, N_s, d], just use directly
        surface = surface_embedding  # [B, N_s, d]

        # ── Cross-Attention: Atoms → Surface Points ──
        atom_enhanced, atom_attn = self.atom_cross_attn(
            query=atom_d,
            key=surface,
            value=surface,
            key_padding_mask=~surface_mask,
        )  # [B, N_a, d], [B, N_a, N_s]

        # ── Cross-Attention: Motifs → Surface Points ──
        motif_enhanced, motif_attn = self.motif_cross_attn(
            query=motif_d,
            key=surface,
            value=surface,
            key_padding_mask=~surface_mask,
        )

        # ── Learned attention-based pooling over cross-attention outputs ──
        atom_scores = self.atom_pool(atom_enhanced).squeeze(-1)       # [B, N_a]
        atom_scores = atom_scores.masked_fill(~atom_mask, float('-inf'))
        atom_w = F.softmax(atom_scores, dim=-1)
        atom_repr = (atom_enhanced * atom_w.unsqueeze(-1)).sum(dim=1)  # [B, d]

        motif_scores = self.motif_pool(motif_enhanced).squeeze(-1)    # [B, N_m]
        motif_scores = motif_scores.masked_fill(~motif_mask, float('-inf'))
        motif_w = F.softmax(motif_scores, dim=-1)
        motif_repr = (motif_enhanced * motif_w.unsqueeze(-1)).sum(dim=1)  # [B, d]

        # ── Fuse drug representation ──
        drug_repr = atom_repr + motif_repr  # [B, d]

        # ── Surface mean as global pocket proxy ──
        surface_mean = (surface * surface_mask.unsqueeze(-1)).sum(dim=1) / \
                       surface_mask.sum(dim=1, keepdim=True).float().clamp(min=1)  # [B, d]

        # ── Precomputed features ──
        fp = self.fp_proj(fingerprint)
        esm = self.esm_proj(esm_global)

        # ── Final: no Hadamard ──
        out = self.mlp(torch.cat([drug_repr, surface_mean, fp, esm], dim=-1))  # [B, 4d]

        attn_dict = {
            'atom_to_surface': atom_attn,      # [B, N_a, N_s]
            'motif_to_surface': motif_attn,    # [B, N_m, N_s]
            'atom_importance': atom_w,
            'motif_importance': motif_w,
        }
        return out, attn_dict


class DTAModel(nn.Module):
    """DTA model with hierarchical drug GNN and surface-point cross-attention."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_tasks: int = 1,
        task: str = "regression",
        dropout: float = 0.1,
        atom_in_dim: int = 37,
        motif_in_dim: int = 50,
        mol_num_layers: int = 2,
        fp_in_dim: int = 1024,
        esm_dim: int = 1152,
    ):
        super().__init__()
        self.task = task
        self.drug_encoder = HierMolGNN(
            atom_in_dim=atom_in_dim,
            motif_in_dim=motif_in_dim,
            hidden_dim=hidden_dim,
            num_layers=mol_num_layers,
            dropout=dropout,
        )
        self.fusion_head = DTASurfaceCrossHead(
            d=hidden_dim,
            num_tasks=num_tasks,
            dropout=dropout,
            fp_in_dim=fp_in_dim,
            esm_dim=esm_dim,
        )

    def forward(self, data: DTABatch) -> tuple[Tensor, dict]:
        (atom_pooled, motif_pooled, atom_nodes, motif_nodes, atom_batch, motif_batch) = self.encode_drug(data)
        return self.fusion_head(
            atom_nodes=atom_nodes,
            atom_batch=atom_batch,
            motif_nodes=motif_nodes,
            motif_batch=motif_batch,
            surface_embedding=data.surface_embedding,
            surface_mask=data.surface_mask,
            fingerprint=data.fingerprint,
            esm_global=data.esm_global,
        )

    def encode_drug(self, data: DTABatch):
        """Return pooled + unpooled atom/motif nodes and batch indices."""
        return self.drug_encoder(data.hetero)

    def count_parameters(self) -> dict:
        def count(module):
            return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
        
        print(self.drug_encoder)
        print(self.fusion_head)
        return {
            "drug_encoder": count(self.drug_encoder),
            "fusion_head": count(self.fusion_head),
            "total": count(self),
        }
