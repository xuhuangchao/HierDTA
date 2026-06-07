"""Top-level DTA model and custom batching for nested PyG graph objects."""

from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch, Data, HeteroData

from .encoder import HierMolGNN, ProteinGraphEncoder, SurfaceEncoder


class DTABatch:
    """Container for drug graphs, contact graphs, surface points, globals, and labels."""

    __slots__ = (
        "hetero",
        "protein_graph",
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
    """Batch nested heterogeneous drug graphs and protein residue graphs."""

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
        protein_graph=Batch.from_data_list([data.protein_graph for data in data_list]),
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


class DTAInteractionHead(nn.Module):
    """Fuse drug and protein views with intra-modal attention.

    Drug side: attention-weighted fusion of [atom_mol, motif_mol] → drug_repr [d]
    Protein side: attention-weighted fusion of [protein_out, surface] → prot_repr [d]
    Precomputed features: fp_proj [d] + esm_proj [d] concatenated at the final stage.
    Final: [drug_repr, prot_repr, fp_proj, esm_proj, drug_repr⊙prot_repr] → MLP → affinity
    """

    def __init__(
        self,
        d: int = 256,
        num_tasks: int = 1,
        dropout: float = 0.2,
        fp_in_dim: int = 1024,
        esm_dim: int = 1152,
    ):
        super().__init__()
        # ── Drug side: 2-view attention (atom + motif) ──
        self.drug_view_attn = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.ReLU(),
            nn.Linear(d // 4, 1),
        )

        # ── Protein side: 2-view attention (protein_out + surface) ──
        self.prot_view_attn = nn.Sequential(
            nn.Linear(d, d // 4),
            nn.ReLU(),
            nn.Linear(d // 4, 1),
        )

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

        # ── Final prediction: 5d input (drug_repr + prot_repr + fp + esm + interaction) ──
        self.mlp = MLPDecoder(
            in_dim=5 * d,
            hidden_dim1=2 * d,
            hidden_dim2=d,
            binary=num_tasks,
            dropout=dropout,
        )

    def forward(
        self,
        atom_mol_out: Tensor,
        motif_mol_out: Tensor,
        protein_out: Tensor,
        fingerprint: Tensor,
        surface: Tensor,
        esm_global: Tensor,
    ) -> Tensor:
        if fingerprint.dim() == 3:
            fingerprint = fingerprint.squeeze(1)

        # ── Drug fusion: attention over 2 GNN views ──
        drug_views = torch.stack([atom_mol_out, motif_mol_out], dim=1)  # [B, 2, d]
        drug_w = F.softmax(self.drug_view_attn(drug_views).squeeze(-1), dim=-1)   # [B, 2]
        drug_repr = (drug_views * drug_w.unsqueeze(-1)).sum(dim=1)                  # [B, d]

        # ── Protein fusion: attention over 2 learned views ──
        prot_views = torch.stack([protein_out, surface], dim=1)        # [B, 2, d]
        prot_w = F.softmax(self.prot_view_attn(prot_views).squeeze(-1), dim=-1)     # [B, 2]
        prot_repr = (prot_views * prot_w.unsqueeze(-1)).sum(dim=1)                    # [B, d]

        # ── Precomputed high-level features (direct, no attention) ──
        fp = self.fp_proj(fingerprint)                                                # [B, d]
        esm = self.esm_proj(esm_global)                                               # [B, d]

        # ── Explicit interaction signal ──
        interaction = drug_repr * prot_repr                                           # [B, d]

        # ── Final: concatenate all 5 components ──
        return self.mlp(torch.cat([drug_repr, prot_repr, fp, esm, interaction], dim=-1))  # [B, 5d]


class DTAModel(nn.Module):
    """DTA model with hierarchical drug GNN and GCNConv→GATConv protein encoder."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_tasks: int = 1,
        task: str = "regression",
        dropout: float = 0.1,
        atom_in_dim: int = 37,
        motif_in_dim: int = 50,
        mol_num_layers: int = 2,
        prot_in_dim: int = 1152,
        prot_num_layers: int = 2,
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
        self.prot_encoder = ProteinGraphEncoder(
            prot_in_dim=prot_in_dim,
            hidden_dim=hidden_dim,
            num_layers=prot_num_layers,
            dropout=dropout,
        )
        self.surface_encoder = SurfaceEncoder()
        self.fusion_head = DTAInteractionHead(
            d=hidden_dim,
            num_tasks=num_tasks,
            dropout=dropout,
            fp_in_dim=fp_in_dim,
            esm_dim=esm_dim,
        )

    def forward(self, data: DTABatch) -> Tensor:
        atom_mol_out, motif_mol_out = self.encode_drug(data)
        protein_out = self.encode_protein(data)
        surface = self.surface_encoder(data.surface_embedding, data.surface_mask)
        return self.fusion_head(
            atom_mol_out=atom_mol_out,
            motif_mol_out=motif_mol_out,
            protein_out=protein_out,
            fingerprint=data.fingerprint,
            surface=surface,
            esm_global=data.esm_global,
        )

    def encode_drug(self, data: DTABatch):
        """Return atom-level and motif-level molecular graph representations."""

        return self.drug_encoder(data.hetero)

    def encode_protein(self, data: DTABatch):
        """Return one pooled representation per full-length protein graph."""

        graph = data.protein_graph
        return self.prot_encoder(
            x = graph.x, 
            edge_index=graph.edge_index,
            edge_weight=graph.edge_weight,
            batch=graph.batch,
        )

    def count_parameters(self) -> dict:
        def count(module):
            return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
        
        print(self.drug_encoder)
        print(self.prot_encoder)
        print(self.surface_encoder)
        print(self.fusion_head)
        return {
            "drug_encoder": count(self.drug_encoder),
            "prot_encoder": count(self.prot_encoder),
            "surface_encoder": count(self.surface_encoder),
            "fusion_head": count(self.fusion_head),
            "total": count(self),
        }
