"""Top-level DTA model and custom batching for nested PyG graph objects."""

from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Batch, Data, HeteroData
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_dense_batch

from .encoder import HeteroMolGNN, ProteinGraphEncoder, SurfaceEncoder
from .fusion import (
    Interaction,
)


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
        self.fc4 = nn.Linear(hidden_dim2, binary)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc4(x)


class DTAFusionHead(nn.Module):
    """Fuse local motif-residue interactions with compact global representations."""

    def __init__(
        self,
        d: int = 256,
        num_tasks: int = 1,
        dropout: float = 0.2,
        fp_in_dim: int = 1024,
    ):
        super().__init__()
        self.local_interaction = Interaction(
            hidden_dim=d,
            dropout=dropout,
        )
        self.fp_proj = nn.Sequential(
            nn.Linear(fp_in_dim, d),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.drug_proj = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.target_proj = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.global_pair_proj = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.local_proj = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.local_gate = nn.Linear(2 * d, d)
        self.decoder = MLPDecoder(
            in_dim=d,
            hidden_dim1=2 * d,
            hidden_dim2=d,
            binary=num_tasks,
            dropout=dropout,
        )
        self.last_motif_residue_attention = None
        self.last_motif_weight = None
        self.last_local_gate = None

    def forward(
        self,
        motif_tokens: Tensor,
        motif_batch: Tensor,
        protein_out: Tensor,
        residue_tokens: Tensor,
        residue_batch: Tensor,
        fingerprint: Tensor,
        surface: Tensor,
    ) -> Tensor:
        if fingerprint.dim() == 3:
            fingerprint = fingerprint.squeeze(1)

        dense_motif, motif_mask = to_dense_batch(motif_tokens, motif_batch)
        dense_residue, residue_mask = to_dense_batch(residue_tokens, residue_batch)
        local, attention, motif_weight = self.local_interaction(
            motif_tokens=dense_motif,
            residue_tokens=dense_residue,
            motif_mask=motif_mask,
            residue_mask=residue_mask,
        )
        # Retain detached interaction weights for interpretation after inference.
        self.last_motif_residue_attention = attention.detach()
        self.last_motif_weight = motif_weight.detach()

        enriched_motif_out = global_mean_pool(motif_tokens, motif_batch)
        drug = self.drug_proj(torch.cat([
            enriched_motif_out,
            self.fp_proj(fingerprint),
        ], dim=-1))
        target = self.target_proj(torch.cat([protein_out, surface], dim=-1))
        global_pair = self.global_pair_proj(torch.cat([drug, target], dim=-1))
        local_context = self.local_proj(local)
        local_gate = torch.sigmoid(
            self.local_gate(torch.cat([global_pair, local_context], dim=-1))
        )
        self.last_local_gate = local_gate.detach()
        fused = global_pair + local_gate * local_context
        return self.decoder(fused)


class DTAModel(nn.Module):
    """DTA model with hierarchical global fusion and motif-residue interaction."""

    def __init__(
        self,
        hidden_dim: int = 256,
        num_tasks: int = 1,
        task: str = "regression",
        dropout: float = 0.1,
        atom_in_dim: int = 37,
        motif_in_dim: int = 50,
        aa_edge_dim: int = 13,
        mm_edge_dim: int = 37,
        atom_num_layers: int = 2,
        motif_num_layers: int = 2,
        prot_in_dim: int = 1152,
        prot_num_layers: int = 2,
        fp_in_dim: int = 1024,
    ):
        super().__init__()
        self.task = task
        self.drug_encoder = HeteroMolGNN(
            atom_in_dim=atom_in_dim,
            motif_in_dim=motif_in_dim,
            hidden_dim=hidden_dim,
            atom_num_layers=atom_num_layers,
            motif_num_layers=motif_num_layers,
            aa_edge_dim=aa_edge_dim,
            mm_edge_dim=mm_edge_dim,
            dropout=dropout,
        )
        self.prot_encoder = ProteinGraphEncoder(
            prot_in_dim=prot_in_dim,
            hidden_dim=hidden_dim,
            num_layers=prot_num_layers,
            dropout=dropout,
        )
        self.surface_encoder = SurfaceEncoder(
            in_dim=128,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.fusion_head = DTAFusionHead(
            d=hidden_dim,
            num_tasks=num_tasks,
            dropout=dropout,
            fp_in_dim=fp_in_dim,
        )

    def forward(self, data: DTABatch) -> Tensor:
        _atom_tokens, motif_tokens = self.encode_drug(data)
        protein_out, residue_tokens = self.encode_protein(data)
        surface = self.surface_encoder(data.surface_embedding, data.surface_mask)
        hetero = data.hetero
        protein_graph = data.protein_graph
        return self.fusion_head(
            motif_tokens=motif_tokens,
            motif_batch=hetero["motif"].batch,
            protein_out=protein_out,
            residue_tokens=residue_tokens,
            residue_batch=protein_graph.batch,
            fingerprint=data.fingerprint,
            surface=surface,
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
            return_tokens=True,
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
