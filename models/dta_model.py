"""Top-level DTA model and custom batching for nested PyG graph objects."""

from typing import Any, List

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch, Data, HeteroData

from .cross_attention import MultiScaleDrugResidueAttention
from .encoder import AtomGNN, ProteinGraphEncoder
from .interaction import BottomUpAtomMotifFusion


class DTABatch:
    """Container for drug graphs, protein graphs, globals, and labels."""

    __slots__ = (
        "hetero",
        "protein_graph",
        "fingerprint",
        "esm_global",
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

    return DTABatch(
        hetero=Batch.from_data_list([data.hetero for data in data_list]),
        protein_graph=Batch.from_data_list([data.protein_graph for data in data_list]),
        fingerprint=torch.cat([data.fingerprint for data in data_list], dim=0),
        esm_global=torch.cat([data.esm_global for data in data_list], dim=0),
        y=torch.cat([data.y for data in data_list], dim=0),
        smiles=[getattr(data, "smiles", "") for data in data_list],
        key=[getattr(data, "key", "") for data in data_list],
    )


# Experimental graph-level fusion retained for a future ablation, but inactive
# in the current cross-attention training path.
#
# class GatedFusionLayer(nn.Module):
#     def __init__(self, v_dim, q_dim, output_dim=128, dropout_rate=0.2):
#         super(GatedFusionLayer, self).__init__()
#         self.v_transform = nn.Linear(v_dim, output_dim)
#         self.q_transform = nn.Linear(q_dim, output_dim)
#         self.gate_transform = nn.Linear(output_dim * 2, output_dim)
#         self.activation = nn.Tanh()
#         self.output_dim = output_dim
#
#     def get_output_shape(self):
#         return self.output_dim
#
#     def forward(self, v, q, return_gate=False):
#         v_proj = self.activation(self.v_transform(v))
#         q_proj = self.activation(self.q_transform(q))
#         concat_proj = torch.cat([v_proj, q_proj], dim=1)
#         gate = torch.sigmoid(self.gate_transform(concat_proj))
#         gated_output = gate * v_proj + (1 - gate) * q_proj
#         if return_gate:
#             return gated_output, gate
#         return gated_output


class FinalFCLayers(nn.Module):
    def __init__(
        self,
        in_dim: int,
        binary: int = 1,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, binary),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class FusionHead(nn.Module):
    """Fuse one structural representation with global descriptors."""

    def __init__(
        self,
        structural_dim: int = 1024,
        fp_dim: int = 2048,
        fp_out_dim: int = 256,
        esm_in_dim: int = 1280,
        esm_out_dim: int = 256,
        dropout: float = 0.5,
        modality_ablation: str = "none",
    ):
        super().__init__()
        self.modality_ablation = modality_ablation
        self.fp_proj = None
        if modality_ablation != "drug_fingerprint":
            self.fp_proj = nn.Sequential(
                nn.Linear(fp_dim, fp_out_dim),
                nn.BatchNorm1d(fp_out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        self.esm_global_proj = None
        if modality_ablation != "protein_seq":
            self.esm_global_proj = nn.Sequential(
                nn.Linear(esm_in_dim, esm_out_dim),
                nn.BatchNorm1d(esm_out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        fusion_dim = structural_dim
        if self.fp_proj is not None:
            fusion_dim += fp_out_dim
        if self.esm_global_proj is not None:
            fusion_dim += esm_out_dim
        self.mlp = FinalFCLayers(fusion_dim, 1, dropout=0.5)

    def forward(
        self,
        structural_repr: Tensor,
        fingerprint: Tensor,
        esm_global: Tensor,
    ) -> Tensor:
        fusion_inputs = [structural_repr]
        if self.fp_proj is not None:
            fusion_inputs.append(self.fp_proj(fingerprint))
        if self.esm_global_proj is not None:
            fusion_inputs.append(self.esm_global_proj(esm_global))

        return self.mlp(torch.cat(fusion_inputs, dim=-1))


class DTAModel(nn.Module):
    """Hierarchical DTA model with drug-residue cross-attention pooling."""

    def __init__(
        self,
        drug_graph_type: str = "atom",
        atom_in_dim: int = 37,
        atom_edge_dim: int = 13,
        motif_in_dim: int = 50,
        motif_edge_dim: int = 37,
        protein_in_dim: int = 41,
        protein_num_layers: int = 1,
        protein_hidden_dim: int = 256,
        atom_num_layers: int = 1,
        motif_num_layers: int = 1,
        graph_pool_type: str = "mean_add_max",
        embed_dim: int = 256,
        num_heads: int = 8,
        cross_out_dim: int = 1024,
        protein_graph_out_dim: int = 1024,
        modality_ablation: str = "none",
        fp_dim: int = 2048,
        fp_out_dim: int = 256,
        esm_in_dim: int = 1280,
        esm_out_dim: int = 256,
        dropout: float = 0.5,
        protein_graph_mode: str = "cov",
    ):
        super().__init__()
        if drug_graph_type not in ("atom", "motif", "dual"):
            raise ValueError("drug_graph_type must be 'atom', 'motif', or 'dual'")
        if protein_graph_mode not in ("cov", "noncov", "dual_view"):
            raise ValueError(
                "protein_graph_mode must be 'cov', 'noncov', or 'dual_view'"
            )
        valid_ablations = {"none", "drug_fingerprint", "protein_seq"}
        if modality_ablation not in valid_ablations:
            raise ValueError(
                f"modality_ablation must be one of {sorted(valid_ablations)}"
            )

        self.drug_graph_type = drug_graph_type
        self.protein_graph_mode = protein_graph_mode
        self.modality_ablation = modality_ablation

        # ── drug encoder(s) ──────────────────────────────────────────
        self.atom_encoder = AtomGNN(
            in_dim=atom_in_dim,
            edge_dim=atom_edge_dim,
            num_layers=atom_num_layers,
            graph_pool_type=graph_pool_type,
            build_graph_head=False,
            node_key="atom",
            edge_key=("atom", "bond", "atom"),
        )

        self.motif_encoder = AtomGNN(
            in_dim=motif_in_dim,
            edge_dim=motif_edge_dim,
            num_layers=motif_num_layers,
            graph_pool_type=graph_pool_type,
            build_graph_head=False,
            node_key="motif",
            edge_key=("motif", "connects", "motif"),
        )

        self.atom_motif_fusion = BottomUpAtomMotifFusion(
            atom_dim=self.atom_encoder.node_out_dim,
            motif_dim=motif_in_dim,
        )

        # ── protein encoder ──────────────────────────────────────────
        self.protein_encoder = ProteinGraphEncoder(
            protein_in_dim=protein_in_dim,
            hidden_dim=protein_hidden_dim,
            num_layers=protein_num_layers,
            graph_pool_type=graph_pool_type,
            graph_out_dim=protein_graph_out_dim,
            protein_edge_dim=10,
            protein_graph_mode=protein_graph_mode,
            build_graph_head=True,
        )

        self.cross_attention = MultiScaleDrugResidueAttention(
            atom_dim=self.atom_encoder.node_out_dim,
            motif_dim=self.motif_encoder.node_out_dim,
            protein_dim=protein_hidden_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            out_dim=cross_out_dim,
        )

        # ── fusion head (input dimension follows the retained modalities) ───
        self.fusion_head = FusionHead(
            structural_dim=cross_out_dim + protein_graph_out_dim,
            fp_dim=fp_dim,
            fp_out_dim=fp_out_dim,
            esm_in_dim=esm_in_dim,
            esm_out_dim=esm_out_dim,
            dropout=dropout,
            modality_ablation=modality_ablation,
        )

    def forward(self, data: DTABatch, return_attention: bool = False):
        atom_h = self.atom_encoder.encode_nodes(data.hetero)
        motif_x = self.atom_motif_fusion(
            atom_h=atom_h,
            motif_x=data.hetero["motif"].x,
            membership_edge_index=data.hetero[
                "atom", "in", "motif"
            ].edge_index,
        )
        motif_h = self.motif_encoder.encode_nodes(
            data.hetero,
            x_override=motif_x,
        )

        protein_h = self.protein_encoder.encode_nodes(
            data.protein_graph.x,
            data.protein_graph.edge_index,
            data.protein_graph.edge_attr,
        )

        cross_repr, attention_info = self.cross_attention(
            atom_h=atom_h,
            atom_batch=data.hetero["atom"].batch,
            motif_h=motif_h,
            motif_batch=data.hetero["motif"].batch,
            protein_h=protein_h,
            protein_batch=data.protein_graph.batch,
            drug_graph_type=self.drug_graph_type,
            return_attention=return_attention,
        )

        protein_graph_repr = self.protein_encoder.readout(
            protein_h,
            data.protein_graph.batch,
        )
        structural_repr = torch.cat(
            [cross_repr, protein_graph_repr],
            dim=-1,
        )

        prediction = self.fusion_head(
            structural_repr=structural_repr,
            fingerprint=data.fingerprint,
            esm_global=data.esm_global,
        )
        if return_attention:
            return prediction, attention_info
        return prediction

    def count_parameters(self) -> dict:
        def count(module):
            if module is None:
                return 0
            return sum(
                parameter.numel()
                for parameter in module.parameters()
                if parameter.requires_grad
            )

        print("── atom_encoder ──")
        if self.atom_encoder is not None:
            print(self.atom_encoder)
        print("── motif_encoder ──")
        if self.motif_encoder is not None:
            print(self.motif_encoder)
        print("── atom_motif_fusion ──")
        print(self.atom_motif_fusion)
        print("── protein_encoder ──")
        print(self.protein_encoder)
        print("── cross_attention ──")
        print(self.cross_attention)
        print("── fusion_head ──")
        print(self.fusion_head)

        return {
            "total": count(self),
        }

    def load_state_dict(self, state_dict, strict=True):
        """Load current or legacy checkpoints with renamed protein encoder keys."""
        legacy_prefix = "pocket_encoder."
        if any(key.startswith(legacy_prefix) for key in state_dict):
            state_dict = state_dict.copy()
            for key in list(state_dict):
                if key.startswith(legacy_prefix):
                    new_key = "protein_encoder." + key[len(legacy_prefix):]
                    state_dict[new_key] = state_dict.pop(key)
        return super().load_state_dict(state_dict, strict=strict)
