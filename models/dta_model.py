"""Top-level hierarchical drug-target affinity model."""

from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from .cross_attention import AttentionBranch
from .encoder import DrugGraphEncoder, ProteinGraphEncoder
from .interaction import CrossLevelExchange


VALID_INTERACTION_TYPES = (
    "all",
    "woatom",
    "womotif",
    "woglobal",
)


def interaction_components(interaction_type: str):
    if interaction_type not in VALID_INTERACTION_TYPES:
        raise ValueError(
            "interaction_type must be one of "
            f"{list(VALID_INTERACTION_TYPES)}"
        )
    modes = {
        "all": frozenset({"atom", "motif", "global"}),
        "woatom": frozenset({"motif", "global"}),
        "womotif": frozenset({"atom", "global"}),
        "woglobal": frozenset({"atom", "motif"}),
    }
    return modes[interaction_type]


class FinalFCLayers(nn.Module):
    def __init__(
        self,
        in_dim: int,
        binary: int = 1,
        hidden_dims: Sequence[int] = (512, 256),
        dropout: float = 0.3,
    ):
        super().__init__()
        if in_dim < 1 or binary < 1:
            raise ValueError("in_dim and binary must be positive")
        if any(hidden_dim < 1 for hidden_dim in hidden_dims):
            raise ValueError("all hidden dimensions must be positive")

        layers = []
        current_dim = in_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, binary))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class FusionHead(nn.Module):
    """Fuse only the representations selected by ``interaction_type``."""

    def __init__(
        self,
        interaction_dim: int = 256,
        fp_dim: int = 2048,
        fp_out_dim: int = 256,
        esm_in_dim: int = 1280,
        esm_out_dim: int = 256,
        global_out_dim: int = 256,
        dropout: float = 0.3,
        interaction_type: str = "all",
    ):
        super().__init__()
        self.interaction_type = interaction_type
        self.components = interaction_components(interaction_type)

        self.fp_proj = None
        self.esm_global_proj = None
        self.global_encoder = None
        if "global" in self.components:
            self.fp_proj = nn.Sequential(
                nn.Linear(fp_dim, fp_out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.esm_global_proj = nn.Sequential(
                nn.Linear(esm_in_dim, esm_out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.global_encoder = FinalFCLayers(
                in_dim=fp_out_dim + esm_out_dim,
                binary=global_out_dim,
                hidden_dims=(512,),
                dropout=dropout,
            )

        fusion_dim = 0
        if "atom" in self.components:
            fusion_dim += interaction_dim
        if "motif" in self.components:
            fusion_dim += interaction_dim
        if "global" in self.components:
            fusion_dim += global_out_dim

        # first_hidden_dim = min(512, fusion_dim)
        # second_hidden_dim = max(128, first_hidden_dim // 2)
        self.mlp = FinalFCLayers(
            in_dim=fusion_dim,
            binary=1,
            hidden_dims=(1024, 256),  # all mode: 768 -> 1024 -> 256
            dropout=dropout,
        )

    def forward(
        self,
        atom_repr: Tensor,
        motif_repr: Tensor,
        fingerprint: Tensor,
        esm_global: Tensor,
    ) -> Tensor:
        fusion_inputs = []
        if "atom" in self.components:
            fusion_inputs.append(atom_repr)
        if "motif" in self.components:
            fusion_inputs.append(motif_repr)
        if "global" in self.components:
            fp_repr = self.fp_proj(fingerprint)
            esm_repr = self.esm_global_proj(esm_global)
            global_repr = self.global_encoder(
                torch.cat([fp_repr, esm_repr], dim=-1)
            )
            fusion_inputs.append(global_repr)

        return self.mlp(torch.cat(fusion_inputs, dim=-1))


class DTAModel(nn.Module):
    """Hierarchical DTA model with drug-residue cross-attention pooling."""

    def __init__(
        self,
        interaction_type: str = "all",
        atom_in_dim: int = 37,
        atom_edge_dim: int = 13,
        motif_in_dim: int = 50,
        motif_edge_dim: int = 37,
        protein_in_dim: int = 41,
        protein_num_layers: int = 2,
        drug_num_layers: int = 2,
        drug_gnn_type: str = "gat",
        hidden_dim: int = 256,
        num_heads: int = 8,
        fp_dim: int = 2048,
        fp_out_dim: int = 256,
        esm_in_dim: int = 1280,
        esm_out_dim: int = 256,
        global_out_dim: int = 256,
        dropout: float = 0.2,
        use_agg: bool = True,
    ):
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")
        if hidden_dim < 1 or num_heads < 1 or hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be positive and divisible by num_heads"
            )
        self.interaction_type = interaction_type
        self.components = interaction_components(interaction_type)
        self.use_atom = "atom" in self.components
        self.use_motif = "motif" in self.components
        self.use_agg = use_agg and self.use_atom and self.use_motif

        # ── drug encoder(s) ──────────────────────────────────────────
        self.atom_encoder = None
        if self.use_atom:
            self.atom_encoder = DrugGraphEncoder(
                in_dim=atom_in_dim,
                edge_dim=atom_edge_dim,
                hidden_dim=hidden_dim,
                num_layers=drug_num_layers,
                input_dropout=dropout,
                build_graph_head=False,
                gnn_type=drug_gnn_type,
                node_key="atom",
                edge_key=("atom", "bond", "atom"),
            )

        self.motif_encoder = None
        if self.use_motif:
            self.motif_encoder = DrugGraphEncoder(
                in_dim=motif_in_dim,
                edge_dim=motif_edge_dim,
                hidden_dim=hidden_dim,
                num_layers=drug_num_layers,
                input_dropout=dropout,
                build_graph_head=False,
                gnn_type=drug_gnn_type,
                node_key="motif",
                edge_key=("motif", "connects", "motif"),
            )
        self.exchange = None
        if self.use_agg:
            self.exchange = CrossLevelExchange(
                atom_dim=self.atom_encoder.node_out_dim,
                motif_dim=self.motif_encoder.node_out_dim,
            )

        # ── protein encoder ──────────────────────────────────────────
        self.protein_encoder = ProteinGraphEncoder(
            protein_in_dim=protein_in_dim,
            hidden_dim=hidden_dim,
            num_layers=protein_num_layers,
            input_dropout=dropout,
            protein_edge_dim=10,
            build_graph_head=False,
        )
        self.atom_attention = None
        if self.use_atom:
            self.atom_attention = AttentionBranch(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
        self.motif_attention = None
        if self.use_motif:
            self.motif_attention = AttentionBranch(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )

        # ── Separate atom/motif interactions + 256-D global pair head ─────
        self.fusion_head = FusionHead(
            interaction_dim=hidden_dim,
            fp_dim=fp_dim,
            fp_out_dim=fp_out_dim,
            esm_in_dim=esm_in_dim,
            esm_out_dim=esm_out_dim,
            global_out_dim=global_out_dim,
            dropout=dropout,
            interaction_type=interaction_type,
        )

    def forward(self, data, return_attention: bool = False):
        atom_h = None
        motif_h = None
        if self.use_atom:
            atom_h = self.atom_encoder.encode_nodes(data.hetero)
        if self.use_motif:
            motif_h = self.motif_encoder.encode_nodes(data.hetero)
        if self.exchange is not None:
            atom_h, motif_h = self.exchange(
                atom_h=atom_h,
                motif_h=motif_h,
                membership_edge_index=data.hetero[
                    "atom", "in", "motif"
                ].edge_index,
            )

        atom_repr = None
        motif_repr = None
        attention_info = {} if return_attention else None
        protein_h = self.protein_encoder.encode_nodes(
            data.protein_graph.x,
            data.protein_graph.edge_index,
            data.protein_graph.edge_attr,
        )
        if self.atom_attention is not None:
            atom_repr, atom_info = self.atom_attention(
                atom_h,
                data.hetero["atom"].batch,
                protein_h,
                data.protein_graph.batch,
                return_attention,
            )
            if return_attention:
                attention_info.update({
                    "atom_weights": atom_info["weights"],
                    "atom_mask": atom_info["drug_mask"],
                    "atom_protein_mask": atom_info["protein_mask"],
                })
        if self.motif_attention is not None:
            motif_repr, motif_info = self.motif_attention(
                motif_h,
                data.hetero["motif"].batch,
                protein_h,
                data.protein_graph.batch,
                return_attention,
            )
            if return_attention:
                attention_info.update({
                    "motif_weights": motif_info["weights"],
                    "motif_mask": motif_info["drug_mask"],
                    "motif_protein_mask": motif_info["protein_mask"],
                })

        prediction = self.fusion_head(
            atom_repr=atom_repr,
            motif_repr=motif_repr,
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
        print("── exchange ──")
        if self.exchange is None:
            print("Disabled (use_agg=False)")
        else:
            print(self.exchange)
        print("── protein_encoder ──")
        print(self.protein_encoder)
        print("── atom_attention ──")
        print(self.atom_attention)
        print("── motif_attention ──")
        print(self.motif_attention)
        print("── fusion_head ──")
        print(self.fusion_head)

        return {
            "total": count(self),
        }
