"""Independent atom–residue and motif–residue interaction branches."""

import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import to_dense_batch


class AttentionBranch(nn.Module):
    """Condition one drug-token scale on protein residue tokens."""

    def __init__(
        self,
        drug_dim: int,
        protein_dim: int,
        embed_dim: int = 256,
        num_heads: int = 8,
    ):
        super().__init__()
        self.drug_proj = nn.Linear(drug_dim, embed_dim)
        self.protein_proj = nn.Linear(protein_dim, embed_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(0.1),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)

    @staticmethod
    def _masked_mean(x: Tensor, mask: Tensor) -> Tensor:
        mask_f = mask.unsqueeze(-1).to(dtype=x.dtype)
        count = mask_f.sum(dim=1).clamp_min(1.0)
        return (x * mask_f).sum(dim=1) / count

    def forward(
        self,
        drug_h: Tensor,
        drug_batch: Tensor,
        protein_h: Tensor,
        protein_batch: Tensor,
        return_attention: bool = False,
    ):
        drug_tokens, drug_mask = to_dense_batch(
            self.drug_proj(drug_h), drug_batch
        )
        protein_tokens, protein_mask = to_dense_batch(
            self.protein_proj(protein_h), protein_batch
        )

        attended, attention_weights = self.cross_attention(
            query=drug_tokens,
            key=protein_tokens,
            value=protein_tokens,
            key_padding_mask=~protein_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        valid_drug = drug_mask.unsqueeze(-1)
        attended = self.attention_norm(drug_tokens + attended)
        attended = attended.masked_fill(~valid_drug, 0.0)
        attended = self.ffn_norm(attended + self.ffn(attended))
        attended = attended.masked_fill(~valid_drug, 0.0)

        pooled = self._masked_mean(attended, drug_mask)
        attention_info = None
        if return_attention:
            attention_info = {
                "weights": attention_weights,
                "drug_mask": drug_mask,
                "protein_mask": protein_mask,
            }
        return pooled, attention_info


class MultiScaleAttention(nn.Module):
    """Run independent atom and motif attention branches on every forward."""

    def __init__(
        self,
        atom_dim: int,
        motif_dim: int,
        protein_dim: int,
        embed_dim: int = 256,
        num_heads: int = 8,
    ):
        super().__init__()
        if embed_dim < 1 or num_heads < 1:
            raise ValueError("embed_dim and num_heads must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.atom_attention = AttentionBranch(
            drug_dim=atom_dim,
            protein_dim=protein_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
        self.motif_attention = AttentionBranch(
            drug_dim=motif_dim,
            protein_dim=protein_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

    def forward(
        self,
        atom_h: Tensor,
        atom_batch: Tensor,
        motif_h: Tensor,
        motif_batch: Tensor,
        protein_h: Tensor,
        protein_batch: Tensor,
        return_attention: bool = False,
    ):
        atom_repr, atom_info = self.atom_attention(
            atom_h,
            atom_batch,
            protein_h,
            protein_batch,
            return_attention,
        )
        motif_repr, motif_info = self.motif_attention(
            motif_h,
            motif_batch,
            protein_h,
            protein_batch,
            return_attention,
        )

        attention_info = None
        if return_attention:
            attention_info = {
                "atom_weights": (
                    atom_info["weights"]
                ),
                "motif_weights": (
                    motif_info["weights"]
                ),
                "atom_mask": (
                    atom_info["drug_mask"]
                ),
                "motif_mask": (
                    motif_info["drug_mask"]
                ),
                "atom_protein_mask": (
                    atom_info["protein_mask"]
                ),
                "motif_protein_mask": (
                    motif_info["protein_mask"]
                ),
            }
        return atom_repr, motif_repr, attention_info
