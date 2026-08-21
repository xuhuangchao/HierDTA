"""Fine-grained interactions between drug nodes and protein residues."""

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.utils import to_dense_batch


class MultiScaleDrugResidueAttention(nn.Module):
    """Drug-node queries attend to residue-node keys and values.

    Atom and motif tokens share the attention space but retain distinct type
    embeddings and are pooled separately after cross-attention.  The final
    representation therefore preserves both chemical scales.
    """

    def __init__(
        self,
        atom_dim: int,
        motif_dim: int,
        protein_dim: int,
        embed_dim: int = 256,
        num_heads: int = 8,
        out_dim: int = 1024,
    ):
        super().__init__()
        if embed_dim < 1 or num_heads < 1:
            raise ValueError("embed_dim and num_heads must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.atom_proj = nn.Linear(atom_dim, embed_dim)
        self.motif_proj = nn.Linear(motif_dim, embed_dim)
        self.protein_proj = nn.Linear(protein_dim, embed_dim)

        self.atom_type = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.motif_type = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.atom_type, std=0.02)
        nn.init.normal_(self.motif_type, std=0.02)

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

        self.pool_proj = nn.Sequential(
            nn.Linear(embed_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

    @staticmethod
    def _masked_mean(x: Tensor, mask: Tensor) -> Tensor:
        """Return the masked mean for a padded node tensor."""
        mask_f = mask.unsqueeze(-1).to(dtype=x.dtype)
        count = mask_f.sum(dim=1).clamp_min(1.0)
        return (x * mask_f).sum(dim=1) / count

    def forward(
        self,
        atom_h: Tensor,
        atom_batch: Tensor,
        motif_h: Tensor,
        motif_batch: Tensor,
        protein_h: Tensor,
        protein_batch: Tensor,
        drug_graph_type: str = "dual",
        return_attention: bool = False,
    ):
        if drug_graph_type not in {"atom", "motif", "dual"}:
            raise ValueError("drug_graph_type must be 'atom', 'motif', or 'dual'")

        atom_tokens, atom_mask = to_dense_batch(
            self.atom_proj(atom_h), atom_batch
        )
        motif_tokens, motif_mask = to_dense_batch(
            self.motif_proj(motif_h), motif_batch
        )
        protein_tokens, protein_mask = to_dense_batch(
            self.protein_proj(protein_h), protein_batch
        )

        atom_tokens = atom_tokens + self.atom_type
        motif_tokens = motif_tokens + self.motif_type
        drug_tokens = torch.cat([atom_tokens, motif_tokens], dim=1)
        drug_mask = torch.cat([atom_mask, motif_mask], dim=1)

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

        atom_width = atom_tokens.size(1)
        atom_out = attended[:, :atom_width]
        motif_out = attended[:, atom_width:]
        atom_mean = self._masked_mean(atom_out, atom_mask)
        motif_mean = self._masked_mean(motif_out, motif_mask)

        if drug_graph_type == "atom":
            pooled = atom_mean
        elif drug_graph_type == "motif":
            pooled = motif_mean
        else:
            pooled = 0.5 * (atom_mean + motif_mean)
        cross_repr = self.pool_proj(pooled)

        attention_info = None
        if return_attention:
            attention_info = {
                "weights": attention_weights,
                "atom_weights": attention_weights[:, :, :atom_width, :],
                "motif_weights": attention_weights[:, :, atom_width:, :],
                "atom_mask": atom_mask,
                "motif_mask": motif_mask,
                "protein_mask": protein_mask,
            }
        return cross_repr, attention_info
