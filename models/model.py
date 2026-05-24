import torch
import torch.nn as nn
import torch.nn.functional as F
from .drug_model import *
from .protein_model import *
from torch_geometric.utils import to_dense_batch

class MLPFusionLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None, dropout_rate=0.2):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = (input_dim + output_dim) // 2
        self.fusion_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, concatenated_features):
        fused_representation = self.fusion_mlp(concatenated_features)
        return fused_representation

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1, dropout=0.2):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.fc4 = nn.Linear(out_dim, binary)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

class D2PInteractionModule(nn.Module):
    """
    单向交互：Drug → Protein (D2P)
    药物侧 (atom / motif) 作为 query，蛋白残基作为 key/value。
    最终 multi_level_reps 与 fingerprint、p_global 拼接后预测。
    """
    def __init__(self, emb_dim, fp_dim=1024, esm_dim=480, dropout=0.2,
                 use_fingerprint=True, use_p_global=True):
        super().__init__()
        self.emb_dim = emb_dim
        self.use_fingerprint = use_fingerprint
        self.use_p_global = use_p_global

        self.interaction_proj = nn.Linear(emb_dim, emb_dim)

        if self.use_fingerprint:
            self.fp_projection = nn.Sequential(
                nn.Linear(fp_dim, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, emb_dim)
            )
        if self.use_p_global:
            self.esm_projection = nn.Sequential(
                nn.Linear(esm_dim, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, emb_dim)
            )

        in_dim = emb_dim
        if use_fingerprint:
            in_dim += emb_dim
        if use_p_global:
            in_dim += emb_dim
        self.predictor = MLPDecoder(in_dim, 1024, 256, 1, dropout=dropout)

    def _masked_softmax(self, scores, mask, dim):
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask_expanded == 0, -1e9)
        attn = F.softmax(scores, dim=dim)
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).expand_as(attn) == 0, 0)
            sum_attn = attn.sum(dim=dim, keepdim=True).clamp(min=1e-9)
            attn = attn / sum_attn
        return attn

    def _masked_mean(self, tensor, mask):
        if mask is None:
            return tensor.mean(dim=1)
        mask_expanded = mask.unsqueeze(-1).float()
        sum_tensor = (tensor * mask_expanded).sum(dim=1)
        count = mask.float().sum(dim=1).unsqueeze(-1).clamp(min=1e-9)
        return sum_tensor / count

    def _to_dense_safe(self, feat, batch, expected_size):
        dense, mask = to_dense_batch(feat, batch)
        if dense.size(0) < expected_size:
            pad = expected_size - dense.size(0)
            dense = torch.cat([dense, dense.new_zeros(pad, dense.size(1), dense.size(2))], dim=0)
            mask = torch.cat([mask, mask.new_zeros(pad, mask.size(1), dtype=torch.bool)], dim=0)
        return dense, mask

    def forward(self, atom_feat, motif_feat, batch_d, node_types, fingerprint, h_node_p, batch_p, esm_feat):
        protein_rep, protein_mask = to_dense_batch(h_node_p, batch_p)
        expected_batch_size = protein_rep.size(0)

        atom_mask_nodes = (node_types == 0)
        motif_mask_nodes = (node_types == 1)
        batch_atoms = batch_d[atom_mask_nodes]
        batch_motifs = batch_d[motif_mask_nodes]

        atom_rep, atom_mask = self._to_dense_safe(atom_feat, batch_atoms, expected_batch_size)
        motif_rep, motif_mask = self._to_dense_safe(motif_feat, batch_motifs, expected_batch_size)

        protein_h = F.relu(self.interaction_proj(protein_rep))
        atom_h = F.relu(self.interaction_proj(atom_rep))
        motif_h = F.relu(self.interaction_proj(motif_rep))

        scores_atom = torch.bmm(atom_h, protein_h.transpose(1, 2)) / (self.emb_dim ** 0.5)
        attn_atom = self._masked_softmax(scores_atom, protein_mask, dim=2)
        atom_enhanced = torch.bmm(attn_atom, protein_rep)
        view_atom = self._masked_mean(atom_enhanced, atom_mask)

        scores_motif = torch.bmm(motif_h, protein_h.transpose(1, 2)) / (self.emb_dim ** 0.5)
        attn_motif = self._masked_softmax(scores_motif, protein_mask, dim=2)
        motif_enhanced = torch.bmm(attn_motif, protein_rep)
        view_motif = self._masked_mean(motif_enhanced, motif_mask)

        # 直接平均：同一语义空间下多粒度描述的线性叠加
        multi_level_reps = (view_atom + view_motif) / 2.0

        reps = [multi_level_reps]
        if self.use_fingerprint:
            fp_emb = self.fp_projection(fingerprint)
            reps.append(fp_emb)
        if self.use_p_global:
            pg_emb = self.esm_projection(esm_feat)
            reps.append(pg_emb)

        final_rep = torch.cat(reps, dim=-1)
        logits = self.predictor(final_rep)
        return logits, {}

class P2DInteractionModule(nn.Module):
    """
    单向交互：Protein → Drug (P2D)
    蛋白残基作为 query，药物侧 (atom / motif) 作为 key/value。
    最终 multi_level_reps 与 fingerprint、p_global 拼接后预测。
    """
    def __init__(self, emb_dim, fp_dim=1024, esm_dim=480, dropout=0.2,
                 use_fingerprint=True, use_p_global=True):
        super().__init__()
        self.emb_dim = emb_dim
        self.use_fingerprint = use_fingerprint
        self.use_p_global = use_p_global

        self.interaction_proj = nn.Linear(emb_dim, emb_dim)

        if self.use_fingerprint:
            self.fp_projection = nn.Sequential(
                nn.Linear(fp_dim, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, emb_dim)
            )
        if self.use_p_global:
            self.esm_projection = nn.Sequential(
                nn.Linear(esm_dim, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, emb_dim)
            )

        in_dim = emb_dim
        if use_fingerprint:
            in_dim += emb_dim
        if use_p_global:
            in_dim += emb_dim
        self.predictor = MLPDecoder(in_dim, 1024, 256, 1, dropout=dropout)

    def _masked_softmax(self, scores, mask, dim):
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask_expanded == 0, -1e9)
        attn = F.softmax(scores, dim=dim)
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).expand_as(attn) == 0, 0)
            sum_attn = attn.sum(dim=dim, keepdim=True).clamp(min=1e-9)
            attn = attn / sum_attn
        return attn

    def _masked_mean(self, tensor, mask):
        if mask is None:
            return tensor.mean(dim=1)
        mask_expanded = mask.unsqueeze(-1).float()
        sum_tensor = (tensor * mask_expanded).sum(dim=1)
        count = mask.float().sum(dim=1).unsqueeze(-1).clamp(min=1e-9)
        return sum_tensor / count

    def _to_dense_safe(self, feat, batch, expected_size):
        dense, mask = to_dense_batch(feat, batch)
        if dense.size(0) < expected_size:
            pad = expected_size - dense.size(0)
            dense = torch.cat([dense, dense.new_zeros(pad, dense.size(1), dense.size(2))], dim=0)
            mask = torch.cat([mask, mask.new_zeros(pad, mask.size(1), dtype=torch.bool)], dim=0)
        return dense, mask

    def forward(self, atom_feat, motif_feat, batch_d, node_types, fingerprint, h_node_p, batch_p, esm_feat):
        protein_rep, protein_mask = to_dense_batch(h_node_p, batch_p)
        expected_batch_size = protein_rep.size(0)

        atom_mask_nodes = (node_types == 0)
        motif_mask_nodes = (node_types == 1)
        batch_atoms = batch_d[atom_mask_nodes]
        batch_motifs = batch_d[motif_mask_nodes]

        atom_rep, atom_mask = self._to_dense_safe(atom_feat, batch_atoms, expected_batch_size)
        motif_rep, motif_mask = self._to_dense_safe(motif_feat, batch_motifs, expected_batch_size)

        protein_h = F.relu(self.interaction_proj(protein_rep))
        atom_h = F.relu(self.interaction_proj(atom_rep))
        motif_h = F.relu(self.interaction_proj(motif_rep))

        scores_prot_atom = torch.bmm(protein_h, atom_h.transpose(1, 2)) / (self.emb_dim ** 0.5)
        attn_prot_atom = self._masked_softmax(scores_prot_atom, atom_mask, dim=2)
        prot_enhanced_atom = torch.bmm(attn_prot_atom, atom_rep)
        view_prot_atom = self._masked_mean(prot_enhanced_atom, protein_mask)

        scores_prot_motif = torch.bmm(protein_h, motif_h.transpose(1, 2)) / (self.emb_dim ** 0.5)
        attn_prot_motif = self._masked_softmax(scores_prot_motif, motif_mask, dim=2)
        prot_enhanced_motif = torch.bmm(attn_prot_motif, motif_rep)
        view_prot_motif = self._masked_mean(prot_enhanced_motif, protein_mask)

        # 直接平均：同一语义空间下多粒度描述的线性叠加
        multi_level_reps = (view_prot_atom + view_prot_motif) / 2.0

        reps = [multi_level_reps]
        if self.use_fingerprint:
            fp_emb = self.fp_projection(fingerprint)
            reps.append(fp_emb)
        if self.use_p_global:
            pg_emb = self.esm_projection(esm_feat)
            reps.append(pg_emb)

        final_rep = torch.cat(reps, dim=-1)
        logits = self.predictor(final_rep)
        return logits, {}

class HierDTA(nn.Module):
    def __init__(self,
                 num_features_xd=133,
                 drug_hidden=64,
                 drug_out=128,
                 n_layers_drug=2,
                 heads=2,
                 dropout=0.2,
                 num_features_xt=608,
                 protein_hidden=128,
                 protein_out=128,
                 n_layers_protein=4,
                 use_surface=True,
                 emb_dim=128,
                 fp_dim=1024,
                 esm_dim=480,
                 use_fingerprint=True,
                 use_p_global=True,
                 interaction_mode='d2p',
                 agg='mean'):
        super().__init__()

        assert drug_out == emb_dim, f"Drug out_channels ({drug_out}) must equal emb_dim ({emb_dim})"
        assert protein_out == emb_dim, f"Protein output_dim ({protein_out}) must equal emb_dim ({emb_dim})"

        self.drug_net = DrugModel(
            in_channels=num_features_xd,
            hidden_channels=drug_hidden,
            out_channels=drug_out,
            num_layers=n_layers_drug,
            heads=heads,
            dropout=dropout,
            agg=agg
        )
        print(self.drug_net)

        self.protein_net = ProteinEGNN(
            num_features_xt=num_features_xt,
            hidden_nf=protein_hidden,
            output_dim=protein_out,
            n_layers=n_layers_protein,
            use_surface=use_surface
        )

        if interaction_mode == 'd2p':
            self.interaction = D2PInteractionModule(
                emb_dim=emb_dim,
                fp_dim=fp_dim,
                esm_dim=esm_dim,
                dropout=dropout,
                use_fingerprint=use_fingerprint,
                use_p_global=use_p_global
            )
        elif interaction_mode == 'p2d':
            self.interaction = P2DInteractionModule(
                emb_dim=emb_dim,
                fp_dim=fp_dim,
                esm_dim=esm_dim,
                dropout=dropout,
                use_fingerprint=use_fingerprint,
                use_p_global=use_p_global
            )
        else:
            raise ValueError(f"Unknown interaction_mode: {interaction_mode}. Choose 'd2p' or 'p2d'.")

    def forward(self, data):
        atom_feat, motif_feat, batch_d, node_types, fingerprint = self.drug_net(data.drug_graph)
        h_node_p, batch_p, esm_feat = self.protein_net(data.protein_graph)

        affinity, output_dict = self.interaction(atom_feat, motif_feat, batch_d, node_types, fingerprint, h_node_p, batch_p, esm_feat)

        if self.training:
            return affinity
        else:
            return affinity, output_dict
