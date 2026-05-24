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
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.2):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class CoAttentionModule(nn.Module):
    """
    双向节点交互 + Bilinear 融合模块。
    Drug↔Protein cross-attention → fp/esm 增强全局表示 → Bilinear fusion → MLP 预测。
    """
    def __init__(self, emb_dim, fp_dim=1024, esm_dim=480, dropout=0.2,
                 use_fingerprint=True, use_p_global=True):
        super().__init__()
        self.emb_dim = emb_dim
        self.use_fingerprint = use_fingerprint
        self.use_p_global = use_p_global

        # 共享投影层（Drug / Protein 共用，保持语义空间一致）
        self.interaction_proj = nn.Linear(emb_dim, emb_dim)

        # 指纹 / ESM 投影到 emb_dim，用于与 cross-attn 读出示拼接
        if self.use_fingerprint:
            self.fp_projection = nn.Sequential(
                nn.Linear(fp_dim, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, emb_dim)
            )
        if self.use_p_global:
            self.esm_projection = nn.Sequential(
                nn.Linear(esm_dim, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, emb_dim)
            )

        drug_in_dim = emb_dim + (emb_dim if use_fingerprint else 0)
        prot_in_dim = emb_dim + (emb_dim if use_p_global else 0)

        # Bilinear 融合：[drug_global + fp] ⊗ [prot_global + esm] → fused
        self.bilinear = nn.Bilinear(drug_in_dim, prot_in_dim, emb_dim)

        self.predictor = MLPDecoder(emb_dim, 256, 1, dropout=dropout)

    def _masked_softmax(self, scores, mask, dim):
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, -1e9)
        return F.softmax(scores, dim=dim)

    def forward(self, h_node_d, batch_d, h_node_p, batch_p, fingerprint, esm_feat):
        # 1. 转换为 dense batch 格式
        drug_rep, drug_mask = to_dense_batch(h_node_d, batch_d)
        prot_rep, prot_mask = to_dense_batch(h_node_p, batch_p)

        # 2. 线性投影
        drug_h = F.relu(self.interaction_proj(drug_rep))
        prot_h = F.relu(self.interaction_proj(prot_rep))

        # 3. Drug → Protein cross-attention: drug 节点关注 protein 残基
        scores_d2p = torch.bmm(drug_h, prot_h.transpose(1, 2)) / (self.emb_dim ** 0.5)
        attn_d2p = self._masked_softmax(scores_d2p, prot_mask, dim=2)
        drug_context = torch.bmm(attn_d2p, prot_rep)

        # 4. Protein → Drug cross-attention: protein 残基关注 drug 节点
        scores_p2d = torch.bmm(prot_h, drug_h.transpose(1, 2)) / (self.emb_dim ** 0.5)
        attn_p2d = self._masked_softmax(scores_p2d, drug_mask, dim=2)
        prot_context = torch.bmm(attn_p2d, drug_rep)

        # 5. 全局读出示（masked mean pooling）
        drug_mask_f = drug_mask.float().unsqueeze(-1)
        prot_mask_f = prot_mask.float().unsqueeze(-1)
        drug_global = (drug_context * drug_mask_f).sum(dim=1) / drug_mask_f.sum(dim=1).clamp(min=1e-9)
        prot_global = (prot_context * prot_mask_f).sum(dim=1) / prot_mask_f.sum(dim=1).clamp(min=1e-9)

        # 6. 拼接指纹/ESM 先验到全局表示中，使得先验参与 Bilinear 交互
        drug_reps = [drug_global]
        if self.use_fingerprint:
            drug_reps.append(self.fp_projection(fingerprint))
        drug_enhanced = torch.cat(drug_reps, dim=-1)

        prot_reps = [prot_global]
        if self.use_p_global:
            prot_reps.append(self.esm_projection(esm_feat))
        prot_enhanced = torch.cat(prot_reps, dim=-1)

        # 7. Bilinear 融合：drug_enhanced ⊗ prot_enhanced
        fused = self.bilinear(drug_enhanced, prot_enhanced)

        # 8. 直接预测（指纹和 ESM 已在 Bilinear 中参与交互，无需后期拼接）
        logits = self.predictor(fused)

        output_dict = {
            'attn_d2p': attn_d2p,
            'attn_p2d': attn_p2d
        }
        return logits, output_dict

class HierDTA(nn.Module):
    def __init__(self,
                 num_features_xd=266,
                 drug_hidden=64,
                 drug_out=128,
                 n_layers_drug=2,
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
                 use_p_global=True):
        super().__init__()

        assert drug_out == emb_dim, f"Drug out_channels ({drug_out}) must equal emb_dim ({emb_dim})"
        assert protein_out == emb_dim, f"Protein output_dim ({protein_out}) must equal emb_dim ({emb_dim})"

        self.drug_net = DrugModel(
            in_channels=num_features_xd,
            hidden_channels=drug_hidden,
            out_channels=drug_out,
            num_layers=n_layers_drug,
            dropout=dropout
        )
        print(self.drug_net)

        self.protein_net = ProteinEGNN(
            num_features_xt=num_features_xt,
            hidden_nf=protein_hidden,
            output_dim=protein_out,
            n_layers=n_layers_protein,
            use_surface=use_surface
        )

        self.interaction = CoAttentionModule(
            emb_dim=emb_dim,
            fp_dim=fp_dim,
            esm_dim=esm_dim,
            dropout=dropout,
            use_fingerprint=use_fingerprint,
            use_p_global=use_p_global
        )
        

    def forward(self, data):
        h_node_d, batch_d, fingerprint = self.drug_net(data.drug_graph)
        h_node_p, batch_p, esm_feat = self.protein_net(data.protein_graph)

        affinity, output_dict = self.interaction(
            h_node_d, batch_d, h_node_p, batch_p, fingerprint, esm_feat
        )

        if self.training:
            return affinity
        else:
            return affinity, output_dict
