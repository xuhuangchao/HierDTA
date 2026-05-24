import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GINEConv
from torch_geometric.data import Batch
from torch_scatter import scatter
import math

# 全局变量定义
num_node_type = 2
EDGE_DIM = 14  # chemprop bond features dim


class LocalAugmentation(nn.Module):
    """
    局部增强层：使用 Multi-Head Attention 融合 fine_messages（原子聚合）和
    coarse_messages（motif 消息传递），以 motif_features 作为 Query。
    """
    def __init__(self, hid_dim, heads=4, dropout=0.2):
        super().__init__()
        assert hid_dim % heads == 0, f"hid_dim {hid_dim} must be divisible by heads {heads}"
        self.hid_dim = hid_dim
        self.heads = heads
        self.d_k = hid_dim // heads

        # Q, K, V 的线性投影
        self.linear_layers = nn.ModuleList([
            nn.Linear(hid_dim, hid_dim, bias=False),
            nn.Linear(hid_dim, hid_dim, bias=False),
            nn.Linear(hid_dim, hid_dim, bias=False)
        ])
        self.W_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, fine_messages, coarse_messages, motif_features):
        """
        fine_messages:    [N_motif, hid_dim]
        coarse_messages:  [N_motif, hid_dim]
        motif_features:   [N_motif, hid_dim]  (作为 Query)
        Returns: motif_messages [N_motif, hid_dim]
        """
        N = motif_features.size(0)

        # Project Q
        Q = self.linear_layers[0](motif_features)  # [N, hid_dim]
        # Stack K/V inputs: [N, 2, hid_dim]
        KV_input = torch.stack([fine_messages, coarse_messages], dim=1)
        K = self.linear_layers[1](KV_input)  # [N, 2, hid_dim]
        V = self.linear_layers[2](KV_input)  # [N, 2, hid_dim]

        # Multi-head reshape
        Q = Q.view(N, self.heads, self.d_k).transpose(0, 1)                 # [heads, N, d_k]
        K = K.view(N, 2, self.heads, self.d_k).permute(2, 0, 3, 1)          # [heads, N, d_k, 2]
        V = V.view(N, 2, self.heads, self.d_k).permute(2, 0, 3, 1)          # [heads, N, d_k, 2]

        # Scaled dot-product attention
        scores = torch.matmul(Q.unsqueeze(2), K).squeeze(2) / math.sqrt(self.d_k)  # [heads, N, 2]
        attn = F.softmax(scores, dim=-1)                                             # [heads, N, 2]
        attn = self.dropout(attn)

        # Weighted sum over two message sources
        out = torch.matmul(V, attn.unsqueeze(-1)).squeeze(-1)  # [heads, N, d_k]

        # Concat heads
        out = out.transpose(0, 1).contiguous().view(N, self.hid_dim)
        motif_messages = self.W_o(out)
        return motif_messages


class DrugMotifGAT(nn.Module):
    def __init__(self, in_channels=133, hidden_channels=64, out_channels=128, num_layers=2, heads=2, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # --- 1. 初始特征嵌入层 ---
        self.atom_feat_embedding = nn.Linear(in_channels, hidden_channels)
        self.motif_feat_embedding = nn.Linear(in_channels, hidden_channels)
        self.node_type_embedding = nn.Embedding(num_node_type, hidden_channels)

        # --- 2. 统一的GNN网络 (GATv2) ---
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skips = nn.ModuleList()

        # 输入层 (输入维度现在是统一后的 hidden_channels)
        self.convs.append(GATv2Conv(hidden_channels, hidden_channels, heads=heads, edge_dim=EDGE_DIM))
        self.norms.append(nn.BatchNorm1d(hidden_channels * heads))
        self.skips.append(nn.Linear(hidden_channels, hidden_channels * heads))

        # 中间隐藏层
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads, edge_dim=EDGE_DIM))
            self.norms.append(nn.BatchNorm1d(hidden_channels * heads))
            self.skips.append(nn.Linear(hidden_channels * heads, hidden_channels * heads))

        # GAT输出层
        self.convs.append(GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False, edge_dim=EDGE_DIM))
        self.norms.append(nn.BatchNorm1d(out_channels))
        self.skips.append(nn.Linear(hidden_channels * heads, out_channels))

    def forward(self, drug_graph):
        if isinstance(drug_graph, list):
            drug_graph = Batch.from_data_list(drug_graph)

        x, edge_index, edge_attr, batch = drug_graph.x, drug_graph.edge_index, drug_graph.edge_attr, drug_graph.batch
        fingerprint = drug_graph.fingerprint
        if fingerprint.dim() > 2:
            fingerprint = fingerprint.squeeze(1)

        # --- 1. 准备统一的初始节点特征 ---
        node_types = x[:, 133].long()
        atom_mask = (node_types == 0)
        motif_mask = (node_types == 1)

        # 创建一个空的特征矩阵
        x_embedded = torch.zeros(x.size(0), self.atom_feat_embedding.out_features, device=x.device)

        # 分别填充原子和Motif的嵌入特征
        x_embedded[atom_mask] = self.atom_feat_embedding(x[atom_mask, :133].float())
        x_embedded[motif_mask] = self.motif_feat_embedding(x[motif_mask, :133].float())
        type_embeddings = self.node_type_embedding(node_types)
        x_embedded = x_embedded + type_embeddings

        # --- 2. 边特征直接使用 chemprop 14维特征 (无需 embedding) ---
        edge_attr_float = edge_attr.float()

        # --- 3. 在完整图上执行统一的消息传递 ---
        x = x_embedded
        for i in range(self.num_layers):
            x_res = self.skips[i](x)
            # GATv2卷积作用于所有节点和所有边
            x = self.convs[i](x, edge_index, edge_attr=edge_attr_float)
            x = self.norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = x + x_res

        # --- 4. 分离最终的节点表示用于输出 ---
        final_atom_feat = x[atom_mask]
        final_motif_feat = x[motif_mask]

        return final_atom_feat, final_motif_feat, batch, node_types, fingerprint


class DrugModel(nn.Module):
    """
    分层 MPNN 药物编码器（PyG 版 DGL-style Local Augmentation）。
    每层流程:
      1) Parallel MP: atom_conv (bond)  &  motif_conv (motif-motif)
      2) Atom -> Motif aggregation (scatter)  ->  fine_messages
      3) LocalAugmentation: 以 motif 自身为 Query，融合 fine + coarse messages
      4) GRUCell Update: 分别更新 atom 和 motif 的隐状态
    Motif 不回传至 Atom，保持 atom 表示的化学纯度。
    """
    def __init__(self, in_channels=133, hidden_channels=64, out_channels=128,
                 num_layers=2, heads=2, dropout=0.2, agg='mean'):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden_channels = hidden_channels
        assert agg in ('mean', 'sum', 'max'), f"agg must be mean/sum/max, got {agg}"
        self.agg = agg

        # ---- 1. 节点初始嵌入 ----
        self.atom_feat_embedding = nn.Linear(in_channels, hidden_channels)
        self.motif_feat_embedding = nn.Linear(in_channels, hidden_channels)
        self.node_type_embedding = nn.Embedding(num_node_type, hidden_channels)

        # ---- 2. 分层顺序卷积 ----
        self.atom_convs = nn.ModuleList()
        self.atom_norms = nn.ModuleList()
        self.motif_convs = nn.ModuleList()
        self.motif_norms = nn.ModuleList()
        self.agg_norms = nn.ModuleList()

        for _ in range(num_layers):
            # GINEConv for atom-level: MLP + edge_feature
            atom_mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.atom_convs.append(GINEConv(atom_mlp, edge_dim=EDGE_DIM))
            self.atom_norms.append(nn.LayerNorm(hidden_channels))

            # GINEConv for motif-level: MLP + edge_feature
            motif_mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.motif_convs.append(GINEConv(motif_mlp, edge_dim=EDGE_DIM))
            self.motif_norms.append(nn.LayerNorm(hidden_channels))

            self.agg_norms.append(nn.LayerNorm(hidden_channels))

        # ---- 3. Local Augmentation (共享) ----
        self.local_aug = LocalAugmentation(hidden_channels, heads=heads, dropout=dropout)

        # ---- 4. GRU Update (共享) ----
        self.atom_update = nn.GRUCell(hidden_channels, hidden_channels)
        self.motif_update = nn.GRUCell(hidden_channels, hidden_channels)

        # ---- 5. 输出投影 ----
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_channels, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, drug_graph):
        if isinstance(drug_graph, list):
            drug_graph = Batch.from_data_list(drug_graph)

        x, edge_index, edge_attr, batch = (
            drug_graph.x, drug_graph.edge_index, drug_graph.edge_attr, drug_graph.batch
        )
        fingerprint = drug_graph.fingerprint
        if fingerprint.dim() > 2:
            fingerprint = fingerprint.squeeze(1)

        # ---- 初始嵌入 ----
        node_types = x[:, 133].long()
        atom_mask = (node_types == 0)
        motif_mask = (node_types == 1)

        x_embedded = torch.zeros(x.size(0), self.hidden_channels, device=x.device)
        x_embedded[atom_mask] = self.atom_feat_embedding(x[atom_mask, :133].float())
        x_embedded[motif_mask] = self.motif_feat_embedding(x[motif_mask, :133].float())
        x_embedded = x_embedded + self.node_type_embedding(node_types)

        # ---- 按 edge_attr 第一维分离边 ----
        edge_attr_float = edge_attr.float()
        edge_type = edge_attr_float[:, 0].long()
        bond_mask = (edge_type == 0) | (edge_type == 1)
        fwd_mask = (edge_type == 2)      # atom -> motif
        mm_mask = (edge_type == 4)       # motif <-> motif

        edge_index_bond = edge_index[:, bond_mask]
        edge_attr_bond = edge_attr_float[bond_mask]
        edge_index_fwd = edge_index[:, fwd_mask]
        edge_index_mm = edge_index[:, mm_mask]
        edge_attr_mm = edge_attr_float[mm_mask]

        # ---- GNN 层: 顺序分层传递 ----
        x = x_embedded
        for i in range(self.num_layers):
            # Phase 1: Parallel MP
            # Atom refinement via chemical bonds
            h_atom = self.atom_convs[i](x, edge_index_bond, edge_attr=edge_attr_bond)
            h_atom = self.atom_norms[i](h_atom)
            h_atom = F.relu(h_atom)
            h_atom = F.dropout(h_atom, p=self.dropout, training=self.training)

            # Motif refinement via motif-motif edges
            h_motif_mp = self.motif_convs[i](x, edge_index_mm, edge_attr=edge_attr_mm)
            h_motif_mp = self.motif_norms[i](h_motif_mp)
            h_motif_mp = F.relu(h_motif_mp)
            h_motif_mp = F.dropout(h_motif_mp, p=self.dropout, training=self.training)

            # Phase 2: Atom -> Motif aggregation (fine messages)
            fine_messages = torch.zeros(x.size(0), self.hidden_channels, device=x.device)
            if edge_index_fwd.size(1) > 0:
                atom_src = edge_index_fwd[0]
                motif_dst = edge_index_fwd[1]

                agg_result = scatter(h_atom[atom_src], motif_dst, dim=0,
                                     dim_size=x.size(0), reduce=self.agg)

                fine_messages[motif_mask] = self.agg_norms[i](agg_result[motif_mask])
                fine_messages[motif_mask] = F.relu(fine_messages[motif_mask])
                fine_messages[motif_mask] = F.dropout(fine_messages[motif_mask],
                                                      p=self.dropout, training=self.training)

            # Phase 3: Local Augmentation
            coarse_messages = h_motif_mp[motif_mask]
            motif_messages = self.local_aug(
                fine_messages=fine_messages[motif_mask],
                coarse_messages=coarse_messages,
                motif_features=x[motif_mask]
            )

            # Phase 4: GRU Update (保留旧知识通过门控)
            new_atom = self.atom_update(h_atom[atom_mask], x[atom_mask])
            new_motif = self.motif_update(motif_messages, x[motif_mask])

            x = torch.zeros_like(x)
            x[atom_mask] = new_atom
            x[motif_mask] = new_motif

        # ---- 输出投影到 out_channels ----
        x = self.out_proj(x)

        # ---- 输出分离（接口与 DrugMotifGAT 完全一致）----
        final_atom_feat = x[atom_mask]
        final_motif_feat = x[motif_mask]

        return final_atom_feat, final_motif_feat, batch, node_types, fingerprint
