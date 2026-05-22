import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Batch

# 全局变量定义
num_node_type = 2
EDGE_DIM = 14  # chemprop bond features dim

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