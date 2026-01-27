import torch
import torch.nn as nn
from torch_geometric.data import Batch
from .egnn_clean import EGNN

class ProteinEGNN(nn.Module):
    """基于 EGNN 的蛋白质口袋图网络"""
    def __init__(self, num_features_xt=608, hidden_nf=128, output_dim=128, n_layers=4, attention=False, normalize=True, tanh=True):  
        super(ProteinEGNN, self).__init__()
        
        self.egnn = EGNN(
            in_node_nf=num_features_xt,
            hidden_nf=hidden_nf,
            out_node_nf=output_dim,
            in_edge_nf=0,
            n_layers=n_layers,
            attention=attention,
            normalize=normalize,
            tanh=tanh,
            residual=True
        )
        print("num_features_xt:", num_features_xt)

    def forward(self, protein_graph):
        if isinstance(protein_graph, list):
            protein_graph = Batch.from_data_list(protein_graph)
            
        h, edge_index, x, batch = (
            protein_graph.x,
            protein_graph.edge_index,
            protein_graph.pos,
            protein_graph.batch
        )
        h = h[:, 41:]  # esm_feat和surface特征
        # h = h[:, 41:521]  # esm_feat
        h_out, x_out = self.egnn(h=h, x=x, edges=edge_index, edge_attr=None)
        
        # ESM特征处理
        target_features = protein_graph.target_features
        if target_features.dim() > 2:
            target_features = target_features.squeeze(1)
        
        return h_out, batch, target_features
    

import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

class ProteinGAT(nn.Module):
    """
    蛋白质残基的图注意力网络 (GAT)。
    该网络使用标准的 GATv2Conv 处理残基节点特征。
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, heads=4, dropout=0.2):
        """
        初始化函数

        参数:
        - in_channels (int): 输入节点特征的维度 (608)
        - hidden_channels (int): GAT层中间隐藏层的维度
        - out_channels (int): GAT最终输出的节点特征维度
        - num_layers (int): GAT层的数量
        - heads (int): 多头注意力的头数
        - dropout (float): Dropout的比率
        """
        super(ProteinGAT, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skips = nn.ModuleList()

        # 输入层
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=heads))
        self.norms.append(nn.BatchNorm1d(hidden_channels * heads))
        self.skips.append(nn.Linear(in_channels, hidden_channels * heads))

        # 中间隐藏层
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels * heads, hidden_channels, heads=heads))
            self.norms.append(nn.BatchNorm1d(hidden_channels * heads))
            self.skips.append(nn.Linear(hidden_channels * heads, hidden_channels * heads))

        # 输出层 (最后一层不使用多头拼接)
        self.convs.append(GATv2Conv(hidden_channels * heads, out_channels, heads=1, concat=False))
        self.norms.append(nn.BatchNorm1d(out_channels))
        self.skips.append(nn.Linear(hidden_channels * heads, out_channels))

    def forward(self, protein_graph):
        """
        前向传播

        参数:
        - x (Tensor): 节点特征张量, 形状 [num_nodes, in_channels]
        - edge_index (Tensor): 边的索引, 形状 [2, num_edges]
        - batch (Tensor): 指示每个节点属于哪个图的批次向量, 形状 [num_nodes]

        返回:
        - final_node_feat (Tensor): 更新后的节点特征
        - graph_feat (Tensor): 全局图级别的特征表示
        """
        if isinstance(protein_graph, list):
            protein_graph = Batch.from_data_list(protein_graph)
            
        x, edge_index, batch = (
            protein_graph.x,
            protein_graph.edge_index,
            protein_graph.batch
        )
        x = x[:, 41:]
        # ESM特征处理
        target_features = protein_graph.target_features
        if target_features.dim() > 2:
            target_features = target_features.squeeze(1)
            
        for i in range(self.num_layers):
            # 保存输入用于残差连接
            x_res = self.skips[i](x)
            
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            # 残差连接
            x = x + x_res
            
        final_node_feat = x
        
        return final_node_feat, batch, target_features