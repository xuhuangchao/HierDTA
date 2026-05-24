import os
import numpy as np
from math import sqrt
from scipy import stats
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric import data as DATA
import torch

class TestbedDatasetHMol(Dataset):
    def __init__(self, xd=None, xt=None, y=None, transform=None,
                 dataset_name=None, cache_dir='data/cache', surface_k=5):
        self.xd = xd
        self.xt = xt
        self.y = y
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.surface_k = surface_k
        self.transform = transform

        assert (self.xd is not None and self.xt is not None and self.y is not None), "The three lists must be the same length!"
        assert self.dataset_name is not None, "Must provide dataset_name"

        # 从全局 cache 加载（与 seed/strategy 无关）
        print(f'Loading global caches from {self.cache_dir} for dataset {self.dataset_name} (surface_k={self.surface_k})...')
        self.smile_graph = torch.load(f'{self.cache_dir}/{self.dataset_name}_smile_graph.pt', weights_only=False)
        self.fingerprint = torch.load(f'{self.cache_dir}/{self.dataset_name}_fingerprint.pt', weights_only=False)
        self.pocket_graph = torch.load(f'{self.cache_dir}/{self.dataset_name}_protein_graphs_k{self.surface_k}.pt', weights_only=False)
        self.esm_feats = torch.load(f'{self.cache_dir}/{self.dataset_name}_esm_feats.pt', weights_only=False)
        print('Cache loaded. Assembling data pairs...')

        self.data_list = []
        data_len = len(self.xd)

        for i in range(data_len):
            if (i + 1) % 1000 == 0 or i == 0:
                print('Converting data pair to graph: {}/{}'.format(i + 1, data_len))
            smiles = self.xd[i]
            key = self.xt[i]
            labels = self.y[i]

            # Process drug data
            smile_data = self.smile_graph[smiles]

            drug_graph = DATA.Data(
                x=torch.FloatTensor(smile_data["x"]),
                edge_index=torch.LongTensor(smile_data["edge_index"]),
                edge_attr=torch.LongTensor(smile_data["edge_attr"]),
            )

            drug_graph.c_size = torch.LongTensor([smile_data["num_part"]])
            drug_graph.smiles = smiles
            drug_graph.fingerprint = torch.FloatTensor([self.fingerprint[smiles]])

            # Process protein data
            pocket_data = self.pocket_graph[key]

            node_features = torch.Tensor(pocket_data[1])
            edge_index_protein = torch.LongTensor(pocket_data[2])
            residue_coords = torch.FloatTensor(pocket_data[3])

            protein_graph = DATA.Data(
                x=node_features,
                edge_index=edge_index_protein,
                pos=residue_coords
            )

            protein_graph.key = key
            target_features = self.esm_feats[key]
            protein_graph.target_features = torch.FloatTensor([target_features])

            data = DATA.Data(
                drug_graph=drug_graph,
                protein_graph=protein_graph,
                y=torch.FloatTensor([labels]),
            )
            self.data_list.append(data)

        print(f'Graph construction done. Total samples: {len(self.data_list)}')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data
        
        
        
# 测试指标
def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs

def ci(gt, pred):
    gt_mask = gt.reshape((1, -1)) > gt.reshape((-1, 1))
    diff = pred.reshape((1, -1)) - pred.reshape((-1, 1))
    h_one = (diff > 0)
    h_half = (diff == 0)
    CI = np.sum(gt_mask * h_one * 1.0 + gt_mask * h_half * 0.5) / np.sum(gt_mask)
    return CI

def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / (float(y_obs_sq * y_pred_sq) + 0.00000001)

def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs * y_pred) / (float(sum(y_pred * y_pred)) + 0.00000001)


def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / (float(down) + 0.00000001))

def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)
    return r2 * (1 - np.sqrt(np.absolute((r2 * r2) - (r02 * r02))))


def rmse_gpu(y, f):
    """Root Mean Squared Error - GPU version"""
    return torch.sqrt(((y - f) ** 2).mean(dim=0))

def mse_gpu(y, f):
    """Mean Squared Error - GPU version"""
    return ((y - f) ** 2).mean(dim=0)

def pearson_gpu(y, f):
    """Pearson correlation coefficient - GPU version"""
    y_mean = torch.mean(y)
    f_mean = torch.mean(f)
    
    y_centered = y - y_mean
    f_centered = f - f_mean
    
    numerator = torch.sum(y_centered * f_centered)
    denominator = torch.sqrt(torch.sum(y_centered ** 2) * torch.sum(f_centered ** 2))
    
    return numerator / (denominator + 1e-8)

# def spearman_gpu(y, f):
#     """Spearman correlation coefficient - GPU version"""
#     # 需要将数据移回CPU以使用scipy的实现
#     y_cpu = y.cpu().numpy()
#     f_cpu = f.cpu().numpy()
    
#     return stats.spearmanr(y_cpu, f_cpu)[0]

def ci_gpu(gt, pred):
    """Concordance index - GPU version"""
    gt_reshaped1 = gt.view(1, -1)
    gt_reshaped2 = gt.view(-1, 1)
    
    pred_reshaped1 = pred.view(1, -1)
    pred_reshaped2 = pred.view(-1, 1)
    
    gt_mask = gt_reshaped1 > gt_reshaped2
    diff = pred_reshaped1 - pred_reshaped2
    
    h_one = (diff > 0).float()
    h_half = (diff == 0).float() * 0.5
    
    return torch.sum(gt_mask.float() * h_one + gt_mask.float() * h_half) / torch.sum(gt_mask.float())

def r_squared_error_gpu(y_obs, y_pred):
    """R-squared error - GPU version"""
    y_obs_mean = torch.mean(y_obs)
    y_pred_mean = torch.mean(y_pred)
    
    covariance = torch.sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = covariance * covariance
    
    y_obs_sq = torch.sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    y_pred_sq = torch.sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))
    
    return mult / (y_obs_sq * y_pred_sq + 1e-8)

def get_k_gpu(y_obs, y_pred):
    """Get scaling factor k - GPU version"""
    return torch.sum(y_obs * y_pred) / (torch.sum(y_pred * y_pred) + 1e-8)

def squared_error_zero_gpu(y_obs, y_pred):
    """Squared error at origin - GPU version"""
    k = get_k_gpu(y_obs, y_pred)
    
    y_obs_mean = torch.mean(y_obs)
    
    upp = torch.sum((y_obs - (k * y_pred)) * (y_obs - (k * y_pred)))
    down = torch.sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))
    
    return 1 - (upp / (down + 1e-8))

def get_rm2_gpu(ys_orig, ys_line):
    """Get RM2 metric - GPU version"""
    r2 = r_squared_error_gpu(ys_orig, ys_line)
    r02 = squared_error_zero_gpu(ys_orig, ys_line)
    
    diff = (r2 * r2) - (r02 * r02)
    abs_diff = torch.abs(diff)
    return r2 * (1 - torch.sqrt(abs_diff))