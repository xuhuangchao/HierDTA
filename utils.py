import os
import numpy as np
from math import sqrt
from scipy import stats
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric import data as DATA
import torch


def load_global_features(dataset_name, cache_dir='data/cache'):
    """Load dataset-wide drug and pocket caches once for sharing across splits."""
    print(f'Loading global caches from {cache_dir} for dataset {dataset_name}...')
    drug_features = torch.load(
        f'{cache_dir}/{dataset_name}_drug_features.pt', weights_only=False
    )
    pocket_features = torch.load(
        f'{cache_dir}/{dataset_name}_pocket_graphs.pt', weights_only=False
    )
    print('Global caches loaded.')
    return drug_features, pocket_features


class TestbedDatasetHMol(Dataset):
    def __init__(self, xd=None, xt=None, y=None, transform=None,
                 dataset_name=None, cache_dir='data/cache',
                 drug_features=None, pocket_features=None):
        self.xd = xd
        self.xt = xt
        self.y = y
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir
        self.transform = transform

        assert (self.xd is not None and self.xt is not None and self.y is not None), "The three lists must be the same length!"
        assert self.dataset_name is not None, "Must provide dataset_name"

        # Reuse dataset-wide caches across train/valid/test when provided.
        if drug_features is None or pocket_features is None:
            drug_features, pocket_features = load_global_features(
                self.dataset_name, self.cache_dir
            )
        self.drug_features = drug_features
        self.pocket_features = pocket_features
        print('Assembling data pairs...')

        self.data_list = []
        data_len = len(self.xd)

        for i in range(data_len):
            if (i + 1) % 1000 == 0 or i == 0:
                print('Converting data pair to graph: {}/{}'.format(i + 1, data_len))
            smiles = self.xd[i]
            key = self.xt[i]
            labels = self.y[i]

            # Process drug data — build from heterogeneous molecular graph cache
            drug_data = self.drug_features[smiles]
            hg = drug_data['hetero_graph']

            # Full heterogeneous graph
            hetero = DATA.HeteroData()
            hetero['atom'].x = torch.FloatTensor(hg['x_atom'])
            hetero['motif'].x = torch.FloatTensor(hg['x_motif'])

            hetero['atom', 'bond', 'atom'].edge_index = torch.LongTensor(hg['aa_edge_index'])
            hetero['atom', 'bond', 'atom'].edge_attr = torch.FloatTensor(hg['aa_edge_attr'])

            hetero['atom', 'in', 'motif'].edge_index = torch.LongTensor(hg['am_edge_index'])
            # Kept for backward compatibility with existing caches. The
            # independent dual-graph encoder does not consume atom-motif edges.
            hetero['atom', 'in', 'motif'].edge_attr = torch.FloatTensor(hg['am_edge_attr'])

            hetero['motif', 'connects', 'motif'].edge_index = torch.LongTensor(hg['mm_edge_index'])
            hetero['motif', 'connects', 'motif'].edge_attr = torch.FloatTensor(hg['mm_edge_attr'])

            # Pocket residue graph from data/cache/{dataset}_pocket_graphs.pt.
            pocket_data = self.pocket_features[key]
            ei = torch.as_tensor(pocket_data['edge_index'], dtype=torch.int64)
            ea = torch.as_tensor(pocket_data['edge_weight'], dtype=torch.float32)
            pocket_graph = DATA.Data(
                x=torch.as_tensor(pocket_data['node_features'], dtype=torch.float32),
                edge_index=ei,
                edge_attr=ea,
                coords=torch.as_tensor(pocket_data['residue_coords'], dtype=torch.float32),
            )

            data = DATA.Data(
                hetero=hetero,
                pocket_graph=pocket_graph,
                fingerprint=torch.as_tensor(drug_data['fingerprint'], dtype=torch.float32).unsqueeze(0),
                esm_global=torch.as_tensor(pocket_data['esm_global'], dtype=torch.float32).unsqueeze(0),
                smiles=smiles,
                key=key,
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
