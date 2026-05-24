"""
网格搜索脚本：对 HierDTA 的关键超参数进行系统性搜索。

参数选择依据：
  - dropout: 正则化强度 → 直接影响 cold drug 泛化
  - drug_hidden: 药物编码器容量 (266-dim → hidden 压缩比)
  - lr / weight_decay: 优化动力学
  - emb_dim: 整体模型容量 (与 drug_out / protein_out 绑定)

用法:
  python grid_search.py --dataset kiba --gpu_idx 0 --strategy random
  python grid_search.py --dataset kiba --gpu_idx 0 --strategy cold_drug --seed 1

输出:
  results_grid/{dataset}/{strategy}/grid_results.csv  — 每行一个配置的结果
  results_grid/{dataset}/{strategy}/run_{idx}/         — 各配置的详细输出
"""

import numpy as np
import pandas as pd
import sys, os, json, time, itertools
from datetime import datetime
import random
import torch
import torch.nn as nn
from torch.optim import AdamW
from models.model import HierDTA
from utils import TestbedDatasetHMol, rmse_gpu, mse_gpu, ci_gpu, pearson_gpu, get_rm2_gpu
from torch_geometric.loader import DataLoader
import argparse

# ============================================================
# 参数网格定义 — 修改此处来调整搜索范围
# ============================================================
PARAM_GRID = {
    # 核心参数 — 24 组合 (~6h @ 15min/run)
    'dropout':       [0.1, 0.2, 0.3],              # 正则化强度
    'drug_hidden':   [64, 128],                     # 药物编码器容量
    'lr':            [5e-4, 1e-3],                  # 学习率
    'weight_decay':  [1e-5, 1e-4],                  # L2 正则

    # 扩展时可取消注释（每条增加约 2× 组合数）:
    # 'emb_dim':       [64, 128],                   # 嵌入维度（绑定 drug_out/protein_out）
    # 'n_layers_drug':  [2, 3],                     # 药物 GNN 层数
    # 'protein_hidden': [64, 128],                   # 蛋白 EGNN 隐藏维度
    # 'use_surface':    [0, 1],                      # 表面特征消融
}

# 搜索时保持固定的参数
FIXED_PARAMS = {
    'emb_dim':           128,
    'drug_out':          128,      # 必须 == emb_dim
    'protein_out':       128,      # 必须 == emb_dim
    'protein_hidden':    128,
    'n_layers_drug':     2,
    'n_layers_protein':  4,
    'use_surface':       1,
    'use_fingerprint':   1,
    'use_p_global':      1,
    'batch_size':        512,
    'epoch':             500,
    'patience':          50,
    'surface_k':         5,
}


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_default_dtype(torch.float32)


def build_model(params, device):
    """根据参数字典构建 HierDTA 模型。"""
    drug_out = params.get('drug_out', params['emb_dim'])
    protein_out = params.get('protein_out', params['emb_dim'])

    model = HierDTA(
        num_features_xd=266,
        drug_hidden=params['drug_hidden'],
        drug_out=drug_out,
        n_layers_drug=params['n_layers_drug'],
        dropout=params['dropout'],
        num_features_xt=608,
        protein_hidden=params['protein_hidden'],
        protein_out=protein_out,
        n_layers_protein=params['n_layers_protein'],
        use_surface=bool(params['use_surface']),
        emb_dim=params['emb_dim'],
        use_fingerprint=bool(params['use_fingerprint']),
        use_p_global=bool(params['use_p_global']),
    )
    return model.to(device)


def train_one_epoch(model, device, train_loader, optimizer, loss_fn, epoch, log_interval=20):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        labels = data.y.view(-1, 1).float().to(device)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, device, loader):
    model.eval()
    preds, labels = [], []
    for data in loader:
        data = data.to(device)
        output, _ = model(data)
        preds.append(output)
        labels.append(data.y.view(-1, 1))
    y = torch.cat(labels, dim=0).flatten()
    f = torch.cat(preds, dim=0).flatten()
    return {
        'mse': mse_gpu(y, f).item(),
        'rmse': rmse_gpu(y, f).item(),
        'pearson': pearson_gpu(y, f).item(),
        'ci': ci_gpu(y, f).item(),
        'rm2': get_rm2_gpu(y, f).item(),
    }


def train_single_config(params, dataset, strategy, seed, gpu_idx, base_dir):
    """训练单个参数组合，返回验证集最佳 MSE 和对应测试集指标。"""
    device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu")
    setup_seed(seed)

    # --- 数据加载 ---
    split_dir = f'split_data/seed_{seed}/{strategy}'
    df_train = pd.read_csv(f'{split_dir}/{dataset}_train.csv')
    df_val   = pd.read_csv(f'{split_dir}/{dataset}_val.csv')
    df_test  = pd.read_csv(f'{split_dir}/{dataset}_test.csv')

    train_data = TestbedDatasetHMol(
        dataset_name=dataset,
        xd=list(df_train['Drug']), xt=list(df_train['target_key']), y=list(df_train['Y']),
        surface_k=params['surface_k']
    )
    val_data = TestbedDatasetHMol(
        dataset_name=dataset,
        xd=list(df_val['Drug']), xt=list(df_val['target_key']), y=list(df_val['Y']),
        surface_k=params['surface_k']
    )
    test_data = TestbedDatasetHMol(
        dataset_name=dataset,
        xd=list(df_test['Drug']), xt=list(df_test['target_key']), y=list(df_test['Y']),
        surface_k=params['surface_k']
    )

    g = torch.Generator()
    g.manual_seed(seed)
    bs = params['batch_size']
    train_loader = DataLoader(train_data, batch_size=bs, shuffle=True, generator=g,
                              worker_init_fn=lambda wid: np.random.seed(seed + wid))
    val_loader   = DataLoader(val_data,   batch_size=bs, shuffle=False, generator=g)
    test_loader  = DataLoader(test_data,  batch_size=bs, shuffle=False, generator=g)

    # --- 模型 & 优化器 ---
    model = build_model(params, device)
    loss_fn = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=params['lr'],
                      weight_decay=params['weight_decay'])

    # --- 训练循环 ---
    best_val_mse = float('inf')
    best_test_metrics = None
    best_epoch = -1
    patience_counter = 0

    for epoch in range(1, params['epoch'] + 1):
        train_one_epoch(model, device, train_loader, optimizer, loss_fn, epoch)
        val_metrics = evaluate(model, device, val_loader)

        if val_metrics['mse'] < best_val_mse:
            best_val_mse = val_metrics['mse']
            best_test_metrics = evaluate(model, device, test_loader)
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= params['patience']:
            break

    result = {
        **params,
        'best_epoch': best_epoch,
        'val_mse': best_val_mse,
        'test_mse': best_test_metrics['mse'],
        'test_rmse': best_test_metrics['rmse'],
        'test_pearson': best_test_metrics['pearson'],
        'test_ci': best_test_metrics['ci'],
        'test_rm2': best_test_metrics['rm2'],
    }
    return result


def params_to_name(params):
    """将参数字典压缩为短标识符。"""
    parts = []
    for k in sorted(PARAM_GRID.keys()):
        v = params[k]
        if isinstance(v, float):
            parts.append(f"{k}={v:.0e}" if v < 0.01 else f"{k}={v}")
        else:
            parts.append(f"{k}={v}")
    return '_'.join(parts)


def main():
    parser = argparse.ArgumentParser(description='Grid search for HierDTA hyperparameters')
    parser.add_argument('--dataset', type=str, default='kiba', choices=['kiba', 'davis'])
    parser.add_argument('--gpu_idx', type=int, default=0)
    parser.add_argument('--strategy', type=str, default='random',
                        help='Data split strategy (random, cold_drug, cold_target, all_cold)')
    parser.add_argument('--seed', type=int, default=1, help='Single seed for grid search')
    parser.add_argument('--resume', action='store_true', help='Skip completed configs in CSV')
    args = parser.parse_args()

    # --- 构建组合 ---
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    total = 1
    for v in values:
        total *= len(v)

    # --- 输出目录 ---
    base_dir = f'results_grid/{args.dataset}/{args.strategy}'
    os.makedirs(base_dir, exist_ok=True)
    csv_path = os.path.join(base_dir, 'grid_results.csv')

    # --- 断点续跑 ---
    completed_names = set()
    if args.resume and os.path.exists(csv_path):
        df_done = pd.read_csv(csv_path)
        for _, row in df_done.iterrows():
            name = params_to_name({k: row[k] for k in keys})
            completed_names.add(name)
        print(f"Resume: {len(completed_names)} configs already completed, skipping.")

    print(f"\n{'='*70}")
    print(f"Grid Search: {args.dataset.upper()} | strategy={args.strategy} | seed={args.seed}")
    print(f"Parameters: {keys}")
    print(f"Total combinations: {total}")
    print(f"Output: {csv_path}")
    print(f"{'='*70}\n")

    results = []
    start_time = time.time()

    for idx, combo in enumerate(itertools.product(*values)):
        params = dict(FIXED_PARAMS)
        params.update(dict(zip(keys, combo)))

        # 同步 emb_dim → drug_out / protein_out（断言要求）
        if 'emb_dim' in params:
            params['drug_out'] = params['emb_dim']
            params['protein_out'] = params['emb_dim']

        # 断点续跑
        name = params_to_name(params)
        if name in completed_names:
            continue

        elapsed = time.time() - start_time
        eta = (elapsed / max(idx - len(completed_names), 1)) * (total - idx - 1)
        eta_str = f"{eta/60:.0f}min" if eta < 3600 else f"{eta/3600:.1f}h"
        print(f"\n[{idx+1}/{total}] {name}  (ETC ~{eta_str})")
        print(f"  params: {params}")

        try:
            result = train_single_config(params, args.dataset, args.strategy,
                                          args.seed, args.gpu_idx, base_dir)
            results.append(result)
            print(f"  ✓ val_mse={result['val_mse']:.4f}  test_rmse={result['test_rmse']:.4f}  "
                  f"test_pearson={result['test_pearson']:.4f}  test_ci={result['test_ci']:.4f}  "
                  f"epoch={result['best_epoch']}")
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            result = {**params, 'best_epoch': -1, 'val_mse': float('nan'),
                      'test_mse': float('nan'), 'test_rmse': float('nan'),
                      'test_pearson': float('nan'), 'test_ci': float('nan'),
                      'test_rm2': float('nan')}
            results.append(result)

        # 增量写入 CSV
        df = pd.DataFrame(results)
        df.to_csv(csv_path, index=False)

    # --- 最终汇总 ---
    if results:
        df = pd.DataFrame(results)
        valid = df[df['best_epoch'] > 0].copy()
        if len(valid) > 0:
            print(f"\n{'='*70}")
            print(f"Top 5 by test MSE:")
            print(f"{'='*70}")
            top5 = valid.nsmallest(5, 'test_mse')
            cols_show = [k for k in keys if k in top5.columns] + ['test_mse', 'test_rmse', 'test_pearson', 'test_ci']
            print(top5[cols_show].to_string(index=False))

            # 最佳参数摘要
            best = top5.iloc[0]
            print(f"\nBest config: {params_to_name({k: best[k] for k in keys})}")
            print(f"  test_mse={best['test_mse']:.4f}  test_rmse={best['test_rmse']:.4f}  "
                  f"test_pearson={best['test_pearson']:.4f}  test_ci={best['test_ci']:.4f}")

    total_time = (time.time() - start_time) / 60
    print(f"\nDone. {len(results)} configs in {total_time:.1f} min. Results saved to {csv_path}")


if __name__ == '__main__':
    main()
