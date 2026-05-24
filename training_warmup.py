import numpy as np
import pandas as pd
import sys, os
import random
from random import shuffle
import torch
import torch.nn as nn
from torch.optim import RAdam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from models.model import HierDTA
from utils import TestbedDatasetHMol, rmse_gpu, mse_gpu, ci_gpu, pearson_gpu, get_rm2_gpu
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import argparse

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='Train DTA models with different seeds')
parser.add_argument('--dataset_idx', type=int, default=0, help='Dataset index: 0 for davis, 1 for kiba')
parser.add_argument('--model_idx', type=int, default=0, help='0-')
parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index to use')
parser.add_argument('--strategy', type=str, default='random', help='Data split strategy')
parser.add_argument('--seed', type=int, default=None, help='Specific seed to use')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate to use')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size to use')
parser.add_argument('--max_norm', type=float, default=1.0, help='max norm in clip to use') # random split:max_norm=5.0 clod split:max_norm=1.0

parser.add_argument('--epoch', type=int, default=500, help='Epoches to use')  # 默认500
parser.add_argument('--patience', type=int, default=50, help='Patience to use')
parser.add_argument('--surface_k', type=int, default=5, help='k for surface-to-residue mapping (must match preprocessing)')

# Drug architecture
parser.add_argument('--drug_hidden', type=int, default=64, help='Drug GAT hidden channels')
parser.add_argument('--drug_out', type=int, default=128, help='Drug GAT output channels (must equal emb_dim)')
parser.add_argument('--n_layers_drug', type=int, default=2, help='Drug GAT num layers')
parser.add_argument('--heads', type=int, default=2, help='Drug GAT attention heads')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate for drug GAT and interaction')

# Protein architecture
parser.add_argument('--protein_hidden', type=int, default=128, help='Protein EGNN hidden dim')
parser.add_argument('--protein_out', type=int, default=128, help='Protein EGNN output dim (must equal emb_dim)')
parser.add_argument('--n_layers_protein', type=int, default=4, help='Protein EGNN num layers')
parser.add_argument('--use_surface', type=int, default=1, help='Use surface features (1=True, 0=False)')

# Interaction & ablation
parser.add_argument('--emb_dim', type=int, default=128, help='Interaction embedding dim')
parser.add_argument('--use_fingerprint', type=int, default=1, help='Use fingerprint branch (1=True, 0=False)')
parser.add_argument('--use_p_global', type=int, default=1, help='Use global ESM protein branch (1=True, 0=False)')
parser.add_argument('--interaction_mode', type=str, default='d2p', help='Interaction mode: d2p (Drug→Protein) or p2d (Protein→Drug)')
parser.add_argument('--agg', type=str, default='mean', choices=['mean', 'sum', 'max'],
                    help='Aggregation for atom-to-motif message passing (mean/sum/max)')

# Optimizer
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for AdamW')

args = parser.parse_args()

def train(model, device, train_loader, optimizer, epoch, max_norm):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        # 获取模型输出和中间表示
        output = model(data)
        labels = data.y.view(-1, 1).float().to(device)
        loss = loss_fn(output, labels)
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)  # max_norms=1.0/5.0
        # print(f"[Grad Clip] Total grad norm before clip: {total_norm:.4f}")
        optimizer.step()

        if batch_idx % LOG_INTERVAL == 0:
            print(f'Train epoch: {epoch} [{batch_idx * train_loader.batch_size}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Total Loss: {loss.item():.6f}')
    return loss
            
def predicting_gpu(model, device, loader, return_attention=False):
    model.eval()
    predictions = []
    labels = []
    attention_records = [] if return_attention else None
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output, attn_dict = model(data)

            if return_attention:
                attention_records.append(attn_dict)
                
            predictions.append(output)
            labels.append(data.y.view(-1, 1))
    
    # 在GPU上连接所有批次
    predictions_tensor = torch.cat(predictions, dim=0)
    labels_tensor = torch.cat(labels, dim=0)
    if return_attention:
        return labels_tensor, predictions_tensor, attention_records
    else:
        return labels_tensor, predictions_tensor

def compute_metrics_gpu(y_tensor, f_tensor):
    # 将张量转换为一维
    y = y_tensor.flatten()
    f = f_tensor.flatten()
    
    # 计算所有指标
    rmse_val = rmse_gpu(y, f).item()
    mse_val = mse_gpu(y, f).item()
    pearson_val = pearson_gpu(y, f).item()
    ci_val = ci_gpu(y, f).item()
    rm2_val = get_rm2_gpu(y, f).item()
    
    return [rmse_val, mse_val, pearson_val, ci_val, rm2_val]

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
        
        
datasets = [['davis','kiba'][args.dataset_idx]]  
modeling = [HierDTA][args.model_idx]
model_st = modeling.__name__

cuda_name = f"cuda:{args.gpu_idx}"
print('cuda_name:', cuda_name)

batch_size = args.batch_size
LR = args.lr
strategy = args.strategy
seeds = [args.seed] if args.seed is not None else [1,2,3,4,0]
max_norm = args.max_norm

NUM_EPOCHS = args.epoch
EARLY_STOPPING_PATIENCE = args.patience

TRAIN_BATCH_SIZE = batch_size
TEST_BATCH_SIZE = batch_size
LOG_INTERVAL = 20

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)
print('Batch size: ', TRAIN_BATCH_SIZE)
print('patience: ', EARLY_STOPPING_PATIENCE)
print('max norm: ', max_norm)
print('weight_decay: ', args.weight_decay)
print('Drug  : drug_hidden={}, drug_out={}, n_layers={}, heads={}, dropout={}'.format(
    args.drug_hidden, args.drug_out, args.n_layers_drug, args.heads, args.dropout))
print('Protein: protein_hidden={}, protein_out={}, n_layers={}, use_surface={}'.format(
    args.protein_hidden, args.protein_out, args.n_layers_protein, bool(args.use_surface)))
print('Interaction: emb_dim={}, dropout={}, fp={}, p_global={}'.format(
    args.emb_dim, args.dropout, bool(args.use_fingerprint), bool(args.use_p_global)))

name = f"runseed_seed"

# Main program: iterate over different seeds and datasets
for dataset in datasets:
    for seed in seeds:
        # 添加配置信息到日志
        print(f'\nRunning on {model_st}_{dataset} with seed {seed}, strategy {strategy}')
        
        setup_seed(seed)

        # 构建路径
        dataset_prefix = dataset
        split_dir = f'split_data/seed_{seed}/{strategy}'

        # 检查 split CSV 是否存在
        if not os.path.isfile(f'{split_dir}/{dataset_prefix}_train.csv'):
            print(f'Split CSV not found for seed {seed}, strategy {strategy}. Please run split_all_dataset.py first!')
            continue

        # 从 CSV 读取数据，TestbedDatasetHMol 会自动从 cache 组装 .pt 文件（惰性生成）
        df_train = pd.read_csv(f'{split_dir}/{dataset_prefix}_train.csv')
        train_drugs, train_prots, train_Y = list(df_train['Drug']), list(df_train['target_key']), list(df_train['Y'])

        df_val = pd.read_csv(f'{split_dir}/{dataset_prefix}_val.csv')
        val_drugs, val_prots, val_Y = list(df_val['Drug']), list(df_val['target_key']), list(df_val['Y'])

        df_test = pd.read_csv(f'{split_dir}/{dataset_prefix}_test.csv')
        test_drugs, test_prots, test_Y = list(df_test['Drug']), list(df_test['target_key']), list(df_test['Y'])

        train_data = TestbedDatasetHMol(
            dataset_name=dataset, xd=train_drugs, xt=train_prots, y=train_Y,
            surface_k=args.surface_k
        )
        val_data = TestbedDatasetHMol(
            dataset_name=dataset, xd=val_drugs, xt=val_prots, y=val_Y,
            surface_k=args.surface_k
        )
        test_data = TestbedDatasetHMol(
            dataset_name=dataset, xd=test_drugs, xt=test_prots, y=test_Y,
            surface_k=args.surface_k
        )

        g = torch.Generator()
        g.manual_seed(seed)

        # make data PyTorch mini-batch processing ready
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
                                  worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id), generator=g)
        val_loader = DataLoader(val_data, batch_size=TEST_BATCH_SIZE, shuffle=False, generator=g)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, generator=g)


        # 创建结果保存目录
        os.makedirs(f'results_{dataset}/{strategy}/seed_{seed}', exist_ok=True)

        # training the model
        device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
        model = modeling(
            drug_hidden=args.drug_hidden,
            drug_out=args.drug_out,
            n_layers_drug=args.n_layers_drug,
            heads=args.heads,
            dropout=args.dropout,
            protein_hidden=args.protein_hidden,
            protein_out=args.protein_out,
            n_layers_protein=args.n_layers_protein,
            use_surface=bool(args.use_surface),
            emb_dim=args.emb_dim,
            use_fingerprint=bool(args.use_fingerprint),
            use_p_global=bool(args.use_p_global),
            interaction_mode=args.interaction_mode,
            agg=args.agg
        ).to(device)
        loss_fn = nn.MSELoss()
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=args.weight_decay)

        # Warmup (5 epochs) + Cosine Annealing
        WARMUP_EPOCHS = 5
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[WARMUP_EPOCHS]
        )

        best_mse = 1000
        best_ci = 0
        best_epoch = -1
        result_file_name = f'results_{dataset}/{strategy}/seed_{seed}/result_{model_st}_{name}.csv'
        checkpoint_file_name = f'results_{dataset}/{strategy}/seed_{seed}/ckpt_{model_st}_{name}.pt'

        for epoch in range(NUM_EPOCHS):
            avg_loss = train(model, device, train_loader, optimizer, epoch+1, max_norm)
        
            G_val, P_val = predicting_gpu(model, device, val_loader)
            val_ret = compute_metrics_gpu(G_val, P_val)
          
            # 使用验证集MSE决定最佳模型
            if val_ret[1] < best_mse:   
                G_test, P_test, test_attention_records = predicting_gpu(model, device, test_loader, return_attention=True)
                test_ret = compute_metrics_gpu(G_test, P_test)
         
                # 保存完整的检查点
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_mse': best_mse,
                    'best_ci': best_ci,
                    'val_metrics': val_ret,
                    'test_metrics': test_ret
                }
                
                # 保存完整检查点
                torch.save(checkpoint, checkpoint_file_name)
                
                # 保存最佳测试集结果
                with open(result_file_name,'w') as f:
                    f.write(','.join(map(str, test_ret)))
                
                # 保存测试集预测结果和attention记录
                test_analysis_data = {
                    'G_test': G_test,
                    'P_test': P_test, 
                    'attention_records': test_attention_records,
                    'test_metrics': test_ret,
                    'epoch': epoch + 1
                }
                attention_file_name = checkpoint_file_name.replace('.pt', '_test_analysis.pt')
                torch.save(test_analysis_data, attention_file_name)
                
                best_epoch = epoch+1
                best_mse = val_ret[1]
                best_ci = val_ret[3]
                best_rm2 = val_ret[4]
                print(f'Val MSE improved at epoch {best_epoch}; best_mse:{best_mse}, best_ci:{best_ci}, best_rm2:{best_rm2}')
                print(f'Corresponding test metrics - RMSE:{test_ret[0]}, MSE:{test_ret[1]}, Pearson:{test_ret[2]}, CI:{test_ret[3]}, RM2:{test_ret[4]}')
                print(f'Saved checkpoint to {checkpoint_file_name}')

            else:
                print(f'Val MSE: {val_ret[1]} - No improvement since epoch {best_epoch}; best_mse:{best_mse}, best_ci:{best_ci}, best_rm2:{best_rm2}')
            
            scheduler.step()
            print(f"Epoch {epoch+1}, Current LR: {optimizer.param_groups[0]['lr']}")
            # 早停策略 - 如果N个周期内没有改善，则停止训练
            if epoch + 1 - best_epoch > EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1} as no improvement for patience")
                break
            
                            