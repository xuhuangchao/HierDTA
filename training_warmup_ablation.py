import numpy as np
import pandas as pd
import sys, os
import random
from random import shuffle
import torch
import torch.nn as nn
from torch.optim import RAdam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from models.model_ablation1 import HierDTA
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
parser.add_argument('--max_norm', type=float, default=5.0, help='max norm in clip to use') 
parser.add_argument('--suffix', type=str, default='hmol_motif', help='create_data file')
parser.add_argument('--epoch', type=int, default=500, help='Epoches to use')  # 默认500
parser.add_argument('--patience', type=int, default=50, help='Patience to use')  
parser.add_argument('--use_fp', action='store_true', help='use_fingerprint')
parser.add_argument('--use_esm', action='store_true', help='use_p_global')
parser.add_argument('--use_multi', action='store_true', help='use_multi_level_reps')

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
modeling = [][args.model_idx]
model_st = modeling.__name__

cuda_name = f"cuda:{args.gpu_idx}"
print('cuda_name:', cuda_name)

batch_size = args.batch_size
LR = args.lr
strategy = args.strategy
seeds = [args.seed] if args.seed is not None else [1,2,3,4,0]
max_norm = args.max_norm
suffix = args.suffix

NUM_EPOCHS = args.epoch
EARLY_STOPPING_PATIENCE = args.patience  
T_max = args.epoch
WARMUP_EPOCHS = 5

TRAIN_BATCH_SIZE = batch_size
TEST_BATCH_SIZE = batch_size
LOG_INTERVAL = 20

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)
print('Batch size: ', TRAIN_BATCH_SIZE)
print('patience: ', EARLY_STOPPING_PATIENCE)
print('max norm: ', max_norm)
print(f"ablation: use_fp {args.use_fp}, use_esm {args.use_esm}, use_multi {args.use_multi}")

name = f"wosurface_pegnn_warmup_epoch{NUM_EPOCHS}_pat{EARLY_STOPPING_PATIENCE}_norm{max_norm}"

# Main program: iterate over different seeds and datasets
for dataset in datasets:
    for seed in seeds:
        # 添加配置信息到日志
        print(f'\nRunning on {model_st}_{dataset} with seed {seed}, strategy {strategy}')
        
        setup_seed(seed)

        # 构建路径
        processed_data_dir = f'data/processed/{strategy}/seed_{seed}'
        dataset_prefix = dataset  
        
        processed_data_file_train = f'{processed_data_dir}/processed/{dataset_prefix}_train_{suffix}.pt'
        processed_data_file_val = f'{processed_data_dir}/processed/{dataset_prefix}_val_{suffix}.pt'
        processed_data_file_test = f'{processed_data_dir}/processed/{dataset_prefix}_test_{suffix}.pt'
        
        if (not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_val)) or (not os.path.isfile(processed_data_file_test)):
            print(f'Data files not found for seed {seed}. Please run create_data.py first!')
            continue
        else:
            train_data = TestbedDatasetHMol(root=processed_data_dir, dataset=f'{dataset_prefix}_train_{suffix}')
            val_data = TestbedDatasetHMol(root=processed_data_dir, dataset=f'{dataset_prefix}_val_{suffix}') 
            test_data = TestbedDatasetHMol(root=processed_data_dir, dataset=f'{dataset_prefix}_test_{suffix}')
            
            g = torch.Generator()
            g.manual_seed(seed)
            
            # make data PyTorch mini-batch processing ready
            train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, 
                                      worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id), generator=g)
            val_loader = DataLoader(val_data, batch_size=TEST_BATCH_SIZE, shuffle=False, generator=g)
            test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, generator=g)

            # 创建结果保存目录
            os.makedirs(f'results_0924_{dataset}/{strategy}/seed_{seed}', exist_ok=True)

            # training the model
            device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
            model = modeling(use_fingerprint=args.use_fp, 
                             use_p_global=args.use_esm, 
                             use_multi_level_reps=args.use_multi).to(device)
            loss_fn = nn.MSELoss()
            optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)  
            warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=WARMUP_EPOCHS)
            main_scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
            scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[WARMUP_EPOCHS])

            all_metrics = []
            best_mse = 1000
            best_ci = 0
            best_epoch = -1
            result_file_name = f'results_0924_{dataset}/{strategy}/seed_{seed}/result_{model_st}_{name}.csv'
            checkpoint_file_name = f'results_0924_{dataset}/{strategy}/seed_{seed}/ckpt_{model_st}_{name}.pt'

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
                
                                