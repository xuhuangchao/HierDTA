import argparse
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models import DTAModel
from models.dta_model import dta_collate_fn
from utils import (
    TestbedDatasetHMol,
    ci_gpu,
    get_rm2_gpu,
    load_global_features,
    mse_gpu,
    pearson_gpu,
    rmse_gpu,
)


METRIC_NAMES = ['rmse', 'mse', 'pearson', 'ci', 'rm2']


def parse_args():
    parser = argparse.ArgumentParser(description='Train and predict with DTAModel')

    parser.add_argument('--dataset_idx', type=int, default=0, help='0 for davis, 1 for kiba')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset name, overrides dataset_idx')
    parser.add_argument('--gpu_idx', type=int, default=0, help='GPU index')
    parser.add_argument('--strategy', type=str, default='warm',
                        choices=['warm', 'unseen_drug', 'unseen_prot', 'unseen_pair'],
                        help='Data split strategy')
    parser.add_argument('--seed', type=int, default=41, choices=[41, 42, 43, 32, 33],
                        help='Random seed used to select the data split')
    parser.add_argument('--cache_dir', type=str, default='data/cache', help='Global cache directory')
    parser.add_argument('--split_root', type=str, default='data', help='Split csv root')
    parser.add_argument('--result_root', type=str, default=None, help='Result root, default results_{dataset}')

    parser.add_argument('--epochs', type=int, default=300, help='Max training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patience')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--log_interval', type=int, default=20, help='Training log interval')

    parser.add_argument('--hidden_dim', type=int, default=128, help='Shared embedding dimension')
    parser.add_argument('--num_tasks', type=int, default=1, help='Prediction output dimension')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout')

    parser.add_argument('--atom_in_dim', type=int, default=37, help='Atom feature dimension')
    parser.add_argument('--motif_in_dim', type=int, default=50, help='Motif feature dimension')
    parser.add_argument('--aa_edge_dim', type=int, default=13, help='Atom-atom edge dimension')
    parser.add_argument('--mm_edge_dim', type=int, default=37, help='Motif-motif edge dimension')
    parser.add_argument('--atom_num_layers', type=int, default=2, help='Atom AttentiveFP layers')
    parser.add_argument('--motif_num_layers', type=int, default=2, help='Motif AttentiveFP layers')

    parser.add_argument('--prot_in_dim', type=int, default=1152, help='ESMC residue feature dimension')
    parser.add_argument('--prot_num_layers', type=int, default=2, help='Protein GAT layers after the initial GCN')
    parser.add_argument('--fp_in_dim', type=int, default=1024, help='Fingerprint dimension')
    parser.add_argument('--run_name', type=str, default='hisurf_dta', help='Output filename suffix')

    return parser.parse_args()


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def read_split_csv(args, dataset, split):
    split_dir = os.path.join(args.split_root, dataset, f'seed_{args.seed}', args.strategy)
    path = os.path.join(split_dir, f'{split}.csv')
    if not os.path.isfile(path):
        print(f'Split CSV not found: {path}')
        print('Please run preprocessing/cold_split.py first.')
        sys.exit(1)
    df = pd.read_csv(path)
    return list(df['compound_iso_smiles']), list(df['target_key']), list(df['affinity'])


def build_dataset(args, dataset, split, drug_features, protein_features):
    drugs, prots, labels = read_split_csv(args, dataset, split)
    return TestbedDatasetHMol(
        xd=drugs,
        xt=prots,
        y=labels,
        dataset_name=dataset,
        cache_dir=args.cache_dir,
        drug_features=drug_features,
        protein_features=protein_features,
    )


def build_loader(data, args, shuffle=False):
    return DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=dta_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


def build_model(args):
    return DTAModel(
        hidden_dim=args.hidden_dim,
        num_tasks=args.num_tasks,
        task='regression',
        dropout=args.dropout,
        atom_in_dim=args.atom_in_dim,
        motif_in_dim=args.motif_in_dim,
        aa_edge_dim=args.aa_edge_dim,
        mm_edge_dim=args.mm_edge_dim,
        atom_num_layers=args.atom_num_layers,
        motif_num_layers=args.motif_num_layers,
        prot_in_dim=args.prot_in_dim,
        prot_num_layers=args.prot_num_layers,
        fp_in_dim=args.fp_in_dim,
    )


def compute_metrics_gpu(y_tensor, pred_tensor):
    y = y_tensor.flatten()
    pred = pred_tensor.flatten()
    return [
        rmse_gpu(y, pred).item(),
        mse_gpu(y, pred).item(),
        pearson_gpu(y, pred).item(),
        ci_gpu(y, pred).item(),
        get_rm2_gpu(y, pred).item(),
    ]


def train_one_epoch(model, device, loader, optimizer, loss_fn, epoch, log_interval):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)
        labels = batch.y.view(-1, 1).float()

        optimizer.zero_grad()
        output = model(batch)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        if batch_idx % log_interval == 0:
            seen = min((batch_idx + 1) * loader.batch_size, len(loader.dataset))
            percent = 100.0 * seen / len(loader.dataset)
            print(
                f'Train epoch: {epoch} [{seen}/{len(loader.dataset)} '
                f'({percent:.0f}%)]\tLoss: {loss.item():.6f}'
            )

    return total_loss / max(total_samples, 1)


def predict(model, device, loader):
    model.eval()
    predictions = []
    labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)

            predictions.append(output)
            labels.append(batch.y.view(-1, 1))

    return torch.cat(labels, dim=0), torch.cat(predictions, dim=0)

def main():
    args = parse_args()
    datasets = ['davis', 'kiba']
    if args.dataset is not None:
        dataset = args.dataset
    else:
        if args.dataset_idx < 0 or args.dataset_idx >= len(datasets):
            raise ValueError('dataset_idx must be 0 or 1')
        dataset = datasets[args.dataset_idx]
    result_root = args.result_root or f'results_{dataset}'
    output_dir = os.path.join(result_root, args.strategy, f'seed_{args.seed}')
    os.makedirs(output_dir, exist_ok=True)

    set_random_seed(0)
    device = torch.device(f'cuda:{args.gpu_idx}' if torch.cuda.is_available() else 'cpu')

    print(f'Dataset: {dataset}')
    print(f'Strategy: {args.strategy}, seed: {args.seed}')
    print(f'Device: {device}')
    print(f'Batch size: {args.batch_size}, lr: {args.lr}, epochs: {args.epochs}')
    print(f'Hidden dim: {args.hidden_dim}, dropout: {args.dropout}')
    print(
        f'Mol layers: atom={args.atom_num_layers}, motif={args.motif_num_layers}; '
        f'Protein GAT layers: {args.prot_num_layers}'
    )
    print('Protein encoder: ESMC contact graph GCN-GAT')
    print(f'Data split seed: {args.seed}, Run seed: 0 for reproducibility')

    model = build_model(args).to(device)
    print('Parameter counts:', model.count_parameters())

    drug_features, protein_features = load_global_features(dataset, args.cache_dir)
    train_data = build_dataset(args, dataset, 'train', drug_features, protein_features)
    val_data = build_dataset(args, dataset, 'valid', drug_features, protein_features)
    test_data = build_dataset(args, dataset, 'test', drug_features, protein_features)

    train_loader = build_loader(train_data, args, shuffle=True)
    val_loader = build_loader(val_data, args, shuffle=False)
    test_loader = build_loader(test_data, args, shuffle=False)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    result_path = os.path.join(output_dir, f'result_{args.run_name}.csv')

    best_mse = float('inf')
    best_epoch = 0
    best_metrics = None
    best_checkpoint_path = None

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(
            model, device, train_loader, optimizer, loss_fn, epoch,
            args.log_interval
        )
        scheduler.step()
        y_val, pred_val = predict(model, device, val_loader)
        val_metrics = compute_metrics_gpu(y_val, pred_val)

        print(
            f'Epoch {epoch}: train_loss={avg_loss:.6f}, '
            f'val_rmse={val_metrics[0]:.6f}, val_mse={val_metrics[1]:.6f}, '
            f'val_pearson={val_metrics[2]:.6f}, val_ci={val_metrics[3]:.6f}, '
            f'lr={optimizer.param_groups[0]["lr"]:.2e}'
        )

        if val_metrics[1] < best_mse:
            y_test, pred_test = predict(model, device, test_loader)
            test_metrics = compute_metrics_gpu(y_test, pred_test)

            checkpoint_path = os.path.join(output_dir, f'ckpt_{args.run_name}_best.pt')
            torch.save(model.state_dict(), checkpoint_path)
            pd.DataFrame([dict(zip(METRIC_NAMES, test_metrics))]).to_csv(result_path, index=False)

            best_mse = val_metrics[1]
            best_epoch = epoch
            best_metrics = test_metrics
            best_checkpoint_path = checkpoint_path
            print(f'Val MSE improved at epoch {epoch}. Saved checkpoint to {checkpoint_path}')
            print('Test metrics:', dict(zip(METRIC_NAMES, test_metrics)))
        else:
            print(f'No improvement. Best epoch: {best_epoch}, best val MSE: {best_mse:.6f}')

        if epoch - best_epoch >= args.patience:
            print(f'Early stopping at epoch {epoch}')
            break

    if best_metrics is None:
        print('Training finished without a saved best checkpoint.')
    else:
        print(f'Best epoch: {best_epoch}')
        print('Best test metrics:', dict(zip(METRIC_NAMES, best_metrics)))
        print(f'Saved best model to {best_checkpoint_path}')

if __name__ == '__main__':
    main()
