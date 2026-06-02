import os
import glob
import pandas as pd
import numpy as np
import argparse
from collections import defaultdict


METRICS = ['RMSE', 'MSE', 'Pearson', 'CI', 'RM2']
METRIC_ALIASES = {
    'rmse': 'RMSE',
    'mse': 'MSE',
    'pearson': 'Pearson',
    'ci': 'CI',
    'rm2': 'RM2',
}


def normalize_metric_name(name):
    key = str(name).strip().lower()
    return METRIC_ALIASES.get(key)


def parse_plain_metric_line(filepath):
    with open(filepath, 'r') as f:
        line = f.read().strip()
    if not line:
        return None
    try:
        values = list(map(float, line.split(',')))
    except ValueError:
        print(f"Warning: Could not parse {filepath}")
        return None
    if len(values) != len(METRICS):
        print(f"Warning: Expected {len(METRICS)} metrics in {filepath}, got {len(values)}")
    return values


def parse_dataframe_metrics(filepath):
    df = pd.read_csv(filepath)
    if df.empty:
        return None

    column_map = {}
    for col in df.columns:
        metric = normalize_metric_name(col)
        if metric is not None:
            column_map[metric] = col

    # Current train.py format: one row with columns rmse,mse,pearson,ci,rm2.
    if column_map:
        row = df.iloc[0]
        values = []
        for metric in METRICS:
            col = column_map.get(metric)
            if col is None:
                print(f"Warning: Missing metric column {metric} in {filepath}")
                return None
            values.append(float(row[col]))
        return values

    # Also support metric/value style CSV files.
    lower_cols = {str(col).strip().lower(): col for col in df.columns}
    metric_col = lower_cols.get('metric') or lower_cols.get('name')
    value_col = lower_cols.get('value') or lower_cols.get('score')
    if metric_col is not None and value_col is not None:
        values_by_metric = {}
        for _, row in df.iterrows():
            metric = normalize_metric_name(row[metric_col])
            if metric is not None:
                values_by_metric[metric] = float(row[value_col])
        if values_by_metric:
            values = []
            for metric in METRICS:
                if metric not in values_by_metric:
                    print(f"Warning: Missing metric {metric} in {filepath}")
                    return None
                values.append(values_by_metric[metric])
            return values

    return None


def parse_result_file(filepath):
    """Parse a single result CSV file in either current headered or legacy plain format."""
    try:
        values = parse_dataframe_metrics(filepath)
        if values is not None:
            return values
    except Exception:
        pass

    try:
        return parse_plain_metric_line(filepath)
    except ValueError:
        print(f"Warning: Could not parse {filepath}")
        return None


def result_pattern(base_dir, strategy, suffix):
    if not suffix:
        return os.path.join(base_dir, strategy, 'seed_*', '*.csv')
    name = suffix
    if not name.endswith('.csv'):
        name = f'{name}.csv'
    if not name.startswith('result_'):
        name = f'result_{name}'
    return os.path.join(base_dir, strategy, 'seed_*', name)


def summarize_results(base_dir, strategy, suffix=None, save_path=None):
    """
    Scan base_dir/strategy/seed_*/result_*.csv and compute mean/std across seeds.
    If suffix is provided, match exactly result_{suffix}.csv.
    """
    pattern = result_pattern(base_dir, strategy, suffix)
    csv_files = glob.glob(pattern)

    if not csv_files:
        print(f"No result files found matching: {pattern}")
        return None

    # Group by experiment configuration (derived from filename)
    groups = defaultdict(list)

    for csv_file in csv_files:
        parts = csv_file.split(os.sep)
        seed_part = [p for p in parts if p.startswith('seed_')]
        seed = int(seed_part[0].split('_')[1]) if seed_part else None

        filename = os.path.basename(csv_file)
        # Expected format: result_{model_st}_{name}.csv
        if filename.startswith('result_') and filename.endswith('.csv'):
            config_name = filename[7:-4]  # strip 'result_' and '.csv'
        else:
            config_name = filename[:-4]

        values = parse_result_file(csv_file)
        if values is None:
            continue

        groups[config_name].append((seed, values))

    all_summaries = []

    for config_name, results in sorted(groups.items()):
        results.sort(key=lambda x: x[0])
        seeds = [r[0] for r in results]
        values_array = np.array([r[1] for r in results])

        # Build per-seed DataFrame
        df = pd.DataFrame(values_array, columns=METRICS)
        df.insert(0, 'Seed', seeds)

        # Compute statistics
        mean_vals = df[METRICS].mean()
        std_vals = df[METRICS].std()

        # Print
        print(f"\n{'='*70}")
        print(f"Config : {config_name}")
        print(f"Seeds  : {len(results)}  ({seeds})")
        print(f"{'='*70}")
        print(df.to_string(index=False))
        print(f"{'-'*70}")
        print("Mean ± Std:")
        for m in METRICS:
            print(f"  {m:8s}: {mean_vals[m]:.6f} ± {std_vals[m]:.6f}")
        print(f"{'='*70}\n")

        # Collect for summary CSV
        summary_row = {'Config': config_name, 'Num_Seeds': len(results)}
        for m in METRICS:
            summary_row[f'{m}_mean'] = mean_vals[m]
            summary_row[f'{m}_std'] = std_vals[m]
        all_summaries.append(summary_row)

    summary_df = pd.DataFrame(all_summaries)

    if save_path:
        summary_df.to_csv(save_path, index=False, float_format='%.6f')
        print(f"Summary saved to: {save_path}")

    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description='Summarize HierDTA experiment results across seeds (mean ± std)'
    )
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name, e.g., kiba or davis')
    parser.add_argument('--strategy', type=str, default='warm',
                        choices=['warm', 'unseen_drug', 'unseen_prot', 'unseen_pair'],
                        help='Split strategy (default: warm)')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='Base results directory. Auto-detected if not provided.')
    parser.add_argument('--suffix', type=str, default="dta_model",
                        help='Result filename suffix to match, e.g., runseed_seed. '
                             'If provided, matches exactly result_{suffix}.csv. '
                             'If omitted, matches all CSV files.')
    parser.add_argument('--save_csv', type=str, default=None,
                        help='Path to save the summary CSV (optional)')
    args = parser.parse_args()

    # Auto-detect results directory
    if args.results_dir:
        base_dir = args.results_dir
    else:
        candidates = [
            f'results_{args.dataset}',
        ]
        base_dir = None
        for c in candidates:
            if os.path.exists(c):
                base_dir = c
                break

        if base_dir is None:
            print(f"Error: Could not find results directory for dataset '{args.dataset}'.")
            print(f"Searched: {candidates}")
            print("Please specify --results_dir explicitly.")
            return

    print(f"Scanning results in: {base_dir}/{args.strategy}/")
    if args.suffix:
        print(f"Matching files: result_{args.suffix}.csv")
    summarize_results(base_dir, args.strategy, suffix=args.suffix, save_path=args.save_csv)


if __name__ == '__main__':
    main()
