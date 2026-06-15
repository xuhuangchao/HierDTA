import argparse
from pathlib import Path

import pandas as pd


DEFAULT_SEEDS = [32, 33, 41, 42, 43]
DEFAULT_METRICS = ["rmse", "mse", "pearson", "ci", "rm2"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize mean/std over random seeds for saved DTA results."
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. davis or kiba")
    parser.add_argument("--strategy", type=str, required=True, help="Split strategy")
    parser.add_argument("--run_name", type=str, required=True, help="Run name used in checkpoint/result filenames")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Seed list to summarize",
    )
    parser.add_argument(
        "--result_root",
        type=str,
        default=None,
        help="Result root. Defaults to results_{dataset}",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional CSV path for the summary table",
    )
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="Skip missing seeds instead of failing",
    )
    return parser.parse_args()


def load_seed_metrics(root, strategy, seed, run_name):
    seed_dir = root / strategy / f"seed_{seed}"
    ckpt_path = seed_dir / f"ckpt_{run_name}_best.pt"
    result_path = seed_dir / f"result_{run_name}.csv"

    missing = []
    if not ckpt_path.is_file():
        missing.append(str(ckpt_path))
    if not result_path.is_file():
        missing.append(str(result_path))
    if missing:
        raise FileNotFoundError("; ".join(missing))

    df = pd.read_csv(result_path)
    if len(df) != 1:
        raise ValueError(f"Expected one row in {result_path}, got {len(df)}")

    row = df.iloc[0].to_dict()
    row["seed"] = seed
    row["checkpoint"] = str(ckpt_path)
    row["result_csv"] = str(result_path)
    return row


def main():
    args = parse_args()
    root = Path(args.result_root or f"results_{args.dataset}")

    rows = []
    missing = []
    for seed in args.seeds:
        try:
            rows.append(load_seed_metrics(root, args.strategy, seed, args.run_name))
        except FileNotFoundError as exc:
            if args.allow_missing:
                missing.append((seed, str(exc)))
                continue
            raise

    if not rows:
        raise RuntimeError("No seed results were loaded.")

    per_seed = pd.DataFrame(rows)
    metrics = [metric for metric in DEFAULT_METRICS if metric in per_seed.columns]
    if not metrics:
        raise ValueError(f"No known metric columns found. Columns: {list(per_seed.columns)}")

    summary_rows = []
    for metric in metrics:
        values = pd.to_numeric(per_seed[metric], errors="raise")
        summary_rows.append(
            {
                "metric": metric,
                "mean": values.mean(),
                "std": values.std(ddof=1) if len(values) > 1 else 0.0,
                "n": len(values),
            }
        )
    summary = pd.DataFrame(summary_rows)

    print("Per-seed metrics:")
    print(per_seed[["seed"] + metrics].to_string(index=False))
    print("\nSummary:")
    print(summary.to_string(index=False))

    if missing:
        print("\nMissing seeds skipped:")
        for seed, reason in missing:
            print(f"  seed {seed}: {reason}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False)
        print(f"\nSaved summary to {output_path}")


if __name__ == "__main__":
    main()
