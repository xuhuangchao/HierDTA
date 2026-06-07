import argparse
import re
from pathlib import Path

import pandas as pd


METRICS = ["rmse", "mse", "pearson", "ci", "rm2"]
RESULT_RE = re.compile(r"result_motif(?P<motif>\d+)_prot(?P<prot>\d+)\.csv$")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize layer ablation results.")
    parser.add_argument("--result_root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def collect_rows(result_root):
    rows = []
    for path in result_root.glob("*/seed_*/result_motif*_prot*.csv"):
        match = RESULT_RE.match(path.name)
        if match is None:
            continue

        strategy = path.parent.parent.name
        seed_text = path.parent.name.replace("seed_", "")
        metrics = pd.read_csv(path).iloc[0].to_dict()

        row = {
            "strategy": strategy,
            "seed": int(seed_text),
            "motif_num_layers": int(match.group("motif")),
            "prot_num_layers": int(match.group("prot")),
        }
        row.update({metric: metrics.get(metric) for metric in METRICS})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    raw = collect_rows(args.result_root)
    if raw.empty:
        raise SystemExit(f"No ablation result files found under {args.result_root}")

    group_cols = ["strategy", "motif_num_layers", "prot_num_layers"]
    summary = raw.groupby(group_cols)[METRICS].agg(["mean", "std", "count"])
    summary.columns = ["_".join(col).rstrip("_") for col in summary.columns]
    summary = summary.reset_index()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    raw.to_csv(args.output.with_name("summary_raw.csv"), index=False)

    print(summary.to_string(index=False))
    print(f"Saved summary to {args.output}")
    print(f"Saved raw rows to {args.output.with_name('summary_raw.csv')}")


if __name__ == "__main__":
    main()
