import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
from experiment_utils import get_results_dir


def export(dataset: str = "pima"):
    results_dir = get_results_dir(dataset)
    out_base    = results_dir / "uncertainty_decomposition"
    json_path   = out_base / "overall_summary.json"
    csv_path    = out_base / "uncertainty_results.csv"

    if not json_path.exists():
        print(f"  [ERROR] JSON not found: {json_path}")
        print("  Run bootstrap_uncertainty.py first.")
        return

    data = json.loads(json_path.read_text())
    rows = data.get("rows", [])
    if not rows:
        print("  [WARN] overall_summary.json contains no rows.")
        return

    df = pd.DataFrame(rows)

    if "subset_name" in df.columns and "feature_subset_name" not in df.columns:
        df = df.rename(columns={"subset_name": "feature_subset_name"})

    dataset_label = "Breast Cancer Wisconsin" if dataset == "breast_cancer" else "Pima Indians Diabetes"
    if "dataset" not in df.columns:
        df["dataset"] = dataset_label

    df.to_csv(csv_path, index=False, float_format="%.6f")
    print(f"  Written: {csv_path}  ({len(df)} rows × {len(df.columns)} cols)")


def main():
    p = argparse.ArgumentParser(description="Convert uncertainty JSON → CSV.")
    p.add_argument("--dataset", default="pima", choices=["pima", "breast_cancer"])
    export(p.parse_args().dataset)


if __name__ == "__main__":
    main()
