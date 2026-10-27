import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
from experiment_utils import RESULTS_DIR, get_results_dir

PERF_METRICS = [
    "auc", "f1", "recall", "precision", "accuracy",
    "brier", "nll", "ece", "mce", "entropy", "composite",
]
UNC_METRICS = ["entropy", "ece", "mce", "brier", "nll"]


def _base_row(model_name, exp_dir):
    cfg_path    = exp_dir / "config.json"
    scores_path = exp_dir / "scores.json"
    if not (cfg_path.exists() and scores_path.exists()):
        return None

    with open(cfg_path)    as f: cfg    = json.load(f)
    with open(scores_path) as f: scores = json.load(f)

    row = {
        "model_name"          : model_name,
        "exp_idx"             : cfg.get("exp_idx"),
        "calibration_method"  : "none",
        "feature_subset_name" : cfg.get("feature_subset_name"),
        "feature_family"      : cfg.get("feature_family"),
        "n_features"          : cfg.get("n_features"),
        "seed"                : cfg.get("seed"),
        "search_stage"        : cfg.get("search_stage", "coarse"),
        "val_auc"             : scores.get("val_auc"),
        "val_composite"       : scores.get("val_composite"),
    }

    hp = {k: v for k, v in cfg.get("hyperparameters", {}).items()
          if v is not None and str(v).lower() != "none"}
    row["hyperparameters"] = json.dumps(hp)

    for m in PERF_METRICS:
        row[f"orig_{m}"] = scores.get(f"test_{m}")
    for m in UNC_METRICS:
        row[f"orig_unc_{m}"] = scores.get(f"test_{m}")
    for m in PERF_METRICS:
        row[f"cal_{m}"]   = float("nan")
        row[f"delta_{m}"] = float("nan")
    for m in UNC_METRICS:
        row[f"cal_unc_{m}"] = float("nan")

    return row


def _calibrated_row(model_name, exp_dir, method_name, cal_scores_path):
    base_cfg_path = exp_dir / "config.json"
    if not (base_cfg_path.exists() and cal_scores_path.exists()):
        return None

    with open(base_cfg_path)   as f: cfg        = json.load(f)
    with open(cal_scores_path) as f: cal_scores  = json.load(f)

    row = {
        "model_name"          : model_name,
        "exp_idx"             : cfg.get("exp_idx"),
        "calibration_method"  : method_name,
        "feature_subset_name" : cfg.get("feature_subset_name"),
        "feature_family"      : cfg.get("feature_family"),
        "n_features"          : cfg.get("n_features"),
        "seed"                : cfg.get("seed"),
        "search_stage"        : cfg.get("search_stage", "coarse"),
        "val_auc"             : None,
        "val_composite"       : None,
    }

    hp = {k: v for k, v in cfg.get("hyperparameters", {}).items()
          if v is not None and str(v).lower() != "none"}
    row["hyperparameters"] = json.dumps(hp)

    for m in PERF_METRICS:
        row[f"orig_{m}"] = cal_scores.get(f"before_{m}")
    for m in PERF_METRICS:
        row[f"cal_{m}"]   = cal_scores.get(f"after_{m}")
        row[f"delta_{m}"] = cal_scores.get(f"delta_{m}")

    unc_before = cal_scores.get("uncertainty_before", {})
    unc_after  = cal_scores.get("uncertainty_after",  {})
    for m in UNC_METRICS:
        row[f"orig_unc_{m}"] = unc_before.get(m)
        row[f"cal_unc_{m}"]  = unc_after.get(m)

    return row


def collect_rows(model_names=None, include_base=True, include_cal=True, results_dir=None):
    rows = []
    root = results_dir if results_dir is not None else RESULTS_DIR

    model_dirs = (
        sorted(d for d in root.iterdir() if d.is_dir())
        if model_names is None
        else [root / n for n in model_names]
    )

    for model_dir in model_dirs:
        if not model_dir.is_dir():
            print(f"  [WARN] Directory not found: {model_dir}")
            continue
        model_name = model_dir.name

        for exp_dir in sorted(d for d in model_dir.iterdir()
                              if d.is_dir() and d.name.startswith("exp_")):
            if include_base:
                row = _base_row(model_name, exp_dir)
                if row:
                    rows.append(row)

            if include_cal:
                cal_root = exp_dir / "calibrated"
                if cal_root.exists():
                    for method_dir in sorted(cal_root.iterdir()):
                        cal_scores_path = method_dir / "scores.json"
                        if not cal_scores_path.exists():
                            continue
                        row = _calibrated_row(model_name, exp_dir, method_dir.name, cal_scores_path)
                        if row:
                            rows.append(row)

        n_cal  = sum(1 for r in rows if r["model_name"] == model_name and r["calibration_method"] != "none")
        n_base = sum(1 for r in rows if r["model_name"] == model_name and r["calibration_method"] == "none")
        print(f"  {model_name:<22}  base={n_base:>5}  calibrated={n_cal:>6}")

    return rows


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export all experiment results to a flat CSV.")
    parser.add_argument("--dataset", default="pima", choices=["pima", "breast_cancer"])
    parser.add_argument("--out",     default=None)
    parser.add_argument("--models",  nargs="+", default=None)
    parser.add_argument("--no-base",       action="store_true")
    parser.add_argument("--no-calibrated", action="store_true")
    args = parser.parse_args()

    results_dir = get_results_dir(args.dataset)
    out_path    = Path(args.out) if args.out else results_dir / "all_results.csv"

    print(f"Collecting results...  (dataset={args.dataset})")
    rows = collect_rows(
        model_names=args.models,
        include_base=not args.no_base,
        include_cal=not args.no_calibrated,
        results_dir=results_dir,
    )

    if not rows:
        print("No rows found. Check that experiments have been run.")
        return

    df = pd.DataFrame(rows)

    id_cols    = ["model_name", "exp_idx", "calibration_method",
                  "feature_subset_name", "feature_family", "n_features",
                  "seed", "search_stage", "val_auc", "val_composite", "hyperparameters"]
    orig_cols  = [f"orig_{m}"     for m in PERF_METRICS]
    cal_cols   = [f"cal_{m}"      for m in PERF_METRICS]
    delta_cols = [f"delta_{m}"    for m in PERF_METRICS]
    ounc_cols  = [f"orig_unc_{m}" for m in UNC_METRICS]
    cunc_cols  = [f"cal_unc_{m}"  for m in UNC_METRICS]

    ordered = id_cols + orig_cols + cal_cols + delta_cols + ounc_cols + cunc_cols
    extra   = [c for c in df.columns if c not in ordered]
    df      = df[ordered + extra]

    df.to_csv(out_path, index=False, float_format="%.6f")

    print(f"\n  Total rows     : {len(df):,}")
    print(f"  Base rows      : {(df['calibration_method'] == 'none').sum():,}")
    print(f"  Calibrated rows: {(df['calibration_method'] != 'none').sum():,}")
    print(f"  Saved to       : {out_path}")


if __name__ == "__main__":
    main()
