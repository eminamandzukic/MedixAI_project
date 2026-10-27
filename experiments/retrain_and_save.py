"""
experiments/retrain_and_save.py
================================
Re-run a specific experiment from its saved config.json, retrain the model,
fit all calibration methods, and save every model as a .pkl file.

Usage
-----
    python3 experiments/retrain_and_save.py --model logistic_regression --idx 5
    python3 experiments/retrain_and_save.py --model random_forest --idx 42
    python3 experiments/retrain_and_save.py --model svm --idx 0 --out-dir my_models/

Saves to (default)
------------------
    saved_models/{model_type}_exp_{idx:04d}/
        uncalibrated.pkl
        platt_scaling.pkl
        isotonic_regression.pkl
        histogram_binning_B5_uniform.pkl
        histogram_binning_B5_quantile.pkl
        histogram_binning_B10_uniform.pkl
        histogram_binning_B10_quantile.pkl
        histogram_binning_B15_uniform.pkl
        histogram_binning_B15_quantile.pkl

How it works
------------
1. Read results/{model_type}/exp_{idx:04d}/config.json
2. Reconstruct the exact sklearn Pipeline from stored hyperparameters
3. Reproduce the exact data splits from the stored seed
4. Retrain the model on the training split
5. Save the fitted pipeline as uncalibrated.pkl
6. Fit each calibration method on the cal split
7. Save each fitted calibrator as {method_name}.pkl
   (stored as the calibrator object alone, not wrapped with the base model)
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

# ── Path setup ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "calibration"))

from experiment_utils import load_pima, make_splits, RESULTS_DIR
from calibration_utils import (
    PlattCalibrator,
    IsotonicCalibrator,
    HistogramBinningCalibrator,
)

# ──────────────────────────────────────────────────────────────────────────────
# Model reconstruction from config
# ──────────────────────────────────────────────────────────────────────────────

def _deserialise_hp(hp: dict) -> dict:
    """
    Undo JSON serialisation artefacts:
      - "None"  → None
      - numeric strings like "0.01", "1" → float / int where appropriate
    Operates on a shallow copy so the caller's dict is unchanged.
    """
    result = {}
    for k, v in hp.items():
        if v == "None":
            result[k] = None
        elif isinstance(v, str):
            # Try int first (e.g. "100"), then float (e.g. "0.01")
            try:
                as_int = int(v)
                result[k] = as_int
            except ValueError:
                try:
                    result[k] = float(v)
                except ValueError:
                    result[k] = v   # keep as string ("scale", "balanced", …)
        else:
            result[k] = v
    return result


def _build_pipeline(model_name: str, hp: dict):
    """
    Reconstruct a sklearn Pipeline from the stored hyperparameter dict.
    Handles type coercions needed because JSON doesn't distinguish
    None / int / float / tuple from their string representations.
    """
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    kw = _deserialise_hp(hp)

    if model_name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        # Remove any legacy human-readable label — not a sklearn parameter
        kw.pop("penalty_label", None)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**kw)),
        ])

    elif model_name == "svm":
        from sklearn.svm import SVC
        # gamma may still be a string like "scale" — that's valid for sklearn
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(**kw)),
        ])

    elif model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        # hidden_layer_sizes: list → tuple (not needed here, just for safety)
        return Pipeline([
            ("clf", RandomForestClassifier(**kw)),
        ])

    elif model_name == "mlp":
        from sklearn.neural_network import MLPClassifier
        # hidden_layer_sizes stored as list in JSON → needs tuple
        if isinstance(kw.get("hidden_layer_sizes"), list):
            kw["hidden_layer_sizes"] = tuple(kw["hidden_layer_sizes"])
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(**kw)),
        ])

    else:
        raise ValueError(f"Unknown model_name: {model_name!r}. "
                         "Expected one of: logistic_regression, svm, "
                         "random_forest, mlp.")


# ──────────────────────────────────────────────────────────────────────────────
# Main retrain + save routine
# ──────────────────────────────────────────────────────────────────────────────

def retrain_and_save(
    model_name : str,
    exp_idx    : int,
    out_dir    : Path | None = None,
) -> Path:
    """
    Read config.json → retrain → save uncalibrated + all calibrated models.

    Returns the output directory path.
    """
    # 1. Load config
    config_path = RESULTS_DIR / model_name / f"exp_{exp_idx:04d}" / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config found at {config_path}\n"
            f"Run the coarse sweep first: python3 experiments/run_all_experiments.py"
        )
    with open(config_path) as f:
        config = json.load(f)

    print(f"\n{'='*60}")
    print(f"Retraining: {model_name}  exp_{exp_idx:04d}")
    print(f"  Feature subset : {config['feature_subset_name']}  "
          f"(family {config['feature_family']})")
    print(f"  Features       : {config['features']}")
    print(f"  Hyperparameters: {config['hyperparameters']}")
    print(f"  Seed           : {config['seed']}")
    print(f"{'='*60}\n")

    # 2. Reproduce the exact splits from config
    seed     = config["seed"]
    features = config["features"]
    X_full, y_full, _ = load_pima()
    splits_full = make_splits(X_full, y_full, seed=seed)
    splits = {
        split: (df[features], labels)
        for split, (df, labels) in splits_full.items()
    }
    X_train, y_train = splits["train"]
    X_cal,   y_cal   = splits["cal"]
    X_test,  y_test  = splits["test"]

    print(f"  Split sizes — train:{len(X_train)}  "
          f"cal:{len(X_cal)}  test:{len(X_test)}")

    # 3. Reconstruct and retrain the base model
    pipeline = _build_pipeline(model_name, config["hyperparameters"])
    pipeline.fit(X_train, y_train)
    print(f"  ✓  Base model retrained.")

    # 4. Set up output directory
    if out_dir is None:
        out_dir = PROJECT_ROOT / "saved_models" / f"{model_name}_exp_{exp_idx:04d}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 5. Save uncalibrated model
    uncal_path = out_dir / "uncalibrated.pkl"
    with open(uncal_path, "wb") as f:
        pickle.dump(pipeline, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  ✓  Saved: {uncal_path.name}")

    # 6. Get raw probabilities on cal and test sets
    p_cal_raw  = pipeline.predict_proba(X_cal)[:, 1]
    p_test_raw = pipeline.predict_proba(X_test)[:, 1]

    # 7. Fit and save each calibration method
    calibrators: dict[str, object] = {}

    # Platt Scaling
    calibrators["platt_scaling"] = PlattCalibrator().fit(p_cal_raw, y_cal)

    # Isotonic Regression
    calibrators["isotonic_regression"] = IsotonicCalibrator().fit(p_cal_raw, y_cal)

    # Histogram Binning (6 variants)
    for n_bins in [5, 10, 15]:
        for strategy in ["uniform", "quantile"]:
            name = f"histogram_binning_B{n_bins}_{strategy}"
            calibrators[name] = HistogramBinningCalibrator(
                n_bins=n_bins, strategy=strategy
            ).fit(p_cal_raw, y_cal)



    # Save each calibrator
    for name, cal in calibrators.items():
        path = out_dir / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(cal, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  ✓  Saved: {path.name}")

    # 8. Summary
    total_size = sum(p.stat().st_size for p in out_dir.iterdir()) / 1024
    print(f"\n  Output directory : {out_dir}")
    print(f"  Files saved      : {len(list(out_dir.iterdir()))}")
    print(f"  Total size       : {total_size:.1f} KB\n")

    return out_dir


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrain an experiment from its config.json and save all models."
    )
    parser.add_argument(
        "--model", required=True,
        choices=["logistic_regression", "svm", "random_forest", "mlp"],
        help="Model type.",
    )
    parser.add_argument(
        "--idx", required=True, type=int,
        help="Experiment index (as stored in results/{model}/exp_{idx:04d}/).",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help=(
            "Output directory. Defaults to "
            "saved_models/{model}_exp_{idx:04d}/ in the project root."
        ),
    )
    args = parser.parse_args()

    out = retrain_and_save(
        model_name=args.model,
        exp_idx=args.idx,
        out_dir=Path(args.out_dir) if args.out_dir else None,
    )
    print(f"✓  Done.  Models saved to: {out}")
