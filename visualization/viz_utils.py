import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
RESULTS_DIR  = RESULTS_ROOT / "pima_indians"
FIGURES_DIR  = PROJECT_ROOT / "visualization" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "Pima Indians Diabetes"   : "#4C72B0",
    "Breast Cancer Wisconsin" : "#DD8452",
}

MODEL_ORDER = ["logistic_regression", "svm", "random_forest", "mlp"]

MODEL_LABELS = {
    "logistic_regression": "Log. Reg.",
    "svm"                : "SVM",
    "random_forest"      : "Rand. Forest",
    "mlp"                : "MLP",
}

FAMILY_LABELS = {
    "A"                   : "All features",
    "B"                   : "Top-k selected",
    "B_mutual_information": "MI top-k",
    "B_mrmr"              : "mRMR top-k",
    "C"                   : "Corr-filtered",
    "D"                   : "Domain groups",
    "E"                   : "Random control",
}


def set_style():
    sns.set_theme(style="whitegrid", font_scale=1.05)
    mpl.rcParams.update({
        "figure.dpi"       : 120,
        "savefig.dpi"      : 150,
        "savefig.bbox"     : "tight",
        "axes.spines.top"  : False,
        "axes.spines.right": False,
    })


def apply_family_labels(df):
    # Split Family B into MI vs mRMR using selection_method if available,
    # otherwise infer from subset name suffix (_mi / _mrmr)
    key = df["feature_family"].copy().astype(str)
    is_b = key == "B"

    if "selection_method" in df.columns:
        key[is_b] = "B_" + df.loc[is_b, "selection_method"].fillna("unknown")
    elif "feature_subset_name" in df.columns:
        def _infer(name):
            n = str(name).lower()
            if n.endswith("_mrmr"):
                return "B_mrmr"
            if n.endswith("_mi"):
                return "B_mutual_information"
            return "B"
        key[is_b] = df.loc[is_b, "feature_subset_name"].map(_infer)

    return key.map(FAMILY_LABELS).fillna(key)


def _load_csv(path, label):
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "dataset" not in df.columns:
        df["dataset"] = label
    return df


def _load_unc_csv(path, label):
    if not path.exists():
        return None
    unc = pd.read_csv(path)
    if "subset_name" in unc.columns and "feature_subset_name" not in unc.columns:
        unc = unc.rename(columns={"subset_name": "feature_subset_name"})
    if "dataset" not in unc.columns:
        unc["dataset"] = label
    return unc


def load_all():
    set_style()

    pima_csv = RESULTS_DIR  / "all_results.csv"
    bc_csv   = RESULTS_ROOT / "breast_cancer" / "all_results.csv"
    pima_unc = RESULTS_DIR  / "uncertainty_decomposition" / "uncertainty_results.csv"
    bc_unc   = RESULTS_ROOT / "breast_cancer" / "uncertainty_decomposition" / "uncertainty_results.csv"

    parts = []
    for path, label in [(pima_csv, "Pima Indians Diabetes"),
                        (bc_csv,   "Breast Cancer Wisconsin")]:
        chunk = _load_csv(path, label)
        if chunk is not None:
            parts.append(chunk)
        else:
            print(f"  [INFO] {path.name} not found for '{label}' — skipping.")

    if not parts:
        raise FileNotFoundError("No all_results.csv found. Run experiments first.")

    df   = pd.concat(parts, ignore_index=True)
    base = df[df["calibration_method"] == "none"].copy()
    cal  = df[df["calibration_method"] != "none"].copy()

    unc_parts = []
    for path, label in [(pima_unc, "Pima Indians Diabetes"),
                        (bc_unc,   "Breast Cancer Wisconsin")]:
        chunk = _load_unc_csv(path, label)
        if chunk is not None:
            unc_parts.append(chunk)
        else:
            print(f"  [INFO] uncertainty_results.csv not found for '{label}' — run bootstrap_uncertainty.py first.")

    unc_df = pd.concat(unc_parts, ignore_index=True) if unc_parts else pd.DataFrame()

    if not unc_df.empty:
        merge_keys = ["dataset", "model_name", "feature_subset_name"]
        drop_cols  = [c for c in unc_df.columns if c not in merge_keys and c in base.columns]
        base_merged = base.merge(unc_df.drop(columns=drop_cols), on=merge_keys, how="left")
    else:
        base_merged = base.copy()

    print(f"  Loaded: df={df.shape}  base={base.shape}  cal={cal.shape}  "
          f"unc={unc_df.shape if not unc_df.empty else '(empty)'}")
    return df, base, cal, unc_df, base_merged


def save_fig(name, fig=None):
    path = FIGURES_DIR / f"{name}.png"
    (fig or plt).savefig(path)
    print(f"  Saved → {path.relative_to(PROJECT_ROOT)}")
    plt.close("all")
