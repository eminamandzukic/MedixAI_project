"""
feature_extraction/generate_feature_subsets.py
================================================
Unified feature subset generation for both datasets.

Families  (identical structure for Pima and Breast Cancer)
----------------------------------------------------------
Family A : All features  (baseline)
Family B : Top-k by Mutual Information      k ∈ {3, 5, 10, 15}
           Top-k by mRMR                    k ∈ {3, 5, 10, 15}
           (Pima uses only k ≤ n_features, so k=10,15 are clipped to 8)
Family C : Correlation-filtered  (|r| > threshold, threshold ∈ {0.80, 0.90, 0.95})
Family D : Domain-based feature groups  (dataset-specific)
Family E : Random subsets  (m=10 draws per k, control group)

Outputs
-------
feature_extraction/feature_subsets_pima.json
feature_extraction/feature_subsets_breast_cancer.json

Usage
-----
    python3 feature_extraction/generate_feature_subsets.py              # both
    python3 feature_extraction/generate_feature_subsets.py --dataset pima
    python3 feature_extraction/generate_feature_subsets.py --dataset breast_cancer
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR     = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Top-k values shared by both datasets (capped per dataset below)
TOP_K_VALUES = [3, 5, 10, 15]
RANDOM_M     = 10   # draws for Family E

# ─────────────────────────────────────────────────────────────────────────────
# mRMR implementation (pure numpy/pandas — no external lib needed)
# Falls back to mrmr-selection package if available.
# ─────────────────────────────────────────────────────────────────────────────

def _mrmr_numpy(X: pd.DataFrame, y: pd.Series, k: int, seed: int = 42) -> list[str]:
    """
    Greedy mRMR (MID variant):
        score(f) = MI(f, y) - (1/|S|) * sum_{s in S} MI(f, s)
    Returns top-k feature names.
    """
    features = list(X.columns)
    mi_target = mutual_info_classif(X, y, discrete_features=False,
                                    random_state=seed)
    mi_t = dict(zip(features, mi_target))

    selected: list[str] = []
    remaining = features.copy()

    for _ in range(min(k, len(features))):
        if not selected:
            # First feature: highest MI with target
            best = max(remaining, key=lambda f: mi_t[f])
        else:
            # Compute pairwise MI between candidates and already-selected
            def score(f: str) -> float:
                redundancy = np.mean([
                    mutual_info_classif(
                        X[[f]], X[s].values,
                        discrete_features=False, random_state=seed,
                    )[0]
                    for s in selected
                ])
                return mi_t[f] - redundancy

            best = max(remaining, key=score)

        selected.append(best)
        remaining.remove(best)

    return selected


def mrmr_select(X: pd.DataFrame, y: pd.Series, k: int, seed: int = 42) -> list[str]:
    """Try mrmr-selection package first; fall back to our numpy implementation."""
    # Always reset index — mrmr_classif requires aligned integer index
    X_   = X.reset_index(drop=True)
    y_   = y.reset_index(drop=True)
    try:
        from mrmr import mrmr_classif   # type: ignore
        return mrmr_classif(X=X_, y=y_, K=k, n_jobs=1)
    except ImportError:
        return _mrmr_numpy(X_, y_, k, seed)


# ─────────────────────────────────────────────────────────────────────────────
# Shared family builders
# ─────────────────────────────────────────────────────────────────────────────

def build_family_a(feature_cols: list[str], mi_dict: dict) -> dict:
    return {
        "all_features": {
            "family"          : "A",
            "description"     : f"All {len(feature_cols)} features — maximum-information baseline.",
            "features"        : feature_cols,
            "n_features"      : len(feature_cols),
            "mi_scores"       : {f: mi_dict[f] for f in feature_cols},
            "selection_method": "none (baseline)",
        }
    }


def build_family_b(
    feature_cols : list[str],
    ranked_mi    : list[str],
    mi_dict      : dict,
    X_train      : pd.DataFrame,
    y_train      : pd.Series,
    k_values     : list[int],
    seed         : int = 42,
) -> dict:
    """MI top-k and mRMR top-k, for each k in k_values (capped to n_features)."""
    subsets: dict = {}
    n = len(feature_cols)

    for k in k_values:
        k_eff = min(k, n)   # Pima has only 8 features

        # ── MI top-k ──────────────────────────────────────────────────────────
        top_k = ranked_mi[:k_eff]
        key = f"top_{k}_mi"
        subsets[key] = {
            "family"          : "B",
            "description"     : (
                f"Top-{k_eff} features by Mutual Information "
                f"(computed on training split, seed={seed})."
            ),
            "features"        : top_k,
            "n_features"      : k_eff,
            "mi_scores"       : {f: mi_dict[f] for f in top_k},
            "selection_method": "mutual_information",
            "k"               : k_eff,
            "ranking"         : [
                {"rank": r + 1, "feature": f, "mi_score": mi_dict[f]}
                for r, f in enumerate(top_k)
            ],
        }

        # ── mRMR top-k ────────────────────────────────────────────────────────
        print(f"    Computing mRMR top-{k_eff} …", end=" ", flush=True)
        mrmr_feats = mrmr_select(X_train, y_train, k=k_eff, seed=seed)
        print("done")
        key_m = f"top_{k}_mrmr"
        subsets[key_m] = {
            "family"          : "B",
            "description"     : (
                f"Top-{k_eff} features by mRMR (Minimum Redundancy Maximum Relevance, "
                f"greedy MID variant, training split, seed={seed})."
            ),
            "features"        : mrmr_feats,
            "n_features"      : k_eff,
            "mi_scores"       : {f: mi_dict[f] for f in mrmr_feats},
            "selection_method": "mrmr",
            "k"               : k_eff,
            "ranking"         : [
                {"rank": r + 1, "feature": f, "mi_score": mi_dict[f]}
                for r, f in enumerate(mrmr_feats)
            ],
        }

    return subsets


def build_family_c(
    X_train     : pd.DataFrame,
    feature_cols: list[str],
    mi_dict     : dict,
    thresholds  : list[float] = (0.80, 0.90, 0.95),
) -> dict:
    subsets: dict = {}
    for threshold in thresholds:
        corr   = X_train[feature_cols].corr().abs()
        cols   = list(feature_cols)
        to_drop: set = set()

        for i, j in itertools.combinations(range(len(cols)), 2):
            fi, fj = cols[i], cols[j]
            if fi in to_drop or fj in to_drop:
                continue
            if corr.loc[fi, fj] > threshold:
                # Drop the one with lower MI
                drop = fj if mi_dict[fi] >= mi_dict[fj] else fi
                to_drop.add(drop)

        retained = [f for f in cols if f not in to_drop]
        key = f"corr_filtered_{str(threshold).replace('.', '')}"
        subsets[key] = {
            "family"           : "C",
            "description"      : (
                f"Correlation-filtered at |r|>{threshold}. "
                f"Within each high-corr pair the feature with lower MI is dropped."
            ),
            "features"         : retained,
            "n_features"       : len(retained),
            "dropped_features" : sorted(to_drop),
            "threshold"        : threshold,
            "mi_scores"        : {f: mi_dict[f] for f in retained},
            "selection_method" : "correlation_filter",
        }
        print(f"    [C] corr_filtered_{threshold}: retained={retained}")
    return subsets


def build_family_e(
    feature_cols: list[str],
    mi_dict     : dict,
    k_values    : list[int],
    seed        : int = 42,
    m           : int = 10,
) -> dict:
    subsets: dict = {}
    n = len(feature_cols)
    for k in k_values:
        k_eff = min(k, n)
        rng = random.Random(seed + k_eff)
        for draw_idx in range(m):
            feats = sorted(rng.sample(feature_cols, k_eff))
            key = f"random_k{k_eff}_{draw_idx:02d}"
            subsets[key] = {
                "family"          : "E",
                "description"     : (
                    f"Random subset #{draw_idx+1} of size {k_eff} "
                    f"(control group, seed={seed+k_eff}, draw={draw_idx})."
                ),
                "features"        : feats,
                "n_features"      : k_eff,
                "mi_scores"       : {f: mi_dict[f] for f in feats},
                "selection_method": "random",
                "k"               : k_eff,
                "draw_index"      : draw_idx,
            }
        print(f"    [E] {m} random subsets of k_eff={k_eff}")
    return subsets


def finalise_json(
    dataset_name: str,
    source_url  : str,
    local_path  : str,
    feature_cols: list[str],
    n_samples   : int,
    pos_ratio   : float,
    ranked_mi   : list[str],
    mi_dict     : dict,
    n_train     : int,
    subsets     : dict,
    extra_meta  : dict | None = None,
) -> dict:
    families = ["A", "B", "C", "D", "E"]
    doc = {
        "metadata": {
            "dataset"               : dataset_name,
            "source_url"            : source_url,
            "local_path"            : local_path,
            "n_samples_total"       : n_samples,
            "n_features"            : len(feature_cols),
            "feature_names"         : feature_cols,
            "positive_class_ratio"  : pos_ratio,
            "seed"                  : SEED,
            "train_size"            : n_train,
            "mi_ranking_note"       : (
                "Mutual Information scores computed by "
                "sklearn.feature_selection.mutual_info_classif on the "
                f"training split (60% of data, seed={SEED})."
            ),
            "mrmr_note"             : (
                "mRMR top-k uses a greedy MID variant: at each step the "
                "feature maximising MI(f,y) − mean MI(f,S) is selected, "
                "where S is the set of already-selected features."
            ),
            "family_b_k_values"     : TOP_K_VALUES,
        },
        "global_mi_ranking": [
            {"rank": r+1, "feature": f, "mi_score": float(mi_dict[f])}
            for r, f in enumerate(ranked_mi)
        ],
        "feature_subsets": subsets,
        "summary": {
            "total_subsets": len(subsets),
            "family_counts": {
                fam: sum(1 for v in subsets.values() if v["family"] == fam)
                for fam in families
            },
        },
    }
    if extra_meta:
        doc["metadata"].update(extra_meta)
    return doc


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-specific loaders & domain groups
# ─────────────────────────────────────────────────────────────────────────────

def _pima_domain_groups() -> dict:
    return {
        "metabolic"              : {
            "features"   : ["Glucose", "Insulin", "BMI"],
            "description": "Metabolic markers: Glucose, Insulin, BMI.",
        },
        "cardiovascular"         : {
            "features"   : ["BloodPressure", "Age"],
            "description": "Cardiovascular/ageing proxies: BloodPressure, Age.",
        },
        "demographic_pregnancy"  : {
            "features"   : ["Pregnancies", "Age"],
            "description": "Demographic/pregnancy history: Pregnancies, Age.",
        },
        "skin_body_composition"  : {
            "features"   : ["SkinThickness", "BMI"],
            "description": "Skin/body composition: SkinThickness, BMI.",
        },
        "genetic_risk_proxy"     : {
            "features"   : ["DiabetesPedigreeFunction", "Age"],
            "description": "Genetic risk proxy: DiabetesPedigreeFunction, Age.",
        },
        "metabolic_plus_risk"    : {
            "features"   : ["Glucose", "Insulin", "BMI",
                            "DiabetesPedigreeFunction", "Age"],
            "description": "Combined metabolic + genetic risk.",
        },
    }


def _bc_domain_groups() -> dict:
    return {
        "size_shape_mean"  : {
            "features"   : ["mean radius", "mean perimeter", "mean area",
                            "mean compactness"],
            "description": "Mean size/shape: radius, perimeter, area, compactness.",
        },
        "texture_symmetry" : {
            "features"   : ["mean texture", "mean symmetry",
                            "mean fractal dimension"],
            "description": "Mean texture, symmetry, fractal dimension.",
        },
        "concavity_mean"   : {
            "features"   : ["mean concavity", "mean concave points",
                            "mean compactness"],
            "description": "Mean concavity-related features.",
        },
        "worst_size"       : {
            "features"   : ["worst radius", "worst perimeter",
                            "worst area", "worst compactness"],
            "description": "Worst-case size features.",
        },
        "worst_shape"      : {
            "features"   : ["worst concavity", "worst concave points",
                            "worst symmetry", "worst fractal dimension"],
            "description": "Worst-case shape/boundary features.",
        },
        "error_features"   : {
            "features"   : ["radius error", "perimeter error",
                            "area error", "texture error"],
            "description": "SE (standard error) measurement features.",
        },
        "core_discriminative": {
            "features"   : ["worst perimeter", "worst area",
                            "worst radius", "mean concave points",
                            "worst concave points"],
            "description": "Top-5 discriminative features by domain + MI consensus.",
        },
    }


def generate_pima():
    print("\n" + "=" * 60)
    print("PIMA INDIANS DIABETES DATASET")
    print("=" * 60)

    pima_path = DATA_DIR / "pima_indians_diabetes.csv"
    pima_url  = (
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
        "pima-indians-diabetes.data.csv"
    )
    pima_cols = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
    ]
    zero_invalid = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

    if not pima_path.exists():
        print(f"  Downloading {pima_url} …")
        urllib.request.urlretrieve(pima_url, pima_path)
    else:
        print(f"  Dataset present: {pima_path}")

    df = pd.read_csv(pima_path, header=None, names=pima_cols)
    feature_cols = pima_cols[:-1]
    X_raw = df[feature_cols].replace(
        {col: {0: np.nan} for col in zero_invalid}
    )
    y = df["Outcome"]

    X_tr_raw, _, y_tr, _ = train_test_split(
        X_raw, y, test_size=0.40, stratify=y, random_state=SEED
    )
    medians = X_tr_raw.median()
    X_train = X_tr_raw.fillna(medians)
    y_train = y_tr.reset_index(drop=True)

    print("  Computing MI scores …")
    mi_scores = mutual_info_classif(X_train, y_train,
                                    discrete_features=False, random_state=SEED)
    mi_dict   = dict(zip(feature_cols, map(float, mi_scores)))
    ranked_mi = sorted(feature_cols, key=lambda f: -mi_dict[f])

    k_vals = [k for k in TOP_K_VALUES if k <= len(feature_cols)]
    subsets: dict = {}

    print("  Family A …")
    subsets.update(build_family_a(feature_cols, mi_dict))

    print("  Family B …")
    subsets.update(build_family_b(
        feature_cols, ranked_mi, mi_dict, X_train, y_train, k_vals, SEED
    ))

    print("  Family C …")
    subsets.update(build_family_c(X_train, feature_cols, mi_dict))

    print("  Family D …")
    for name, info in _pima_domain_groups().items():
        feats = info["features"]
        subsets[f"domain_{name}"] = {
            "family"          : "D",
            "description"     : info["description"],
            "features"        : feats,
            "n_features"      : len(feats),
            "mi_scores"       : {f: mi_dict[f] for f in feats},
            "selection_method": "domain_knowledge",
        }
        print(f"    [D] domain_{name}: {feats}")

    print("  Family E …")
    subsets.update(build_family_e(feature_cols, mi_dict, k_vals, SEED, RANDOM_M))

    doc = finalise_json(
        dataset_name = "Pima Indians Diabetes Database",
        source_url   = pima_url,
        local_path   = str(pima_path),
        feature_cols = feature_cols,
        n_samples    = len(df),
        pos_ratio    = float(y.mean()),
        ranked_mi    = ranked_mi,
        mi_dict      = mi_dict,
        n_train      = len(X_train),
        subsets      = subsets,
        extra_meta   = {
            "missing_value_handling": (
                "Biologically-invalid zeros in Glucose, BloodPressure, "
                "SkinThickness, Insulin, BMI replaced with NaN, "
                "then median-imputed using training split only."
            )
        },
    )

    out = SCRIPT_DIR / "feature_subsets_pima.json"
    out.write_text(json.dumps(doc, indent=2))
    print(f"\n  ✓ Written: {out}  ({len(subsets)} subsets)")
    return doc


def generate_breast_cancer():
    print("\n" + "=" * 60)
    print("BREAST CANCER WISCONSIN DATASET")
    print("=" * 60)

    bc   = load_breast_cancer()
    X_df = pd.DataFrame(bc.data, columns=bc.feature_names.tolist())
    y    = pd.Series(bc.target)
    feature_cols = bc.feature_names.tolist()

    # Also save raw CSV for reproducibility
    bc_path = DATA_DIR / "breast_cancer_wisconsin.csv"
    if not bc_path.exists():
        combined = X_df.copy()
        combined["target"] = y.values
        combined.to_csv(bc_path, index=False)
        print(f"  Saved CSV: {bc_path}")
    else:
        print(f"  Dataset present: {bc_path}")

    X_tr, _, y_tr, _ = train_test_split(
        X_df, y, test_size=0.40, stratify=y, random_state=SEED
    )
    X_train = X_tr.reset_index(drop=True)
    y_train = y_tr.reset_index(drop=True)

    print("  Computing MI scores …")
    mi_scores = mutual_info_classif(X_train, y_train,
                                    discrete_features=False, random_state=SEED)
    mi_dict   = dict(zip(feature_cols, map(float, mi_scores)))
    ranked_mi = sorted(feature_cols, key=lambda f: -mi_dict[f])

    # BC has 30 features — all k values are valid
    k_vals = TOP_K_VALUES

    subsets: dict = {}

    print("  Family A …")
    subsets.update(build_family_a(feature_cols, mi_dict))

    print("  Family B …")
    subsets.update(build_family_b(
        feature_cols, ranked_mi, mi_dict, X_train, y_train, k_vals, SEED
    ))

    print("  Family C …")
    subsets.update(build_family_c(X_train, feature_cols, mi_dict))

    print("  Family D …")
    for name, info in _bc_domain_groups().items():
        feats = info["features"]
        subsets[f"domain_{name}"] = {
            "family"          : "D",
            "description"     : info["description"],
            "features"        : feats,
            "n_features"      : len(feats),
            "mi_scores"       : {f: mi_dict[f] for f in feats},
            "selection_method": "domain_knowledge",
        }
        print(f"    [D] domain_{name}: {feats}")

    print("  Family E …")
    subsets.update(build_family_e(feature_cols, mi_dict, k_vals, SEED, RANDOM_M))

    doc = finalise_json(
        dataset_name = "Breast Cancer Wisconsin Diagnostic",
        source_url   = "https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset",
        local_path   = str(bc_path),
        feature_cols = feature_cols,
        n_samples    = len(X_df),
        pos_ratio    = float(y.mean()),
        ranked_mi    = ranked_mi,
        mi_dict      = mi_dict,
        n_train      = len(X_train),
        subsets      = subsets,
    )

    out = SCRIPT_DIR / "feature_subsets_breast_cancer.json"
    out.write_text(json.dumps(doc, indent=2))
    print(f"\n  ✓ Written: {out}  ({len(subsets)} subsets)")
    return doc


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Generate feature subset JSONs for Pima and/or Breast Cancer."
    )
    p.add_argument(
        "--dataset", default="both",
        choices=["pima", "breast_cancer", "both"],
        help="Which dataset to process. Default: both.",
    )
    args = p.parse_args()

    if args.dataset in ("pima", "both"):
        generate_pima()

    if args.dataset in ("breast_cancer", "both"):
        generate_breast_cancer()

    print("\n✓  All done.")


if __name__ == "__main__":
    main()
