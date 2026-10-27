
import os
import json
import random
import itertools
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
PIMA_PATH = DATA_DIR / "pima_indians_diabetes.csv"

PIMA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)

PIMA_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome",
]

print("Downloading Pima Indians Diabetes Dataset")

if not PIMA_PATH.exists():
    print(f"  Downloading from {PIMA_URL} …")
    urllib.request.urlretrieve(PIMA_URL, PIMA_PATH)
    print(f"  Saved to {PIMA_PATH}")
else:
    print(f"  Dataset already present at {PIMA_PATH}")

df = pd.read_csv(PIMA_PATH, header=None, names=PIMA_COLUMNS)
print(f"  Shape: {df.shape}")

print("\nPreprocessing...")

BIOLOGICALLY_INVALID_ZERO_COLS = [
    "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"
]

# Replace invalid zeros with NaN
df_clean = df.copy()
df_clean[BIOLOGICALLY_INVALID_ZERO_COLS] = df_clean[
    BIOLOGICALLY_INVALID_ZERO_COLS
].replace(0, np.nan)

print(f"  Missing values after zero-to-NaN conversion:")
print(df_clean.isnull().sum().to_string())

# Feature matrix and target
feature_cols = PIMA_COLUMNS[:-1]   # 8 features
X_raw = df_clean[feature_cols].copy()
y     = df_clean["Outcome"].values

# Train / (val+cal+test) split — 60 % train, 40 % rest  (stratified)
X_train_raw, X_rest_raw, y_train, y_rest = train_test_split(
    X_raw, y, test_size=0.40, stratify=y, random_state=SEED
)

# Median imputation fitted ONLY on training data
train_medians = X_train_raw.median()
X_train = X_train_raw.fillna(train_medians)
X_rest  = X_rest_raw.fillna(train_medians)
X_full  = X_raw.fillna(train_medians)   # used for MI computation on full data

print("\nComputing Mutual Information scores...")

mi_scores = mutual_info_classif(
    X_train, y_train, discrete_features=False, random_state=SEED
)
mi_series = pd.Series(mi_scores, index=feature_cols).sort_values(ascending=False)

print("\n  MI scores (descending):")
for feat, score in mi_series.items():
    print(f"    {feat:<30s}  MI = {score:.6f}")

print("\nBuilding feature subsets (families A-E)...")

subsets = {}

# Family A: All features (baseline)
subsets["all_features"] = {
    "family"      : "A",
    "description" : "All 8 features — maximum-information baseline.",
    "features"    : feature_cols,
    "n_features"  : len(feature_cols),
    "mi_scores"   : {f: mi_dict[f] for f in feature_cols},
    "selection_method" : "none (baseline)",
}
print("  [A] all_features :", feature_cols)

# Family B: Top-k by Mutual Information
for k in [3, 5, 8]:
    top_k_feats = ranked_features[:k]
    key = f"top_{k}_mi"
    subsets[key] = {
        "family"      : "B",
        "description" : (
            f"Top-{k} features ranked by Mutual Information "
            f"(computed on training split, seed={SEED})."
        ),
        "features"    : top_k_feats,
        "n_features"  : k,
        "mi_scores"   : {f: mi_dict[f] for f in top_k_feats},
        "selection_method" : "mutual_information",
        "k"           : k,
        "ranking"     : [
            {"rank": r + 1, "feature": f, "mi_score": mi_dict[f]}
            for r, f in enumerate(top_k_feats)
        ],
    }
    print(f"  [B] top_{k}_mi       :", top_k_feats)

# Family C: Correlation-Filtered
def correlation_filter(X_train_df, all_cols, mi_scores_dict, threshold):
    """
    Remove one of each highly-correlated pair (|r| > threshold).
    Among the pair, drop the feature with the LOWER MI score.
    Returns list of retained features.
    """
    corr_matrix = X_train_df[all_cols].corr().abs()
    cols = list(all_cols)
    to_drop = set()

    for i, j in itertools.combinations(range(len(cols)), 2):
        fi, fj = cols[i], cols[j]
        if fi in to_drop or fj in to_drop:
            continue
        if corr_matrix.loc[fi, fj] > threshold:
            # Drop the one with lower MI
            if mi_scores_dict[fi] >= mi_scores_dict[fj]:
                to_drop.add(fj)
            else:
                to_drop.add(fi)

    retained = [f for f in cols if f not in to_drop]
    return retained, to_drop


for threshold in [0.80, 0.90, 0.95]:
    retained, dropped = correlation_filter(
        X_train, feature_cols, mi_dict, threshold
    )
    key = f"corr_filtered_{str(threshold).replace('.', '')}"
    subsets[key] = {
        "family"      : "C",
        "description" : (
            f"Correlation-filtered subset at |r| > {threshold}. "
            f"When two features exceed the threshold, the one with "
            f"lower MI is dropped."
        ),
        "features"    : retained,
        "n_features"  : len(retained),
        "dropped_features" : list(dropped),
        "threshold"   : threshold,
        "mi_scores"   : {f: mi_dict[f] for f in retained},
        "selection_method" : "correlation_filter",
    }
    print(f"  [C] corr_filtered_{threshold} : retained={retained}  dropped={list(dropped)}")

# Family D: Domain-Based Groups
domain_groups = {
    "metabolic": {
        "features"    : ["Glucose", "Insulin", "BMI"],
        "description" : "Metabolic markers: Glucose, Insulin, BMI.",
    },
    "cardiovascular": {
        "features"    : ["BloodPressure", "Age"],
        "description" : "Cardiovascular / ageing proxies: BloodPressure, Age.",
    },
    "demographic_pregnancy": {
        "features"    : ["Pregnancies", "Age"],
        "description" : "Demographic / pregnancy history: Pregnancies, Age.",
    },
    "skin_body_composition": {
        "features"    : ["SkinThickness", "BMI"],
        "description" : "Skin / body composition: SkinThickness, BMI.",
    },
    "genetic_risk_proxy": {
        "features"    : ["DiabetesPedigreeFunction", "Age"],
        "description" : "Genetic / risk proxy: DiabetesPedigreeFunction, Age.",
    },
    "metabolic_plus_risk": {
        "features"    : [
            "Glucose", "Insulin", "BMI",
            "DiabetesPedigreeFunction", "Age"
        ],
        "description" : (
            "Combined metabolic + genetic risk: "
            "Glucose, Insulin, BMI, DiabetesPedigreeFunction, Age."
        ),
    },
}

for group_name, group_info in domain_groups.items():
    feats = group_info["features"]
    key   = f"domain_{group_name}"
    subsets[key] = {
        "family"      : "D",
        "description" : group_info["description"],
        "features"    : feats,
        "n_features"  : len(feats),
        "mi_scores"   : {f: mi_dict[f] for f in feats},
        "selection_method" : "domain_knowledge",
    }
    print(f"  [D] domain_{group_name:<25s} : {feats}")

# Family E: Random Subsets (m=10 draws per k)
m = 10
# Valid k values for Pima (same as Family B)
for k in [3, 5, 8]:
    rng = random.Random(SEED + k)   # deterministic but different per k
    for draw_idx in range(m):
        sampled_feats = sorted(rng.sample(feature_cols, k))
        key = f"random_k{k}_{draw_idx:02d}"
        subsets[key] = {
            "family"      : "E",
            "description" : (
                f"Random subset #{draw_idx + 1} of size {k} "
                f"(control group, seed={SEED + k}, draw={draw_idx})."
            ),
            "features"    : sampled_feats,
            "n_features"  : k,
            "mi_scores"   : {f: mi_dict[f] for f in sampled_feats},
            "selection_method" : "random",
            "k"           : k,
            "draw_index"  : draw_idx,
        }
    print(f"  [E] {m} random subsets of k={k}")

print("\nAssembling JSON output...")

output = {
    "metadata": {
        "dataset"           : "Pima Indians Diabetes Database",
        "source_url"        : PIMA_URL,
        "local_path"        : str(PIMA_PATH),
        "n_samples_total"   : int(len(df)),
        "n_features"        : len(feature_cols),
        "feature_names"     : feature_cols,
        "positive_class_ratio" : float(y.mean()),
        "seed"              : SEED,
        "train_size"        : int(len(X_train)),
        "missing_value_handling" : (
            "Biologically-invalid zeros in Glucose, BloodPressure, "
            "SkinThickness, Insulin, BMI replaced with NaN, then "
            "median-imputed using training split only."
        ),
        "mi_ranking_note"   : (
            "Mutual Information scores computed by "
            "sklearn.feature_selection.mutual_info_classif on the "
            "training split (60 % of data, seed=42). "
            "Family B (Top-k) uses this ranking as the primary "
            "feature selection method, as specified in "
            "research_plan Section 3.2."
        ),
    },
    "global_mi_ranking": [
        {"rank": r + 1, "feature": f, "mi_score": float(mi_dict[f])}
        for r, f in enumerate(ranked_features)
    ],
    "feature_subsets": subsets,
    "summary": {
        "total_subsets"   : len(subsets),
        "family_counts"   : {
            fam: sum(1 for v in subsets.values() if v["family"] == fam)
            for fam in ["A", "B", "C", "D", "E"]
        },
    },
}

OUT_PATH = Path(__file__).parent / "feature_subsets_pima.json"
with open(OUT_PATH, "w") as fh:
    json.dump(output, fh, indent=2)

print(f"\n  JSON saved to: {OUT_PATH}")
print(f"\n  Total subsets documented: {len(subsets)}")
print("  Family breakdown:")
for fam, cnt in output["summary"]["family_counts"].items():
    descriptions = {
        "A": "All features",
        "B": "Top-k MI",
        "C": "Correlation-filtered",
        "D": "Domain-based",
        "E": "Random (control)",
    }
    print(f"    Family {fam} ({descriptions[fam]}): {cnt} subsets")

print("\n✓  Done — feature_subsets_pima.json written successfully.")
