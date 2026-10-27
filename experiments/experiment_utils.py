import json
import math
import pickle
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score,
    precision_score, accuracy_score, brier_score_loss, log_loss,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results" / "pima_indians"

FEATURE_SUBSETS_JSON    = PROJECT_ROOT / "feature_extraction" / "feature_subsets_pima.json"
FEATURE_SUBSETS_BC_JSON = PROJECT_ROOT / "feature_extraction" / "feature_subsets_breast_cancer.json"

COARSE_FEATURE_FAMILIES = {"A", "B", "C", "D"}

PIMA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
)
PIMA_COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]
BIOLOGICALLY_INVALID_ZERO_COLS = [
    "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
]


def load_pima():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pima_path = DATA_DIR / "pima_indians_diabetes.csv"
    if not pima_path.exists():
        print(f"  Downloading Pima dataset -> {pima_path}")
        urllib.request.urlretrieve(PIMA_URL, pima_path)
    df = pd.read_csv(pima_path, header=None, names=PIMA_COLUMNS)
    df[BIOLOGICALLY_INVALID_ZERO_COLS] = df[BIOLOGICALLY_INVALID_ZERO_COLS].replace(0, np.nan)
    feature_cols = PIMA_COLUMNS[:-1]
    X = df[feature_cols].copy()
    y = df["Outcome"].values
    return X, y, feature_cols


def load_breast_cancer():
    from sklearn.datasets import load_breast_cancer as _load_bc
    data = _load_bc()
    X = pd.DataFrame(data.data, columns=list(data.feature_names))
    y = data.target
    return X, y, list(data.feature_names)


def load_dataset(dataset="pima"):
    if dataset == "pima":
        return load_pima()
    elif dataset == "breast_cancer":
        return load_breast_cancer()
    else:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose 'pima' or 'breast_cancer'.")


def load_feature_subsets():
    with open(FEATURE_SUBSETS_JSON) as f:
        data = json.load(f)
    return data["feature_subsets"]


def load_feature_subsets_for(dataset="pima"):
    path = FEATURE_SUBSETS_JSON if dataset == "pima" else FEATURE_SUBSETS_BC_JSON
    if dataset not in ("pima", "breast_cancer"):
        raise ValueError(f"Unknown dataset '{dataset}'.")
    with open(path) as f:
        data = json.load(f)
    return data["feature_subsets"]


def make_splits(X, y, seed=42):
    # 60/15/15/10 split with median imputation fitted on train only
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=seed
    )
    X_train, X_rest, y_train, y_rest = train_test_split(
        X_pool, y_pool, test_size=(30 / 90), stratify=y_pool, random_state=seed
    )
    X_val, X_cal, y_val, y_cal = train_test_split(
        X_rest, y_rest, test_size=0.50, stratify=y_rest, random_state=seed
    )
    train_medians = X_train.median()
    def impute(df):
        return df.fillna(train_medians)
    return {
        "train": (impute(X_train), y_train),
        "val"  : (impute(X_val),   y_val),
        "cal"  : (impute(X_cal),   y_cal),
        "test" : (impute(X_test),  y_test),
    }


def make_splits_bc(X, y, seed=42):
    # 60/15/15/10 split, no imputation needed
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=seed
    )
    X_train, X_rest, y_train, y_rest = train_test_split(
        X_pool, y_pool, test_size=(30 / 90), stratify=y_pool, random_state=seed
    )
    X_val, X_cal, y_val, y_cal = train_test_split(
        X_rest, y_rest, test_size=0.50, stratify=y_rest, random_state=seed
    )
    return {
        "train": (X_train, y_train),
        "val"  : (X_val,   y_val),
        "cal"  : (X_cal,   y_cal),
        "test" : (X_test,  y_test),
    }


def make_splits_for(X, y, dataset="pima", seed=42):
    if dataset == "pima":
        return make_splits(X, y, seed)
    elif dataset == "breast_cancer":
        return make_splits_bc(X, y, seed)
    else:
        raise ValueError(f"Unknown dataset '{dataset}'.")


def _ece_mce(y_true, y_prob, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        gap  = abs(acc - conf)
        ece += (mask.sum() / n) * gap
        mce  = max(mce, gap)
    return float(ece), float(mce)


def _mean_entropy(y_prob, eps=1e-12):
    p = np.clip(y_prob, eps, 1 - eps)
    H = -p * np.log(p) - (1 - p) * np.log(1 - p)
    return float(H.mean())


def compute_scores(y_true, y_prob, split_name="val", n_bins=10, lambda1=0.5, lambda2=0.1):
    y_pred    = (y_prob >= 0.5).astype(int)
    auc       = float(roc_auc_score(y_true, y_prob))
    f1        = float(f1_score(y_true, y_pred, zero_division=0))
    recall    = float(recall_score(y_true, y_pred, zero_division=0))
    prec      = float(precision_score(y_true, y_pred, zero_division=0))
    acc       = float(accuracy_score(y_true, y_pred))
    brier     = float(brier_score_loss(y_true, y_prob))
    nll       = float(log_loss(y_true, y_prob))
    ece, mce  = _ece_mce(y_true, y_prob, n_bins)
    entropy   = _mean_entropy(y_prob)
    composite = auc - lambda1 * ece - lambda2 * nll
    return {
        f"{split_name}_auc"      : auc,
        f"{split_name}_f1"       : f1,
        f"{split_name}_recall"   : recall,
        f"{split_name}_precision": prec,
        f"{split_name}_accuracy" : acc,
        f"{split_name}_brier"    : brier,
        f"{split_name}_nll"      : nll,
        f"{split_name}_ece"      : ece,
        f"{split_name}_mce"      : mce,
        f"{split_name}_entropy"  : entropy,
        f"{split_name}_composite": composite,
    }


def get_results_dir(dataset="pima"):
    if dataset == "pima":
        return RESULTS_DIR
    elif dataset == "breast_cancer":
        return PROJECT_ROOT / "results" / "breast_cancer"
    else:
        raise ValueError(f"Unknown dataset '{dataset}'.")


def save_experiment(model_name, exp_idx, model, config, scores, save_model=False, results_dir=None):
    root    = results_dir if results_dir is not None else RESULTS_DIR
    exp_dir = root / model_name / f"exp_{exp_idx:04d}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    if save_model:
        with open(exp_dir / "model.pkl", "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)
    with open(exp_dir / "scores.json", "w") as f:
        json.dump(scores, f, indent=2, default=str)
    return exp_dir


def save_experiment_for(dataset, model_name, exp_idx, model, config, scores, save_model=False):
    return save_experiment(
        model_name, exp_idx, model, config, scores,
        save_model=save_model,
        results_dir=get_results_dir(dataset),
    )


def next_exp_idx(model_name):
    base = RESULTS_DIR / model_name
    base.mkdir(parents=True, exist_ok=True)
    existing = [
        int(d.name.split("_")[1])
        for d in base.iterdir()
        if d.is_dir() and d.name.startswith("exp_")
    ]
    return max(existing, default=-1) + 1


def next_exp_idx_for(model_name, dataset="pima"):
    base = get_results_dir(dataset) / model_name
    base.mkdir(parents=True, exist_ok=True)
    existing = [
        int(d.name.split("_")[1])
        for d in base.iterdir()
        if d.is_dir() and d.name.startswith("exp_")
    ]
    return max(existing, default=-1) + 1


def load_experiment_splits(config_path):
    config_path = Path(config_path)
    with open(config_path) as f:
        config = json.load(f)
    seed     = config["seed"]
    features = config["features"]
    X_full, y_full, _ = load_pima()
    splits_full = make_splits(X_full, y_full, seed=seed)
    return {s: (df[features], lbl) for s, (df, lbl) in splits_full.items()}


def load_experiment_splits_for(config_path):
    config_path = Path(config_path)
    with open(config_path) as f:
        config = json.load(f)
    seed     = config["seed"]
    features = config["features"]
    dataset  = config.get("dataset", "pima_indians_diabetes")
    if dataset in ("pima_indians_diabetes", "pima"):
        dataset = "pima"
    X_full, y_full, _ = load_dataset(dataset)
    splits_full = make_splits_for(X_full, y_full, dataset=dataset, seed=seed)
    return {s: (df[features], lbl) for s, (df, lbl) in splits_full.items()}


def load_experiment(exp_dir):
    exp_dir = Path(exp_dir)
    with open(exp_dir / "model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(exp_dir / "config.json") as f:
        config = json.load(f)
    with open(exp_dir / "scores.json") as f:
        scores = json.load(f)
    splits = load_experiment_splits(exp_dir / "config.json")
    return model, config, scores, splits
