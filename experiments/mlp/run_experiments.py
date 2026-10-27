"""
mlp/run_experiments.py
HP grid (3 values per parameter):
  hidden_layer_sizes : (16,), (32,), (64, 32)
  alpha              : 0.0001, 0.001, 0.01
  learning_rate_init : 0.001, 0.005, 0.01
  activation         : relu  (single default for coarse pass)
Total: 3×3×3×1 = 27 configs × N subsets (13 Pima / 21 BC)
Pipeline: train → calibrate all methods → save scores only
Datasets: pima (default) | breast_cancer
"""
import sys
from itertools import product
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calibration"))

from experiment_utils import (
    load_dataset, load_feature_subsets_for, make_splits_for,
    compute_scores, save_experiment_for, next_exp_idx_for,
    COARSE_FEATURE_FAMILIES,
)
from calibration_utils import calibrate_and_save_all

MODEL_NAME = "mlp"
SEEDS      = [42]

HLS_VALUES = [(16,), (32,), (64, 32)]
ALPHA_VALUES = [0.0001, 0.001, 0.01]
LR_VALUES    = [0.001, 0.005, 0.01]


def build_configs():
    return [
        dict(hidden_layer_sizes=hls, alpha=alpha, learning_rate_init=lr,
             activation="relu", early_stopping=True,
             max_iter=500, random_state=42, solver="adam")
        for hls, alpha, lr in product(HLS_VALUES, ALPHA_VALUES, LR_VALUES)
    ]


def run(save_model: bool = False, dataset: str = "pima"):
    print("=" * 70)
    print(f"MODEL: {MODEL_NAME}  |  dataset={dataset}  |  save_model={save_model}")
    print("=" * 70)
    X_full, y_full, _ = load_dataset(dataset)
    feature_subsets = {k: v for k, v in load_feature_subsets_for(dataset).items()
                       if v["family"] in COARSE_FEATURE_FAMILIES}
    hp_configs = build_configs()
    print(f"  HP configs: {len(hp_configs)}  |  Subsets: {len(feature_subsets)}")
    print(f"  Total: {len(hp_configs) * len(feature_subsets) * len(SEEDS)}\n")

    dataset_id = "breast_cancer_wisconsin" if dataset == "breast_cancer" else "pima_indians_diabetes"
    exp_idx = next_exp_idx_for(MODEL_NAME, dataset)
    for seed in SEEDS:
        splits_full = make_splits_for(X_full, y_full, dataset=dataset, seed=seed)
        for subset_name, subset_info in feature_subsets.items():
            features = subset_info["features"]
            splits   = {s: (df[features], lbl) for s, (df, lbl) in splits_full.items()}
            X_train, y_train = splits["train"]
            X_val,   y_val   = splits["val"]
            X_test,  y_test  = splits["test"]

            for hp in hp_configs:
                pipeline = Pipeline([("scaler", StandardScaler()),
                                     ("clf", MLPClassifier(**hp))])
                try:
                    pipeline.fit(X_train, y_train)
                except Exception as exc:
                    print(f"  [SKIP] {subset_name} → {exc}")
                    continue

                scores = {}
                scores.update(compute_scores(y_val,  pipeline.predict_proba(X_val)[:,1],  "val"))
                scores.update(compute_scores(y_test, pipeline.predict_proba(X_test)[:,1], "test"))

                hp_rec = dict(hp)
                hp_rec["hidden_layer_sizes"] = list(hp["hidden_layer_sizes"])
                config = {
                    "model": MODEL_NAME, "exp_idx": exp_idx, "seed": seed,
                    "dataset": dataset_id,
                    "split_ratios": {"train":0.60,"val":0.15,"cal":0.15,"test":0.10},
                    "feature_subset_name": subset_name,
                    "feature_family": subset_info["family"],
                    "feature_description": subset_info.get("description", ""),
                    "features": features, "n_features": len(features),
                    "feature_mi_scores": subset_info.get("mi_scores", {}),
                    "selection_method": subset_info.get("selection_method"),
                    "hyperparameters": hp_rec,
                    "n_train": int(len(X_train)),
                    "n_val":   int(len(X_val)),
                    "n_test":  int(len(X_test)),
                    "search_stage": "coarse",
                }
                exp_dir = save_experiment_for(dataset, MODEL_NAME, exp_idx,
                                              pipeline, config, scores,
                                              save_model=save_model)
                calibrate_and_save_all(pipeline, splits, exp_dir, scores, config)

                print(f"  exp_{exp_idx:04d}  {subset_name:<28}  "
                      f"hls={str(hp['hidden_layer_sizes']):<10}  alpha={hp['alpha']:<7}  "
                      f"val_auc={scores['val_auc']:.4f}")
                exp_idx += 1

    print(f"\n✓  {MODEL_NAME}: {exp_idx} experiments + calibrations done.\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--save-model", action="store_true")
    p.add_argument("--dataset", default="pima", choices=["pima", "breast_cancer"])
    a = p.parse_args()
    run(save_model=a.save_model, dataset=a.dataset)
