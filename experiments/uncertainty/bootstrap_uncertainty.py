import json
import sys
from pathlib import Path
from copy import deepcopy
import argparse

import numpy as np
from sklearn.utils import resample

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "calibration"))

from experiment_utils import load_dataset, make_splits_for, get_results_dir
from retrain_and_save import _build_pipeline

M_BOOTSTRAP = 30
EPSILON     = 1e-10
MODEL_NAMES = ["logistic_regression", "svm", "random_forest", "mlp"]


def _binary_entropy(p):
    p = np.clip(p, EPSILON, 1.0 - EPSILON)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def _scan_best(model_names, results_dir):
    best = {m: {} for m in model_names}

    for model_name in model_names:
        model_dir = results_dir / model_name
        if not model_dir.is_dir():
            print(f"  [WARN] No results found for {model_name}, skipping.")
            continue

        for exp_dir in sorted(model_dir.iterdir()):
            if not (exp_dir.is_dir() and exp_dir.name.startswith("exp_")):
                continue
            cfg_path    = exp_dir / "config.json"
            scores_path = exp_dir / "scores.json"
            if not (cfg_path.exists() and scores_path.exists()):
                continue

            with open(cfg_path)    as f: config = json.load(f)
            with open(scores_path) as f: scores = json.load(f)

            subset = config["feature_subset_name"]
            comp   = scores.get("val_composite", -999)

            if subset not in best[model_name] or comp > best[model_name][subset]["val_composite"]:
                best[model_name][subset] = {
                    "exp_idx"       : config["exp_idx"],
                    "exp_dir"       : str(exp_dir),
                    "val_composite" : comp,
                    "val_auc"       : scores.get("val_auc"),
                    "test_auc"      : scores.get("test_auc"),
                    "config"        : config,
                    "source"        : "base_experiment",
                }

            cal_root = exp_dir / "calibrated"
            if cal_root.exists():
                for method_dir in cal_root.iterdir():
                    cal_s = method_dir / "scores.json"
                    if not cal_s.exists():
                        continue
                    with open(cal_s) as f: cs = json.load(f)
                    cal_comp = cs.get("after_auc", -999)
                    if cal_comp > best[model_name][subset]["val_composite"]:
                        best[model_name][subset] = {
                            "exp_idx"       : config["exp_idx"],
                            "exp_dir"       : str(exp_dir),
                            "val_composite" : cal_comp,
                            "val_auc"       : cal_comp,
                            "test_auc"      : cs.get("after_auc"),
                            "config"        : config,
                            "source"        : f"calibrated/{method_dir.name}",
                        }

    return best


def _bootstrap_probs(model_name, hp, X_train, y_train, X_test, M=M_BOOTSTRAP, seed=42):
    probs = np.zeros((M, len(X_test)))
    rng   = np.random.RandomState(seed)
    for m in range(M):
        X_b, y_b = resample(X_train, y_train, replace=True,
                             n_samples=len(X_train),
                             random_state=rng.randint(0, 2**31))
        pipe = _build_pipeline(model_name, deepcopy(hp))
        pipe.fit(X_b, y_b)
        probs[m] = pipe.predict_proba(X_test)[:, 1]
    return probs


def _decompose(probs):
    p_mean  = probs.mean(axis=0)
    u_total = _binary_entropy(p_mean)
    u_alea  = _binary_entropy(probs).mean(axis=0)
    u_epis  = np.maximum(u_total - u_alea, 0.0)
    var_p   = ((probs - p_mean[None, :]) ** 2).mean(axis=0)

    u_t = float(u_total.mean())
    u_a = float(u_alea.mean())
    u_e = float(u_epis.mean())
    v_p = float(var_p.mean())

    return {
        "per_sample": {
            "p_mean"  : p_mean.tolist(),
            "u_total" : u_total.tolist(),
            "u_alea"  : u_alea.tolist(),
            "u_epis"  : u_epis.tolist(),
            "var_p"   : var_p.tolist(),
        },
        "dataset_level": {
            "u_total_mean" : u_t,
            "u_alea_mean"  : u_a,
            "u_epis_mean"  : u_e,
            "var_p_mean"   : v_p,
            "r_epis"       : u_e / (u_t + EPSILON),
            "r_alea"       : u_a / (u_t + EPSILON),
            "r_check"      : (u_a + u_e) / (u_t + EPSILON),
        },
    }


def run(dataset="pima"):
    results_dir   = get_results_dir(dataset)
    dataset_label = "breast_cancer_wisconsin" if dataset == "breast_cancer" else "pima_indians_diabetes"
    out_base      = results_dir / "uncertainty_decomposition"

    print("ALEATORIC + EPISTEMIC UNCERTAINTY DECOMPOSITION")
    print(f"Dataset: {dataset_label}")

    print("\nScanning experiments...")
    best = _scan_best(MODEL_NAMES, results_dir)

    for model_name in MODEL_NAMES:
        print(f"\n  [{model_name}]")
        for subset, info in sorted(best[model_name].items()):
            print(f"    {subset:<35} exp_{info['exp_idx']:04d}  "
                  f"composite={info['val_composite']:.4f}  ({info['source']})")

    X_full, y_full, _ = load_dataset(dataset)
    overall_summary   = []

    for model_name in MODEL_NAMES:
        model_out = out_base / model_name
        model_out.mkdir(parents=True, exist_ok=True)

        model_summary = []
        subsets       = best[model_name]
        if not subsets:
            print(f"\n[SKIP] {model_name} - no experiments found.")
            continue

        print(f"\n# {model_name.upper()}  ({len(subsets)} subsets)")

        for subset_name, info in sorted(subsets.items()):
            config   = info["config"]
            seed     = config["seed"]
            features = config["features"]
            hp       = config["hyperparameters"]

            splits_full = make_splits_for(X_full, y_full, dataset=dataset, seed=seed)
            splits = {s: (df[features], lbl) for s, (df, lbl) in splits_full.items()}
            X_train, y_train = splits["train"]
            X_test,  y_test  = splits["test"]

            print(f"\n  {subset_name:<35} M={M_BOOTSTRAP} bootstraps...", end=" ", flush=True)

            probs  = _bootstrap_probs(model_name, hp, X_train, y_train, X_test, seed=seed)
            decomp = _decompose(probs)
            dl     = decomp["dataset_level"]

            print(f"U_total={dl['u_total_mean']:.4f}  "
                  f"U_alea={dl['u_alea_mean']:.4f}  "
                  f"U_epis={dl['u_epis_mean']:.4f}  "
                  f"R_epis={dl['r_epis']:.3f}")

            result = {
                "model_name"      : model_name,
                "subset_name"     : subset_name,
                "best_exp_idx"    : info["exp_idx"],
                "best_exp_dir"    : info["exp_dir"],
                "val_composite"   : info["val_composite"],
                "feature_family"  : config["feature_family"],
                "features"        : features,
                "n_features"      : len(features),
                "hyperparameters" : hp,
                "seed"            : seed,
                "M_bootstrap"     : M_BOOTSTRAP,
                "n_train"         : int(len(X_train)),
                "n_test"          : int(len(X_test)),
                "y_test"          : y_test.tolist(),
                "source"          : info["source"],
                "uncertainty"     : decomp,
            }

            out_path = model_out / f"{subset_name}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2, default=str)

            row = {
                "model_name"     : model_name,
                "subset_name"    : subset_name,
                "feature_family" : config["feature_family"],
                "n_features"     : len(features),
                "val_composite"  : info["val_composite"],
                **dl,
            }
            model_summary.append(row)
            overall_summary.append(row)

        with open(model_out / "summary.json", "w") as f:
            json.dump({"model_name": model_name, "M_bootstrap": M_BOOTSTRAP, "subsets": model_summary},
                      f, indent=2, default=str)
        print(f"\n  {model_name}: saved {len(model_summary)} subsets -> {model_out}")

    with open(out_base / "overall_summary.json", "w") as f:
        json.dump({"rows": overall_summary}, f, indent=2, default=str)

    try:
        import pandas as _pd
        unc_csv_path = out_base / "uncertainty_results.csv"
        _pd.DataFrame(overall_summary).to_csv(unc_csv_path, index=False, float_format="%.6f")
        print(f"  CSV -> {unc_csv_path}")
    except Exception as e:
        print(f"  [WARN] Could not write uncertainty CSV: {e}")

    print(f"\nDONE. {len(overall_summary)} total (model x subset) results.")
    print(f"Results -> {out_base}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="pima", choices=["pima", "breast_cancer"])
    run(dataset=p.parse_args().dataset)
