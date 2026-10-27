import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression as _SklearnIR

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiment_utils import RESULTS_DIR, compute_scores


class PlattCalibrator:
    def __init__(self):
        self._lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)

    def fit(self, p, y):
        self._lr.fit(p.reshape(-1, 1), y)
        return self

    def transform(self, p):
        return self._lr.predict_proba(p.reshape(-1, 1))[:, 1]

    def params(self):
        return {
            "A": float(self._lr.coef_[0][0]),
            "B": float(self._lr.intercept_[0]),
        }


class IsotonicCalibrator:
    def __init__(self):
        self._ir = _SklearnIR(out_of_bounds="clip")

    def fit(self, p, y):
        self._ir.fit(p, y)
        return self

    def transform(self, p):
        return self._ir.predict(p).clip(0.0, 1.0)


class HistogramBinningCalibrator:
    def __init__(self, n_bins=10, strategy="uniform"):
        self.n_bins   = n_bins
        self.strategy = strategy
        self._edges   = None
        self._freqs   = None

    def fit(self, p, y):
        if self.strategy == "uniform":
            self._edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        else:
            pcts = np.linspace(0, 100, self.n_bins + 1)
            self._edges = np.percentile(p, pcts)
            self._edges[0]  = 0.0
            self._edges[-1] = 1.0 + 1e-8

        self._freqs = np.zeros(self.n_bins)
        for b in range(self.n_bins):
            mask = (p >= self._edges[b]) & (p < self._edges[b + 1])
            self._freqs[b] = float(y[mask].mean()) if mask.sum() > 0 \
                else (self._edges[b] + self._edges[b + 1]) / 2.0
        return self

    def transform(self, p):
        out = np.empty_like(p)
        for b in range(self.n_bins):
            mask = (p >= self._edges[b]) & (p < self._edges[b + 1])
            out[mask] = self._freqs[b]
        out[p >= self._edges[-1]] = self._freqs[-1]
        return out.clip(0.0, 1.0)


def _build_scores(y_test, p_raw, p_cal, raw_scores):
    after = compute_scores(y_test, p_cal, split_name="after")
    keys  = ["auc", "f1", "recall", "precision", "accuracy",
             "brier", "nll", "ece", "mce", "entropy", "composite"]
    out = {}
    for k in keys:
        out[f"before_{k}"] = raw_scores.get(f"test_{k}")
        out[f"after_{k}"]  = after.get(f"after_{k}")
    for k in keys:
        b, a = out.get(f"before_{k}"), out.get(f"after_{k}")
        if b is not None and a is not None:
            out[f"delta_{k}"] = a - b
    out["uncertainty_before"] = {k: out[f"before_{k}"]
                                  for k in ["entropy", "ece", "mce", "brier", "nll"]}
    out["uncertainty_after"]  = {k: out[f"after_{k}"]
                                  for k in ["entropy", "ece", "mce", "brier", "nll"]}
    out["calibration_improvement"] = {
        "ece_delta":     out.get("delta_ece"),
        "brier_delta":   out.get("delta_brier"),
        "entropy_delta": out.get("delta_entropy"),
        "note": "Negative delta = improvement.",
    }
    return out


def _save(exp_dir, method, config, scores, save_model=False, wrapper=None):
    out = exp_dir / "calibrated" / method
    out.mkdir(parents=True, exist_ok=True)
    if save_model and wrapper is not None:
        with open(out / "calibrated_model.pkl", "wb") as f:
            pickle.dump(wrapper, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)
    with open(out / "scores.json", "w") as f:
        json.dump(scores, f, indent=2, default=str)


def calibrate_and_save_all(model, splits, exp_dir, raw_scores, base_config, save_model=False):
    X_cal,  y_cal  = splits["cal"]
    X_test, y_test = splits["test"]

    p_cal_raw  = model.predict_proba(X_cal)[:, 1]
    p_test_raw = model.predict_proba(X_test)[:, 1]

    def _base_cfg(method, extra):
        return {
            "calibration_method"  : method,
            "base_experiment_dir" : str(exp_dir),
            "base_model"          : base_config["model"],
            "feature_subset_name" : base_config["feature_subset_name"],
            "feature_family"      : base_config["feature_family"],
            "features"            : base_config["features"],
            "n_features"          : base_config["n_features"],
            "hyperparameters"     : base_config["hyperparameters"],
            "seed"                : base_config["seed"],
            "dataset"             : base_config["dataset"],
            "n_cal"               : int(len(X_cal)),
            "n_test"              : int(len(X_test)),
            "method_params"       : extra,
        }

    # Platt Scaling
    cal = PlattCalibrator().fit(p_cal_raw, y_cal)
    p   = cal.transform(p_test_raw)
    _save(exp_dir, "platt_scaling",
          _base_cfg("platt_scaling", cal.params()),
          _build_scores(y_test, p_test_raw, p, raw_scores))

    # Isotonic Regression
    cal = IsotonicCalibrator().fit(p_cal_raw, y_cal)
    p   = cal.transform(p_test_raw)
    _save(exp_dir, "isotonic_regression",
          _base_cfg("isotonic_regression", {}),
          _build_scores(y_test, p_test_raw, p, raw_scores))

    # Histogram Binning (6 variants)
    for n_bins in [5, 10, 15]:
        for strategy in ["uniform", "quantile"]:
            name = f"histogram_binning_B{n_bins}_{strategy}"
            cal  = HistogramBinningCalibrator(n_bins, strategy).fit(p_cal_raw, y_cal)
            p    = cal.transform(p_test_raw)
            _save(exp_dir, name,
                  _base_cfg(name, {"n_bins": n_bins, "strategy": strategy,
                                   "bin_edges": cal._edges.tolist(),
                                   "bin_freqs": cal._freqs.tolist()}),
                  _build_scores(y_test, p_test_raw, p, raw_scores))


class CalibrationWrapper:
    def __init__(self, base_pipeline, calibrator, method_name):
        self.base_pipeline = base_pipeline
        self.calibrator    = calibrator
        self.method_name   = method_name

    def predict_proba_raw(self, X):
        return self.base_pipeline.predict_proba(X)[:, 1]

    def predict_proba_calibrated(self, X):
        return self.calibrator.transform(self.predict_proba_raw(X))


def iter_experiments(model_names=None):
    dirs = (sorted(d for d in RESULTS_DIR.iterdir() if d.is_dir())
            if model_names is None
            else [RESULTS_DIR / n for n in model_names])
    for md in dirs:
        if not md.is_dir():
            continue
        for ed in sorted(md.iterdir()):
            if (ed.is_dir() and ed.name.startswith("exp_")
                    and (ed / "config.json").exists()
                    and (ed / "scores.json").exists()):
                yield md.name, ed


def load_experiment(exp_dir):
    from experiment_utils import load_pima, make_splits
    exp_dir = Path(exp_dir)
    with open(exp_dir / "config.json") as f:
        config = json.load(f)
    with open(exp_dir / "scores.json") as f:
        scores = json.load(f)
    model = None
    if (exp_dir / "model.pkl").exists():
        with open(exp_dir / "model.pkl", "rb") as f:
            model = pickle.load(f)
    X_full, y_full, _ = load_pima()
    splits_full = make_splits(X_full, y_full, seed=config["seed"])
    features = config["features"]
    splits = {s: (df[features], lbl) for s, (df, lbl) in splits_full.items()}
    return model, config, scores, splits


def save_calibration(exp_dir, method_name, cal_wrapper, cal_config, cal_scores, save_model=False):
    _save(Path(exp_dir), method_name, cal_config, cal_scores,
          save_model=save_model, wrapper=cal_wrapper)
    return Path(exp_dir) / "calibrated" / method_name


# Aliases kept for backward compatibility
build_calibration_scores = _build_scores
