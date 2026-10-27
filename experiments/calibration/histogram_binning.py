import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_utils import load_experiment
from calibration_utils import CalibrationWrapper, iter_experiments, save_calibration, build_calibration_scores

N_BINS_VALUES = [5, 10, 15]
STRATEGIES    = ["uniform", "quantile"]


class HistogramBinningCalibrator:
    def __init__(self, n_bins=10, strategy="uniform"):
        self.n_bins    = n_bins
        self.strategy  = strategy
        self._bin_edges = None
        self._bin_freqs = None

    def fit(self, p_raw, y):
        if self.strategy == "uniform":
            self._bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)
        else:
            percentiles = np.linspace(0, 100, self.n_bins + 1)
            self._bin_edges = np.percentile(p_raw, percentiles)
            self._bin_edges[0]  = 0.0
            self._bin_edges[-1] = 1.0 + 1e-8

        self._bin_freqs = np.zeros(self.n_bins)
        for b in range(self.n_bins):
            lo, hi = self._bin_edges[b], self._bin_edges[b + 1]
            mask   = (p_raw >= lo) & (p_raw < hi)
            if mask.sum() > 0:
                self._bin_freqs[b] = float(y[mask].mean())
            else:
                self._bin_freqs[b] = (lo + hi) / 2.0
        return self

    def transform(self, p_raw):
        p_cal = np.empty_like(p_raw)
        for b in range(self.n_bins):
            lo, hi = self._bin_edges[b], self._bin_edges[b + 1]
            mask   = (p_raw >= lo) & (p_raw < hi)
            p_cal[mask] = self._bin_freqs[b]
        p_cal[p_raw >= self._bin_edges[-1]] = self._bin_freqs[-1]
        return p_cal.clip(0.0, 1.0)


def run(model_names=None):
    print("CALIBRATION METHOD: HISTOGRAM BINNING")
    total = 0
    for n_bins in N_BINS_VALUES:
        for strategy in STRATEGIES:
            method_name = f"histogram_binning_B{n_bins}_{strategy}"
            count = 0
            for model_name, exp_dir in iter_experiments(model_names):
                try:
                    model, config, raw_scores, splits = load_experiment(exp_dir)
                except Exception as exc:
                    print(f"  [SKIP] {exp_dir} - could not load: {exc}")
                    continue

                X_cal,  y_cal  = splits["cal"]
                X_test, y_test = splits["test"]

                p_cal_raw  = model.predict_proba(X_cal)[:, 1]
                p_test_raw = model.predict_proba(X_test)[:, 1]

                calibrator = HistogramBinningCalibrator(n_bins=n_bins, strategy=strategy).fit(p_cal_raw, y_cal)
                p_test_cal = calibrator.transform(p_test_raw)

                cal_scores = build_calibration_scores(y_test, p_test_raw, p_test_cal, raw_scores)
                wrapper    = CalibrationWrapper(model, calibrator, method_name)

                cal_config = {
                    "calibration_method"  : method_name,
                    "base_experiment_dir" : str(exp_dir),
                    "base_model"          : config["model"],
                    "feature_subset_name" : config["feature_subset_name"],
                    "feature_family"      : config["feature_family"],
                    "features"            : config["features"],
                    "n_features"          : config["n_features"],
                    "hyperparameters"     : config["hyperparameters"],
                    "seed"                : config["seed"],
                    "dataset"             : config["dataset"],
                    "n_cal"               : int(len(X_cal)),
                    "n_test"              : int(len(X_test)),
                    "method_params"       : {
                        "n_bins"    : n_bins,
                        "strategy"  : strategy,
                        "bin_edges" : calibrator._bin_edges.tolist(),
                        "bin_freqs" : calibrator._bin_freqs.tolist(),
                    },
                }

                save_calibration(exp_dir, method_name, wrapper, cal_config, cal_scores)
                count += 1
                total += 1

            print(f"  [{method_name}]  {count} experiments calibrated.")

    print(f"\nhistogram_binning: {total} total calibrations saved.\n")


if __name__ == "__main__":
    run()
