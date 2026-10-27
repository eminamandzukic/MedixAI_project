import sys
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression as _IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from experiment_utils import load_experiment
from calibration_utils import CalibrationWrapper, iter_experiments, save_calibration, build_calibration_scores

METHOD_NAME = "isotonic_regression"


class IsotonicCalibrator:
    def __init__(self):
        self._ir = _IsotonicRegression(out_of_bounds="clip")

    def fit(self, p_raw, y):
        self._ir.fit(p_raw, y)
        return self

    def transform(self, p_raw):
        return self._ir.predict(p_raw).clip(0.0, 1.0)


def run(model_names=None):
    print(f"CALIBRATION METHOD: {METHOD_NAME.upper()}")
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

        calibrator = IsotonicCalibrator().fit(p_cal_raw, y_cal)
        p_test_cal = calibrator.transform(p_test_raw)

        cal_scores = build_calibration_scores(y_test, p_test_raw, p_test_cal, raw_scores)
        wrapper    = CalibrationWrapper(model, calibrator, METHOD_NAME)

        cal_config = {
            "calibration_method"  : METHOD_NAME,
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
            "method_params"       : {},
        }

        save_calibration(exp_dir, METHOD_NAME, wrapper, cal_config, cal_scores)
        print(f"  {model_name}/{exp_dir.name}  "
              f"ece: {raw_scores.get('test_ece', 0):.4f} -> {cal_scores['after_ece']:.4f}  "
              f"brier: {raw_scores.get('test_brier', 0):.4f} -> {cal_scores['after_brier']:.4f}")
        count += 1

    print(f"\n{METHOD_NAME}: {count} experiments calibrated.\n")


if __name__ == "__main__":
    run()
