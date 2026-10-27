
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import platt_scaling
import isotonic_regression
import histogram_binning

METHODS = {
    "platt"      : platt_scaling,
    "isotonic"   : isotonic_regression,
    "histogram"  : histogram_binning,
}


def main():
    parser = argparse.ArgumentParser(
        description="Run calibration methods over all saved experiments."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHODS.keys()) + ["all"],
        default=["all"],
        help=(
            "Which calibration methods to run. "
            "Options: platt, isotonic, histogram, all. "
            "Default: all."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=(
            "Which base model results to process. "
            "Options: logistic_regression, svm, random_forest, mlp. "
            "Default: all available."
        ),
    )
    args = parser.parse_args()

    selected_methods = (
        list(METHODS.keys()) if "all" in args.methods else args.methods
    )

    for key in selected_methods:
        print(f"\nCalibration: {key.upper()}")
        METHODS[key].run(model_names=args.models)

    print("\nALL CALIBRATION METHODS COMPLETE")


if __name__ == "__main__":
    main()
