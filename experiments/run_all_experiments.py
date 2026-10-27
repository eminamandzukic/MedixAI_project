
import argparse
import importlib.util
import sys
from pathlib import Path

MODELS = {
    "lr"  : "logistic_regression/run_experiments.py",
    "svm" : "svm/run_experiments.py",
    "rf"  : "random_forest/run_experiments.py",
    "mlp" : "mlp/run_experiments.py",
}

EXPERIMENTS_DIR = Path(__file__).resolve().parent


def load_and_run(rel_path, **kwargs):
    full_path = EXPERIMENTS_DIR / rel_path
    spec = importlib.util.spec_from_file_location("_exp_module", full_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.run(**kwargs)


def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning experiments.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()) + ["all"],
        default=["all"],
        help="Which models to run. Options: lr, svm, rf, mlp, all. Default: all.",
    )
    parser.add_argument(
        "--dataset",
        default="pima",
        choices=["pima", "breast_cancer"],
        help="Dataset to run experiments on. Default: pima.",
    )
    args = parser.parse_args()

    selected = list(MODELS.keys()) if "all" in args.models else args.models

    for key in selected:
        print(f"\nRunning: {key.upper()}  (dataset={args.dataset})")
        load_and_run(MODELS[key], dataset=args.dataset)

    print("\nALL SELECTED EXPERIMENTS COMPLETE")


if __name__ == "__main__":
    main()
