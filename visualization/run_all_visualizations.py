"""
visualization/run_all_visualizations.py
=========================================
Run all five visualization scripts in sequence.

Usage
-----
    python3 visualization/run_all_visualizations.py

Output
------
All 14+ figures written to visualization/figures/
"""
import importlib.util
import sys
from pathlib import Path

SCRIPTS = [
    "q1_dataset_nature.py",
    "q2_learning_capability.py",
    "q3_feature_engineering.py",
    "q4_model_selection.py",
    "q5_calibration.py",
]

VIZ_DIR = Path(__file__).resolve().parent


def run_script(name: str):
    path = VIZ_DIR / name
    print(f"\n{'#' * 60}")
    print(f"# Running: {name}")
    print(f"{'#' * 60}")
    spec = importlib.util.spec_from_file_location("_viz", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


def main():
    print("=" * 60)
    print("RUNNING ALL VISUALIZATION SCRIPTS")
    print("=" * 60)
    for script in SCRIPTS:
        run_script(script)
    print(f"\n{'=' * 60}")
    print(f"ALL DONE — figures in visualization/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
