import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_utils import load_all, save_fig, PALETTE, MODEL_LABELS, MODEL_ORDER


def fig09_model_performance_boxplot(base):
    plot_df = base.copy()
    plot_df["Model"] = plot_df["model_name"].map(MODEL_LABELS).fillna(plot_df["model_name"])
    model_order = [MODEL_LABELS[m] for m in MODEL_ORDER if MODEL_LABELS[m] in plot_df["Model"].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    for ax, (col, ylabel) in zip(axes, [
        ("orig_composite", "Composite score  (AUC - 0.5*ECE - 0.1*NLL)"),
        ("orig_auc",       "Test AUC"),
    ]):
        sns.boxplot(
            data=plot_df, x="Model", y=col, hue="dataset",
            order=model_order, palette=PALETTE,
            linewidth=0.8, fliersize=2, ax=ax,
        )
        ax.axhline(0, linestyle="--", linewidth=0.7, color="grey", alpha=0.5)
        ax.set_xlabel("Model")
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
        ax.legend(title="Dataset", fontsize=8, framealpha=0.7)

    fig.suptitle("Model Selection Effect on Learning Performance", fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig("fig09_model_performance_boxplot")


def fig10_model_uncertainty_boxplot(base_merged):
    for col, label, fname, kind in [
        ("u_alea_mean", "Mean aleatoric uncertainty",
         "fig10a_model_uncertainty_alea", "Aleatoric"),
        ("u_epis_mean", "Mean epistemic uncertainty",
         "fig10b_model_uncertainty_epis", "Epistemic"),
    ]:
        if col not in base_merged.columns:
            print(f"  [SKIP] {fname} - column not available.")
            continue

        plot_df = base_merged.dropna(subset=[col]).copy()
        plot_df["Model"] = plot_df["model_name"].map(MODEL_LABELS).fillna(plot_df["model_name"])
        model_order = [MODEL_LABELS[m] for m in MODEL_ORDER if MODEL_LABELS[m] in plot_df["Model"].unique()]

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.boxplot(
            data=plot_df, x="Model", y=col, hue="dataset",
            order=model_order, palette=PALETTE,
            linewidth=0.8, fliersize=2, ax=ax,
        )
        ax.set_xlabel("Model")
        ax.set_ylabel(label)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
        ax.legend(title="Dataset", fontsize=8, framealpha=0.7)
        ax.set_title(f"Model Selection Effect on {kind} Uncertainty", fontweight="bold", pad=12)
        plt.tight_layout()
        save_fig(fname)


def fig11_model_stability_scatter(base):
    stability = base.groupby(["dataset", "model_name"], as_index=False).agg(
        mean_score=("orig_composite", "mean"),
        std_score =("orig_composite", "std"),
        mean_auc  =("orig_auc",       "mean"),
        mean_ece  =("orig_ece",       "mean"),
    )
    stability["Model"] = stability["model_name"].map(MODEL_LABELS).fillna(stability["model_name"])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=stability,
        x="std_score", y="mean_score",
        hue="dataset", style="Model",
        palette=PALETTE, s=180, alpha=0.9, ax=ax,
    )

    for _, row in stability.iterrows():
        ax.annotate(row["Model"], (row["std_score"], row["mean_score"]),
                    xytext=(5, 3), textcoords="offset points", fontsize=7.5, alpha=0.8)

    xmed = stability["std_score"].median()
    ax.axvline(xmed, linestyle=":", linewidth=0.9, color="grey")
    ax.axhline(stability["mean_score"].median(), linestyle=":", linewidth=0.9, color="grey")

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    for txt, x, y in [
        ("Stable\n& weak",      xmin + 0.002, ymin + 0.005),
        ("Best\n& stable",      xmin + 0.002, ymax - 0.015),
        ("Unstable\n& weak",    xmax - 0.002, ymin + 0.005),
        ("Powerful\nbut noisy", xmax - 0.002, ymax - 0.015),
    ]:
        ax.text(x, y, txt, fontsize=7, color="grey",
                ha="left" if x < xmed else "right", alpha=0.7)

    ax.set_xlabel("Std. deviation of composite score")
    ax.set_ylabel("Mean composite score")
    ax.set_title("Model Stability across Configurations", fontweight="bold", pad=12)
    ax.legend(title="", loc="upper left", fontsize=8, framealpha=0.7)
    plt.tight_layout()
    save_fig("fig11_model_stability_scatter")


def main():
    print("Q4 - Model Selection")
    _, base, _, _, base_merged = load_all()
    fig09_model_performance_boxplot(base)
    fig10_model_uncertainty_boxplot(base_merged)
    fig11_model_stability_scatter(base)
    print("Done.")


if __name__ == "__main__":
    main()
