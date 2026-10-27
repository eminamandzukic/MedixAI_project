import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_utils import load_all, save_fig, PALETTE, MODEL_LABELS, MODEL_ORDER


def _draw_performance_plot(base, kind):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    metrics = [
        ("orig_auc",       "Test AUC"),
        ("orig_composite", "Test Composite Score"),
    ]

    for ax, (col, ylabel) in zip(axes, metrics):
        plot_df = base.copy()
        plot_df["model"] = plot_df["model_name"].map(MODEL_LABELS)
        model_order = [MODEL_LABELS[m] for m in MODEL_ORDER
                       if MODEL_LABELS[m] in plot_df["model"].unique()]

        if kind == "violin":
            sns.violinplot(
                data=plot_df, x="model", y=col, hue="dataset",
                order=model_order, palette=PALETTE,
                linewidth=0.8, inner="quartile", density_norm="width", ax=ax,
            )
        else:
            sns.boxplot(
                data=plot_df, x="model", y=col, hue="dataset",
                order=model_order, palette=PALETTE,
                linewidth=0.8, fliersize=2, ax=ax,
            )

        ax.axhline(0, linestyle="--", linewidth=0.7, color="grey", alpha=0.5)
        ax.set_xlabel("Model")
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
        if ax == axes[0]:
            ax.get_legend().remove()
        else:
            ax.legend(title="Dataset", loc="lower right", framealpha=0.7, fontsize=8)

    fig.suptitle("Learning Performance by Dataset and Model", fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig(f"fig04_performance_{kind}")


def fig04_performance_plots(base):
    _draw_performance_plot(base, "violin")
    _draw_performance_plot(base, "boxplot")


def fig05_performance_vs_uncertainty(base_merged):
    if "u_alea_mean" not in base_merged.columns:
        print("  [SKIP] fig05 — no uncertainty data.")
        return

    plot_df = base_merged.dropna(subset=["u_alea_mean", "orig_composite"]).copy()
    plot_df["model"] = plot_df["model_name"].map(MODEL_LABELS)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=plot_df,
        x="u_alea_mean", y="orig_composite",
        hue="dataset", style="model",
        palette=PALETTE, alpha=0.75, s=70, ax=ax,
    )
    ax.annotate(
        "Ideal region\n(high perf, low unc)",
        xy=(plot_df["u_alea_mean"].min() + 0.005, plot_df["orig_composite"].max() - 0.01),
        fontsize=8, color="#2e7d32",
        arrowprops=dict(arrowstyle="->", color="#2e7d32"),
        xytext=(plot_df["u_alea_mean"].min() + 0.04, plot_df["orig_composite"].max() - 0.05),
    )
    ax.set_xlabel("Mean aleatoric uncertainty $\\bar{U}_{\\mathrm{alea}}$")
    ax.set_ylabel("Test composite score")
    ax.set_title("Learning Performance vs Aleatoric Uncertainty", fontweight="bold", pad=12)
    ax.legend(title="", loc="lower left", framealpha=0.7, fontsize=8)
    plt.tight_layout()
    save_fig("fig05_performance_vs_uncertainty")


def main():
    print("Q2 — Learning Capability")
    _, base, _, _, base_merged = load_all()
    fig04_performance_plots(base)
    fig05_performance_vs_uncertainty(base_merged)
    print("Done.")


if __name__ == "__main__":
    main()
