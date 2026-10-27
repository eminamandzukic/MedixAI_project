import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_utils import load_all, save_fig, PALETTE, set_style

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
from experiment_utils import load_dataset


def fig01_uncertainty_decomposition(unc_df):
    if unc_df.empty:
        print("  [SKIP] fig01 — no uncertainty data.")
        return

    summary = unc_df.groupby("dataset", as_index=False)[
        ["u_total_mean", "u_alea_mean", "u_epis_mean"]
    ].mean()

    long = summary.melt(
        id_vars="dataset",
        value_vars=["u_total_mean", "u_alea_mean", "u_epis_mean"],
        var_name="component",
        value_name="uncertainty",
    )
    label_map = {
        "u_total_mean": "Total  $U_{\\mathrm{total}}$",
        "u_alea_mean" : "Aleatoric  $U_{\\mathrm{alea}}$",
        "u_epis_mean" : "Epistemic  $U_{\\mathrm{epis}}$",
    }
    long["component"] = long["component"].map(label_map)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=long, x="dataset", y="uncertainty",
        hue="component", palette=["#2196F3", "#FF9800", "#9C27B0"],
        ax=ax, edgecolor="white", linewidth=0.8,
    )
    ax.set_xlabel("Dataset", labelpad=8)
    ax.set_ylabel("Mean uncertainty (entropy scale)", labelpad=8)
    ax.set_title("Dataset-Level Uncertainty Decomposition", fontweight="bold", pad=12)
    ax.legend(title="Component", framealpha=0.7)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8)
    plt.tight_layout()
    save_fig("fig01_uncertainty_decomposition")


def fig02_uncertainty_ratios(unc_df):
    if unc_df.empty:
        print("  [SKIP] fig02 — no uncertainty data.")
        return

    ratio = unc_df.groupby("dataset", as_index=False)[["r_alea", "r_epis"]].mean()
    datasets = ratio["dataset"].tolist()
    x = np.arange(len(datasets))
    width = 0.5

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_alea = ax.bar(x, ratio["r_alea"], width, label="$R_{\\mathrm{alea}}$",
                       color="#FF9800", edgecolor="white")
    bars_epis = ax.bar(x, ratio["r_epis"], width, bottom=ratio["r_alea"],
                       label="$R_{\\mathrm{epis}}$", color="#9C27B0", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Uncertainty ratio")
    ax.set_xlabel("Dataset", labelpad=8)
    ax.set_title("Aleatoric and Epistemic Uncertainty Ratios", fontweight="bold", pad=12)
    ax.legend(framealpha=0.7)
    ax.axhline(1.0, linestyle="--", linewidth=0.8, color="grey")

    for bar, val in zip(bars_alea, ratio["r_alea"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val / 2,
                f"{val:.3f}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    for bar, bot, val in zip(bars_epis, ratio["r_alea"], ratio["r_epis"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bot + val / 2,
                f"{val:.3f}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    plt.tight_layout()
    save_fig("fig02_uncertainty_ratios")


def fig03_correlation_heatmaps():
    for ds_key, ds_label, fname in [
        ("pima",          "Pima Indians Diabetes",  "fig03a_correlation_heatmap_pima"),
        ("breast_cancer", "Breast Cancer Wisconsin", "fig03b_correlation_heatmap_bc"),
    ]:
        X, _, _ = load_dataset(ds_key)
        corr = X.fillna(X.median()).corr()

        short = {c: c.replace("worst ", "w_").replace("mean ", "m_")
                      .replace("error ", "e_").replace(" ", "_")[:14]
                 for c in corr.columns}
        corr_plot = corr.rename(index=short, columns=short)

        n = len(corr.columns)
        fig_size = max(8, n * 0.45)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.9))
        sns.heatmap(
            corr_plot, ax=ax,
            center=0, vmin=-1, vmax=1, cmap="vlag",
            annot=(n <= 12), fmt=".2f" if n <= 12 else "",
            linewidths=0.3 if n <= 12 else 0, square=True,
        )
        ax.set_title(f"Feature Correlation: {ds_label}", fontweight="bold", pad=12)
        ax.tick_params(axis="x", rotation=45)
        ax.tick_params(axis="y", rotation=0)
        plt.tight_layout()
        save_fig(fname)


def main():
    print("Q1 — Dataset Nature")
    _, _, _, unc_df, _ = load_all()
    fig01_uncertainty_decomposition(unc_df)
    fig02_uncertainty_ratios(unc_df)
    fig03_correlation_heatmaps()
    print("Done.")


if __name__ == "__main__":
    main()
