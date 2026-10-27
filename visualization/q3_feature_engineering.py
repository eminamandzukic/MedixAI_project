import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_utils import load_all, save_fig, PALETTE, FAMILY_LABELS, apply_family_labels


def _subset_summary(base):
    return base.groupby(
        ["dataset", "feature_subset_name", "feature_family", "n_features"],
        as_index=False,
    ).agg(
        mean_composite=("orig_composite", "mean"),
        mean_auc      =("orig_auc",       "mean"),
        mean_ece      =("orig_ece",       "mean"),
        mean_entropy  =("orig_entropy",   "mean"),
    )


def _subset_unc_summary(base_merged):
    if "u_alea_mean" not in base_merged.columns:
        return pd.DataFrame()
    group_cols = ["dataset", "feature_subset_name", "n_features"]
    if "feature_family" in base_merged.columns:
        group_cols.insert(2, "feature_family")
    return base_merged.dropna(subset=["u_alea_mean"]).groupby(
        group_cols, as_index=False,
    ).agg(
        mean_composite=("orig_composite", "mean"),
        mean_u_alea   =("u_alea_mean",    "mean"),
        mean_u_epis   =("u_epis_mean",    "mean"),
    )


def fig06_features_vs_performance(base):
    ss = _subset_summary(base)
    ss["Family"] = apply_family_labels(ss)

    datasets = ss["dataset"].unique()
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(7 * n_ds, 5), sharey=True)
    if n_ds == 1:
        axes = [axes]

    for ax, ds in zip(axes, sorted(datasets)):
        plot_df = ss[ss["dataset"] == ds]
        sns.scatterplot(
            data=plot_df, x="n_features", y="mean_composite",
            hue="Family", style="Family", s=110, alpha=0.85, ax=ax,
        )
        if len(plot_df) > 2:
            z = np.polyfit(plot_df["n_features"], plot_df["mean_composite"], 1)
            xs = sorted(plot_df["n_features"].unique())
            ax.plot(xs, np.poly1d(z)(xs), linestyle="--", linewidth=1, color="grey", alpha=0.6)
        ax.set_title(ds, fontsize=11)
        ax.set_xlabel("Number of features")
        ax.set_ylabel("Mean composite score" if ax == axes[0] else "")
        ax.legend(title="Feature family", fontsize=8, framealpha=0.7)

    fig.suptitle("Feature Subset Size vs Learning Performance", fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig("fig06_features_vs_performance")


def fig07_features_vs_uncertainty(base_merged):
    sus = _subset_unc_summary(base_merged)
    if sus.empty:
        print("  [SKIP] fig07 — no uncertainty data.")
        return

    sus["Family"] = apply_family_labels(sus)
    datasets = sus["dataset"].unique()
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(7 * n_ds, 5), sharey=True)
    if n_ds == 1:
        axes = [axes]

    for ax, ds in zip(axes, sorted(datasets)):
        plot_df = sus[sus["dataset"] == ds]
        sns.scatterplot(
            data=plot_df, x="n_features", y="mean_u_alea",
            hue="Family", style="Family", s=110, alpha=0.85, ax=ax,
        )
        ax.set_title(ds, fontsize=11)
        ax.set_xlabel("Number of features")
        ax.set_ylabel("Mean aleatoric uncertainty $\\bar{U}_{\\mathrm{alea}}$" if ax == axes[0] else "")
        ax.legend(title="Feature family", fontsize=8, framealpha=0.7)

    fig.suptitle("Feature Subset Size vs Aleatoric Uncertainty", fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig("fig07_features_vs_uncertainty")


def fig08_pareto_feature_engineering(base_merged):
    sus = _subset_unc_summary(base_merged)
    if sus.empty:
        print("  [SKIP] fig08 — no uncertainty data.")
        return

    sus["Family"] = apply_family_labels(sus)
    sus["Dataset"] = sus["dataset"].apply(lambda x: "Pima" if "Pima" in x else "BC")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=sus,
        x="mean_u_alea", y="mean_composite",
        hue="Family", style="Dataset",
        size="n_features", sizes=(60, 350),
        alpha=0.8, ax=ax,
    )
    ax.annotate(
        "Pareto frontier\n(upper-left)",
        xy=(sus["mean_u_alea"].min() + 0.005, sus["mean_composite"].max() - 0.01),
        fontsize=8, color="#1b5e20",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f5e9", alpha=0.8),
    )
    ax.set_xlabel("Mean aleatoric uncertainty $\\bar{U}_{\\mathrm{alea}}$")
    ax.set_ylabel("Mean composite score")
    ax.set_title("Feature Engineering Pareto Plot", fontweight="bold", pad=12)
    ax.legend(loc="lower left", fontsize=8, framealpha=0.7, ncol=2)
    plt.tight_layout()
    save_fig("fig08_pareto_feature_engineering")


def main():
    print("Q3 — Feature Engineering")
    _, base, _, _, base_merged = load_all()
    fig06_features_vs_performance(base)
    fig07_features_vs_uncertainty(base_merged)
    fig08_pareto_feature_engineering(base_merged)
    print("Done.")


if __name__ == "__main__":
    main()
