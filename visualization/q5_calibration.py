import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from viz_utils import load_all, save_fig, PALETTE, MODEL_LABELS, MODEL_ORDER

CAL_LABELS = {
    "platt_scaling"                  : "Platt",
    "isotonic_regression"            : "Isotonic",
    "histogram_binning_B5_uniform"   : "Hist B5 uni",
    "histogram_binning_B10_uniform"  : "Hist B10 uni",
    "histogram_binning_B15_uniform"  : "Hist B15 uni",
    "histogram_binning_B5_quantile"  : "Hist B5 qtl",
    "histogram_binning_B10_quantile" : "Hist B10 qtl",
    "histogram_binning_B15_quantile" : "Hist B15 qtl",
}


def fig12_calibration_heatmaps(cal):
    cal_summary = cal.groupby(
        ["dataset", "model_name", "calibration_method"], as_index=False
    ).agg(
        mean_delta_ece      =("delta_ece",       "mean"),
        mean_delta_brier    =("delta_brier",     "mean"),
        mean_delta_nll      =("delta_nll",       "mean"),
        mean_delta_composite=("delta_composite", "mean"),
    )
    cal_summary["Model"]  = cal_summary["model_name"].map(MODEL_LABELS).fillna(cal_summary["model_name"])
    cal_summary["Method"] = cal_summary["calibration_method"].map(CAL_LABELS).fillna(cal_summary["calibration_method"])

    for ds in sorted(cal_summary["dataset"].unique()):
        temp  = cal_summary[cal_summary["dataset"] == ds]
        pivot = temp.pivot(index="Model", columns="Method", values="mean_delta_ece")
        row_order = [MODEL_LABELS[m] for m in MODEL_ORDER if MODEL_LABELS[m] in pivot.index]
        pivot = pivot.reindex(row_order)

        fig, ax = plt.subplots(figsize=(13, 4))
        sns.heatmap(
            pivot, ax=ax,
            annot=True, fmt=".4f",
            center=0, cmap="vlag",
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "ΔECE  (negative = improvement)"},
        )
        ax.set_title(f"Calibration Effect on ECE: {ds}", fontweight="bold", pad=10)
        ax.set_ylabel("Model")
        ax.set_xlabel("Calibration method")
        ax.tick_params(axis="x", rotation=40)
        plt.tight_layout()
        ds_short = "pima" if "Pima" in ds else "bc"
        save_fig(f"fig12_calibration_heatmap_{ds_short}")

    return cal_summary


def fig13_delta_ece_vs_delta_nll(cal_summary):
    datasets = sorted(cal_summary["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(8 * len(datasets), 6), sharey=True)
    if not hasattr(axes, "__iter__"):
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        plot_df = cal_summary[cal_summary["dataset"] == ds]
        sns.scatterplot(
            data=plot_df,
            x="mean_delta_ece", y="mean_delta_nll",
            hue="Method", style="Model",
            s=120, alpha=0.85, ax=ax,
        )
        ax.axvline(0, linestyle="--", linewidth=0.9, color="grey")
        ax.axhline(0, linestyle="--", linewidth=0.9, color="grey")

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        for txt, x_side, y_side in [
            ("[OK] Both improve", "left",  "bottom"),
            ("ECE up  NLL down",  "right", "bottom"),
            ("ECE down  NLL up",  "left",  "top"),
            ("[!!] Both worsen",  "right", "top"),
        ]:
            ax.text(
                xlim[0]*0.95 if "left" in x_side else xlim[1]*0.95,
                ylim[0]*0.95 if "bottom" in y_side else ylim[1]*0.95,
                txt, fontsize=7.5, color="grey", alpha=0.8,
                ha="left" if "left" in x_side else "right",
                va="bottom" if "bottom" in y_side else "top",
            )
        ax.set_xlabel("ΔECE = calibrated − original")
        ax.set_ylabel("ΔNLL = calibrated − original" if ax == axes[0] else "")
        ax.set_title(ds, fontsize=10)
        ax.legend(fontsize=7, framealpha=0.7, loc="lower right", ncol=2)

    plt.suptitle("Calibration Trade-Off: Reliability vs Likelihood", fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig("fig13_delta_ece_vs_delta_nll")


def fig14_calibration_effectiveness(cal):
    plot_df = cal.copy()
    plot_df["Method"] = plot_df["calibration_method"].map(CAL_LABELS).fillna(plot_df["calibration_method"])
    method_order = (
        plot_df.groupby("Method")["delta_ece"].mean()
        .sort_values()
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(13, 5))
    sns.boxplot(
        data=plot_df, x="Method", y="delta_ece",
        hue="dataset", order=method_order,
        palette=PALETTE, linewidth=0.8, fliersize=2, ax=ax,
    )
    ax.axhline(0, linestyle="--", linewidth=0.9, color="black", alpha=0.6)
    ax.set_ylabel("ΔECE  (negative = improvement)")
    ax.set_xlabel("Calibration method")
    ax.set_title("Calibration Effectiveness by Dataset", fontweight="bold", pad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    ax.legend(title="Dataset", fontsize=8, framealpha=0.7)
    plt.tight_layout()
    save_fig("fig14_calibration_effectiveness")


def main():
    print("Q5 - Calibration Effectiveness")
    _, _, cal, _, _ = load_all()
    cal_summary = fig12_calibration_heatmaps(cal)
    fig13_delta_ece_vs_delta_nll(cal_summary)
    fig14_calibration_effectiveness(cal)
    print("Done.")


if __name__ == "__main__":
    main()
