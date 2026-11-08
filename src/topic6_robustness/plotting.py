from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def create_plots(metrics_csv: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    metrics_csv = Path(metrics_csv)
    output_dir = Path(output_dir) if output_dir else metrics_csv.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(metrics_csv)
    clean = frame[frame["corruption"] == "clean"]

    generated: list[Path] = []
    for metric, ylabel in (("map_50_95", "mAP@0.5:0.95"), ("map_50", "mAP@0.5")):
        clean_metric = float(clean.iloc[0][metric]) if not clean.empty else None
        figure, axis = plt.subplots(figsize=(9, 5.5))
        corrupted = frame[frame["corruption"] != "clean"]
        grouped = (
            corrupted.groupby(["corruption", "severity", "recovered"], as_index=False)[metric]
            .agg(["mean", "std"])
            .reset_index()
        )
        for (corruption, recovered), group in grouped.groupby(["corruption", "recovered"]):
            label = f"{corruption} + recovery" if recovered else corruption
            axis.errorbar(
                group["severity"],
                group["mean"],
                yerr=group["std"].fillna(0.0),
                marker="o",
                capsize=3,
                label=label,
            )
        if clean_metric is not None:
            axis.axhline(clean_metric, color="black", linestyle="--", linewidth=1.2, label="clean baseline")
        axis.set_xlabel("Severity")
        axis.set_ylabel(ylabel)
        axis.set_xticks([1, 2, 3, 4, 5])
        axis.grid(alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
        figure.tight_layout()
        destination = output_dir / f"{metric}_vs_severity.png"
        figure.savefig(destination, dpi=180)
        plt.close(figure)
        generated.append(destination)

    corrupted = frame[frame["corruption"] != "clean"].copy()
    if "repeat" not in corrupted:
        # Pilot tables created before repeat was included remain plottable.
        corrupted["repeat"] = 0
    paired = corrupted.pivot_table(
        index=["corruption", "severity", "repeat"],
        columns="recovered",
        values="map_50_95",
    )
    if {False, True}.issubset(paired.columns):
        recovery_gain = (paired[True] - paired[False]).rename("recovery_gain_pp").mul(100.0).reset_index()
        figure, axis = plt.subplots(figsize=(9, 5.5))
        for corruption, group in recovery_gain.groupby("corruption"):
            summary = group.groupby("severity", as_index=False)["recovery_gain_pp"].agg(["mean", "std"]).reset_index()
            axis.errorbar(
                summary["severity"], summary["mean"], yerr=summary["std"].fillna(0.0),
                marker="o", capsize=3, label=corruption,
            )
        axis.axhline(0.0, color="black", linewidth=1.0)
        axis.set_xlabel("Severity")
        axis.set_ylabel("Recovery gain in mAP@0.5:0.95 (percentage points)")
        axis.set_xticks([1, 2, 3, 4, 5])
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        figure.tight_layout()
        destination = output_dir / "recovery_gain_vs_severity.png"
        figure.savefig(destination, dpi=180)
        plt.close(figure)
        generated.append(destination)

    latency_column = "end_to_end_milliseconds_per_image"
    if latency_column in corrupted:
        figure, axis = plt.subplots(figsize=(9, 5.5))
        grouped_latency = corrupted.groupby(["corruption", "severity", "recovered"], as_index=False)[latency_column].agg(["mean", "std"]).reset_index()
        for (corruption, recovered), group in grouped_latency.groupby(["corruption", "recovered"]):
            label = f"{corruption} + recovery" if recovered else corruption
            axis.errorbar(group["severity"], group["mean"], yerr=group["std"].fillna(0.0), marker="o", capsize=3, label=label)
        axis.set_xlabel("Severity")
        axis.set_ylabel("End-to-end latency (ms/image)")
        axis.set_xticks([1, 2, 3, 4, 5])
        axis.grid(alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
        figure.tight_layout()
        destination = output_dir / "end_to_end_latency_vs_severity.png"
        figure.savefig(destination, dpi=180)
        plt.close(figure)
        generated.append(destination)

    if not clean.empty:
        clean_map = float(clean.iloc[0]["map_50_95"])
        frame["relative_map_drop_percent"] = 100.0 * (clean_map - frame["map_50_95"]) / max(clean_map, 1e-12)
        corrupted = frame[frame["corruption"] != "clean"].copy()
        if "repeat" not in corrupted:
            corrupted["repeat"] = 0
        aggregations: dict[str, tuple[str, str]] = {
            "mean_map": ("map_50_95", "mean"),
            "std_map": ("map_50_95", "std"),
            "relative_map_drop_percent": ("relative_map_drop_percent", "mean"),
        }
        for column in (
            "inference_milliseconds_per_image",
            "preprocessing_milliseconds_per_image",
            "end_to_end_milliseconds_per_image",
            "false_negative_unmatched",
            "false_positive_unmatched",
            # Compatibility with pilot metrics generated before timing was split.
            "milliseconds_per_image",
        ):
            if column in corrupted:
                aggregations[column] = (column, "mean")
        summary = corrupted.groupby(["corruption", "severity", "recovered"], as_index=False).agg(**aggregations)
        summary_path = output_dir / "aggregated_metrics.csv"
        summary.to_csv(summary_path, index=False)
        generated.append(summary_path)
    return generated

