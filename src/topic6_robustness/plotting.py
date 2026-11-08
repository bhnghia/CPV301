from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    from scipy.stats import ttest_rel
except ImportError:
    ttest_rel = None


CORRUPTIONS = ("gaussian_noise", "salt_pepper", "fog", "low_light")


def plot_confusion_heatmap(
    confusion_matrix_csv: str | Path,
    output_path: str | Path,
    top_n_classes: int = 15,
    title: str = "Confusion matrix",
) -> Path:
    """Save a heatmap of the most frequent classes in a square confusion matrix."""
    if sns is None:
        raise RuntimeError("seaborn is required for confusion heatmaps.")
    matrix = np.loadtxt(confusion_matrix_csv, delimiter=",")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.size == 0:
        raise ValueError(f"Expected a non-empty square matrix: {confusion_matrix_csv}")
    count = min(max(top_n_classes, 1), matrix.shape[0])
    class_ids = np.argsort(matrix.sum(axis=1))[::-1][:count]
    labels = ["background" if i == matrix.shape[0] - 1 else str(i) for i in class_ids]
    figure, axis = plt.subplots(figsize=(12, 10))
    sns.heatmap(matrix[np.ix_(class_ids, class_ids)], cmap="Blues", annot=False, cbar=True,
                xticklabels=labels, yticklabels=labels, ax=axis)
    axis.set(xlabel="Predicted class ID", ylabel="Ground-truth class ID", title=title)
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)
    return destination


def statistical_significance_test(
    metrics_csv: str | Path, output_json: str | Path, alpha: float = 0.05
) -> dict[str, dict[str, float | int | bool | None]]:
    """Run paired raw/recovery tests using only matching severity/repeat pairs."""
    frame = pd.read_csv(metrics_csv)
    results: dict[str, dict[str, float | int | bool | None]] = {}
    for corruption in CORRUPTIONS:
        subset = frame[frame["corruption"] == corruption].copy()
        if "repeat" not in subset:
            subset["repeat"] = 0
        raw = subset[~subset["recovered"]][["severity", "repeat", "map_50_95"]]
        recovered = subset[subset["recovered"]][["severity", "repeat", "map_50_95"]]
        pairs = raw.merge(recovered, on=["severity", "repeat"], suffixes=("_raw", "_recovered"))
        raw_values = pairs["map_50_95_raw"].to_numpy(float)
        recovered_values = pairs["map_50_95_recovered"].to_numpy(float)
        n_pairs = len(pairs)
        statistic = p_value = None
        if n_pairs >= 2 and ttest_rel is not None:
            test = ttest_rel(raw_values, recovered_values)
            statistic, p_value = float(test.statistic), float(test.pvalue)
        elif n_pairs < 2:
            warnings.warn(f"{corruption}: fewer than two paired observations; no t-test.", stacklevel=2)
        else:
            warnings.warn("scipy is unavailable; no t-test.", stacklevel=2)
        mean_raw = float(raw_values.mean()) if n_pairs else None
        mean_recovered = float(recovered_values.mean()) if n_pairs else None
        improvement = mean_recovered - mean_raw if mean_raw is not None and mean_recovered is not None else None
        significant = bool(p_value is not None and p_value < alpha)
        results[corruption] = {"t_statistic": statistic, "p_value": p_value,
                               "significant_at_0_05": significant, "n_pairs": n_pairs,
                               "mean_raw": mean_raw, "mean_recovered": mean_recovered,
                               "improvement": improvement}
        print(f"{corruption}: t={statistic}, p={p_value} {'SIGNIFICANT' if significant else 'NOT'}")
    destination = Path(output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def plot_per_class_recovery_gain(metrics_dir: str | Path, output_dir: str | Path) -> list[Path]:
    """Save severity-5 per-class raw/recovery AP gain charts and tables."""
    conditions = Path(metrics_dir) / "conditions"
    output_dir = Path(output_dir)
    generated: list[Path] = []
    for corruption in CORRUPTIONS:
        raw_path = conditions / f"{corruption}_s5_r0" / "class_ap.json"
        recovered_paths = sorted(conditions.glob(f"{corruption}_s5_r0_recovered*/class_ap.json"))
        if not raw_path.is_file() or not recovered_paths:
            warnings.warn(f"Skipping {corruption}: severity-5 class AP files are missing.")
            continue
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        recovered = json.loads(recovered_paths[0].read_text(encoding="utf-8"))
        names = [name for name in raw if name in recovered]
        if not names:
            warnings.warn(f"Skipping {corruption}: no shared classes.")
            continue
        result = pd.DataFrame({"class_id": range(len(names)), "raw_ap": [raw[name] for name in names],
                               "recovered_ap": [recovered[name] for name in names]})
        result["gain"] = result["recovered_ap"] - result["raw_ap"]
        result = result.sort_values("gain", ascending=False, ignore_index=True)
        colors = np.where(result["gain"] >= 0, "forestgreen", "firebrick")
        figure, axis = plt.subplots(figsize=(14, 6))
        axis.bar(result["class_id"].astype(str), result["gain"], color=colors)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(xlabel="COCO class ID", ylabel="AP recovery gain",
                 title=f"{corruption}: severity-5 per-class recovery gain")
        axis.tick_params(axis="x", labelrotation=90, labelsize=7)
        figure.tight_layout()
        png_path = output_dir / f"{corruption}_class_recovery_gain.png"
        csv_path = output_dir / f"{corruption}_class_recovery_gain.csv"
        figure.savefig(png_path, dpi=150)
        plt.close(figure)
        result.to_csv(csv_path, index=False)
        generated.extend((png_path, csv_path))
    return generated


def _save_line_plot(frame: pd.DataFrame, clean: pd.DataFrame, metric: str, ylabel: str, output_dir: Path) -> Path:
    clean_metric = float(clean.iloc[0][metric]) if not clean.empty else None
    figure, axis = plt.subplots(figsize=(9, 5.5))
    grouped = frame.groupby(["corruption", "severity", "recovered"], as_index=False)[metric].agg(["mean", "std"]).reset_index()
    for (corruption, recovered), group in grouped.groupby(["corruption", "recovered"]):
        axis.errorbar(group["severity"], group["mean"], yerr=group["std"].fillna(0), marker="o", capsize=3,
                      label=f"{corruption} + recovery" if recovered else corruption)
    if clean_metric is not None:
        axis.axhline(clean_metric, color="black", linestyle="--", linewidth=1.2, label="clean baseline")
    axis.set(xlabel="Severity", ylabel=ylabel, xticks=[1, 2, 3, 4, 5])
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    destination = output_dir / f"{metric}_vs_severity.png"
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def create_plots(metrics_csv: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    """Create aggregate, confusion, statistical and per-class recovery outputs."""
    metrics_csv = Path(metrics_csv)
    output_dir = Path(output_dir) if output_dir else metrics_csv.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(metrics_csv)
    clean = frame[frame["corruption"] == "clean"]
    corrupted = frame[frame["corruption"] != "clean"].copy()
    if "repeat" not in corrupted:
        corrupted["repeat"] = 0
    generated = [_save_line_plot(corrupted, clean, metric, label, output_dir)
                 for metric, label in (("map_50_95", "mAP@0.5:0.95"), ("map_50", "mAP@0.5"))]
    paired = corrupted.pivot_table(index=["corruption", "severity", "repeat"], columns="recovered", values="map_50_95")
    if {False, True}.issubset(paired.columns):
        gains = (paired[True] - paired[False]).mul(100).rename("gain").reset_index()
        figure, axis = plt.subplots(figsize=(9, 5.5))
        for name, group in gains.groupby("corruption"):
            summary = group.groupby("severity")["gain"].agg(["mean", "std"]).reset_index()
            axis.errorbar(summary["severity"], summary["mean"], yerr=summary["std"].fillna(0), marker="o",
                          capsize=3, label=name)
        axis.axhline(0, color="black", linewidth=1)
        axis.set(xlabel="Severity", ylabel="Recovery gain in mAP@0.5:0.95 (percentage points)", xticks=[1, 2, 3, 4, 5])
        axis.grid(alpha=0.25); axis.legend(fontsize=8); figure.tight_layout()
        path = output_dir / "recovery_gain_vs_severity.png"; figure.savefig(path, dpi=180); plt.close(figure); generated.append(path)
    if "end_to_end_milliseconds_per_image" in corrupted:
        figure, axis = plt.subplots(figsize=(9, 5.5))
        grouped = corrupted.groupby(["corruption", "severity", "recovered"], as_index=False)["end_to_end_milliseconds_per_image"].agg(["mean", "std"]).reset_index()
        for (name, recovered), group in grouped.groupby(["corruption", "recovered"]):
            axis.errorbar(group["severity"], group["mean"], yerr=group["std"].fillna(0), marker="o", capsize=3,
                          label=f"{name} + recovery" if recovered else name)
        axis.set(xlabel="Severity", ylabel="End-to-end latency (ms/image)", xticks=[1, 2, 3, 4, 5])
        axis.grid(alpha=0.25); axis.legend(ncol=2, fontsize=8); figure.tight_layout()
        path = output_dir / "end_to_end_latency_vs_severity.png"; figure.savefig(path, dpi=180); plt.close(figure); generated.append(path)
    if not clean.empty:
        baseline = float(clean.iloc[0]["map_50_95"])
        corrupted["relative_map_drop_percent"] = 100 * (baseline - corrupted["map_50_95"]) / max(baseline, 1e-12)
        summary = corrupted.groupby(["corruption", "severity", "recovered"], as_index=False).agg(
            mean_map=("map_50_95", "mean"), std_map=("map_50_95", "std"),
            relative_map_drop_percent=("relative_map_drop_percent", "mean"))
        path = output_dir / "aggregated_metrics.csv"; summary.to_csv(path, index=False); generated.append(path)
    for matrix in sorted((metrics_csv.parent / "conditions").glob("*/confusion_matrix.csv")):
        try:
            generated.append(plot_confusion_heatmap(matrix, output_dir / f"{matrix.parent.name}_confusion_heatmap.png",
                                                    title=f"{matrix.parent.name}: top confusion classes"))
        except (OSError, RuntimeError, ValueError) as error:
            warnings.warn(f"Skipping confusion heatmap for {matrix.parent.name}: {error}")
    significance = output_dir / "significance_test.json"
    statistical_significance_test(metrics_csv, significance); generated.append(significance)
    generated.extend(plot_per_class_recovery_gain(metrics_csv.parent, output_dir))
    return generated
