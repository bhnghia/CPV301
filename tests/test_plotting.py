from pathlib import Path

import pandas as pd

from topic6_robustness.plotting import create_plots


def test_create_plots_headless(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        [
            {"corruption": "clean", "severity": 0, "recovered": False, "map_50_95": 0.5, "map_50": 0.7, "milliseconds_per_image": 10.0},
            {"corruption": "fog", "severity": 1, "recovered": False, "map_50_95": 0.4, "map_50": 0.6, "milliseconds_per_image": 11.0},
            {"corruption": "fog", "severity": 1, "recovered": True, "map_50_95": 0.45, "map_50": 0.65, "milliseconds_per_image": 13.0},
        ]
    )
    metrics = tmp_path / "metrics.csv"
    frame.to_csv(metrics, index=False)
    outputs = create_plots(metrics)
    assert len(outputs) == 4
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)

