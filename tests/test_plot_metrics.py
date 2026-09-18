import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from plot_metrics import plot, series  # noqa: E402

from manas.training.metrics import read_metrics


def test_series_skips_missing_keys():
    rows = [{"step": 1, "loss": 3.0}, {"step": 2, "val_loss": 2.9}, {"step": 3, "loss": 2.5, "grad_norm": None}]
    assert series(rows, "loss") == ([1, 3], [3.0, 2.5])
    assert series(rows, "val_loss") == ([2], [2.9])
    assert series(rows, "grad_norm") == ([], [])


def test_plot_writes_a_png(tmp_path):
    metrics = tmp_path / "demo_64_metrics.jsonl"
    with open(metrics, "w", encoding="utf-8") as f:
        for step in range(1, 11):
            row = {"step": step, "loss": 9 / step, "lr": 1e-3, "grad_norm": 1.0, "tokens_per_sec": 500}
            f.write(json.dumps(row) + "\n")
            if step % 5 == 0:
                f.write(json.dumps({"step": step, "val_loss": 9.5 / step}) + "\n")
    rows = read_metrics(metrics)
    out = plot(rows, "demo", str(tmp_path / "demo_curves.png"))
    assert Path(out).exists() and Path(out).stat().st_size > 10_000
