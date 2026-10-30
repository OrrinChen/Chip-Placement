from __future__ import annotations

import json
from pathlib import Path

import pytest

from chip.experiments.ppo_vs_sa import display_mean_std
from chip.experiments.ppo_vs_sa import main
from chip.experiments.ppo_vs_sa import summarize_numeric


ROOT = Path(__file__).resolve().parents[2]


def test_sample_std_is_zero_for_identical_values():
    summary = summarize_numeric([3.0, 3.0, 3.0, 3.0, 3.0])

    assert summary["count"] == 5
    assert summary["mean"] == 3.0
    assert summary["std"] == 0.0
    assert display_mean_std(summary) == "3 ± 0"


def test_sample_std_is_undefined_for_one_value():
    summary = summarize_numeric([1.5])

    assert summary["std"] is None
    assert "std undefined" in display_mean_std(summary)


def test_ariane_config_writes_no_metrics(tmp_path):
    output_dir = tmp_path / "ariane"
    code = main(
        [
            "--config",
            str(ROOT / "experiments" / "ppo_vs_sa" / "configs" / "ariane.json"),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert code == 2
    assert not output_dir.exists()


def test_smoke_run_reports_both_methods_without_a_std(tmp_path):
    output_dir = tmp_path / "smoke"
    code = main(
        [
            "--config",
            str(ROOT / "experiments" / "ppo_vs_sa" / "configs" / "smoke.json"),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert code == 0
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    placements = json.loads((output_dir / "placements.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed"
    assert summary["metric"] == "analytic_hpwl"
    assert "ariane133" not in summary["benchmarks"]
    tiny = summary["benchmarks"]["tiny"]
    ppo = tiny["methods"]["masked_ppo"]["final_hpwl"]
    sa = tiny["methods"]["simulated_annealing"]["final_hpwl"]
    assert ppo["count"] == 1
    assert sa["count"] == 1
    assert ppo["std"] is None
    assert sa["std"] is None
    assert ppo["mean"] == pytest.approx(tiny["methods"]["masked_ppo"]["per_seed"][0]["final_hpwl"])
    assert placements["tiny"]["masked_ppo"]
    assert placements["tiny"]["simulated_annealing"]
