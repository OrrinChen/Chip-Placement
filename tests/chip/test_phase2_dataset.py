import json

from chip.experiments.phase2_dataset import (
    build_sampling_plan,
    collect_sample_result,
    summarize_dataset_row,
)


def test_build_sampling_plan_expands_random_and_sa_samples():
    plan = build_sampling_plan(
        benchmark_name="mgc_pci_bridge32_1",
        batch="smoke",
        random_samples=2,
        sa_truncation_steps=(1, 3),
    )

    assert [item["source"] for item in plan] == [
        "random_perturbation",
        "random_perturbation",
        "sa_truncated",
        "sa_truncated",
    ]
    assert [item["sample_id"] for item in plan] == [
        "rand_0000",
        "rand_0001",
        "sa_0001",
        "sa_0003",
    ]
    assert {item["batch"] for item in plan} == {"smoke"}


def test_summarize_dataset_row_keeps_metrics_and_gate_state():
    row = summarize_dataset_row(
        sample={"benchmark": "mgc_pci_bridge32_1", "sample_id": "rand_0001", "source": "random_perturbation"},
        metrics={
            "total_routed_wirelength": 100.0,
            "wns": -0.1,
            "tns": -1.0,
            "drc_violations": 0,
            "runtime_seconds": 5.0,
        },
        gate_status="infra_pass / quality_pending",
    )

    assert row["sample_id"] == "rand_0001"
    assert row["total_routed_wirelength"] == 100.0
    assert row["phase1_gate_status"] == "infra_pass / quality_pending"


def test_load_phase1_gate_map_reads_benchmark_statuses(tmp_path):
    results_path = tmp_path / "phase1_acceptance_results.json"
    results_path.write_text(
        json.dumps(
            [
                {"benchmark": "mgc_pci_bridge32_1", "phase1_gate_status": "infra_pass / quality_pending"},
                {"benchmark": "mgc_des_perf_2", "phase1_gate_status": "infra_pass / quality_pass"},
            ]
        )
    )

    from chip.experiments.phase2_dataset import load_phase1_gate_map

    gate_map = load_phase1_gate_map(results_path)

    assert gate_map == {
        "mgc_pci_bridge32_1": "infra_pass / quality_pending",
        "mgc_des_perf_2": "infra_pass / quality_pass",
    }


def test_collect_sample_result_marks_timeout_without_losing_metadata():
    sample = {
        "benchmark": "mgc_pci_bridge32_1",
        "sample_id": "rand_0001",
        "source": "random_perturbation",
        "batch": "smoke",
    }

    def timeout_evaluator(_sample, timeout_seconds):
        raise TimeoutError(f"timed out after {timeout_seconds}s")

    row = collect_sample_result(
        sample=sample,
        gate_status="infra_pass / quality_pending",
        evaluate_fn=timeout_evaluator,
        timeout_seconds=30,
    )

    assert row["status"] == "timeout"
    assert row["sample_id"] == "rand_0001"
    assert row["batch"] == "smoke"
    assert row["phase1_gate_status"] == "infra_pass / quality_pending"
    assert "timed out" in row["error"]


def test_collect_sample_result_marks_errors_without_dropping_sample_identity():
    sample = {
        "benchmark": "mgc_pci_bridge32_1",
        "sample_id": "sa_0005",
        "source": "sa_truncated",
        "batch": "smoke",
    }

    def failing_evaluator(_sample, timeout_seconds):
        raise RuntimeError(f"router failed inside {timeout_seconds}s window")

    row = collect_sample_result(
        sample=sample,
        gate_status="infra_pass / quality_pending",
        evaluate_fn=failing_evaluator,
        timeout_seconds=45,
    )

    assert row["status"] == "error"
    assert row["sample_id"] == "sa_0005"
    assert row["batch"] == "smoke"
    assert "router failed" in row["error"]
