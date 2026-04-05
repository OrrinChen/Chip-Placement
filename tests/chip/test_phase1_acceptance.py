from chip.experiments.phase1_acceptance import summarize_acceptance_row


def test_summarize_acceptance_row_combines_baseline_and_openroad_metrics():
    row = summarize_acceptance_row(
        benchmark_name="ibm01",
        scale_bucket="small",
        design_stats={"node_count": 1000, "net_count": 1200},
        sa_summary={"hpwl": 10.0, "is_valid": True, "runtime_seconds": 1.5},
        built_in_metrics={
            "total_routed_wirelength": 100.0,
            "wns": -0.1,
            "tns": -1.0,
            "drc_violations": 0,
            "runtime_seconds": 5.0,
        },
        sa_metrics={
            "total_routed_wirelength": 110.0,
            "wns": -0.2,
            "tns": -2.0,
            "drc_violations": 1,
            "runtime_seconds": 2.5,
        },
    )

    assert row["benchmark"] == "ibm01"
    assert row["scale_bucket"] == "small"
    assert row["sa_hpwl"] == 10.0
    assert row["built_in_total_routed_wirelength"] == 100.0
    assert row["sa_drc_violations"] == 1


def test_run_acceptance_suite_surfaces_benchmark_failures(monkeypatch, tmp_path):
    from chip.experiments import phase1_acceptance as module

    monkeypatch.setattr(
        module,
        "run_single_benchmark_acceptance",
        lambda *args, **kwargs: {"benchmark": "ibm01", "error": "routing_failed"},
    )

    rows = module.run_acceptance_suite(
        benchmark_root=tmp_path,
        output_path=tmp_path / "results.json",
        benchmark_names=["ibm01"],
    )

    assert rows == [{"benchmark": "ibm01", "error": "routing_failed"}]
