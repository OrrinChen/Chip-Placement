from chip.openroad_eval import build_evaluation_request
from chip.openroad_eval import build_openroad_script
from chip.openroad_eval import parse_openroad_metrics


def test_build_evaluation_request_preserves_mode_and_artifacts():
    request = build_evaluation_request(
        benchmark_name="ibm01",
        lef_path="design.lef",
        def_path="design.def",
        placement_mode="built_in",
    )

    assert request.benchmark_name == "ibm01"
    assert request.placement_mode == "built_in"
    assert request.lef_path == "design.lef"


def test_parse_openroad_metrics_extracts_routed_wirelength_timing_and_drc():
    metrics = parse_openroad_metrics(
        """
        total routed wire length = 12345
        report_tns -12.5
        report_wns -0.9
        drc violation count = 7
        """
    )

    assert metrics["total_routed_wirelength"] == 12345.0
    assert metrics["tns"] == -12.5
    assert metrics["wns"] == -0.9
    assert metrics["drc_violations"] == 7


def test_build_openroad_script_changes_only_the_injection_section():
    built_in = build_openroad_script("design.lef", "design.def", placement_mode="built_in")
    injected = build_openroad_script(
        "design.lef",
        "design.def",
        placement_mode="sa_injected",
        placement={"U1": (1.0, 2.0)},
        orientations={"U1": "N"},
    )

    assert "place_inst U1 1.0 2.0 N" in injected
    assert built_in.split("global_route", maxsplit=1)[1] == injected.split("global_route", maxsplit=1)[1]
