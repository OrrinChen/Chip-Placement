from pathlib import Path

from chip.openroad_bridge import OpenROADBridge


def test_openroad_bridge_reports_unavailable_when_binary_missing():
    bridge = OpenROADBridge(binary_name="definitely-missing-openroad")

    assert bridge.is_available() is False


def test_openroad_bridge_builds_docker_command_with_mapped_paths():
    bridge = OpenROADBridge(docker_image="openroad/orfs")
    root = Path("C:/workspace/project")
    script_path = root / "tmp" / "run.tcl"

    command = bridge.build_command(
        root / "chip" / "benchmarks" / "sample.lef",
        root / "chip" / "benchmarks" / "sample.def",
        script_path,
    )

    assert command[0] == "docker"
    assert "openroad/orfs" in command
    assert "/workspace/tmp/run.tcl" == command[-1]


def test_build_script_injects_place_inst_commands_for_movable_instances():
    bridge = OpenROADBridge()
    script = bridge.build_script(
        "design.lef",
        "design.def",
        {"U1": (10.0, 20.0), "U2": (30.0, 40.0)},
        placement_mode="sa_injected",
        orientations={"U1": "N", "U2": "FS"},
    )

    assert "place_inst U1 10.0 20.0 N" in script
    assert "place_inst U2 30.0 40.0 FS" in script


def test_build_script_uses_same_tail_flow_for_built_in_and_sa():
    bridge = OpenROADBridge()

    built_in = bridge.build_script("design.lef", "design.def", {}, placement_mode="built_in")
    sa = bridge.build_script("design.lef", "design.def", {"U1": (1.0, 2.0)}, placement_mode="sa_injected")

    assert "global_route" in built_in
    assert "global_route" in sa
    assert built_in.split("global_route", maxsplit=1)[1] == sa.split("global_route", maxsplit=1)[1]
