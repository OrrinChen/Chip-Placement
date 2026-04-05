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
