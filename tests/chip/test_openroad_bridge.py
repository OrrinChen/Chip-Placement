from pathlib import Path
import subprocess

import pytest

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
    assert "--gpus" in command
    assert "all" in command
    assert "/workspace/tmp/run.tcl" == command[-1]
    assert "/workspace/tmp" == command[6]


def test_openroad_bridge_mounts_all_referenced_inputs_and_runs_from_artifact_dir():
    bridge = OpenROADBridge(docker_image="openroad/orfs")
    root = Path("C:/workspace/project")
    artifact_dir = root / "tmp" / "mgc_pci_bridge32_1"
    script_path = artifact_dir / ".built_in_acceptance.tcl"

    command = bridge.build_command(
        artifact_dir / "tech.lef",
        artifact_dir / "floorplan_clean.def",
        script_path,
        referenced_paths=[
            root / "tmp" / "ispd2012" / "lib" / "contest.lib",
            root / "tmp" / "ispd2012" / "pci_bridge32_fast" / "pci_bridge32_fast.sdc",
            artifact_dir / "cells.lef",
        ],
        working_directory=artifact_dir,
    )

    assert "C:/workspace/project/tmp:/workspace" == command[4]
    assert "/workspace/mgc_pci_bridge32_1" == command[6]
    assert "/workspace/mgc_pci_bridge32_1/.built_in_acceptance.tcl" == command[-1]


def test_openroad_bridge_normalizes_parent_relative_references_before_mounting():
    bridge = OpenROADBridge(docker_image="openroad/orfs")
    root = Path("C:/workspace/project")
    artifact_dir = root / "tmp" / "mgc_pci_bridge32_1"
    script_path = artifact_dir / ".built_in_acceptance.tcl"

    command = bridge.build_command(
        artifact_dir / "tech.lef",
        artifact_dir / "floorplan_clean.def",
        script_path,
        referenced_paths=[
            artifact_dir / ".." / "ispd2012" / "lib" / "contest.lib",
            artifact_dir / ".." / "ispd2012" / "pci_bridge32_fast" / "pci_bridge32_fast.sdc",
        ],
        working_directory=artifact_dir,
    )

    assert "C:/workspace/project/tmp:/workspace" == command[4]


def test_build_script_injects_place_inst_commands_for_movable_instances():
    bridge = OpenROADBridge()
    script = bridge.build_script(
        "design.lef",
        "design.def",
        {"U1": (10.0, 20.0), "U2": (30.0, 40.0)},
        placement_mode="sa_injected",
        orientations={"U1": "N", "U2": "FS"},
    )

    assert 'place_inst -name {U1} -location {10.0 20.0} -orientation R0 -status PLACED' in script
    assert 'place_inst -name {U2} -location {30.0 40.0} -orientation MX -status PLACED' in script


def test_build_script_braces_hierarchical_instance_names_for_tcl_safety():
    bridge = OpenROADBridge()

    script = bridge.build_script(
        "design.lef",
        "design.def",
        {"top/block[0]/macro": (10.0, 20.0)},
        placement_mode="sa_injected",
        orientations={"top/block[0]/macro": "N"},
    )

    assert "place_inst -name {top/block[0]/macro}" in script


def test_build_script_uses_same_tail_flow_for_built_in_and_sa():
    bridge = OpenROADBridge()

    built_in = bridge.build_script("design.lef", "design.def", {}, placement_mode="built_in")
    sa = bridge.build_script("design.lef", "design.def", {"U1": (1.0, 2.0)}, placement_mode="sa_injected")

    assert "global_route" in built_in
    assert "global_route" in sa
    assert built_in.split("global_route", maxsplit=1)[1] == sa.split("global_route", maxsplit=1)[1]


def test_build_script_can_write_seed_def_after_placement_before_tail_flow():
    bridge = OpenROADBridge()

    script = bridge.build_script(
        "design.lef",
        "design.def",
        {},
        placement_mode="built_in",
        write_def_path="seed_placed.def",
    )

    assert "write_def seed_placed.def" in script
    assert script.index("write_def seed_placed.def") < script.index("global_route")


def test_build_script_can_skip_route_tail_for_legalization_only():
    bridge = OpenROADBridge()

    script = bridge.build_script(
        "design.lef",
        "design.def",
        {"U1": (10.0, 20.0)},
        placement_mode="sa_injected",
        orientations={"U1": "N"},
        write_def_path="legalized.def",
        run_routing_flow=False,
    )

    assert "detailed_placement" in script
    assert "write_def legalized.def" in script
    assert "global_route" not in script
    assert "detailed_route" not in script


def test_build_script_can_skip_detailed_route_for_fast_ground_truth():
    bridge = OpenROADBridge()

    script = bridge.build_script(
        "design.lef",
        "design.def",
        {"U1": (10.0, 20.0)},
        placement_mode="sa_injected",
        orientations={"U1": "N"},
        run_detailed_route=False,
    )

    assert "global_route" in script
    assert "estimate_parasitics -global_routing" in script
    assert "report_wire_length -net * -global_route -summary" in script
    assert "detailed_route" not in script


def test_openroad_bridge_retries_without_gpu_when_runtime_rejects_gpus(monkeypatch, tmp_path):
    bridge = OpenROADBridge(docker_image="openroad/orfs")
    lef_path = tmp_path / "design.lef"
    def_path = tmp_path / "design.def"
    script_path = tmp_path / "run.tcl"
    lef_path.write_text("")
    def_path.write_text("")
    script_path.write_text("")
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=command,
                stderr="could not select device driver \"nvidia\" with capabilities: [[gpu]]",
            )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr("chip.openroad_bridge.subprocess.run", fake_run)

    completed = bridge.run_script(lef_path, def_path, script_path, capture_output=True, text=True, check=True)

    assert completed.stdout == "ok"
    assert calls[0].count("--gpus") == 1
    assert "--gpus" not in calls[1]


def test_openroad_bridge_force_removes_timed_out_docker_container(monkeypatch, tmp_path):
    bridge = OpenROADBridge(docker_image="openroad/orfs")
    lef_path = tmp_path / "design.lef"
    def_path = tmp_path / "design.def"
    script_path = tmp_path / "run.tcl"
    lef_path.write_text("")
    def_path.write_text("")
    script_path.write_text("")
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[:3] == ["docker", "run", "--rm"]:
            raise subprocess.TimeoutExpired(cmd=command, timeout=5)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("chip.openroad_bridge.subprocess.run", fake_run)

    with pytest.raises(subprocess.TimeoutExpired):
        bridge.run_script(lef_path, def_path, script_path, timeout=5, capture_output=True, text=True, check=True)

    assert "--name" in calls[0]
    container_name = calls[0][calls[0].index("--name") + 1]
    assert calls[1] == ["docker", "rm", "-f", container_name]
