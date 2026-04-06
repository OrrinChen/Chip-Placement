from pathlib import Path
import subprocess

from chip.legalization import legalize_placement


class FakeBridge:
    def __init__(self):
        self.script_calls: list[dict[str, object]] = []
        self.run_calls: list[dict[str, object]] = []

    def build_script(self, *args, **kwargs):
        self.script_calls.append({"args": args, "kwargs": kwargs})
        return "read_def seed.def\ndetailed_placement\nwrite_def legalized.def\n"

    def run_script(self, lef_path, def_path, script_path, referenced_paths=None, working_directory=None, **kwargs):
        self.run_calls.append(
            {
                "lef_path": lef_path,
                "def_path": def_path,
                "script_path": script_path,
                "referenced_paths": referenced_paths,
                "working_directory": working_directory,
            }
        )
        Path(working_directory, "legalized.def").write_text("VERSION 5.8 ;\nEND DESIGN\n")
        return subprocess.CompletedProcess(["openroad"], 0, stdout="", stderr="")


def test_legalize_placement_runs_lightweight_openroad_flow_and_returns_def(tmp_path):
    artifact_dir = tmp_path / "bench"
    artifact_dir.mkdir()
    for name in ("tech.lef", "cells.lef", "seed.def"):
        (artifact_dir / name).write_text("")
    bridge = FakeBridge()

    legalized_path = legalize_placement(
        artifact_dir=artifact_dir,
        lef_paths=("tech.lef", "cells.lef"),
        def_path="seed.def",
        placement={"U1": (10.2, 2.0)},
        orientations={"U1": "N"},
        bridge=bridge,
    )

    assert legalized_path == artifact_dir / "legalized.def"
    assert bridge.script_calls[0]["kwargs"]["run_routing_flow"] is False
    assert bridge.run_calls[0]["working_directory"] == artifact_dir
