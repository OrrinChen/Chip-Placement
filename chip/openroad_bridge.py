from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


class OpenROADUnavailableError(RuntimeError):
    """Raised when OpenROAD is not installed or not on PATH."""


class OpenROADBridge:
    def __init__(
        self,
        binary_name: str = "openroad",
        docker_image: str | None = None,
        docker_binary: str = "docker",
        docker_openroad_path: str = "/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad",
    ) -> None:
        self.binary_name = binary_name
        self.docker_image = docker_image
        self.docker_binary = docker_binary
        self.docker_openroad_path = docker_openroad_path

    def is_available(self) -> bool:
        if self.docker_image is not None:
            return shutil.which(self.docker_binary) is not None
        return shutil.which(self.binary_name) is not None

    def build_script(
        self,
        lef_path: str,
        def_path: str,
        placement: dict[str, tuple[float, float]],
    ) -> str:
        lines = [f"read_lef {lef_path}", f"read_def {def_path}"]
        for macro_name, (x, y) in placement.items():
            lines.append(f"# place {macro_name} at ({x}, {y})")
        lines.extend(["global_route", "report_tns", "report_wns"])
        return "\n".join(lines) + "\n"

    def build_command(
        self,
        lef_path: str | Path,
        def_path: str | Path,
        script_path: str | Path,
    ) -> list[str]:
        if self.docker_image is None:
            return [self.binary_name, "-exit", str(script_path)]

        mount_root = Path(os.path.commonpath([Path(lef_path), Path(def_path), Path(script_path)]))
        container_script_path = self._to_container_path(Path(script_path), mount_root)
        return [
            self.docker_binary,
            "run",
            "--rm",
            "-v",
            f"{mount_root.as_posix()}:/workspace",
            "-w",
            "/workspace",
            self.docker_image,
            self.docker_openroad_path,
            "-exit",
            container_script_path,
        ]

    def evaluate_placement(
        self,
        lef_path: str | Path,
        def_path: str | Path,
        placement: dict[str, tuple[float, float]],
    ) -> dict[str, str]:
        if not self.is_available():
            target = self.docker_image or self.binary_name
            raise OpenROADUnavailableError(f"OpenROAD target '{target}' is not available")

        host_lef = Path(lef_path).resolve()
        host_def = Path(def_path).resolve()
        mount_root = Path(os.path.commonpath([host_lef, host_def]))
        script_dir = mount_root / ".openroad_tmp"
        script_dir.mkdir(exist_ok=True)
        script_path = script_dir / "openroad_run.tcl"

        script_lef = self._to_container_path(host_lef, mount_root) if self.docker_image else str(host_lef)
        script_def = self._to_container_path(host_def, mount_root) if self.docker_image else str(host_def)
        script = self.build_script(script_lef, script_def, placement)
        script_path.write_text(script)

        try:
            completed = subprocess.run(
                self.build_command(host_lef, host_def, script_path),
                check=True,
                capture_output=True,
                text=True,
            )
            return {"stdout": completed.stdout, "stderr": completed.stderr}
        finally:
            script_path.unlink(missing_ok=True)

    def _to_container_path(self, path: Path, mount_root: Path) -> str:
        relative_path = path.relative_to(mount_root).as_posix()
        return f"/workspace/{relative_path}"
