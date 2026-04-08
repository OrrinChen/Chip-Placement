from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


class OpenROADUnavailableError(RuntimeError):
    """Raised when OpenROAD is not installed or not on PATH."""


class OpenROADBridge:
    DEF_TO_ODB_ORIENTATION = {
        "N": "R0",
        "S": "R180",
        "E": "R270",
        "W": "R90",
        "FN": "MY",
        "FS": "MX",
        "FE": "MYR90",
        "FW": "MXR90",
    }

    def __init__(
        self,
        binary_name: str = "openroad",
        docker_image: str | None = None,
        docker_binary: str = "docker",
        docker_openroad_path: str = "/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad",
        docker_gpus: str | None = "all",
    ) -> None:
        self.binary_name = binary_name
        self.docker_image = docker_image
        self.docker_binary = docker_binary
        self.docker_openroad_path = docker_openroad_path
        self.docker_gpus = docker_gpus

    def is_available(self) -> bool:
        if self.docker_image is not None:
            return shutil.which(self.docker_binary) is not None
        return shutil.which(self.binary_name) is not None

    def build_script(
        self,
        lef_path: str | list[str] | tuple[str, ...],
        def_path: str,
        placement: dict[str, tuple[float, float]],
        placement_mode: str = "sa_injected",
        orientations: dict[str, str] | None = None,
        liberty_path: str | None = None,
        sdc_path: str | None = None,
        use_detailed_placement: bool = True,
        detailed_route_end_iter: int = 0,
        global_placement_density: float | None = None,
        write_def_path: str | None = None,
        run_routing_flow: bool = True,
        run_detailed_route: bool = True,
    ) -> str:
        lef_paths = [lef_path] if isinstance(lef_path, str) else list(lef_path)
        lines = [f"read_lef {path}" for path in lef_paths]
        lines.append(f"read_def {def_path}")
        if liberty_path is not None:
            lines.append(f"read_liberty {liberty_path}")
        if sdc_path is not None:
            lines.append(f"read_sdc {sdc_path}")
        lines.append("set_wire_rc -layer metal2")
        if placement_mode == "sa_injected":
            orientation_by_macro = orientations or {}
            for macro_name, (x, y) in placement.items():
                orient = self._to_odb_orientation(orientation_by_macro.get(macro_name, "N"))
                lines.append(
                    f"place_inst -name {{{macro_name}}} -location {{{x} {y}}} -orientation {orient} -status PLACED"
                )
        elif placement_mode != "built_in":
            raise ValueError(f"Unknown placement mode: {placement_mode}")
        elif placement_mode == "built_in":
            placement_command = "global_placement -skip_io"
            if global_placement_density is not None:
                placement_command += f" -density {global_placement_density}"
            lines.append(placement_command)
        if use_detailed_placement:
            lines.append("detailed_placement")
        if write_def_path is not None:
            lines.append(f"write_def {write_def_path}")
        if run_routing_flow:
            tail_lines = [
                "global_route",
                "estimate_parasitics -global_routing",
                "report_tns",
                "report_wns",
                "report_wire_length -net * -global_route -summary",
            ]
            if run_detailed_route:
                tail_lines.append(f"detailed_route -output_drc drc.rpt -droute_end_iter {detailed_route_end_iter}")
            lines.extend(tail_lines)
        return "\n".join(lines) + "\n"

    def build_command(
        self,
        lef_path: str | Path,
        def_path: str | Path,
        script_path: str | Path,
        referenced_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
        working_directory: str | Path | None = None,
        use_gpu: bool | None = None,
        container_name: str | None = None,
    ) -> list[str]:
        if self.docker_image is None:
            return [self.binary_name, "-exit", str(script_path)]

        host_paths = [self._normalize_host_path(lef_path), self._normalize_host_path(def_path), self._normalize_host_path(script_path)]
        if referenced_paths is not None:
            host_paths.extend(self._normalize_host_path(path) for path in referenced_paths)
        if working_directory is not None:
            host_paths.append(self._normalize_host_path(working_directory))

        mount_root = Path(os.path.commonpath(host_paths))
        container_script_path = self._to_container_path(self._normalize_host_path(script_path), mount_root)
        container_workdir = self._to_container_path(
            self._normalize_host_path(working_directory or Path(script_path).parent),
            mount_root,
        )
        command = [
            self.docker_binary,
            "run",
            "--rm",
        ]
        if container_name is not None:
            command.extend(["--name", container_name])
        command.extend(
            [
            "-v",
            f"{mount_root.as_posix()}:/workspace",
            "-w",
            container_workdir,
            ]
        )
        if use_gpu is None:
            use_gpu = self.docker_gpus is not None
        if use_gpu and self.docker_gpus is not None:
            command.extend(["--gpus", self.docker_gpus])
        command.extend(
            [
                self.docker_image,
                self.docker_openroad_path,
                "-exit",
                container_script_path,
            ]
        )
        return command

    def run_script(
        self,
        lef_path: str | Path,
        def_path: str | Path,
        script_path: str | Path,
        referenced_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
        working_directory: str | Path | None = None,
        **subprocess_kwargs,
    ) -> subprocess.CompletedProcess[str]:
        container_name = None
        if self.docker_image is not None:
            container_name = self._build_container_name(script_path)
        command = self.build_command(
            lef_path,
            def_path,
            script_path,
            referenced_paths=referenced_paths,
            working_directory=working_directory,
            use_gpu=True,
            container_name=container_name,
        )
        try:
            return subprocess.run(command, **subprocess_kwargs)
        except subprocess.TimeoutExpired:
            self._force_remove_container(container_name)
            raise
        except subprocess.CalledProcessError as error:
            if self._should_retry_without_gpu(error):
                fallback_command = self.build_command(
                    lef_path,
                    def_path,
                    script_path,
                    referenced_paths=referenced_paths,
                    working_directory=working_directory,
                    use_gpu=False,
                    container_name=container_name,
                )
                return subprocess.run(fallback_command, **subprocess_kwargs)
            raise

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
            completed = self.run_script(host_lef, host_def, script_path, check=True, capture_output=True, text=True)
            return {"stdout": completed.stdout, "stderr": completed.stderr}
        finally:
            script_path.unlink(missing_ok=True)

    def _to_container_path(self, path: Path, mount_root: Path) -> str:
        relative_path = path.relative_to(mount_root).as_posix()
        return f"/workspace/{relative_path}"

    def _normalize_host_path(self, path: str | Path) -> Path:
        return Path(path).resolve(strict=False)

    def _to_odb_orientation(self, orientation: str) -> str:
        return self.DEF_TO_ODB_ORIENTATION.get(orientation, orientation)

    def _build_container_name(self, script_path: str | Path) -> str:
        script = Path(script_path)
        script_stem = "".join(character if character.isalnum() else "-" for character in script.stem.lower()).strip("-")
        if not script_stem:
            script_stem = "openroad"
        suffix = abs(hash(str(script.resolve(strict=False)))) % 10_000_000
        return f"codex-openroad-{script_stem}-{os.getpid()}-{suffix}"

    def _force_remove_container(self, container_name: str | None) -> None:
        if self.docker_image is None or container_name is None:
            return
        subprocess.run(
            [self.docker_binary, "rm", "-f", container_name],
            capture_output=True,
            text=True,
            check=False,
        )

    def _should_retry_without_gpu(self, error: subprocess.CalledProcessError) -> bool:
        if self.docker_image is None or self.docker_gpus is None:
            return False
        details = " ".join(filter(None, [getattr(error, "stderr", None), getattr(error, "stdout", None)])).lower()
        gpu_markers = [
            "could not select device driver",
            "capabilities: [[gpu]]",
            "unknown flag: --gpus",
            "nvidia-container-runtime",
            "no gpu devices available",
        ]
        return any(marker in details for marker in gpu_markers)
