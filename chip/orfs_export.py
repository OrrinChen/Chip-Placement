from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ORFSExportBundle:
    design_name: str
    odb_path: Path
    output_dir: Path
    def_path: Path
    sdc_path: Path
    tech_lef_path: Path
    macro_lef_path: Path


class ORFSDesignExporter:
    def __init__(
        self,
        work_root: Path,
        export_root: Path,
        platform: str = "nangate45",
        docker_image: str = "openroad/orfs",
        docker_binary: str = "docker",
        docker_openroad_path: str = "/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad",
    ) -> None:
        self.work_root = Path(work_root)
        self.export_root = Path(export_root)
        self.platform = platform
        self.docker_image = docker_image
        self.docker_binary = docker_binary
        self.docker_openroad_path = docker_openroad_path

    def bundle_paths(self, design_name: str) -> ORFSExportBundle:
        output_dir = self.export_root / self.platform / design_name
        return ORFSExportBundle(
            design_name=design_name,
            odb_path=self.work_root / "results" / self.platform / design_name / "base" / "3_place.odb",
            output_dir=output_dir,
            def_path=output_dir / "3_place.def",
            sdc_path=output_dir / "3_place.sdc",
            tech_lef_path=output_dir / "NangateOpenCellLibrary.tech.lef",
            macro_lef_path=output_dir / "NangateOpenCellLibrary.macro.mod.lef",
        )

    def build_export_command(self, bundle: ORFSExportBundle, script_path: Path) -> list[str]:
        mount_root = self._mount_root()
        container_script = self._to_container_path(Path(script_path), mount_root)
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
            container_script,
        ]

    def export_design(self, design_name: str) -> dict[str, object]:
        bundle = self.bundle_paths(design_name)
        if not bundle.odb_path.exists():
            return {"status": "missing_odb", "bundle": bundle}

        bundle.output_dir.mkdir(parents=True, exist_ok=True)
        if self._bundle_is_ready(bundle):
            self._copy_host_sdc(bundle)
            return {"status": "ready", "bundle": bundle}

        if shutil.which(self.docker_binary) is None:
            return {"status": "tool_unavailable", "bundle": bundle, "tool": self.docker_binary}

        mount_root = self._mount_root()
        script_path = bundle.output_dir / "export.tcl"
        script_path.write_text(self.build_export_script(bundle, mount_root))
        try:
            subprocess.run(
                self.build_export_command(bundle, script_path),
                check=True,
                capture_output=True,
                text=True,
            )
            self.copy_reference_lefs(bundle, mount_root)
            self._copy_host_sdc(bundle)
        finally:
            script_path.unlink(missing_ok=True)

        return {"status": "exported", "bundle": bundle}

    def build_export_script(self, bundle: ORFSExportBundle, mount_root: Path) -> str:
        container_odb = self._to_container_path(bundle.odb_path, mount_root)
        container_def = self._to_container_path(bundle.def_path, mount_root)
        return "\n".join(
            [
                f"read_db {container_odb}",
                f"write_def {container_def}",
                "exit",
            ]
        )

    def copy_reference_lefs(self, bundle: ORFSExportBundle, mount_root: Path) -> None:
        container_macro = self._to_container_path(bundle.macro_lef_path, mount_root)
        container_tech = self._to_container_path(bundle.tech_lef_path, mount_root)
        script = " && ".join(
            [
                f"cp /OpenROAD-flow-scripts/flow/platforms/{self.platform}/lef/NangateOpenCellLibrary.tech.lef {container_tech}",
                f"cp /OpenROAD-flow-scripts/flow/platforms/{self.platform}/lef/NangateOpenCellLibrary.macro.mod.lef {container_macro}",
            ]
        )
        subprocess.run(
            [
                self.docker_binary,
                "run",
                "--rm",
                "-v",
                f"{mount_root.as_posix()}:/workspace",
                self.docker_image,
                "bash",
                "-lc",
                script,
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def _bundle_is_ready(self, bundle: ORFSExportBundle) -> bool:
        return bundle.def_path.exists() and bundle.tech_lef_path.exists() and bundle.macro_lef_path.exists()

    def _copy_host_sdc(self, bundle: ORFSExportBundle) -> None:
        source_sdc = bundle.odb_path.with_name("3_place.sdc")
        if source_sdc.exists():
            shutil.copy2(source_sdc, bundle.sdc_path)

    def _mount_root(self) -> Path:
        return Path(os.path.commonpath([self.work_root.resolve(), self.export_root.resolve()]))

    def _to_container_path(self, path: Path, mount_root: Path) -> str:
        return f"/workspace/{path.resolve().relative_to(mount_root).as_posix()}"
