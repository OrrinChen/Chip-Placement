from __future__ import annotations

from pathlib import Path

from chip.openroad_bridge import OpenROADBridge


def legalize_placement(
    artifact_dir: str | Path,
    lef_paths: tuple[str, ...],
    def_path: str,
    placement: dict[str, tuple[float, float]],
    orientations: dict[str, str],
    bridge: OpenROADBridge | None = None,
    output_def_name: str = "legalized.def",
) -> Path:
    artifact_root = Path(artifact_dir)
    evaluator = bridge or OpenROADBridge(docker_image="openroad/orfs")
    script_path = artifact_root / ".sa_legalize.tcl"
    script_path.write_text(
        evaluator.build_script(
            lef_paths,
            def_path,
            placement,
            placement_mode="sa_injected",
            orientations=orientations,
            write_def_path=output_def_name,
            run_routing_flow=False,
        )
    )

    host_lef_path = artifact_root / lef_paths[0]
    host_def_path = artifact_root / def_path
    try:
        evaluator.run_script(
            host_lef_path,
            host_def_path,
            script_path,
            referenced_paths=[artifact_root / path for path in lef_paths],
            working_directory=artifact_root,
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        script_path.unlink(missing_ok=True)

    output_path = artifact_root / output_def_name
    if not output_path.exists():
        raise FileNotFoundError(f"OpenROAD legalization did not produce {output_path}")
    return output_path
