from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


class OpenROADUnavailableError(RuntimeError):
    """Raised when OpenROAD is not installed or not on PATH."""


class OpenROADBridge:
    def __init__(self, binary_name: str = "openroad") -> None:
        self.binary_name = binary_name

    def is_available(self) -> bool:
        return shutil.which(self.binary_name) is not None

    def build_script(
        self,
        lef_path: str | Path,
        def_path: str | Path,
        placement: dict[str, tuple[float, float]],
    ) -> str:
        lines = [f"read_lef {Path(lef_path)}", f"read_def {Path(def_path)}"]
        for macro_name, (x, y) in placement.items():
            lines.append(f"# place {macro_name} at ({x}, {y})")
        lines.extend(["global_route", "report_tns", "report_wns"])
        return "\n".join(lines) + "\n"

    def evaluate_placement(
        self,
        lef_path: str | Path,
        def_path: str | Path,
        placement: dict[str, tuple[float, float]],
    ) -> dict[str, str]:
        if not self.is_available():
            raise OpenROADUnavailableError(f"OpenROAD binary '{self.binary_name}' is not available")

        script = self.build_script(lef_path, def_path, placement)
        with tempfile.NamedTemporaryFile("w", suffix=".tcl", delete=False) as handle:
            handle.write(script)
            script_path = handle.name

        completed = subprocess.run(
            [self.binary_name, "-exit", script_path],
            check=True,
            capture_output=True,
            text=True,
        )
        return {"stdout": completed.stdout, "stderr": completed.stderr}
