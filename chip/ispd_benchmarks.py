from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ISPDBenchmark:
    name: str
    source_family: str
    scale_bucket: str
    artifact_dir: str
    compatibility_status: str = "candidate"


ACCEPTANCE_BENCHMARKS = [
    ISPDBenchmark("ibm01", "ISPD 1998", "small", "ibm01"),
    ISPDBenchmark("adaptec1", "ISPD 2005", "medium", "adaptec1"),
    ISPDBenchmark("bigblue1", "ISPD 2005", "large", "bigblue1"),
]


def resolve_artifact_paths(benchmark_root: Path, benchmark_name: str) -> dict[str, Path]:
    root = Path(benchmark_root) / benchmark_name
    return {
        "lef_path": root / "design.lef",
        "def_path": root / "design.def",
        "sdc_path": root / "design.sdc",
    }
