from __future__ import annotations

import argparse
import json
from pathlib import Path

from chip.ispd_benchmarks import ACCEPTANCE_BENCHMARKS


DEFAULT_RANDOM_SAMPLES = 4
DEFAULT_SA_TRUNCATION_STEPS = (1, 5, 10)


def build_sampling_plan(
    benchmark_name: str,
    random_samples: int = DEFAULT_RANDOM_SAMPLES,
    sa_truncation_steps: tuple[int, ...] = DEFAULT_SA_TRUNCATION_STEPS,
) -> list[dict[str, object]]:
    plan: list[dict[str, object]] = []
    for index in range(random_samples):
        plan.append(
            {
                "benchmark": benchmark_name,
                "sample_id": f"rand_{index:04d}",
                "source": "random_perturbation",
            }
        )
    for steps in sa_truncation_steps:
        plan.append(
            {
                "benchmark": benchmark_name,
                "sample_id": f"sa_{steps:04d}",
                "source": "sa_truncated",
                "sa_iterations": steps,
            }
        )
    return plan


def summarize_dataset_row(
    sample: dict[str, object],
    metrics: dict[str, float | int],
    gate_status: str,
) -> dict[str, object]:
    return {
        "benchmark": sample["benchmark"],
        "sample_id": sample["sample_id"],
        "source": sample["source"],
        "phase1_gate_status": gate_status,
        "total_routed_wirelength": metrics["total_routed_wirelength"],
        "wns": metrics["wns"],
        "tns": metrics["tns"],
        "drc_violations": metrics["drc_violations"],
        "runtime_seconds": metrics["runtime_seconds"],
    }


def load_phase1_gate_map(results_path: str | Path) -> dict[str, str]:
    rows = json.loads(Path(results_path).read_text())
    return {row["benchmark"]: row["phase1_gate_status"] for row in rows}


def build_dataset_manifest(
    phase1_results_path: str | Path,
    random_samples: int = DEFAULT_RANDOM_SAMPLES,
    sa_truncation_steps: tuple[int, ...] = DEFAULT_SA_TRUNCATION_STEPS,
) -> list[dict[str, object]]:
    gate_map = load_phase1_gate_map(phase1_results_path)
    rows: list[dict[str, object]] = []
    for benchmark in ACCEPTANCE_BENCHMARKS:
        for sample in build_sampling_plan(
            benchmark_name=benchmark.name,
            random_samples=random_samples,
            sa_truncation_steps=sa_truncation_steps,
        ):
            rows.append(
                {
                    **sample,
                    "scale_bucket": benchmark.scale_bucket,
                    "phase1_gate_status": gate_map.get(benchmark.name, "unknown"),
                }
            )
    return rows


def write_dataset_manifest(
    output_path: str | Path,
    phase1_results_path: str | Path,
    random_samples: int = DEFAULT_RANDOM_SAMPLES,
    sa_truncation_steps: tuple[int, ...] = DEFAULT_SA_TRUNCATION_STEPS,
) -> list[dict[str, object]]:
    output = Path(output_path)
    rows = build_dataset_manifest(
        phase1_results_path=phase1_results_path,
        random_samples=random_samples,
        sa_truncation_steps=sa_truncation_steps,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2))
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the Phase 2 placement-sampling manifest.")
    parser.add_argument(
        "--phase1-results",
        default=str(_repo_root() / "experiments" / "chip" / "phase1_acceptance_results.json"),
    )
    parser.add_argument(
        "--output",
        default=str(_repo_root() / "experiments" / "chip" / "phase2_dataset_manifest.json"),
    )
    parser.add_argument("--random-samples", type=int, default=DEFAULT_RANDOM_SAMPLES)
    parser.add_argument("--sa-truncation-steps", type=int, nargs="*", default=list(DEFAULT_SA_TRUNCATION_STEPS))
    args = parser.parse_args(argv)

    rows = write_dataset_manifest(
        output_path=args.output,
        phase1_results_path=args.phase1_results,
        random_samples=args.random_samples,
        sa_truncation_steps=tuple(args.sa_truncation_steps),
    )
    print(f"wrote {len(rows)} phase2 samples to {args.output}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    main()
