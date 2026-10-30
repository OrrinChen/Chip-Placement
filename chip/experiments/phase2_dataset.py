from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from chip.cost import hpwl_cost
from chip.experiments.baseline import _neighbor
from chip.experiments.phase1_acceptance import _build_sa_initial_placement
from chip.experiments.phase1_acceptance import _relative_path
from chip.experiments.phase1_acceptance import _run_sa_baseline
from chip.experiments.phase1_acceptance import _select_injectable_instances
from chip.ispd_benchmarks import ACCEPTANCE_BENCHMARKS
from chip.ispd_benchmarks import ISPDBenchmark
from chip.ispd_benchmarks import prepare_benchmark_artifacts
from chip.legalization import legalize_placement
from chip.netlist_parser import load_chip_problem
from chip.openroad_bridge import OpenROADBridge
from chip.openroad_eval import build_evaluation_request
from chip.openroad_eval import evaluate_openroad_request


DEFAULT_RANDOM_SAMPLES = 4
DEFAULT_SA_TRUNCATION_STEPS = (1, 5, 10)
DEFAULT_RANDOM_PERTURBATION_STEPS = 5
DEFAULT_RANDOM_MOVE_PROBABILITY = 0.02
DEFAULT_RANDOM_MAX_X_FRACTION = 0.05
DEFAULT_RANDOM_ROW_WINDOW = 1
DEFAULT_FULL_RANDOM_TIERS = (
    {
        "move_probability": 0.1,
        "max_x_fraction": 0.1,
        "row_window": 1,
        "weight": 0.7,
    },
    {
        "move_probability": 0.2,
        "max_x_fraction": 0.1,
        "row_window": 1,
        "weight": 0.3,
    },
)
DEFAULT_FULL_RANDOM_TIER_OVERRIDES = {
    "mgc_des_perf_2": (
        {
            "move_probability": 0.1,
            "max_x_fraction": 0.1,
            "row_window": 1,
            "weight": 0.7,
        },
        {
            "move_probability": 0.15,
            "max_x_fraction": 0.1,
            "row_window": 1,
            "weight": 0.3,
        },
    ),
}
DEFAULT_TIMEOUT_SECONDS = 900
MAX_RANDOM_NEIGHBOR_ATTEMPTS = 32


def build_sampling_plan(
    benchmark_name: str,
    batch: str = "smoke",
    random_samples: int = DEFAULT_RANDOM_SAMPLES,
    sa_truncation_steps: tuple[int, ...] = DEFAULT_SA_TRUNCATION_STEPS,
    random_move_probability: float = DEFAULT_RANDOM_MOVE_PROBABILITY,
    random_max_x_fraction: float = DEFAULT_RANDOM_MAX_X_FRACTION,
    random_row_window: int = DEFAULT_RANDOM_ROW_WINDOW,
    random_tiers: tuple[dict[str, float | int], ...] | None = None,
) -> list[dict[str, object]]:
    plan: list[dict[str, object]] = []
    tiers = _resolve_random_tiers(
        benchmark_name=benchmark_name,
        batch=batch,
        random_move_probability=random_move_probability,
        random_max_x_fraction=random_max_x_fraction,
        random_row_window=random_row_window,
        random_tiers=random_tiers,
    )
    tier_counts = _allocate_random_samples(random_samples, tiers)
    index = 0
    for tier, count in zip(tiers, tier_counts):
        for _ in range(count):
            plan.append(
                {
                    "benchmark": benchmark_name,
                    "sample_id": f"rand_{index:04d}",
                    "source": "random_perturbation",
                    "batch": batch,
                    "random_steps": DEFAULT_RANDOM_PERTURBATION_STEPS,
                    "move_probability": tier["move_probability"],
                    "max_x_fraction": tier["max_x_fraction"],
                    "row_window": tier["row_window"],
                }
            )
            index += 1
    for steps in sa_truncation_steps:
        plan.append(
            {
                "benchmark": benchmark_name,
                "sample_id": f"sa_{steps:04d}",
                "source": "sa_truncated",
                "batch": batch,
                "sa_iterations": steps,
            }
        )
    return plan


def _resolve_random_tiers(
    benchmark_name: str,
    batch: str,
    random_move_probability: float,
    random_max_x_fraction: float,
    random_row_window: int,
    random_tiers: tuple[dict[str, float | int], ...] | None,
) -> tuple[dict[str, float | int], ...]:
    if random_tiers is not None:
        return random_tiers
    if batch == "full":
        if benchmark_name in DEFAULT_FULL_RANDOM_TIER_OVERRIDES:
            return DEFAULT_FULL_RANDOM_TIER_OVERRIDES[benchmark_name]
        return DEFAULT_FULL_RANDOM_TIERS
    return (
        {
            "move_probability": random_move_probability,
            "max_x_fraction": random_max_x_fraction,
            "row_window": random_row_window,
            "weight": 1.0,
        },
    )


def _allocate_random_samples(
    random_samples: int,
    tiers: tuple[dict[str, float | int], ...],
) -> list[int]:
    if random_samples <= 0:
        return [0 for _ in tiers]
    weights = [float(tier.get("weight", 0.0)) for tier in tiers]
    total_weight = sum(weights)
    if total_weight <= 0.0:
        raise ValueError("Random tier weights must sum to a positive value")
    exact_counts = [(random_samples * weight) / total_weight for weight in weights]
    base_counts = [int(count) for count in exact_counts]
    remaining = random_samples - sum(base_counts)
    remainders = sorted(
        ((exact_counts[index] - base_counts[index], index) for index in range(len(tiers))),
        reverse=True,
    )
    for _remainder, index in remainders[:remaining]:
        base_counts[index] += 1
    return base_counts


def summarize_dataset_row(
    sample: dict[str, object],
    metrics: dict[str, float | int],
    gate_status: str,
) -> dict[str, object]:
    row = {
        "benchmark": sample["benchmark"],
        "sample_id": sample["sample_id"],
        "source": sample["source"],
        "batch": sample.get("batch", "smoke"),
        "status": "success",
        "phase1_gate_status": gate_status,
        "total_routed_wirelength": metrics["total_routed_wirelength"],
        "wns": metrics["wns"],
        "tns": metrics["tns"],
        "drc_violations": metrics["drc_violations"],
        "runtime_seconds": metrics["runtime_seconds"],
    }
    for key, value in metrics.items():
        if key in row:
            continue
        row[key] = value
    return row


def collect_sample_result(
    sample: dict[str, object],
    gate_status: str,
    evaluate_fn,
    timeout_seconds: int,
) -> dict[str, object]:
    try:
        metrics = evaluate_fn(sample, timeout_seconds)
    except TimeoutError as error:
        return {
            "benchmark": sample["benchmark"],
            "sample_id": sample["sample_id"],
            "source": sample["source"],
            "batch": sample["batch"],
            "status": "timeout",
            "phase1_gate_status": gate_status,
            "timeout_seconds": timeout_seconds,
            "error": str(error),
        }
    except Exception as error:
        return {
            "benchmark": sample["benchmark"],
            "sample_id": sample["sample_id"],
            "source": sample["source"],
            "batch": sample["batch"],
            "status": "error",
            "phase1_gate_status": gate_status,
            "timeout_seconds": timeout_seconds,
            "error": str(error),
        }
    return summarize_dataset_row(sample=sample, metrics=metrics, gate_status=gate_status)


def load_phase1_gate_map(results_path: str | Path) -> dict[str, str]:
    rows = json.loads(Path(results_path).read_text())
    return {row["benchmark"]: row["phase1_gate_status"] for row in rows}


def build_dataset_manifest(
    phase1_results_path: str | Path,
    batch: str = "smoke",
    random_samples: int = DEFAULT_RANDOM_SAMPLES,
    sa_truncation_steps: tuple[int, ...] = DEFAULT_SA_TRUNCATION_STEPS,
    random_move_probability: float = DEFAULT_RANDOM_MOVE_PROBABILITY,
    random_max_x_fraction: float = DEFAULT_RANDOM_MAX_X_FRACTION,
    random_row_window: int = DEFAULT_RANDOM_ROW_WINDOW,
    benchmark_names: tuple[str, ...] | None = None,
) -> list[dict[str, object]]:
    gate_map = load_phase1_gate_map(phase1_results_path)
    rows: list[dict[str, object]] = []
    selected_names = set(benchmark_names or ())
    benchmarks = [
        benchmark
        for benchmark in ACCEPTANCE_BENCHMARKS
        if not selected_names or benchmark.name in selected_names
    ]
    for benchmark in benchmarks:
        for sample in build_sampling_plan(
            benchmark_name=benchmark.name,
            batch=batch,
            random_samples=random_samples,
            sa_truncation_steps=sa_truncation_steps,
            random_move_probability=random_move_probability,
            random_max_x_fraction=random_max_x_fraction,
            random_row_window=random_row_window,
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
    batch: str = "smoke",
    random_samples: int = DEFAULT_RANDOM_SAMPLES,
    sa_truncation_steps: tuple[int, ...] = DEFAULT_SA_TRUNCATION_STEPS,
    random_move_probability: float = DEFAULT_RANDOM_MOVE_PROBABILITY,
    random_max_x_fraction: float = DEFAULT_RANDOM_MAX_X_FRACTION,
    random_row_window: int = DEFAULT_RANDOM_ROW_WINDOW,
    benchmark_names: tuple[str, ...] | None = None,
) -> list[dict[str, object]]:
    output = Path(output_path)
    rows = build_dataset_manifest(
        phase1_results_path=phase1_results_path,
        batch=batch,
        random_samples=random_samples,
        sa_truncation_steps=sa_truncation_steps,
        random_move_probability=random_move_probability,
        random_max_x_fraction=random_max_x_fraction,
        random_row_window=random_row_window,
        benchmark_names=benchmark_names,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2))
    return rows


def collect_ground_truth(
    samples: list[dict[str, object]],
    output_path: str | Path,
    evaluate_fn: Callable[[dict[str, object], int, dict[str, Path]], dict[str, float | int]],
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> list[dict[str, object]]:
    output = Path(output_path)
    samples_root = output.parent / "phase2_samples"
    output.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = json.loads(output.read_text()) if output.exists() else []
    rows_by_key = {_sample_key(row): row for row in existing_rows}
    ordered_sample_keys = [_sample_key(sample) for sample in samples]
    for sample in samples:
        sample_key = _sample_key(sample)
        if sample_key in rows_by_key:
            continue
        artifact_paths = _sample_artifact_paths(samples_root=samples_root, sample=sample)

        def wrapped_evaluator(sample_row: dict[str, object], sample_timeout_seconds: int):
            return evaluate_fn(sample_row, sample_timeout_seconds, artifact_paths)

        row = collect_sample_result(
            sample=sample,
            gate_status=str(sample.get("phase1_gate_status", "unknown")),
            evaluate_fn=wrapped_evaluator,
            timeout_seconds=timeout_seconds,
        )
        if row["status"] == "success":
            row["placement_def"] = artifact_paths["placement_def"].resolve().as_posix()
            row["metrics_json"] = artifact_paths["metrics_json"].resolve().as_posix()
            artifact_paths["metrics_json"].write_text(json.dumps(row, indent=2))
        rows_by_key[sample_key] = row
        output.write_text(json.dumps(_ordered_rows(rows_by_key, ordered_sample_keys), indent=2))
    rows = _ordered_rows(rows_by_key, ordered_sample_keys)
    output.write_text(json.dumps(rows, indent=2))
    return rows


def run_ground_truth_collection(
    benchmark_root: str | Path,
    phase1_results_path: str | Path,
    output_path: str | Path,
    batch: str = "smoke",
    random_samples: int = DEFAULT_RANDOM_SAMPLES,
    sa_truncation_steps: tuple[int, ...] = DEFAULT_SA_TRUNCATION_STEPS,
    random_move_probability: float = DEFAULT_RANDOM_MOVE_PROBABILITY,
    random_max_x_fraction: float = DEFAULT_RANDOM_MAX_X_FRACTION,
    random_row_window: int = DEFAULT_RANDOM_ROW_WINDOW,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    benchmark_names: tuple[str, ...] | None = None,
) -> list[dict[str, object]]:
    samples = build_dataset_manifest(
        phase1_results_path=phase1_results_path,
        batch=batch,
        random_samples=random_samples,
        sa_truncation_steps=sa_truncation_steps,
        random_move_probability=random_move_probability,
        random_max_x_fraction=random_max_x_fraction,
        random_row_window=random_row_window,
        benchmark_names=benchmark_names,
    )

    def default_evaluator(sample: dict[str, object], sample_timeout_seconds: int, artifact_paths: dict[str, Path]):
        return _evaluate_sample(
            sample=sample,
            benchmark_root=Path(benchmark_root),
            artifact_paths=artifact_paths,
            timeout_seconds=sample_timeout_seconds,
        )

    return collect_ground_truth(
        samples=samples,
        output_path=output_path,
        evaluate_fn=default_evaluator,
        timeout_seconds=timeout_seconds,
    )


def _evaluate_sample(
    sample: dict[str, object],
    benchmark_root: Path,
    artifact_paths: dict[str, Path],
    timeout_seconds: int,
) -> dict[str, float | int]:
    benchmark = _benchmark_by_name(str(sample["benchmark"]))
    prepared = prepare_benchmark_artifacts(benchmark_root=benchmark_root, benchmark=benchmark)
    artifact_dir = Path(prepared["artifact_dir"])
    lef_paths = tuple(
        _relative_path(path, artifact_dir)
        for path in [prepared["tech_lef_path"], prepared["site_fix_lef_path"], prepared["cells_lef_path"]]
    )
    bridge = OpenROADBridge(docker_image="openroad/orfs")
    seed_def_path = _ensure_seed_placed_def(prepared=prepared, benchmark=benchmark, bridge=bridge)
    graph, problem = load_chip_problem(prepared["cells_lef_path"], seed_def_path)
    sample_placement = _build_sample_placement(problem=problem, graph=graph, sample=sample)
    injectable_placement, orientations = _select_injectable_instances(
        problem,
        sample_placement,
        changed_only=True,
    )
    legalized_name = f"{sample['sample_id']}.def"
    legalized_def_path = legalize_placement(
        artifact_dir=artifact_dir,
        lef_paths=lef_paths,
        def_path=_relative_path(seed_def_path, artifact_dir),
        placement=injectable_placement,
        orientations=orientations,
        bridge=bridge,
        output_def_name=legalized_name,
    )
    shutil.copyfile(legalized_def_path, artifact_paths["placement_def"])
    _, legalized_problem = load_chip_problem(prepared["cells_lef_path"], legalized_def_path)
    request = build_evaluation_request(
        benchmark_name=benchmark.name,
        artifact_dir=str(artifact_dir),
        lef_path=lef_paths[0],
        lef_paths=lef_paths,
        def_path=_relative_path(legalized_def_path, artifact_dir),
        liberty_path=_relative_path(prepared["liberty_path"], artifact_dir),
        sdc_path=_relative_path(prepared["sdc_path"], artifact_dir),
        placement_mode="sa_injected",
        detailed_route_end_iter=benchmark.detailed_route_end_iter,
        placement_density=benchmark.placement_density,
        script_stem=f"{sample['sample_id']}_sa_legalized",
        run_detailed_route=False,
    )
    try:
        metrics = evaluate_openroad_request(
            request,
            bridge=bridge,
            timeout_seconds=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(f"timed out after {timeout_seconds}s while evaluating {sample['sample_id']}") from error
    metrics["hpwl"] = hpwl_cost(legalized_problem, legalized_problem.placement())
    metrics.update(
        summarize_displacement_distribution(
            problem=problem,
            baseline_placement=problem.placement(),
            candidate_placement=legalized_problem.placement(),
        )
    )
    return metrics


def _ensure_seed_placed_def(
    prepared: dict[str, Path],
    benchmark: ISPDBenchmark,
    bridge: OpenROADBridge,
) -> Path:
    artifact_dir = Path(prepared["artifact_dir"])
    seed_def_path = artifact_dir / "seed_placed.def"
    if seed_def_path.exists():
        return seed_def_path

    lef_paths = tuple(
        _relative_path(path, artifact_dir)
        for path in [prepared["tech_lef_path"], prepared["site_fix_lef_path"], prepared["cells_lef_path"]]
    )
    request = build_evaluation_request(
        benchmark_name=benchmark.name,
        artifact_dir=str(artifact_dir),
        lef_path=lef_paths[0],
        lef_paths=lef_paths,
        def_path=_relative_path(prepared["def_path"], artifact_dir),
        liberty_path=_relative_path(prepared["liberty_path"], artifact_dir),
        sdc_path=_relative_path(prepared["sdc_path"], artifact_dir),
        placement_mode="built_in",
        detailed_route_end_iter=benchmark.detailed_route_end_iter,
        placement_density=benchmark.placement_density,
        write_def_path="seed_placed.def",
        script_stem="seed_placed",
        run_detailed_route=False,
    )
    evaluate_openroad_request(request, bridge=bridge)
    if not seed_def_path.exists():
        raise FileNotFoundError(f"OpenROAD did not produce {seed_def_path}")
    return seed_def_path


def _build_sample_placement(problem, graph, sample: dict[str, object]) -> dict[str, tuple[float, float]]:
    source = sample["source"]
    if source == "sa_truncated":
        initial_placement = _build_sa_initial_placement(graph, problem)
        placement, summary = _run_sa_baseline(
            problem,
            initial_placement=initial_placement,
            iterations=int(sample.get("sa_iterations", 1)),
        )
        return placement if bool(summary.get("is_valid", False)) else initial_placement
    if source == "random_perturbation":
        rng = random.Random(_stable_sample_seed(sample))
        placement = _build_dense_random_perturbation(problem, rng, sample)
        return placement
    raise ValueError(f"Unsupported sample source: {source}")


def _next_valid_neighbor(problem, placement, bounds, rng, constraints):
    for _ in range(MAX_RANDOM_NEIGHBOR_ATTEMPTS):
        candidate = _neighbor(problem, placement, bounds, rng)
        if constraints.check_all(candidate).is_valid:
            return candidate
    return placement


def _build_dense_random_perturbation(
    problem,
    rng: random.Random,
    sample: dict[str, object],
) -> dict[str, tuple[float, float]]:
    placement = dict(problem.placement())
    movable_macros = [name for name, macro in problem.macros.items() if not macro.is_fixed]
    move_probability = float(sample.get("move_probability", DEFAULT_RANDOM_MOVE_PROBABILITY))
    max_x_fraction = float(sample.get("max_x_fraction", DEFAULT_RANDOM_MAX_X_FRACTION))
    row_window = int(sample.get("row_window", DEFAULT_RANDOM_ROW_WINDOW))
    die_width = problem.die_area.width
    x_span = max(die_width * max_x_fraction, problem.site_pitch())

    for macro_name in movable_macros:
        if rng.random() > move_probability:
            continue
        current_x, current_y = placement[macro_name]
        next_coords = _sample_macro_perturbation(
            problem=problem,
            macro_name=macro_name,
            current_x=current_x,
            current_y=current_y,
            rng=rng,
            x_span=x_span,
            row_window=row_window,
        )
        placement[macro_name] = next_coords
    return placement


def _sample_macro_perturbation(
    problem,
    macro_name: str,
    current_x: float,
    current_y: float,
    rng: random.Random,
    x_span: float,
    row_window: int = DEFAULT_RANDOM_ROW_WINDOW,
) -> tuple[float, float]:
    rows = list(problem.rows)
    current_row_index = min(range(len(rows)), key=lambda index: abs(rows[index].origin_y - current_y)) if rows else -1
    candidate_rows = [
        row
        for index, row in enumerate(rows)
        if index != current_row_index and abs(index - current_row_index) <= max(row_window, 0)
    ]
    if not candidate_rows:
        candidate_rows = [row for row in rows if abs(row.origin_y - current_y) > 1e-9] or rows
    for _ in range(MAX_RANDOM_NEIGHBOR_ATTEMPTS):
        target_x = current_x + rng.uniform(-x_span, x_span)
        if candidate_rows:
            target_row = rng.choice(candidate_rows)
            target_y = target_row.origin_y
        else:
            target_y = current_y
        next_x, next_y, _ = problem.snap_movable_macro(macro_name, target_x, target_y)
        if abs(next_x - current_x) > 1e-9 or abs(next_y - current_y) > 1e-9:
            return (next_x, next_y)

    min_x = problem.die_area.min_x
    max_x = problem.die_area.max_x - problem.macros[macro_name].width
    fallback_x = min(max(current_x + x_span, min_x), max_x)
    next_x, next_y, _ = problem.snap_movable_macro(macro_name, fallback_x, current_y)
    return (next_x, next_y)


def _sample_artifact_paths(samples_root: Path, sample: dict[str, object]) -> dict[str, Path]:
    sample_dir = samples_root / str(sample["benchmark"])
    sample_dir.mkdir(parents=True, exist_ok=True)
    sample_id = str(sample["sample_id"])
    return {
        "sample_dir": sample_dir,
        "placement_def": sample_dir / f"{sample_id}.def",
        "metrics_json": sample_dir / f"{sample_id}.metrics.json",
    }


def _stable_sample_seed(sample: dict[str, object]) -> int:
    digest = hashlib.sha256(f"{sample['benchmark']}:{sample['sample_id']}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _ordered_rows(rows_by_key: dict[str, dict[str, object]], ordered_keys: list[str]) -> list[dict[str, object]]:
    ordered_rows = [rows_by_key[key] for key in ordered_keys if key in rows_by_key]
    remaining_keys = [key for key in rows_by_key if key not in ordered_keys]
    ordered_rows.extend(rows_by_key[key] for key in sorted(remaining_keys))
    return ordered_rows


def _sample_key(sample: dict[str, object]) -> str:
    return f"{sample['benchmark']}::{sample['sample_id']}"


def _benchmark_by_name(benchmark_name: str) -> ISPDBenchmark:
    for benchmark in ACCEPTANCE_BENCHMARKS:
        if benchmark.name == benchmark_name:
            return benchmark
    raise KeyError(f"Unknown benchmark: {benchmark_name}")


def summarize_displacement_distribution(
    problem,
    baseline_placement: dict[str, tuple[float, float]],
    candidate_placement: dict[str, tuple[float, float]],
) -> dict[str, float]:
    site_pitch = problem.site_pitch()
    displacements = []
    moved = 0
    movable_count = 0
    for macro_name, macro in problem.macros.items():
        if macro.is_fixed:
            continue
        movable_count += 1
        base_x, base_y = baseline_placement[macro_name]
        cand_x, cand_y = candidate_placement[macro_name]
        displacement_sites = (abs(cand_x - base_x) + abs(cand_y - base_y)) / site_pitch
        displacements.append(displacement_sites)
        if displacement_sites > 1e-9:
            moved += 1
    if not displacements:
        return {
            "displacement_median_sites": 0.0,
            "displacement_p90_sites": 0.0,
            "displacement_max_sites": 0.0,
            "moved_fraction": 0.0,
        }
    sorted_displacements = sorted(displacements)
    return {
        "displacement_median_sites": float(statistics.median(sorted_displacements)),
        "displacement_p90_sites": float(_percentile(sorted_displacements, 0.9)),
        "displacement_max_sites": float(sorted_displacements[-1]),
        "moved_fraction": moved / movable_count,
    }


def _percentile(sorted_values: list[float], fraction: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * fraction
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = position - lower_index
    return sorted_values[lower_index] * (1.0 - weight) + sorted_values[upper_index] * weight


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build or collect the Phase 2 placement dataset.")
    parser.add_argument(
        "--phase1-results",
        default=str(_repo_root() / "experiments" / "chip" / "phase1_acceptance_results.json"),
    )
    parser.add_argument(
        "--output",
        default=str(_repo_root() / "experiments" / "chip" / "phase2_dataset_manifest.json"),
    )
    parser.add_argument("--benchmark-root", default=str(_repo_root() / "tmp"))
    parser.add_argument("--batch", default="smoke")
    parser.add_argument("--random-samples", type=int, default=DEFAULT_RANDOM_SAMPLES)
    parser.add_argument("--sa-truncation-steps", type=int, nargs="*", default=list(DEFAULT_SA_TRUNCATION_STEPS))
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--random-move-probability", type=float, default=DEFAULT_RANDOM_MOVE_PROBABILITY)
    parser.add_argument("--random-max-x-fraction", type=float, default=DEFAULT_RANDOM_MAX_X_FRACTION)
    parser.add_argument("--random-row-window", type=int, default=DEFAULT_RANDOM_ROW_WINDOW)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--benchmarks", nargs="*")
    args = parser.parse_args(argv)

    if args.collect:
        rows = run_ground_truth_collection(
            benchmark_root=args.benchmark_root,
            phase1_results_path=args.phase1_results,
            output_path=args.output,
            batch=args.batch,
            random_samples=args.random_samples,
            sa_truncation_steps=tuple(args.sa_truncation_steps),
            random_move_probability=args.random_move_probability,
            random_max_x_fraction=args.random_max_x_fraction,
            random_row_window=args.random_row_window,
            timeout_seconds=args.timeout_seconds,
            benchmark_names=tuple(args.benchmarks) if args.benchmarks else None,
        )
        print(f"wrote {len(rows)} phase2 ground-truth rows to {args.output}")
        return

    rows = write_dataset_manifest(
        output_path=args.output,
        phase1_results_path=args.phase1_results,
        batch=args.batch,
        random_samples=args.random_samples,
        sa_truncation_steps=tuple(args.sa_truncation_steps),
        random_move_probability=args.random_move_probability,
        random_max_x_fraction=args.random_max_x_fraction,
        random_row_window=args.random_row_window,
        benchmark_names=tuple(args.benchmarks) if args.benchmarks else None,
    )
    print(f"wrote {len(rows)} phase2 samples to {args.output}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


if __name__ == "__main__":
    main()
