from __future__ import annotations

from chip.experiments.phase3_comparison import run_phase3_comparison
from chip.problem import ChipProblem


def run_phase3_teacher_strength_sweep(
    *,
    problem: ChipProblem,
    grid_shape: tuple[int, int] = (32, 32),
    hidden_dim: int = 64,
    bc_epochs: int = 120,
    ppo_epochs: int = 1,
    reward_source: str = "analytic_hpwl",
    beam_widths: tuple[int, ...] = (1, 2, 4),
    seeds: tuple[int, ...] = (0,),
) -> dict[str, object]:
    rows = []
    reports: dict[str, dict[str, object]] = {}
    for beam_width in beam_widths:
        report = run_phase3_comparison(
            problem=problem,
            grid_shape=grid_shape,
            hidden_dim=hidden_dim,
            bc_epochs=bc_epochs,
            ppo_epochs=ppo_epochs,
            reward_source=reward_source,
            beam_width=beam_width,
            seeds=seeds,
        )
        reports[str(beam_width)] = report
        rows.append(_summarize_teacher_row(report))
    return {
        "beam_widths": [int(width) for width in beam_widths],
        "rows": rows,
        "reports": reports,
    }


def _summarize_teacher_row(report: dict[str, object]) -> dict[str, float]:
    methods = report["methods"]
    beam = methods["beam"]["aggregate"]
    warm = methods["ppo_warm_start"]["aggregate"]
    scratch = methods["ppo_scratch"]["aggregate"]
    return {
        "beam_width": float(report["beam_width"]),
        "beam_final_hpwl_mean": float(beam["final_hpwl"]),
        "warm_start_final_hpwl_mean": float(warm["final_hpwl"]),
        "scratch_final_hpwl_mean": float(scratch["final_hpwl"]),
        "warm_start_updates_to_beam_mean": float(warm["updates_to_beam_level"]),
        "scratch_updates_to_beam_mean": float(scratch["updates_to_beam_level"]),
        "warm_start_gap_to_beam_mean": float(warm["final_hpwl"] - beam["final_hpwl"]),
        "scratch_gap_to_beam_mean": float(scratch["final_hpwl"] - beam["final_hpwl"]),
    }
