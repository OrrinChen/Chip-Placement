from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chip.experiments.phase3_macroplacement_ppo import run_ariane_fragile_decision_probe


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / "phase3_ariane_fragile_decision_probe.json"
    start = time.time()
    artifact = run_ariane_fragile_decision_probe(output_path=out_path)
    summary = {
        "output_path": str(out_path),
        "elapsed_seconds": time.time() - start,
        "baseline_materiality_counts": artifact["panel_summaries"]["baseline"]["materiality_counts"],
        "alpha10_materiality_counts": artifact["panel_summaries"]["alpha10_overlap"]["materiality_counts"],
    }
    (out_dir / "phase3_ariane_fragile_decision_probe_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
