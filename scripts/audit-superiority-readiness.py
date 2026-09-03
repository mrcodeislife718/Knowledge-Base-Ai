#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

REQUIRED = [
    "configs/competitive_systems.json",
    "configs/superiority_preregistration.json",
    "src/knowledge_base_ai/competitive_proof.py",
    "src/knowledge_base_ai/superiority_certainty.py",
    "src/knowledge_base_ai/proof_assurance.py",
    "scripts/bootstrap-competitors.py",
    "scripts/run-competitive-proof.py",
]


def main() -> int:
    missing = [path for path in REQUIRED if not Path(path).exists()]
    config = json.loads(Path("configs/superiority_preregistration.json").read_text(encoding="utf-8"))
    competitors = json.loads(Path("configs/competitive_systems.json").read_text(encoding="utf-8"))
    names = {item["name"] for item in competitors["systems"]}
    missing_baselines = [name for name in config["required_baselines"] if name not in names]
    result = {
        "ready_for_execution": not missing and not missing_baselines,
        "missing_files": missing,
        "missing_pinned_baselines": missing_baselines,
        "minimum_cases": config["minimum_cases"],
        "minimum_trials": config["minimum_trials"],
        "claim_policy": config["claim_policy"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ready_for_execution"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
