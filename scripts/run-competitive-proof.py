#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from knowledge_base_ai.benchmarking import KnowledgeBaseAIAdapter
from knowledge_base_ai.competitive_proof import (
    ExternalCompetitorAdapter,
    deterministic_heldout_cases,
    load_specs,
    run_repeated,
    technical_superiority_gate,
    write_report,
)


class _NativeKBAdapter:
    def __init__(self):
        self.inner = KnowledgeBaseAIAdapter()

    def run(self, case):
        return self.inner.run(case)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage B held-out competitive proof experiments.")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--per-category", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--config", default="configs/competitive_systems.json")
    parser.add_argument("--output", default="benchmark-results/stage-b")
    parser.add_argument("--external", action="store_true", help="Execute configured official competitor bridge commands.")
    args = parser.parse_args()

    cases = deterministic_heldout_cases(seed=args.seed, per_category=args.per_category)
    adapters = {"knowledge-base-ai-level5-6": _NativeKBAdapter()}
    specs = load_specs(args.config)
    if args.external:
        for spec in specs:
            adapters[spec.name] = ExternalCompetitorAdapter(spec)

    observations = run_repeated(adapters, cases, trials=args.trials)
    json_path, md_path = write_report(observations, args.output)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")

    if args.external:
        gate = technical_superiority_gate(
            observations,
            "knowledge-base-ai-level5-6",
            [spec.name for spec in specs],
        )
        gate_path = Path(args.output) / "superiority-gate.json"
        gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {gate_path}")
        print("technical-superiority claim gate:", "PASS" if gate["claim_allowed"] else "FAIL")
    else:
        print("official competitors were not executed; no competitive claim is permitted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
