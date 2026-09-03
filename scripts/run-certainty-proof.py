#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

from knowledge_base_ai.benchmarking import KnowledgeBaseAIAdapter
from knowledge_base_ai.certainty_cases import certainty_heldout_cases
from knowledge_base_ai.competitive_proof import ExternalCompetitorAdapter, load_specs, run_repeated, write_report
from knowledge_base_ai.proof_assurance import build_reproducibility_bundle, write_bundle
from knowledge_base_ai.superiority_certainty import (
    FrozenRunManifest,
    ResourceEnvelope,
    build_claim_certificate,
    capture_environment,
    detect_contamination,
    hash_records,
    negative_controls,
    verify_resource_parity,
    write_certificate,
)


class _NativeKBAdapter:
    def __init__(self):
        self.inner = KnowledgeBaseAIAdapter()

    def run(self, case):
        return self.inner.run(case)


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resource_envelope(args) -> ResourceEnvelope:
    return ResourceEnvelope(
        model=args.model,
        embedding_model=args.embedding_model,
        reranker=args.reranker or None,
        max_context_tokens=args.max_context_tokens,
        max_output_tokens=args.max_output_tokens,
        concurrency=args.concurrency,
        cache_policy=args.cache_policy,
        retry_policy=args.retry_policy,
        hardware_class=args.hardware_class,
        provider=args.provider,
        temperature=args.temperature,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen superiority-certainty proof. No proxy execution is permitted.")
    parser.add_argument("--preregistration", default="configs/superiority_preregistration.json")
    parser.add_argument("--systems", default="configs/competitive_systems.json")
    parser.add_argument("--output", default="benchmark-results/certainty")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--per-category", type=int, default=40)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--model", required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--reranker", default="")
    parser.add_argument("--max-context-tokens", type=int, default=8192)
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--cache-policy", default="cold")
    parser.add_argument("--retry-policy", default="none")
    parser.add_argument("--hardware-class", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tuning-corpus", action="append", default=[])
    parser.add_argument("--known-training-text", action="append", default=[])
    parser.add_argument("--allow-unreplicated-certificate", action="store_true", help="Write a local-only certificate; public claim remains forbidden.")
    args = parser.parse_args()

    prereg = _load_json(args.preregistration)
    specs = load_specs(args.systems)
    spec_by_name = {spec.name: spec for spec in specs}
    required_baselines = list(prereg["required_baselines"])
    missing = sorted(set(required_baselines) - set(spec_by_name))
    if missing:
        raise SystemExit(f"preregistration references unpinned baselines: {missing}")
    if args.trials < int(prereg["minimum_trials"]):
        raise SystemExit(f"trials {args.trials} < preregistered minimum {prereg['minimum_trials']}")

    cases = certainty_heldout_cases(seed=args.seed, per_category=args.per_category)
    if len(cases) < int(prereg["minimum_cases"]):
        raise SystemExit(f"cases {len(cases)} < preregistered minimum {prereg['minimum_cases']}")
    categories = {case.category for case in cases}
    missing_categories = sorted(set(prereg["required_categories"]) - categories)
    if missing_categories:
        raise SystemExit(f"held-out suite missing preregistered categories: {missing_categories}")

    case_payloads = [
        {
            "case_id": c.case_id,
            "category": c.category,
            "query": c.query,
            "expected_ids": c.expected_ids,
        }
        for c in cases
    ]
    contamination = detect_contamination(case_payloads, args.tuning_corpus, args.known_training_text)
    if not contamination.clean:
        raise SystemExit(f"contamination gate failed: {asdict(contamination)}")

    envelope = _resource_envelope(args)
    envelopes = {prereg["challenger"]: envelope, **{name: envelope for name in required_baselines}}
    parity = verify_resource_parity(envelopes)
    if not parity.equal:
        raise SystemExit(f"resource parity failed: {parity.differences}")

    competitor_commits = {name: spec_by_name[name].commit for name in required_baselines}
    seeds = tuple(args.seed + i for i in range(args.trials))
    manifest = FrozenRunManifest(
        benchmark_commit=_git_sha(),
        dataset_hash=hash_records([{"documents": [asdict(d) for d in c.documents]} for c in cases]),
        query_hash=hash_records(case_payloads),
        random_seeds=seeds,
        competitor_commits=competitor_commits,
        resource_envelope=envelope,
        preregistered_metrics=tuple(prereg["required_metrics"]),
        minimum_cases=int(prereg["minimum_cases"]),
        minimum_trials=int(prereg["minimum_trials"]),
        alpha=float(prereg["alpha"]),
        superiority_margin=float(prereg["superiority_margin"]),
        noninferiority_margin=float(prereg["noninferiority_margin"]),
    )

    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    (root / "frozen-manifest.json").write_text(json.dumps({**asdict(manifest), "fingerprint": manifest.fingerprint()}, indent=2, sort_keys=True, default=list) + "\n", encoding="utf-8")
    (root / "environment.json").write_text(json.dumps(capture_environment(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Certainty mode makes incomplete native competitor output a hard failure.
    os.environ["KBAI_CERTAINTY_MODE"] = "1"
    os.environ["KBAI_RESOURCE_ENVELOPE_FINGERPRINT"] = envelope.fingerprint()
    adapters = {prereg["challenger"]: _NativeKBAdapter()}
    for name in required_baselines:
        adapters[name] = ExternalCompetitorAdapter(spec_by_name[name])

    observations = run_repeated(adapters, cases, trials=args.trials)
    json_path, md_path = write_report(observations, root)

    official_verified = {name: bool(spec_by_name[name].official) for name in required_baselines}
    external_execution_verified = {
        name: any(r.system == name and r.provenance.get("commit") == spec_by_name[name].commit for r in observations)
        for name in required_baselines
    }
    controls = negative_controls(observations)
    certificate = build_claim_certificate(
        observations=observations,
        challenger=prereg["challenger"],
        baselines=required_baselines,
        manifest=manifest,
        parity=parity,
        contamination=contamination,
        official_verified=official_verified,
        external_execution_verified=external_execution_verified,
        negative_control=controls,
        minimum_power=float(prereg["minimum_power"]),
    )

    replication_required = bool(prereg["claim_policy"].get("independent_replication_required_for_public_claim", True))
    if replication_required:
        certificate.permitted = False
        certificate.blockers = sorted(set(certificate.blockers + ["independent replication required before public superiority claim"]))
        certificate.scope = "local experimental evidence only; public superiority claim forbidden pending independent replication"
        certificate.claim_text = "Independent replication is still required before any public technical-superiority claim."
    if certificate.permitted or args.allow_unreplicated_certificate:
        write_certificate(certificate, root / "claim-certificate.json")
    else:
        write_certificate(certificate, root / "claim-certificate.blocked.json")

    bundle = build_reproducibility_bundle(
        manifest_fingerprint=manifest.fingerprint(),
        files={
            "preregistration": args.preregistration,
            "systems": args.systems,
            "manifest": root / "frozen-manifest.json",
            "results": json_path,
            "report": md_path,
        },
        environment=capture_environment(),
        command=" ".join(os.sys.argv),
    )
    write_bundle(bundle, root / "reproducibility-bundle.json")

    print(json.dumps({
        "manifest_fingerprint": manifest.fingerprint(),
        "cases": len(cases),
        "trials": args.trials,
        "systems": list(adapters),
        "statistical_gate_passed_before_replication": not any("competitive evidence" in b for b in certificate.blockers),
        "public_claim_permitted": certificate.permitted,
        "blockers": certificate.blockers,
    }, indent=2))
    return 0 if not any(b for b in certificate.blockers if b != "independent replication required before public superiority claim") else 2


if __name__ == "__main__":
    raise SystemExit(main())
