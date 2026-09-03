from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

PINS = {
    "microsoft-graphrag": "f40e9a26ce62ba0b3fef8837d24aafdcc6e6c704",
    "lightrag": "c1248646e4eda4d89054926af2e094730daf23fe",
    "raptor": "7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767",
    "llamaindex-conventional-rag": "949f2b83dbff276238fbdf0490cf06fe06a28189",
}

ENV_COMMAND = {
    "microsoft-graphrag": "KBAI_GRAPHRAG_NATIVE_COMMAND",
    "lightrag": "KBAI_LIGHTRAG_NATIVE_COMMAND",
    "raptor": "KBAI_RAPTOR_NATIVE_COMMAND",
    "llamaindex-conventional-rag": "KBAI_LLAMAINDEX_NATIVE_COMMAND",
}

CERTAINTY_REQUIRED = {
    "answer",
    "retrieved_ids",
    "cited_ids",
    "confidence",
    "abstained",
    "latency_ms",
    "estimated_cost",
    "independent_origins",
}


def _verify_pin(system: str) -> Path:
    checkout = Path(".competitors") / system
    if not checkout.exists():
        raise RuntimeError(f"missing pinned checkout {checkout}; run scripts/bootstrap-competitors.py")
    actual = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if actual != PINS[system]:
        raise RuntimeError(f"pin mismatch for {system}: {actual} != {PINS[system]}")
    return checkout


def _validate_certainty_result(result: dict) -> None:
    missing = sorted(CERTAINTY_REQUIRED - set(result))
    if missing:
        raise RuntimeError(f"certainty-mode native output missing required fields: {', '.join(missing)}")
    if not isinstance(result["answer"], str):
        raise RuntimeError("certainty-mode answer must be a string")
    if not isinstance(result["abstained"], bool):
        raise RuntimeError("certainty-mode abstained must be boolean")
    for name in ("retrieved_ids", "cited_ids"):
        if not isinstance(result[name], list) or not all(isinstance(x, str) for x in result[name]):
            raise RuntimeError(f"certainty-mode {name} must be a list of strings")
    confidence = float(result["confidence"])
    if not 0 <= confidence <= 1:
        raise RuntimeError("certainty-mode confidence must be in [0,1]")


def run_bridge(system: str) -> int:
    checkout = _verify_pin(system)
    command_text = os.getenv(ENV_COMMAND[system], "").strip()
    if not command_text:
        raise RuntimeError(
            f"{ENV_COMMAND[system]} is not configured. Stage B refuses to substitute a proxy for {system}. "
            "Set it to a native runner command that reads one BenchmarkCase JSON object on stdin and emits the normalized JSON result."
        )
    payload = sys.stdin.read()
    json.loads(payload)
    env = os.environ.copy()
    env["KBAI_COMPETITOR_CHECKOUT"] = str(checkout.resolve())
    env["KBAI_COMPETITOR_COMMIT"] = PINS[system]
    proc = subprocess.run(shlex.split(command_text), input=payload, text=True, capture_output=True, env=env)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return proc.returncode
    result = json.loads(proc.stdout)
    if os.getenv("KBAI_CERTAINTY_MODE", "0") == "1":
        _validate_certainty_result(result)
    result.setdefault("provenance", {})
    result["provenance"].update({
        "system": system,
        "commit": PINS[system],
        "checkout": str(checkout),
        "certainty_mode": os.getenv("KBAI_CERTAINTY_MODE", "0"),
    })
    sys.stdout.write(json.dumps(result))
    return 0
