#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Clone exact competitor commits for Stage B benchmarking.")
    parser.add_argument("--config", default="configs/competitive_systems.json")
    parser.add_argument("--root", default=".competitors")
    args = parser.parse_args()

    payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = []

    for spec in payload["systems"]:
        target = root / spec["name"]
        if not target.exists():
            run(["git", "clone", "--filter=blob:none", "--no-checkout", spec["repository"], str(target)])
        run(["git", "-C", str(target), "fetch", "origin", spec["commit"], "--depth", "1"])
        run(["git", "-C", str(target), "checkout", "--detach", spec["commit"]])
        actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
        if actual != spec["commit"]:
            raise SystemExit(f"pin verification failed for {spec['name']}: {actual} != {spec['commit']}")
        manifest.append({"name": spec["name"], "repository": spec["repository"], "commit": actual, "path": str(target)})
        print(f"pinned {spec['name']} @ {actual[:12]}")

    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
