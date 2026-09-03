import json
from pathlib import Path


def test_preregistered_baselines_match_pinned_systems():
    prereg = json.loads(Path("configs/superiority_preregistration.json").read_text(encoding="utf-8"))
    systems = json.loads(Path("configs/competitive_systems.json").read_text(encoding="utf-8"))
    names = {item["name"] for item in systems["systems"]}
    assert set(prereg["required_baselines"]).issubset(names)
    assert prereg["minimum_cases"] >= 300
    assert prereg["minimum_trials"] >= 10
    assert prereg["claim_policy"]["universal_superiority_forbidden"] is True
