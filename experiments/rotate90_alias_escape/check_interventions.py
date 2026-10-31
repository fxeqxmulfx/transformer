"""Verify frozen tensors against the pinned original query, using saved artifacts.

Run from python/: uv run --locked python ../experiments/rotate90_alias_escape/check_interventions.py
This reads serialized tensors directly; it does not use the restart initializer.
"""

import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checks = []
    for directory in sorted((ROOT / "runs").glob("frozen-*")):
        manifest = json.loads((directory / "experiment.json").read_text())
        spec = manifest["description"]["model"]
        source_path = ROOT.parent.parent / spec["checkpoint"]
        assert sha256(source_path) == spec["sha256"]
        source = torch.load(source_path, map_location="cpu", weights_only=False)["model"]
        final_path = directory / "checkpoint.pt"
        final = torch.load(final_path, map_location="cpu", weights_only=False)
        frozen = [name for name in source
                  if (name == "E" and not spec["train_embedding"])
                  or (name != "E" and not spec["train_branch"])]
        assert frozen
        for name in frozen:
            assert torch.equal(source[name], final["model"][name]), (directory.name, name)
        assert len(final["optimizer"]["param_groups"][0]["params"]) == len(source) - len(frozen)
        checks.append({"label": directory.name, "source_sha256": spec["sha256"],
                       "final_sha256": sha256(final_path), "bitwise_unchanged": frozen,
                       "optimizer_tensor_count": len(source) - len(frozen)})
    assert len(checks) == 12
    (ROOT / "intervention_audit.json").write_text(json.dumps({
        "script_sha256": sha256(Path(__file__)), "checks": checks}, indent=2) + "\n")
    print("Verified all 12 frozen-parameter interventions against pinned source tensors")


if __name__ == "__main__":
    main()
