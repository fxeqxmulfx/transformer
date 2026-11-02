"""Independent same-assistant audit of the closed Basis records."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

root = Path(sys.argv[1]).resolve()
here = root / "experiments/basis_residual"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
protocol = json.loads((here / "protocol.json").read_text())
result = json.loads((here / "results.json").read_text())
phase = json.loads((here / "runs/phase.json").read_text())
assert phase["status"] == "learning_closed" and phase == result["phase"]
assert result["protocol_sha256"] == sha(here / "protocol.json")
expected = {case["label"]: case for case in protocol["cases"]}
assert set(expected) == set(phase["cases"]) == {case["label"] for case in result["cases"]}
raw_path = here / result["storage"]["full_result_path"]
assert sha(raw_path) == result["storage"]["full_result_sha256"]
cases = []
torch.set_num_threads(1)
for case in result["cases"]:
    label = case["label"]
    directory = here / "runs" / label
    assert sha(directory / "checkpoint.pt") == case["checkpoint_sha256"]
    state = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert state["step"] == case["step"] == case["stop"]["step"]
    assert all(torch.isfinite(t).all().item() for t in state["model"].values())
    history = [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines()]
    assert history[-1]["step"] == state["step"]
    assert all("test" not in key for row in history for key in row)
    for raw, compact in zip(history, case["history"], strict=True):
        for value in raw.values():
            if isinstance(value, dict):
                value.pop("classes", None)
        assert raw == compact
    benchmark = expected[label]["experiment"]["benchmark"]
    name = "validation" if benchmark["select"] is None else "validation/" + benchmark["select"]
    crossings = [row["step"] for row in history if row[name]["sequence_accuracy"] >= .99]
    if case["status"] == "validation99":
        assert crossings == [state["step"]]
    else:
        assert case["status"] == "censored" and not crossings
        assert case["censor_reason"].strip()
    count, zero = 0, 0
    with (directory / "gradients.jsonl").open() as stream:
        for count, line in enumerate(stream, 1):
            row = json.loads(line)
            assert row["step"] == count and math.isfinite(row["gradient_l2"])
            zero += row["gradient_l2"] == 0
    assert count == case["gradient_count"] == state["step"]
    assert zero == case["gradient_zero_count"] == 0
    cuda, cpu = history[-1][name], case["final_cpu_replay"][name]
    count_fields = ("exact", "correct", "example_count", "target_count", "generated_tokens", "extra_tokens")
    differences = {key: {"cuda": cuda[key], "cpu": cpu[key]} for key in count_fields
                   if key in cuda and cuda[key] != cpu[key]}
    cases.append({"label": label, "status": case["status"], "step": state["step"],
                  "gradient_records": count, "zero_gradients": zero,
                  "first99_step": crossings[0] if crossings else None,
                  "validation_count_differences_cpu_cuda": differences,
                  "test_sequence": case["final_cpu_replay"]["test"]["sequence_accuracy"],
                  "novel_test_sequence": case["final_cpu_replay"]["test/novel"]["sequence_accuracy"],
                  "checkpoint_sha256": case["checkpoint_sha256"]})
audit = {"reviewed_utc": datetime.now(timezone.utc).isoformat(),
         "result_sha256": sha(here / "results.json"), "audit_source_sha256": sha(Path(__file__)),
         "status_counts": dict(Counter(c["status"] for c in cases)),
         "total_updates": sum(c["step"] for c in cases),
         "all_gradients_finite_nonzero_and_complete": True,
         "all_saved_models_finite": True, "every_observed_first_crossing_retained": True,
         "all_history_aggregates_unchanged_by_storage_reduction": True,
         "cases": cases}
output = here / "analysis/final_audit.json"
if output.exists():
    previous = json.loads(output.read_text())
    assert {k: v for k, v in previous.items() if k != "reviewed_utc"} == {
        k: v for k, v in audit.items() if k != "reviewed_utc"}
else:
    output.write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit, indent=2))
