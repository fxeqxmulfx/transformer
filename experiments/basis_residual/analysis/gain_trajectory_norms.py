"""Task161: retain sampled gate-gradient/update norms and a real-arithmetic bound.

Source: task160's unmodified diagnostics. For x=RMS(u), ||x||<=sqrt(d),
so |w.x|<=sqrt(d)||w||. For g=2sigmoid(z), its derivative divided by its
maximum1/2 is g(2-g). Less than10% requires |z|>2acosh(sqrt(10)). This
bound uses the real formula and recorded FP32 norms; it is not a formal
floating-point certificate and cannot reconstruct a past activation vector.
"""

import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    result = json.loads((HERE / "results.json").read_text())
    protocol = json.loads((HERE / "protocol.json").read_text())
    assert result["phase"]["status"] == "learning_closed"
    threshold = 2 * math.acosh(math.sqrt(10))
    cases = []
    for case in result["cases"]:
        if "-depth-" not in case["label"]:
            continue
        label = case["label"]
        expected = next(c for c in protocol["cases"] if c["label"] == label)
        width = expected["experiment"]["model"]["width"]
        cadence = expected["experiment"]["diagnostics"]["every"]
        folder = HERE / "runs" / label
        rows = [json.loads(s) for s in (folder / "diagnostics.jsonl").read_text().splitlines()]
        gradients = [json.loads(s) for s in (folder / "gradients.jsonl").read_text().splitlines()]
        assert [r["step"] for r in rows] == list(range(cadence, case["step"] + 1, cadence))
        assert [r["step"] for r in gradients] == list(range(1, case["step"] + 1))
        assert all(math.isfinite(r["gradient_l2"]) and r["gradient_l2"] > 0 for r in gradients)
        samples, gate_gradients = [], []
        for row in rows:
            gates = {name: dict(values) for name, values in row["parameters"].items() if "_gain." in name}
            for values in gates.values():
                assert all(math.isfinite(value) and value >= 0 for value in values.values())
                gate_gradients.append(values["gradient_l2"])
                values["real_abs_logit_upper_bound"] = math.sqrt(width) * values["parameter_l2"]
                values["fixed_input_logit_update_upper_bound"] = math.sqrt(width) * values["update_l2"]
            samples.append({"step": row["step"], "learning_rate": row["learning_rate"],
                            "training_batch_loss": row["loss"], "gates": gates})
        possible = [r["step"] for r in samples if any(
            value["real_abs_logit_upper_bound"] >= threshold for value in r["gates"].values())]
        cases.append({"label": label, "step": case["step"], "width": width,
                      "first_sample_where_bound_allows_saturation": possible[0] if possible else None,
                      "sampled_gate_gradient_count": len(gate_gradients),
                      "sampled_gate_zero_gradients": sum(g == 0 for g in gate_gradients),
                      "sampled_gate_gradient_range": [min(gate_gradients), max(gate_gradients)] if gate_gradients else None,
                      "all_global_gradients_finite_nonzero": True,
                      "first_learning_rate": gradients[0]["learning_rate"],
                      "raw_sha256": {name: sha(folder / name) for name in ("diagnostics.jsonl", "gradients.jsonl")},
                      "samples": samples})
    output = {"scope": "Post-hoc recorded training diagnostics. No new predictions, optimizer updates or checkpoint changes.",
              "source_sha256": sha(Path(__file__)), "closed_result_sha256": sha(HERE / "results.json"),
              "saturation_abs_logit_threshold": threshold,
              "limitation": "Only sampled norms, not gradient vectors or intermediate weights, are available. A bound above the threshold permits but does not establish saturation. The real-arithmetic bound omits rounding error. Fixed-input update bounds omit changes to the other model weights and features.",
              "cases": cases}
    path = HERE / "runs/gain_transfer/trajectory_norms.json"
    if path.exists():
        assert json.loads(path.read_text()) == output
    else:
        path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print("Trajectory-norm report sha256", sha(path), flush=True)


if __name__ == "__main__":
    main()
