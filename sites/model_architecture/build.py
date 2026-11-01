"""Build the local diagram from the architecture spec and recorded experiments.

No training or external assets: the chart contains the recorded validation
observations from residual_generalization, with no extrapolation.
"""

import hashlib
import json
from pathlib import Path
import shutil

SITE = Path(__file__).resolve().parent
ROOT = SITE.parents[1]


def build():
    output = SITE / "dist"
    output.mkdir(exist_ok=True)
    for name in ("index.html", "style.css", "app.js"):
        shutil.copyfile(SITE / name, output / name)
    shutil.copyfile(ROOT / "MODEL_ARCHITECTURE.md", output / "architecture.md")
    sources = {}
    reports = {}
    for experiment, filename in (
        ("residual_generalization", "transformer-results.json"),
        ("residual_gradient_paths", "synthetic-results.json"),
    ):
        source = ROOT / "experiments" / experiment / "results.json"
        content = source.read_bytes()
        (output / filename).write_bytes(content)
        sources[filename] = {
            "path": str(source.relative_to(ROOT)),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        reports[experiment] = json.loads(content)
    transformer = reports["residual_generalization"]
    continuation = ROOT / "experiments/residual_scaled_long/results.json"
    continued = {}
    if continuation.exists():
        content = continuation.read_bytes()
        (output / "continuation-results.json").write_bytes(content)
        sources["continuation-results.json"] = {
            "path": str(continuation.relative_to(ROOT)),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        continued = {case["parent_label"]: case for case in json.loads(content)["cases"]}
    cases = []
    for case in transformer["cases"]:
        if case["mode"] not in ("add", "scaled"):
            continue
        case = continued.get(case["label"], case)
        cases.append({
            "label": case["label"], "mode": case["mode"],
            "prime": case["prime"], "seed": case["seed"],
            "parameters": case["parameters"], "stop": case["stop"],
            "stable_validation": case.get("stable_validation", case.get("stable_validation999", False)),
            "test_accuracy": case["final_cpu_fp32"]["test"]["accuracy"],
            "validation_accuracy": case["final_cpu_fp32"]["validation"]["accuracy"],
            "curve": [
                {"step": row["step"], "accuracy": row["validation"]["accuracy"]}
                for row in case["learning_curve"]
            ],
        })
    gates = reports["residual_gradient_paths"]["cases"]
    synthetic = {}
    for placement in ("before", "after"):
        matching = [case for case in gates if f"-{placement}-" in case["label"]]
        synthetic[placement] = {
            "recovered": sum(bool(case["recovered"]) for case in matching),
            "total": len(matching),
        }
    evidence = {
        "sources": sources, "scope": transformer["scope"],
        "python": transformer["python"], "torch": transformer["torch"],
        "cases": cases, "synthetic_gate_recovery": synthetic,
        "continuation": [{
            "seed": case["seed"], "step": case["step"],
            "validation_accuracy": case["final_cpu_fp32"]["validation"]["accuracy"],
            "ordinary_first99_update": case["ordinary_first99"]["step"],
            "additional_updates": case["additional_updates"],
        } for case in continued.values()],
        "chart_note": "Recorded validation observations connected linearly; no extrapolation.",
    }
    (output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(f"Built {output}: {len(cases)} measured curves, {len(gates)} synthetic cases", flush=True)
    return output


if __name__ == "__main__":
    build()
