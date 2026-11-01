"""Compare every fixed-batch residual diagnostic at early and long endpoints."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent


def summarize(batch):
    return {
        "accuracy": batch["accuracy"],
        "loss": batch["loss"],
        "final_state_mean_rms": batch["state_rms_mean_by_position"][-1][-1],
        "largest_parameter_gradient_rms": max(p["gradient_rms"] for p in batch["parameters"]),
        "largest_ambient_state_gradient_rms": max(batch["state_gradient_rms_all_positions"]),
        "initial_ambient_state_gradient_rms": batch["state_gradient_rms_all_positions"][0],
        "branch_to_state_rms_means": {r["branch"]: r["branch_to_residual_rms_mean"] for r in batch["branches"]},
        "branch_residual_cosine_means": {r["branch"]: r["cosine_with_residual_mean"] for r in batch["branches"]},
        "prefix_tied_readout_accuracy": batch["prefix_tied_readout_accuracy"],
        "input_strata": batch["input_strata"],
    }


def main():
    early_path = HERE.with_name("modular_composition") / "residual_diagnostics.json"
    long_path = HERE / "residual_diagnostics.json"
    early = {r["label"]: r for r in json.loads(early_path.read_text())["cases"]}
    long = json.loads(long_path.read_text())["cases"]
    comparisons = []
    for case in long:
        before = early[case["label"]]
        assert case["status"] == before["status"] == "measured"
        assert case["task"] == before["task"]
        assert before["step"] == 2000 and case["step"] == 100000
        for split in ("train", "heldout"):
            assert before["batches"][split]["manual_forward_bitwise_equal"]
            assert case["batches"][split]["manual_forward_bitwise_equal"]
            assert before["batches"][split]["examples"] == case["batches"][split]["examples"] == 512
        comparisons.append({
            "label": case["label"], "early_step": before["step"], "long_step": case["step"],
            "early_checkpoint_sha256": before["checkpoint_sha256"],
            "long_checkpoint_sha256": case["checkpoint_sha256"],
            "splits": {split: {"early": summarize(before["batches"][split]),
                               "long": summarize(case["batches"][split])}
                       for split in ("train", "heldout")},
        })
    result = {
        "scope": "Same first512 rows per split at2000 and100000 updates; original arithmetic, CPU FP32; all eight pairs retained",
        "limits": "Ambient state gradients can cancel under shared parameters. Tiny gradients at solved endpoints reflect low CE loss, not proof of trainability. Existing early exits are not trained classifiers. No causal residual ablation.",
        "input_sha256": {"early": hashlib.sha256(early_path.read_bytes()).hexdigest(),
                         "long": hashlib.sha256(long_path.read_bytes()).hexdigest()},
        "cases": comparisons,
    }
    (HERE / "residual_comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
