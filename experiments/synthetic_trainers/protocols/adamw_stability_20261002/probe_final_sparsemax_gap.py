"""Read-only final-checkpoint measurements of the strict-gap Lean premises."""

import json
from pathlib import Path
from unittest.mock import patch

import torch

from experiments.synthetic_trainers import sparsemax_attention
from experiments.synthetic_trainers.attention_layout import digest
from experiments.synthetic_trainers.protocols.adamw_stability_20261002 import probe_sparsemax_routes
from experiments.synthetic_trainers.stability_report import write_json


ARCHIVE = Path("experiments/synthetic_trainers/baselines/adamw_sparsemax_final_certificate_20261003")
CHECKPOINT = Path("experiments/runs/adamw_stability_20261002/attention_mod193_fraction25_lr0003_budget300k/adamw-sparsemax/checkpoint.pt")
SNAPSHOT = Path("experiments/runs/adamw_stability_20261002/bootstrap/sparsemax-final-gap-checkpoint-20261003.pt")


def main():
    receipt = json.loads((ARCHIVE / "CPU-export-validation.json").read_text())
    assert digest(CHECKPOINT) == receipt["checkpoint_sha256"]
    original = sparsemax_attention.causal_sparsemax
    observations = []

    def record_gap(scores):
        weights = original(scores)
        with torch.no_grad():
            width = scores.shape[-1]
            future = torch.ones(width, width, dtype=torch.bool).triu(1)
            ordered = scores.detach().masked_fill(future, -torch.inf).topk(2, dim=-1).values
            gaps = (ordered[..., 0] - ordered[..., 1])[:, :, 4:]
            singletons = ((weights > 0).sum(-1) == 1)[:, :, 4:]
            strict = gaps > 1
            assert not (strict & ~singletons).any()
            observations.append({"directly_supervised_rows": gaps.numel(),
                "strict_unit_gap_rows": int(strict.sum()),
                "singleton_support_rows": int(singletons.sum()),
                "singleton_without_strict_unit_gap_rows": int((singletons & ~strict).sum()),
                "minimum_gap_among_strict_rows": float(gaps[strict].min()) if strict.any() else None})
        return weights

    with patch.object(sparsemax_attention, "causal_sparsemax", record_gap):
        results = probe_sparsemax_routes.execute(ARCHIVE / "final-routing-probe", SNAPSHOT, CHECKPOINT)
    assert len(results) == 2 and len(observations) == 8
    for batch_index, result in enumerate(results):
        start = batch_index * 4
        observed = observations[start:start + 2]
        assert observed == observations[start + 2:start + 4]
        for layer, gaps in zip(result["layers"], observed):
            assert layer["directly_supervised_rows"] == gaps["directly_supervised_rows"]
            assert layer["directly_supervised_support_histogram"].get("1", 0) == gaps["singleton_support_rows"]
            layer.update(gaps)
    output = {"scope": "actual_final_checkpoint_CPU_two_training_batches; no_causal_accuracy_claim",
        "checkpoint_sha256": receipt["checkpoint_sha256"], "checkpoint_step": 300000,
        "strictness_threshold": 1, "score_scale": "original_GPTMini_attention_scores",
        "optimizer_updates_performed": 0, "CUDA_initialized": torch.cuda.is_initialized(),
        "gap_instrumentation_matches_uninstrumented_results": True,
        "program_sha256": digest(__file__), "probes": results}
    assert not output["CUDA_initialized"]
    write_json(ARCHIVE / "final-strict-gap-probes.json", output)
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
