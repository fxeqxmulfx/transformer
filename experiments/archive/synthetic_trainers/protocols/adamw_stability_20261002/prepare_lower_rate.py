"""Check the lower-rate AdamW control on CPU before scientific freezing.

Convexifying Transformers, Section 4: this is a follow-up rate intervention.
Run with CUDA_VISIBLE_DEVICES='' to avoid the optimizer's accelerator query.
"""

from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from experiments.synthetic_trainers.paper_reproduction.diagnostics import DiagnosticsConfig
from experiments.synthetic_trainers.paper_reproduction.grokking import RunConfig, make_model
from experiments.synthetic_trainers.paper_reproduction.modular_data import make_corpus
from experiments.synthetic_trainers.persistence import PersistenceConfig
from experiments.synthetic_trainers.stability import freeze, run_stage
from experiments.synthetic_trainers.stability_integrity import expected_batches
from experiments.synthetic_trainers.stability_report import save_run
from .prepare_fraction25 import state_digest


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        raise ValueError("Run this CPU preparation with CUDA_VISIBLE_DEVICES=''")
    repository = Path.cwd()
    protocol = Path(__file__).resolve().parent
    parent = json.loads((protocol / "fraction25-plan.json").read_text())
    destination = protocol / "lower-rate-preparation"
    output = repository / "experiments/runs/adamw_stability_20261002/bootstrap/lower_rate_cpu_preparation"
    if destination.exists() or output.exists():
        raise FileExistsError("Lower-rate preparation requires fresh destinations")
    frozen = {name: value for key in ("source_hashes", "analysis_and_driver_source_hashes", "papers")
              for name, value in parent[key].items()}
    check_frozen = lambda: all(digest(repository / name) == value for name, value in frozen.items())
    assert check_frozen(), "Frozen scientific sources changed"
    primary = RunConfig(**parent["recipes"][0]["config"])
    candidate = replace(primary, learning_rate=.0003)
    assert [key for key in asdict(primary) if asdict(primary)[key] != asdict(candidate)[key]] == ["learning_rate"]
    original, corpus = [make_corpus(c.prime, c.train_fraction, c.data_seed) for c in (primary, candidate)]
    assert original.summary() == corpus.summary() == parent["corpus"]
    assert original.tokens == corpus.tokens
    assert np.array_equal(original.train, corpus.train) and np.array_equal(original.heldout, corpus.heldout)
    models = [make_model(replace(c, device="cpu"), len(corpus.tokens)) for c in (primary, candidate)]
    assert all(torch.equal(models[0].state_dict()[name], tensor) for name, tensor in models[1].state_dict().items())
    initial_hash = state_digest(models[0])
    assert initial_hash == state_digest(models[1])
    prepared_parent = json.loads((protocol / "fraction25-preparation/preparation.json").read_text())
    assert initial_hash == prepared_parent["common_initial_state_cpu_sha256"]
    parameters = sum(t.numel() for t in models[0].parameters())
    del models
    assert expected_batches(asdict(primary), primary.steps) == expected_batches(asdict(candidate), candidate.steps)
    prepared = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_manifest_sha256": digest(protocol / "fraction25-plan.json"),
        "candidate_scientific_configuration": asdict(candidate), "changed_fields": ["learning_rate"],
        "corpus": corpus.summary(), "identical_complete_corpus_verified": True,
        "common_initial_state_verified": True, "common_initial_state_cpu_sha256": initial_hash,
        "parameters_both_models": parameters, "budget_exposure": expected_batches(asdict(candidate), candidate.steps),
        "frozen_fingerprints": {"checked_files": len(frozen), "all_unchanged": True},
        "scope": "CPU_preparation_only; scientific_plan_not_frozen_or_launched"}
    smoke = replace(candidate, device="cpu", steps=20, eval_every=19)
    name = "lower-rate-smoke"
    recipe = {"name": name, "config": asdict(smoke), "changed_mechanism": "CPU_pipeline_smoke_only"}
    manifest = freeze(output / "source", [recipe], DiagnosticsConfig(19, True, True), PersistenceConfig())
    run_stage(output / "source", manifest)
    summary = save_run(output / "source", name, output / "portable", render=False)
    code = ("from experiments.synthetic_trainers.stability_report import verify_archive; import json,sys; "
            "verify_archive(sys.argv[1]); print(json.dumps({'verified':True,'torch_imported':'torch' in sys.modules}))")
    result = subprocess.run(["/usr/bin/python3", "-c", code, str(output / "portable")],
                            check=True, capture_output=True, text=True)
    verified = json.loads(result.stdout)
    assert verified["verified"] and not verified["torch_imported"]
    run = output / "source" / name
    gradients = [json.loads(line) for line in (run / "gradients.jsonl").read_text().splitlines()]
    for point in gradients:
        assert point["learning_rate"] == .0003 * min(1, (point["step"] - 1) / 10)
    assert gradients[18]["batch_size"] == 48 and gradients[18]["epoch_tail"]
    assert gradients[19]["batch_size"] == 512 and not gradients[19]["epoch_tail"]
    assert summary["gradient_trace"]["observations"] == 20 and summary["canonical_observations"] == 3
    assert not torch.cuda.is_initialized() and check_frozen()
    destination.mkdir()
    smoke_record = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "updates": 20,
        "canonical_steps": [0, 19, 20], "dense_gradients": 20, "device": "cpu",
        "warmup_rates": {str(p["step"]): p["learning_rate"] for p in gradients[:11]},
        "short_tail": gradients[18], "next_full_batch": gradients[19], "parameters": parameters,
        "portable_archive_verified_without_torch": True, "cuda_context_created": False,
        "archive": str((output / "portable").relative_to(repository)),
        "scope": "implementation_smoke_only; no_learning_or_stability_claim"}
    for filename, value in (("preparation.json", prepared), ("cpu-smoke.json", smoke_record)):
        (destination / filename).write_text(json.dumps(value, indent=2) + "\n")
    (destination / "history.jsonl").write_bytes((run / "history.jsonl").read_bytes())
    (destination / "source-snapshot.py").write_bytes(Path(__file__).read_bytes())
    hashes = {p.name: digest(p) for p in destination.iterdir()}
    (destination / "artifact-hashes.json").write_text(json.dumps({"files": hashes}, indent=2) + "\n")
    print(json.dumps({"prepared": True, "scientific_training_started": False,
                      "cpu_smoke_verified": True, "frozen_files_unchanged": len(frozen)}), flush=True)


if __name__ == "__main__":
    main()
