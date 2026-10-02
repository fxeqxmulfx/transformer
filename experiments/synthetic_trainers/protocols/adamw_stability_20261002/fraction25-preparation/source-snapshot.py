"""Check a mod-193 fraction intervention on CPU before a future scientific freeze.

Setting: Convexifying Transformers, Section 4. A 25% exhaustive training split
is a follow-up adaptation; this preparation cannot establish learning or stability.
Launch with CUDA_VISIBLE_DEVICES='' so CPU checkpoint RNG capture cannot open CUDA.
"""

from dataclasses import replace, asdict
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
from .csv_verification import linear_csv_verification


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_digest(model):
    value = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value.update(name.encode())
        value.update(str((str(tensor.dtype), tuple(tensor.shape))).encode())
        value.update(tensor.detach().cpu().numpy().tobytes())
    return value.hexdigest()


def main():
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        raise ValueError("Run this CPU-only preparation with CUDA_VISIBLE_DEVICES=''")
    repository = Path.cwd()
    protocol = Path(__file__).resolve().parent
    parent = json.loads((protocol / "larger-modulus-plan.json").read_text())
    destination = protocol / "fraction25-preparation"
    output = repository / "experiments/runs/adamw_stability_20261002/bootstrap/fraction25_cpu_preparation"
    if destination.exists() or output.exists():
        raise FileExistsError("Fraction preparation requires fresh destinations")
    frozen = {name: value for key in ("source_hashes", "analysis_and_driver_source_hashes", "papers")
              for name, value in parent[key].items()}
    check_frozen = lambda: all(digest(repository / name) == value for name, value in frozen.items())
    assert check_frozen(), "The current campaign's frozen files changed"
    primary = RunConfig(**parent["recipes"][0]["config"])
    candidate = replace(primary, train_fraction=.25)
    assert [key for key in asdict(primary) if asdict(primary)[key] != asdict(candidate)[key]] == ["train_fraction"]
    corpora = [make_corpus(c.prime, c.train_fraction, c.data_seed) for c in (primary, candidate)]
    original, reduced = corpora
    assert original.tokens == reduced.tokens
    assert np.array_equal(reduced.train, original.train[:len(reduced.train)])
    assert np.array_equal(reduced.heldout[-len(original.heldout):], original.heldout)
    coverage = {}
    for config, corpus in zip((primary, candidate), corpora):
        splits = {}
        all_pairs = set()
        for name, rows in (("train", corpus.train), ("heldout", corpus.heldout)):
            values = np.array([[int(corpus.tokens[int(row[i])]) for i in (1, 3, 5)] for row in rows])
            for row, (a, b, answer) in zip(rows, values):
                assert (int(a), int(b)) not in all_pairs
                all_pairs.add((int(a), int(b)))
                assert 0 <= a < config.prime and 0 < b < config.prime
                assert a * pow(int(b), -1, config.prime) % config.prime == answer
                assert b * answer % config.prime == a
                assert [corpus.tokens[int(row[i])] for i in (0, 2, 4, 6)] == ["<|eos|>", "/", "=", "<|eos|>"]
            classes = {}
            for index, label in enumerate(("numerator", "denominator", "answer")):
                unique, counts = np.unique(values[:, index], return_counts=True)
                expected = np.arange(1, config.prime) if label == "denominator" else np.arange(config.prime)
                assert np.array_equal(unique, expected)
                classes[label] = {"classes": len(unique), "minimum": int(counts.min()), "maximum": int(counts.max())}
            splits[name] = classes
        assert len(all_pairs) == config.prime * (config.prime - 1)
        coverage[str(config.train_fraction)] = splits
    models = [make_model(replace(c, device="cpu"), len(reduced.tokens)) for c in (primary, candidate)]
    assert all(torch.equal(models[0].state_dict()[name], tensor) for name, tensor in models[1].state_dict().items())
    initial_hash = state_digest(models[0])
    assert initial_hash == state_digest(models[1])
    parameters = sum(t.numel() for t in models[0].parameters())
    del models
    prepared = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "parent_manifest_sha256": digest(protocol / "larger-modulus-plan.json"),
                "candidate_scientific_configuration": asdict(candidate), "changed_fields": ["train_fraction"],
                "corpora": [c.summary() for c in corpora], "split_class_coverage": coverage,
                "complete_domain_oracles_verified": True, "nested_train_and_common_heldout_verified": True,
                "common_initial_state_verified": True, "common_initial_state_cpu_sha256": initial_hash,
                "parameters_both_models": parameters,
                "budget_exposure": {str(c.train_fraction): expected_batches(asdict(c), c.steps)
                                    for c in (primary, candidate)},
                "frozen_fingerprints": {"checked_files": len(frozen), "all_unchanged": True},
                "scope": "CPU_preparation_only; scientific_plan_not_frozen_or_launched"}
    smoke = replace(candidate, device="cpu", steps=20, eval_every=19)
    recipes = [{"name": "fraction25-smoke", "config": asdict(smoke), "changed_mechanism": "CPU_pipeline_smoke_only"}]
    manifest = freeze(output / "source", recipes, DiagnosticsConfig(19, True, True), PersistenceConfig())
    run_stage(output / "source", manifest)
    with linear_csv_verification():
        summary = save_run(output / "source", "fraction25-smoke", output / "portable", render=False)
    result = subprocess.run(["/usr/bin/python3", "-m",
                            "experiments.synthetic_trainers.protocols.adamw_stability_20261002.csv_verification",
                            "archive", str(output / "portable")], check=True, capture_output=True, text=True)
    verified = json.loads(result.stdout)
    assert verified["verified"] and not verified["torch_imported"]
    run = output / "source/fraction25-smoke"
    gradients = [json.loads(line) for line in (run / "gradients.jsonl").read_text().splitlines()]
    assert gradients[18]["batch_size"] == 48 and gradients[18]["epoch_tail"]
    assert gradients[19]["batch_size"] == 512 and not gradients[19]["epoch_tail"]
    assert summary["gradient_trace"]["observations"] == 20 and summary["canonical_observations"] == 3
    assert not torch.cuda.is_initialized() and check_frozen()
    destination.mkdir()
    (destination / "preparation.json").write_text(json.dumps(prepared, indent=2) + "\n")
    smoke_record = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "updates": 20,
                    "canonical_steps": [0, 19, 20], "dense_gradients": 20, "device": "cpu",
                    "short_tail": gradients[18], "next_full_batch": gradients[19],
                    "parameters": parameters, "portable_archive_verified_without_torch": True,
                    "cuda_context_created": False, "archive": str((output / "portable").relative_to(repository)),
                    "scope": "implementation_smoke_only; no_learning_or_stability_claim"}
    (destination / "cpu-smoke.json").write_text(json.dumps(smoke_record, indent=2) + "\n")
    (destination / "history.jsonl").write_bytes((run / "history.jsonl").read_bytes())
    (destination / "source-snapshot.py").write_bytes(Path(__file__).read_bytes())
    hashes = {p.name: digest(p) for p in destination.iterdir()}
    (destination / "artifact-hashes.json").write_text(json.dumps({"files": hashes}, indent=2) + "\n")
    print(json.dumps({"prepared": True, "scientific_training_started": False,
                      "cpu_smoke_verified": True, "frozen_files_unchanged": len(frozen)}), flush=True)


if __name__ == "__main__":
    main()
