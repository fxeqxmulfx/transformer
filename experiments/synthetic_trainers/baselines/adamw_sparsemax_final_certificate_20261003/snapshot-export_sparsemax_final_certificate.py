"""Export actual final-checkpoint CPU predictions for a finite Lean certificate."""

import csv
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch

from experiments.synthetic_trainers.attention_layout import current_sources, digest
from experiments.synthetic_trainers.attention_training import AttentionRunConfig, make_model, training_sources
from experiments.synthetic_trainers.paper_reproduction.grokking import logits
from experiments.synthetic_trainers.paper_reproduction.modular_data import answer, make_corpus
from experiments.synthetic_trainers.stability_report import write_json


PROTOCOL = Path("experiments/synthetic_trainers/protocols/adamw_stability_20261002")


def export(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Final prediction certificates require a fresh destination")
    if torch.cuda.is_initialized():
        raise RuntimeError("The final certificate uses CPU inspection only")
    plan = json.loads((PROTOCOL / "attention-pair-plan.json").read_text())
    recipe = next(row for row in plan["recipes"] if row["name"] == "adamw-sparsemax")
    case = Path(plan["output_directory"]) / "adamw-sparsemax"
    archive = Path("experiments/synthetic_trainers/baselines") / (
        "adamw_attention_adamw-sparsemax_mod193_fraction25_lr0003_budget300k_seed0_data0_20261002")
    report = json.loads((case / "measurements.json").read_text())
    manifest = json.loads((archive / "artifact-hashes.json").read_text())
    checkpoint_path = case / "checkpoint.pt"
    checkpoint_hash = digest(checkpoint_path)
    assert checkpoint_hash == manifest["raw_checkpoint_sha256"]
    assert current_sources() == plan["source_hashes"] and training_sources() == plan["training_source_hashes"]
    for field in ("Lean_specification_hashes", "papers"):
        assert all(digest(path) == sha for path, sha in plan[field].items())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["step"] == report["completed_steps"] == recipe["config"]["steps"] == 300000
    assert report["plan"]["config"] == recipe["config"]
    config = AttentionRunConfig(**recipe["config"])
    corpus = make_corpus(config.prime, config.train_fraction, config.data_seed)
    assert corpus.summary() == plan["corpus"]
    torch.set_num_threads(1)
    model = make_model(replace(config, device="cpu"), len(corpus.tokens))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    assert sum(p.numel() for p in model.parameters()) == report["plan"]["parameters"] == 436104
    assert all(torch.isfinite(p).all() for p in model.parameters())
    output.mkdir(parents=True)
    summaries = {}
    numbers = {str(number) for number in range(config.prime)}
    with torch.no_grad():
        for partition, rows in (("train", corpus.train), ("heldout", corpus.heldout)):
            predictions, correct, EOS_correct = [], 0, 0
            for start in range(0, len(rows), config.batch_size):
                batch = torch.tensor(rows[start:start + config.batch_size], dtype=torch.long)
                prediction = logits(model, batch[:, :-1])[:, 4:, :].argmax(dim=-1)
                flags = (prediction == batch[:, 5:]).all(dim=1)
                correct += int(flags.sum())
                EOS_correct += int((prediction[:, 1] == 0).sum())
                for row, pred, flag in zip(batch.tolist(), prediction.tolist(), flags.tolist()):
                    numerator, denominator, truth = [int(corpus.tokens[row[index]]) for index in (1, 3, 5)]
                    assert corpus.tokens[row[2]] == "/" and answer(numerator, denominator, config.prime) == truth
                    token = corpus.tokens[pred[0]]
                    value = int(token) if token in numbers else config.prime
                    stop = int(pred[1] == 0)
                    oracle_flag = value < config.prime and stop == 1 and denominator * value % config.prime == numerator
                    assert oracle_flag == flag
                    predictions.append((numerator, denominator, value, stop))
            with (output / f"{partition}-predictions.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("numerator", "denominator", "predicted_answer", "EOS_correct"))
                writer.writerows(predictions)
            measured = report["final"][partition]
            summaries[partition] = {"examples": len(rows), "correct_complete_RHS": correct,
                "EOS_correct": EOS_correct, "CPU_accuracy": correct / len(rows),
                "archived_GPU_accuracy": measured["accuracy"],
                "CPU_correct_count_matches_archived_GPU_accuracy": correct / len(rows) == measured["accuracy"],
                "CSV_sha256": digest(output / f"{partition}-predictions.csv")}
            print(json.dumps({"partition": partition, **summaries[partition]}), flush=True)
    assert not torch.cuda.is_initialized()
    receipt = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "actual_final_checkpoint_CPU_prediction_table; no_full_PyTorch_trajectory_proof",
        "checkpoint_sha256": checkpoint_hash, "checkpoint_step": 300000,
        "corpus": corpus.summary(), "parameters": 436104, "predictions": summaries,
        "all_frozen_attention_training_Lean_and_paper_sources_unchanged": True,
        "optimizer_updates_performed": 0, "GPU_context_initialized": False,
        "Torch_version": str(torch.__version__), "export_program_sha256": digest(__file__),
        "CPU_GPU_prediction_table_identity_claimed": False}
    write_json(output / "validation.json", receipt)
    return receipt


if __name__ == "__main__":
    print(json.dumps(export(sys.argv[1])))
