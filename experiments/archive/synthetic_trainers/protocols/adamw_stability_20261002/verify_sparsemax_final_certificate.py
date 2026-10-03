"""Independently verify actual CSV coverage, corpus fingerprints and Lean packing."""

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import sys


ROOT = Path("experiments/synthetic_trainers/baselines/adamw_sparsemax_final_certificate_20261003")
LEAN = Path("src/Transformer/GPTMini/Sparsemax/Certificate")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lean_batches(filename):
    source = (LEAN / filename).read_text()
    payload = "[" + source.rsplit(" := [", 1)[1].split("\n\nend ", 1)[0]
    batches = ast.literal_eval(payload)
    assert all(0 < len(batch) <= 128 for batch in batches)
    return [code for batch in batches for code in batch]


def verify():
    receipt = json.loads((ROOT / "CPU-export-validation.json").read_text())
    parent = Path("experiments/synthetic_trainers/baselines/adamw_attention_adamw-sparsemax_mod193_fraction25_lr0003_budget300k_seed0_data0_20261002/artifact-hashes.json")
    assert json.loads(parent.read_text())["raw_checkpoint_sha256"] == receipt["checkpoint_sha256"]
    plan = json.loads(Path("experiments/synthetic_trainers/protocols/adamw_stability_20261002/attention-pair-plan.json").read_text())
    assert receipt["corpus"] == plan["corpus"]
    corpus_source = Path("experiments/synthetic_trainers/paper_reproduction/modular_data.py")
    assert digest(corpus_source) == plan["training_source_hashes"][str(corpus_source)]
    tree = ast.parse(corpus_source.read_text())
    operators = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "OPERATORS" for target in node.targets))
    number_offset, division_token = 2 + len(operators), 2 + sorted(operators).index("/")
    summaries, keys = {}, {}
    for partition, filenames in (("train", ["Train.lean"]),
                                 ("heldout", ["HeldoutFirst.lean", "HeldoutLast.lean"])):
        path = ROOT / f"{partition}-predictions.csv"
        measured = receipt["predictions"][partition]
        assert digest(path) == measured["CSV_sha256"]
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            rows = [tuple(int(row[key]) for key in ("numerator", "denominator", "predicted_answer", "EOS_correct"))
                    for row in reader]
        codes = [code for filename in filenames for code in lean_batches(filename)]
        decoded = []
        for code in codes:
            eos = code % 2
            intermediate = code // 2
            prediction = intermediate % 194
            intermediate //= 194
            denominator = intermediate % 193
            numerator = intermediate // 193
            decoded.append((numerator, denominator, prediction, eos))
        assert decoded == rows
        keys[partition] = {(numerator, denominator) for numerator, denominator, _, _ in rows}
        assert len(keys[partition]) == len(rows)
        oracle_correct, fingerprint = 0, hashlib.sha256()
        for numerator, denominator, prediction, eos in rows:
            assert 0 <= numerator < 193 and 0 < denominator < 193
            assert 0 <= prediction < 194 and eos == 1
            truth = numerator * pow(denominator, -1, 193) % 193
            assert (prediction == truth) == (prediction < 193 and denominator * prediction % 193 == numerator)
            oracle_correct += prediction == truth
            fingerprint.update(struct.pack("<7q", 0, number_offset + numerator, division_token,
                                           number_offset + denominator, 1, number_offset + truth, 0))
        assert fingerprint.hexdigest() == receipt["corpus"][f"{partition}_fingerprint"]
        assert len(rows) == measured["examples"] and oracle_correct == measured["correct_complete_RHS"]
        summaries[partition] = {"rows": len(rows), "correct_complete_RHS": oracle_correct,
            "CSV_sha256": digest(path), "corpus_fingerprint": fingerprint.hexdigest(),
            "Lean_packing_matches_every_CSV_record": True, "distinct_inputs": len(keys[partition])}
    assert keys["train"].isdisjoint(keys["heldout"])
    assert keys["train"] | keys["heldout"] == {(x, y) for x in range(193) for y in range(1, 193)}
    return {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "independent_supplied_table_coverage_corpus_and_serialization_check",
        "checkpoint_sha256": receipt["checkpoint_sha256"], "partitions": summaries,
        "exhaustive_disjoint_prime_field_domain_checked": True,
        "inverse_oracle_agrees_with_multiplication_predicate_on_every_record": True,
        "Torch_imported": "torch" in sys.modules, "optimizer_updates_performed": 0,
        "program_sha256": digest(__file__)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional fresh receipt; default inspection writes no files")
    args = parser.parse_args()
    result = verify()
    assert not result["Torch_imported"]
    if args.output is not None:
        with args.output.open("x") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
