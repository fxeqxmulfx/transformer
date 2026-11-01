"""Check the actual division splits against the ideal recovery theorem.

Source: review/flat_spectrum.md, Rigidity is connectivity; matching search
adapted from rotate90_power_learning/theory.py at33e412cb. Integer witnesses
certify a connected observation graph, a perfect matching and a faithful
character on nonzero residues. These are premises of ideal nuclear-norm
recovery, not premises sufficient for Schedule-Free learning.
"""

import json
from pathlib import Path

import torch

from lab.infrastructure.benchmarks import build_task
from lab.infrastructure.loader import load

HERE = Path(__file__).parent


def check(experiment):
    prime = experiment.benchmark.prime
    task = build_task(experiment.benchmark, experiment.seeds.data, torch.device("cpu"))
    n = prime - 1
    neighbors = [set() for _ in range(n)]
    adjacency = [set() for _ in range(2 * n)]
    for a, b, answer in task.splits["train"].tolist():
        assert (answer * b - a) % prime == 0
        if a:
            neighbors[a - 1].add(b - 1)
            adjacency[a - 1].add(n + b - 1)
            adjacency[n + b - 1].add(a - 1)
    parents, pending = {0: None}, [0]
    while pending:
        vertex = pending.pop()
        for other in sorted(adjacency[vertex]):
            if other not in parents:
                parents[other] = vertex
                pending.append(other)
    matching = [-1] * n

    def augment(a, visited):
        for b in sorted(neighbors[a]):
            if b not in visited:
                visited.add(b)
                if matching[b] == -1 or augment(matching[b], visited):
                    matching[b] = a
                    return True
        return False

    perfect = all(augment(a, set()) for a in range(n))
    connected = len(parents) == 2 * n
    # Independently verify the recorded combinatorial certificates.
    if perfect:
        assert sorted(matching) == list(range(n))
        assert all(b in neighbors[a] for b, a in enumerate(matching))
    for vertex in parents:
        path, node = set(), vertex
        while node:
            assert node not in path and parents[node] in adjacency[node]
            path.add(node)
            node = parents[node]
    generator = next(g for g in range(1, prime)
                     if len({pow(g, i, prime) for i in range(n)}) == n)
    logs = {pow(generator, i, prime): i for i in range(n)}
    assert sorted(logs.values()) == list(range(n))
    # Integer phases certify chi(a/b) = chi(a) conjugate(chi(b)) exactly.
    for a in range(1, prime):
        for b in range(1, prime):
            assert logs[a * pow(b, -1, prime) % prime] == (logs[a] - logs[b]) % n
    return {"prime": prime, "data_seed": experiment.seeds.data,
            "training_fingerprint": task.summary()["train"]["fingerprint"],
            "nonzero_training_cells": sum(map(len, neighbors)), "generator": generator,
            "faithful_character_integer_phase_check": True,
            "connected": connected, "perfect_matching": perfect,
            "column_to_row_zero_indexed": matching if perfect else None,
            "spanning_tree_parent": [parents.get(i) for i in range(2 * n)],
            "ideal_nonzero_character_kernel_uniquely_recovered": connected and perfect}


def main():
    study = load(HERE)
    cases = [check(study.experiments[f"pairs-division-p{p}-w256d1"]) for p in (17, 97, 193)]
    result = {"scope": "Ideal faithful character recovery on nonzero operands; no optimizer conclusion.",
              "zero_numerator": "chi(0)=0 distinguishes zero from every nonzero output; the zero row is excluded from the unit-modulus graph.",
              "cases": cases}
    (HERE / "recovery_patterns.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in row.items()
                       if k not in ("column_to_row_zero_indexed", "spanning_tree_parent")} for row in cases]))


if __name__ == "__main__":
    main()
