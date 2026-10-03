"""Transformer experiments composed from swappable blocks.

An experiment is a folder under `experiments/` whose `experiment.py` composes
a model and the conditions of its benchmark runs in the language of
`lab.dsl`; `lab run <experiment>` trains every run it defines. The package is
layered by dependency:

- `lab.domain`: the language itself and pure rules (schedules, cadences,
  continuation, analysis); no PyTorch, NumPy or filesystem;
- `lab.application`: use cases over ports; still no PyTorch;
- `lab.infrastructure`: PyTorch builders and engines, data, storage;
- `lab.interfaces`: the command line.

Inner layers never import outer ones; `tests/test_architecture.py` checks it.
"""
