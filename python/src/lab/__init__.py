"""Transformer experiments composed from swappable blocks.

An experiment file under `experiments/` composes a model and the conditions
of its benchmark run in the language of `lab.dsl`; `lab run <file>` trains
every experiment it defines. The package is layered by dependency:

- `lab.domain`: the language itself and pure rules (schedules, cadences,
  continuation, analysis); no PyTorch, NumPy or filesystem;
- `lab.application`: use cases over ports; still no PyTorch;
- `lab.infrastructure`: PyTorch builders and engines, data, storage;
- `lab.interfaces`: the command line.

Inner layers never import outer ones; `tests/test_architecture.py` checks it.
"""
