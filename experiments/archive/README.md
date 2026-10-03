# Archive

The records of the three codebases the [lab](../../python/README.md)
replaced: what they planned, measured and concluded. Their code is no longer
in the tree; the lab ports what the experiments need, and
[`python/README.md`](../../python/README.md#ports) says at which commit each
cited source can be read. Nothing here runs. The one script left is the
reference [`gpt_mini/gpt_mini.py`](gpt_mini/gpt_mini.py), which the Lean
formalization and the lab cite as GPTMini.

| Folder | Codebase | Code last in |
| --- | --- | --- |
| [`gpt_mini/`](gpt_mini) | the reference GPTMini and its Tiny Shakespeare optimizer benchmarks | `2aec5b9` (`legacy/gpt_mini/src/`) |
| [`synthetic_trainers/`](synthetic_trainers) | the synthetic task suite, its studies and the stability protocols | `5d64147` |
| [`convex_mqar/`](convex_mqar) | convex content routing against a RoPE transformer on MQAR | `9416d03` |

## Removed scripts

The records also kept the scripts that prepared, ran, checked and plotted
their experiments, and the source snapshots those froze: 270 `.py` files,
removed on 2026-10-03. Read one with `git show a64c896:<path>`, at its path
here. Links to them in the records were made plain names. The sparsemax
generalization inspector, which no commit had held, was recorded in
`441f47b` before it was removed; read it there.

## Old paths

The records are kept as they were written, and name the paths they had
then. They moved here on 2026-10-03:

| Written as | Now |
| --- | --- |
| `experiments/gpt_mini.py` | `experiments/archive/gpt_mini/gpt_mini.py` |
| `experiments/gpt_mini.md` | `experiments/archive/gpt_mini/README.md` |
| `experiments/<name>_benchmark/` | `experiments/archive/gpt_mini/<name>_benchmark/` |
| `experiments/synthetic_trainers/` | `experiments/archive/synthetic_trainers/` |
| `experiments/convex_mqar/` | `experiments/archive/convex_mqar/` |
| `experiments/runs/` (not tracked) | `experiments/archive/synthetic_trainers/runs/` (not tracked) |

On the move, Markdown links were pointed at the new places, a note atop the
READMEs of the synthetic trainers and of the convex MQAR comparison says
so, and each benchmark got a README; paths in prose, JSON and logs were
left as written. A path under one of these folders that is not in the tree
is either a removed script (above, under its new path) or the codebase's
code: read that with `git show <commit>:<path>` at the commit of the first
table. `synthetic_trainers/runs/` holds the raw runs of the synthetic
trainers' stability protocols (1.1 GB, never committed), and exists only on
the machine that trained them.
