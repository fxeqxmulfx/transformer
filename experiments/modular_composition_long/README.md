# Long continuation of the normalized RoPE arithmetic screen

Does an early memorization plateau conceal delayed grokking? This is an
adaptive follow-up to the frozen2000-update screen, declared before any
continuation. Only Schedule-Free AdamW is used. The fixed source model,
labels, splits, seeds, batch size, rate, decay and moment parameters stay
the same. Query, base, second moments, averaging clock and sampler state
are loaded from SHA256-pinned checkpoints. This is not a fresh optimizer
restart. Python3.15/Torch2.15 and CUDA max-autotune remain required.

| Cases | Count | Purpose |
| --- | ---: | --- |
| Division and mixed pairs, P97/P193, W256D1/W128D4 | 8 | Rank-one versus flat character routes; early width/depth differences |
| Full-table mixed P193, W256D1 | 1 | Whether incomplete fitting at2000 updates persists |
| One-step chain P97, W128D4 | 1 | Whether a longer budget learns the local mixed update |

Each target is100000 **absolute** updates, continuing from2000. Evaluation
every250 updates is retained, including its normal Schedule-Free mode
roundtrips. Checkpoints every10000 updates are retained, together with the
initial2000 state. There is no early stopping or held-out selection; report
the final model, every measured transition above99%, and subsequent relapse.
For the chain, doubled length is evaluated only at its final endpoint.

The cohort shares one **two-hour wall-time cap**, including compilation and
evaluation. The fixed order is recorded in `experiment.py`. If it runs out,
stop the active process and preserve its last complete atomic checkpoint;
report it as unfinished, alongside not-started cases. A timeout is not
evidence that a task cannot be learned. `run_bounded.py` refuses to allocate
a second budget automatically and records start/end times and case status
in root-linked ignored `runs/phase.json`. The original screen remains intact.

No long outcomes have been observed yet. Profile the warmed source model before
starting this phase. The counterexample criteria and limits are inherited
from `../modular_composition/falsification.json` and repeated in the frozen
`hypotheses.json`. The target is to test missing learning implications, not
to weaken valid exact kernel-completion theorems.

Validation: the complete CPU-visible `./run.py test` passed446 lab tests
(13 CUDA-dependent skips) and104 optimizer tests (14 skips). The new
continuation test compares query, complete optimizer state and exact next
sampler state with uninterrupted training on both pairs and chains. The
budget-supervisor test checks successful exit and termination while keeping
the child's log. Earlier CUDA validation retains the two documented Muon
allocation failures; no new failure is hidden by this CPU validation.

```sh
./run.py check experiments/modular_composition_long
# From python/, after profiling and protocol commit:
uv run --locked python ../experiments/modular_composition_long/run_bounded.py
```
