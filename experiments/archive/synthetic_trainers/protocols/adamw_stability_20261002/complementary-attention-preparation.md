# Native paired normalizers for complementary tasks

The opt-in [bridge](../../complementary_attention.py) applies softmax/sparsemax
to the existing native complementary AdamW adapter. Only attention normalization
changes within a pair. Serialization, clean disjoint pools, short final batches,
all-parameter decay, betas 0.9/0.98, epsilon 1e-8, initialization, warmup and the
selected rate schedule stay unchanged. Scientific selection remains gated.

Four targeted checks pass in 3.669 seconds. Both factories preserve original
parameter tensors, names, ties and RNG at width 128/two layers/four heads.
Real softmax final/best tensors, every native moment, sampling/rate history and
non-time scores exactly match the previous adapter at constant and cosine-tail
rates. Unknown normalizers, rehashed wrong tags and changed-seed pairs are
rejected. Interrupted files remain intact and generic factories restore.
The complementary trainer remains fresh-run-only: this bridge adds no resume,
and retry cannot overwrite or repeat an interrupted budget. A future driver
must keep incomplete cases rather than silently restart the same run identity.

Eight prospectively planned CPU pairs execute both normalizers in alternating
order. Model/data seeds are 0/0; width 8, one layer/head; four examples per
train/validation/test pool; batch two; sixteen updates; cadence four; warmup ten.
Each case has five canonical observations and sixteen actual rate/decay records.
Both final and validation-selected checkpoints are independently reloaded with
the actual normalizer; all ID/length predictions agree, excluding generation
time. Native final optimizer reloads confirm groups, steps, finite moments and
state shapes. No optimizer update is used for these reloads.

| CPU pair | Training / longer input | Mechanism |
| --- | --- | --- |
| MQAR | 24 / 48 | Four associations, two queries; greater spacing |
| Lookup, one hop | 24 / 48 | Greater spacing at fixed hop depth |
| Lookup, two hops | 24 / 48 | Greater spacing at fixed composition depth |
| Copy | 4 / 8 | Longer input and complete generated answer |
| Direct parity | 4 / 8 | More input bits |
| Running parity | 4 / 8 | More input bits and scratchpad outputs |
| C-RASP, depth two | 8 / 16 | Longer prefix horizon at fixed program |
| Copy, cosine tail | 4 / 8 | CPU annealing bounds 10–14 |

The complete [archive](complementary-attention-preparation/summary.json) retains
sixteen cases, 256 updates and eighty canonical observations. Parameter counts
are 1,057–1,313, equal within each pair. Each case has four novel ID validation
inputs and four novel longer-input probes. Novel ID means complete inputs absent
from training, not unseen token IDs. Every final and selected ID/length sequence
accuracy is zero: these fixtures establish execution, not learning, stable
grokking, architecture improvement or algorithmic transfer.

The portable verifier requires identical task/config/model/source settings,
data, parameter count and exposure within a pair. Final/selected checkpoints
and ID/length outcomes remain separate. Costs are descriptive; no scientific
effect or speedup is certified. NoTorch verification checks saved replay receipts
rather than independently recomputing neural predictions.

The archive includes raw pools, both model checkpoints, native optimizer states,
full histories/rates, ninety-four repository source snapshots and the executed
preparation program. The native PyTorch implementation is fingerprinted without
copying dependency source. Both PNGs and actual PDF renders are reviewed and
labelled CPU fixtures. The final full CPU suite passes 288 tests in 90.250 seconds,
with no failures/errors/skips. See the
[review receipt](complementary-attention-preparation-validation.json).

The first preparation stopped after one sixteen-update CPU case because its
replay helper compared native tuple fields against JSON list fields. Its raw
case, checkpoint, plan/program and failure record remain under
`experiments/runs/adamw_stability_20261002/bootstrap/complementary-attention-preparation-initial-serialization/`.
The corrected helper compares the actual saved audit representation. The complete
preparation uses a fresh directory; no scientific run was retrained.

```bash
env CUDA_VISIBLE_DEVICES='' .venv/bin/python -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.prepare_complementary_attention --output /tmp/complementary-normalizer-pairs
python3 -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.verify_complementary_attention /tmp/complementary-normalizer-pairs
```

All active 43 Python/nine Lean and prepared 55/both 62/66/70-source sets remain
unchanged. No scientific complementary task, architecture or campaign is selected,
frozen or started. Complete the full softmax control/pair and unchanged all-six
benchmark/architecture requirements before selecting scientific task difficulties,
seeds, targets and budgets.
