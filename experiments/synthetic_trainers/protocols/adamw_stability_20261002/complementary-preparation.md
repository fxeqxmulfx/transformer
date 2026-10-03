# Existing complementary trainers: AdamW and transfer preparation

The continuing objective includes MQAR, copy/parity, lookup/composition and
prefix counting after an independently repeatable modular benchmark and a
targeted architecture comparison. No scientific architecture has been selected
or tested. The current GPU optimizer control and frozen mod-193 queue continue.

The existing common trainer already supports native AdamW. Its defaults differ
from the modular manuscript adaptation and must not be silently reused as the
same optimizer protocol:

| Control | General trainer default | Current modular primary |
| --- | --- | --- |
| Betas | (0.9, 0.999), configurable | (0.9, 0.98) |
| Epsilon | 1e-8, configurable | 1e-8 |
| Weight decay | 0.01, matrices only | 0.1, all trainable parameters |
| Gradient clipping | Norm 1.0, can disable | Disabled |
| Warmup | Constant learning rate | Ten updates |
| Matrix initialization | Constructor default, std configurable | Normal std 0.02 |

The CLI can set betas, epsilon, decay amount, clipping and matrix initialization.
The existing general AdamW factory still decays only matrices, and its loop
has no warmup option. The separate opt-in native adapter below now implements
matching controls without changing those general defaults or any frozen modular
source. A later scientific complementary protocol must explicitly select and
freeze the adapter and preserve identical optimizer settings within each
architecture pair. The historical probes below used the general factory.

The study profile already retains both final and validation-selected checkpoint
test scores. Novel ID means complete inputs absent from training, evaluated
at training lengths; it does not mean unseen token identifiers. Longer inputs
are separate probes, each with its own novelty support. Zero novel support
cannot certify transfer. Generative tasks use free rollout metrics; teacher
forcing is a separate train/diagnostic measure. These are the existing rules
in [STUDIES.md](../../STUDIES.md).

Seven real [CPU implementation probes](complementary-preparation/validation.json)
completed two updates each at width 8, one layer, one head, betas (0.9, 0.98),
matrix decay 0.1, no clipping and no warmup. Their disjoint train/validation/test
pools each have four examples. Every ID and longer-input metric exactly matched
independent reloads of both final and selected checkpoints, excluding measured
generation time. All have four novel ID validation examples and four novel
longer-input probe examples. Complete [results](complementary-preparation/results.json),
[three-point histories](complementary-preparation/histories.json), source hashes
and the exact executed [probe snapshot](complementary-preparation/probe-snapshot.py)
are retained with checksums. They are implementation checks, not learning,
grokking, stability, architecture-improvement or algorithmic-transfer evidence.

| Probe | Train / longer length | What grows |
| --- | --- | --- |
| MQAR | 24 / 48 | Spacing with four associations and two queries fixed |
| Lookup, one and two hops | 24 / 48 | Spacing with association/query counts and hop depth fixed |
| Copy | 4 / 8 | Input and complete generated answer |
| Direct/running-state parity | 4 / 8 | Input, and running answer for the scratchpad control |
| C-RASP depth 2 | 8 / 16 | Prefix horizon with the counting formula depth fixed |

The MQAR/lookup probes above do not increase association count or composition
depth. Those harder mechanisms need separately frozen data settings. Scientific
task difficulties, targets, budgets, selection rules, seeds/splits and paired
execution remain to be chosen after the unchanged benchmark gate passes.

The [prefix-oracle audit](complementary-prefix-oracle-audit.json) checks the
same depth-2, program-seed-0 C-RASP formula against the existing independent
point-semantics oracle. All 2,187 words of length seven (input length eight
including BOS), with 15,309 inclusive prefix labels, agree. A separate bank
of 128 words of length fifteen contributes 1,920 checked prefix labels;
diagnostic RNG seed 713 is unrelated to scientific model/data seeds. The
local manuscript *Knee-Deep in C-RASP*, Section 2.3, explicitly includes the
current position in past counts and permits derived integer operations.
The audit records its manuscript, implementation and independent-oracle hashes.

Order-sensitive support is explicit: `aaabbbc` and `aabbbac` each contain
three `a`, three `b` and one `c`, but the last truth values differ. Their
first six labels are true; the last is false and true, respectively. Thus
the formula is not a function of the final symbol histogram alone. The
supervised outputs are Boolean predicates at every inclusive prefix;
sequence accuracy requires the whole labelled sequence. This is a finite
oracle check, not evidence of a trained algorithm, a depth lower bound or
length transfer. The later scientific plan must include the C-RASP manuscript
hash and separate novel support. At fixed input length eight there are only
2,187 possible words, so a large training pool can exhaust ID novelty.

Recompute this audit without PyTorch into a fresh file:

```bash
python3 -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.audit_prefix_counting --output /tmp/complementary-prefix-oracle-audit.json
```

## Explicit native AdamW adapter

[ComplementaryConfig](../../complementary_config.py) and
[complementary_training.train](../../complementary_training.py) are a separate,
opt-in path through the existing common study loop. They require native AdamW,
betas (0.9, 0.98), epsilon 1e-8, all-trainable-parameter decay, no gradient
clipping, Normal(std=0.02) matrix initialization, clean disjoint pools, the
complete declared budget and at most 300,000 total updates. The defaults use
rate 0.0003, decay 0.1 and ten-update warmup. Constant and explicit fixed
cosine-tail schedules follow the primary next-update convention: update one
uses rate zero while still updating both moments and the native step counter.
The common loop retains its finite-gradient check at an infinite norm limit.
The adapter records every actual applied group rate and decay, and saves the
audited native optimizer state on CPU after the complete budget.

The task serialization, masked loss, finite disjoint pools and free rollout
evaluations differ from modular division. Matching AdamW does not establish
matched scientific task difficulty or any transfer or architecture effect.
The adapter starts a fresh run; it does not add resume to the general trainer.
The independent all-six benchmark gate and later architecture selection remain
external requirements for a scientific complementary campaign.

Four [targeted tests](../../tests/test_complementary_training.py) pass in 3.125
seconds. Actual final parameters and every native moment/step exactly match an
independent loop for both rate schedules, including short final batches. The
tests check the zero-rate update, reject changed optimizer semantics and the
300,000-update cap before model construction, restore the generic factory on
success/interruption, and reject self-consistently rehashed wrong-rate and
wrong-dataset archives. Portable verification imports no PyTorch; prediction
agreement is established separately by actual checkpoint reloads.

The complete [native CPU preparation](complementary-native-preparation/validation.json)
retains eight real width-8, one-layer, one-head cases: the seven mechanics above
at constant rate after warmup, plus a copy cosine-tail fixture with bounds
10–14. Every case completes 16 updates and five canonical observations; all
eight retain four novel ID validation inputs and four novel longer-input
validation probes. The archived total is 128 updates. The small models have
1,057–1,313 parameters. Every final and selected ID/length sequence accuracy is
zero; this preparation demonstrates execution and metric agreement, not
learning, stable grokking, algorithmic transfer or improvement.

All final and selected test metrics reproduce after independent checkpoint
reloads, excluding generation time. All eight native optimizer checkpoints
reload on CPU with the audited steps. Archives retain both model checkpoints,
the optimizer checkpoint/audit, raw generated data, complete histories and
rate traces, 82 repository source snapshots and the exact executed probe.
The native PyTorch implementation fingerprint is recorded without copying
the dependency into repository snapshots. The raw curves of these
five-observation implementation fixtures remain in each `history.jsonl`.

Reproduce into a fresh directory and verify without Torch:

```bash
.venv/bin/python -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.prepare_complementary_native --output /tmp/complementary-native-preparation
python3 -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.verify_complementary_native /tmp/complementary-native-preparation
```

The [review receipt](complementary-native-preparation-validation.json) records
the actual tests, portable check, raw archive checksums and unchanged active
43 Python/nine Lean sources, prepared 55-source schedule and both 62-source
confirmation fixtures. No scientific complementary plan or architecture is
selected, frozen or launched. Complete the current sparsemax-first GPU pair,
retain and review both outcomes, and follow the unchanged primary benchmark
gate before selecting scientific follow-up tasks.
