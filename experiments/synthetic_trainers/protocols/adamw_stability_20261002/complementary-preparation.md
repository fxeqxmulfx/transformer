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
has no warmup option. A later frozen complementary protocol must explicitly
document these differences, or add and verify matching options after the live
frozen campaigns finish. Preserve identical optimizer settings within each
architecture pair. No production source or current frozen plan changed here.

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
