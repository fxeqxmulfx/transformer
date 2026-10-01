# Synthetic trainers for mini GPT architecture experiments

## Objective

Build a suite of synthetic trainers inspired by MQAR and the RASP family that
tests transformer capabilities more comprehensively than associative recall alone.
Use them to evaluate architectural improvements to mini GPT. Success means
either better task quality at the same training budget or less time to reach
a predefined quality target.

The generators, exact oracles, shared mini GPT training harness, and CPU tests
are implemented; see [README.md](README.md) for commands. [TASKS.md](TASKS.md)
explains task selection using the RASP, RASP-L, and C-RASP results. The initial
pair has expanded to 17 tasks and 37 comparison variants, including free answer
generation and scratchpad controls. Calibrated
difficulty ranges, evaluation targets, and architecture ablations remain open.
Memorization and double-descent profiles, a uniform random sequence control,
fixed noise, paired capacity/data-size sweeps, and delayed-transfer diagnostics
are also implemented; see [STUDIES.md](STUDIES.md). A 61-run AMSGradW/softmax
baseline is saved under [baselines/](baselines/amsgradw_softmax_20261002/metrics.csv).
It did not demonstrate delayed algorithmic length transfer. The subsequent
[27-run data/width/depth study](baselines/amsgradw_softmax_scaling_20261002/metrics.csv)
is complete: larger models solve ID copy but still fail length transfer;
no repeated mean size-double-descent effect appears on direct parity. The
paper reproduction loop below is now running separate reference protocols.

## Paper reproduction loop

The requested next objective is to reproduce grokking and double-descent effects
from the local papers through existing or additional trainers. The baseline
and scaling measurements remain part of this work. Increasing model size alone
does not count as a reproduction.

For grokking, use the modular division mod 97 experiment in
[Convexifying Transformers](../../papers/arXiv-2211.11052v1/arxiv.tex), Section 4,
as an initial reference: its standard transformer fits train around 1,000 updates
but generalizes after more than 100,000 updates. Recover missing implementation
details from the authors' published reference code, preserve attribution and
licenses, and record every difference from the paper. Add a complete, oracle
checked modular arithmetic corpus with disjoint splits. Compare the reference
transformer with GPTMini and the requested raw AMSGradW optimizer; label those
architecture/optimizer substitutions as adaptations. Use sufficient budgets to
observe the reported delay, rather than stopping when train has been fitted.

For double descent, reproduce an interpolation transition and both descending
branches of held-out risk from [Deep Double Descent](../../papers/arXiv-1912.02292v1/paper.txt).
Its random-feature case study supplies a simpler fixed-feature reference with
a measurable linear interpolation threshold. Its CNN and translation results
use different data and model families; synthetic GPTMini analogues should not
be described as exact numerical reproductions of those experiments. Check the
published protocols and reference implementations before freezing the plan.

The [random-feature confirmation](baselines/fashion_rff_20261002/summary.json)
completed 123 fits and reproduced both sample-wise and model-wise
classification-error double descent in all three data/feature seeds, with the
peak at n=d=1000. Missing paper details and numerical differences are explicit
in [STUDIES.md](STUDIES.md#reproduced-random-feature-double-descent-2026-10-02).
The modular division reference is running its fixed 150,000-update budget;
passing its CPU tests or reaching train accuracy does not complete grokking.

Freeze the data, targets, evaluation cadence, grid, and budget before each
confirmation experiment. Separate exploratory calibration from repetitions,
use several initialization seeds, and retain per-seed fit failures as well as
mean curves. Record final and selected scores, complete histories, source/data
hashes, compute/memory, and the exact source figure or section being compared.
A finite curve witness alone is insufficient evidence of a robust reproduction;
report repeatability, the interpolation region, and the size of each effect.
Do not infer a universal data count from GPT parameter count or call an
unconfirmed run successful. Continue the reproduction loop while useful
experiments remain, and document failures and remaining uncertainty explicitly.

## Trainer design

Use complementary task families with procedurally generated examples,
exact algorithmic answer oracles, and adjustable difficulty. Retain MQAR as a
reference task and reuse the existing infrastructure in
[`convex_mqar`](../convex_mqar/README.md) where appropriate.

The initial pair is composed associative lookup and causal prefix computation,
with Dyck-1 status and neutral-letter block recognition as separately scored
prefix modes. RASP and C-RASP supply algorithmic references and a proved depth
hierarchy for the block languages. Test composition, dependency distance, and
generalization to longer sequences independently.

The complete suite adds global frequency aggregation and selection, full
sequence copying/reordering, typed bracket matching, counting, carry arithmetic,
parity with external state, positional distribution transfer, and varied nested
counting formulas. Keep scores separate by task, format, and difficulty.
Free-generation evaluation includes stopping behavior and the cost of the full
answer. Hard-carry tests and AND position transfer complement length extension.

## Comparison protocol

Establish a baseline using [`gpt_mini.py`](../gpt_mini.py), then compare
architecture variants on the same generated splits, optimizer protocol,
tuning budget, and hardware. Record parameter counts, compute, and memory
when a change affects their cost. Choose validation targets and budgets
before the final comparisons; use held-out tests after model selection.

Measure task accuracy and exact-answer accuracy where applicable, quality at
a fixed budget, and wall-clock time, updates, and examples needed to reach
the target. Report results across seeds with learning curves. Include unseen
sequence lengths and harder dependency structures to assess generalization.
Keep throughput and time to target separate: faster steps only achieve the
objective if the required quality is reached sooner.

Distinguish finite-pool fitting from reusable algorithmic behavior. Track the lag
between clean train fit and confirmed success on novel held-out and transfer
inputs. Compare that lag with observed epoch-wise second descents, while keeping
temporal association separate from a causal explanation. Assess random-data
coding gain and noise fitting independently; memorizing noise can coexist with
algorithmic transfer. Use both final and validation-selected checkpoint curves,
and state whether sample-size sweeps hold updates or epochs fixed.

## Deliverables

1. Reproducible generators, answer oracles, and difficulty configurations for
   the complete suite (implemented).
2. Calibrated baseline mini GPT measurements on the suite and MQAR reference.
3. Architecture ablations showing quality at a fixed budget and time to target.
4. A comparison report identifying which architectural changes help which
   capabilities and whether those gains generalize.

The next step is to calibrate the baseline to set difficulty ranges and
evaluation targets, then compare architecture variants. Formal expressivity
results guide task design; learning speed and quality require experiments.
