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
are also implemented; see [STUDIES.md](STUDIES.md). Their empirical effects still
need calibration and sustained training.

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
