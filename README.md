# Transformer

This project has two goals:

1. Prove end-to-end convergence to generalization for the training process
   studied in **Grokking: Generalization Beyond Overfitting on Small Algorithmic
   Datasets** (Power et al., [arXiv:2201.02177](https://arxiv.org/abs/2201.02177)).
2. Build a **convex model that passes the GPT stress tests**, retaining their
   original inputs, splits, answer losses and evaluation criteria.

Lean proofs live in [src/](src/Transformer.lean); native Rust/Kernite
experiments live in [experiments/](experiments/README.md). Both goals remain
unfinished. The status below is as of **2026-11-12**.

Readiness percentages are rough assessments of project progress, not proof
coverage or benchmark pass rates. They include the proved spectral foundation
below; the remaining milestones are not equally difficult.

## Proved foundation: spectrum, recovery and classification

[Flat spectrum](review/flat_spectrum.md) contains substantial proved foundations
for both goals. Its results determine which recovery objectives can work,
which representations admit recovery, and what additional assumptions are
needed. The next steps build on these results.

The spectrum belongs to a task kernel `K(a,b) = phi(f(a,b))`, so the output
readout `phi` matters. The same task can have a flat kernel in one readout and
a rank-one kernel in another. The arithmetic statements below use a prime
modulus `p`.

| Result already proved | What it supplies |
| --- | --- |
| **Theorem A and the flat-kernel extension** | A flat kernel with no zero entries cannot be recovered by minimum-nuclear-norm interpolation from any proper subset of its entries. This rules out that recovery objective for that kernel. |
| **Theorem B** | For `f(a,b) = a^3 + ab` under uniform inputs, every unit-norm function of the answer has only `O(p^(-1/2))` correlation with the specified nonconstant additive characters of the intermediate `ab`. This bounds a particular learning signal. |
| **Theorem C** | For `f(a,b) = P(a) + ab`, bounds on the centered operator norm hold for every output readout, with degree-dependent bounds for nonlinear `P`. This extends the analysis beyond a single chosen character. |
| **Theorem E** | A square unit-modulus rank-one kernel is uniquely recovered by minimum-nuclear-norm interpolation exactly when the observation graph is connected and contains a perfect matching. This is a positive recovery criterion. |
| **Feature codes and margins** | A separating bank of recovered features gives correct classification with a positive margin; sufficiently small score errors preserve it. Multiplicative characters give rank-one features for division on nonzero residues. |
| **Convex spectral bounds (SB1–SB3)** | Operator-norm upper bounds admit flat targets while preserving convexity. They provably recover one flat two-by-two example; a separate ambiguity example proves that the bound alone cannot guarantee unseen answers in general. |
| **Pure-factorization training** | For convex analytic losses bounded below, gradient descent with a positive coupled quadratic penalty converges to a global minimum under the stated step-size, width and initialization hypotheses. WD13 proves why this result cannot be transferred to AdamW in general. |

These theorems address different parts of learning. Theorem A concerns exact
kernel recovery; Theorems B and C bound specified correlations and operators.
They do not establish a general impossibility of classification, convex
learning or transformer convergence. The positive recovery and margin results
already provide endpoints to which a training theorem can connect.

## Goal 1: end-to-end grokking convergence

**Estimated readiness: 30%.** Spectral recovery, the protocol and conditional
correctness results are in place; the central positive convergence argument
for the actual training process remains open.

The first target is division modulo 97 from the grokking paper: train on half
of the equation table and prove that held-out accuracy reaches and stays at
100%, under explicit assumptions on the split, initialization and optimizer.
The theorem must connect the actual training trajectory to these predictions.

### Ready

- **The training protocol is formalized.** The exact-real decoder, tokenization,
  supervised loss, Adam/AdamW updates, initialization law and shuffled batches
  are defined in [Grokking/Protocol](src/Transformer/Grokking/Protocol.lean).
  The trajectory and accuracy are measurable; finite accuracy tending to one
  is equivalent to eventually answering every held-out equation correctly.
- **Representation and conditional correctness are proved.** Constructed
  parameters of the paper's decoder solve all 9,312 division equations.
  Margin and quotient-symmetry certificates imply held-out correctness when
  their premises hold. These parameters are constructed; training is not
  proved to reach them.
- **The limits of the original target are proved.** Unconditional almost-sure
  convergence over random half-table splits is false. Requiring every residue
  to appear as a training answer is still insufficient. Split symmetries give
  explicit obstructions and upper bounds on the probability of eventual
  perfect generalization.
- **A concrete spectral route for division is proved.** Theorem E recovers
  its multiplicative-character features on nonzero residues under the graph
  conditions above. The feature-code results turn recovery of a separating
  bank into correct answers with a margin. The actual-decoder construction
  separately covers zero numerators as part of all 9,312 equations.
- **Finite training evidence is available.** The native
  [division experiment](experiments/grok_division/README.md) records training
  runs, checkpoint checks and later losses of accuracy.

### Not ready

- **A positive end-to-end theorem.** We have no proof that this training process
  eventually generalizes permanently, or even a positive lower bound on the
  probability of that event. A valid split condition or a quantitative success
  guarantee still has to be established.
- **Discovery and persistence under the actual optimizer.** We have not proved
  that Adam/AdamW discovers the required representation and preserves correct
  held-out decisions through later updates. A correct representation, an ideal
  completion optimum or one successful checkpoint does not supply this step.
- **The complete execution bridge.** The exact-real training trajectory is
  not connected to the full floating-point run. Long-run behavior at ReLU
  kinks and the corresponding numerical guarantees remain open.

The next proof step is to connect the actual updates to sufficient learned
score structure, using the existing division features and correctness
certificates, and then prove that structure persists. Correct score ordering
can suffice; recovering every entry of a chosen kernel is a stronger demand.
The pure-factorization convergence theorem does not supply this Adam/AdamW
argument.

The [convergence review](review/grok_division_convergence.md) gives the precise
target, proved refutations, missing steps and audit commands. The paper reports
an empirical phenomenon; this project is seeking a theorem explaining it.

## Goal 2: a convex model for the GPT stress tests

**Estimated readiness: 40%.** Convex theory, native learners and benchmark
replay exist; complete-suite performance and a compact general learner remain open.

The [GPT stress-test catalog](stress_tests_for_gpt.md) describes **165 cases**:
arithmetic, sequence algorithms, recall, recurrences, forecasting, numeric
probes, image classification and language modeling. Their original targets
differ. Random-label and random-language cases retain their role as negative
controls. The current experimental milestone is the **35-task short/long
subset**: 25 short and 10 long tasks, judged by their original held-out metrics.

### Ready

- **Convex models with ordinary answer cross-entropy are proved.**
  [Flat spectrum](review/flat_spectrum.md) develops categorical routing and
  generated-output channels, coupled complex moments with spectral bounds,
  and recovery under explicit structure and observation conditions. It also
  proves examples where a global training optimum gives a wrong unseen answer.
- **The spectral recovery theorems are already reused.** CM13 directly applies
  Theorem E: when the true table has the specified separating rank-one code
  bank, a connected observation graph with a perfect matching and nuclear
  budgets identify the complete probability table at ideal observed CE. The
  [Lean proof](src/Transformer/Grokking/FlatSpectrum/CoupledMoments/Schatten/GraphRecovery.lean)
  explicitly invokes the existing recovery theorem. CM18 adds conditional
  certificates for decoded answers; their linear answer constraints preserve
  convexity even when the spectral relaxation alone admits a wrong optimum.
- **A common causal probability interface is formalized.** The
  [full-catalog theory](review/convex_causal.md) supports calibrated ordinary
  CE, stochastic outputs, bounded-context interactions and free generation
  with EOS. A finite candidate learner has proved recovery when the true rule
  belongs to its fixed bank and the observed inputs distinguish the candidates.
  The [depth extension](review/convex_depth.md) retains nonlinear interactions
  by learning joint weights over complete paths through fixed block libraries.
- **Native training and replay are implemented.** Full-batch, persistent
  minibatch and forward-only learners reconstruct the 35 original datasets
  and replay saved predictions. The
  [certified paired-decoder implementation](experiments/convex_stress_certified/README.md)
  checks training-derived certificates and preserves their answer constraints
  during updates; all three modes now recover every held-out absolute-difference
  answer, while seven other tested tasks remain uncertified.

These are separate empirical results:

| Evidence | Result | Scope |
| --- | --- | --- |
| [Causal counts](experiments/convex_stress_counts/README.md) | 10/35 primary passes for full-batch and minibatch at scale 10; 9/35 for forward-only | Complete 210-cell inventory and replay; stricter all-split counts are lower |
| [Spectral sharing](experiments/convex_stress_sharing/README.md) | All six selected tasks pass in all three modes: 18/18 choices | Targeted arithmetic follow-up, selected by training CE |
| [Calibrated causal readout](experiments/convex_stress_calibrated/README.md) | 8/35 primary passes, versus 9/35 for the scaled control | First cohort; memory failures counted as failures |
| [Certified paired decoder](experiments/convex_stress_certified/README.md) | 100% held-out absolute difference in all three modes; seven other tasks uncertified | All 24 outcomes replay; three returned models pass raw audits; integration with earlier candidates pending |

### Not ready

- **A complete benchmark pass.** No single completed convex-model protocol
  passes all 35 tasks, and the full 165-case catalog has not been implemented
  and evaluated by a common native convex learner. Combining the best scores
  from separate pilots is not a whole-suite result. The
  [combined spectral/causal cohort](experiments/convex_stress_hull/README.md)
  retains interrupted fits and has no complete suite result.
- **A compact representation that can be learned across the catalog.** The
  universal prefix construction has exponential size. Fixed path and candidate
  banks require selection and coverage assumptions; convexity of their joint
  coordinates does not make arbitrary transformer weights jointly convex.
- **General unseen-answer identification.** Low training CE and spectral
  feasibility alone do not determine missing labels. The new coding and
  paired-decoder certificates need their structural assumptions and coverage
  verified for the original tasks; fixture success is not stress-test success.
- **Practical convergence and numerical guarantees.** The exact-real results
  do not establish convergence rates for every native solver or turn a small
  numerical CE gap into a recovery certificate. Memory-bounded execution,
  replay of every outcome and the remaining matched transformer comparisons
  are part of the unfinished benchmark work.

The next model step is to apply these existing recovery results to the
original tasks: establish their structural assumptions, audit observation
coverage, and use the resulting certificates in the learner. Flat targets
need constraints that admit their spectral shape. Claiming universal recovery
by minimum-nuclear-norm interpolation would contradict Theorem A. Capacity,
identification of unseen answers and solver convergence each need their own
check before a benchmark pass can be claimed.

The [experiment plan](EXPERIMENT_PLAN.md) records the frozen cohorts and next
steps. Each experiment README records its own protocol, results and limits.

## Working in this repository

Use the Lean toolchain pinned in [lean-toolchain](lean-toolchain) and native
Rust/Kernite tooling. Existing Python code is historical reference material;
new work and verification do not use Python.

```sh
LAKE_ARTIFACT_CACHE=false lake build
LAKE_ARTIFACT_CACHE=false lake env lean scripts/Axioms.lean
```

[INDEX.md](INDEX.md) locates declarations and records unfinished proofs.
Regenerate it with the [native index generator](scripts/native_index/README.md)
whenever `src/` changes. Experiment READMEs supply their native test and
reproduction commands. Generated runs, reports, plots and logs remain local
and ignored; source, protocols and authored findings are committed.
