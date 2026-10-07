# Grokking: an active experiment-to-Lean research cycle

Started 2026-10-07 at the user's explicit request. Status: active. Work solo.
The [Basis/convex cycle](plan.md) remains paused and unfinished; ANSR stays
stopped. Continue the [measurement study](grokking_plan.md) with ordinary
softmax and unchanged native AdamW. Every declared run retains 150,000
updates. The research goal has no requested token or time limit.

## Objective and limits of the claims

Explain delayed generalization through precise, competing mathematical
formulations and formalize the defensible statements in Lean. Systematically
expand the formulations below as new evidence warrants. A finite list is
not a claim to exhaust every possible explanation. An empirical transition,
a simplified model and a theorem about the actual GPTMini training system
are separate results; the transfer must be proved, not hidden in a definition.

## Formulations to investigate

| Route | First mathematical question | Evidence / source | Status |
| --- | --- | --- | --- |
| Operational delayed generalization | Define train fit, a sustained held-out plateau, later generalization, and finite-budget censoring without future information entering a detector | Power et al., arXiv:2201.02177; pinned causal histories | Definition/proof work next |
| Confidence versus decisions | Positive logit scaling preserves every ordering but can strictly decrease cross-entropy; quantify the missing conditions and counterexamples | Prieto et al., arXiv:2501.04697v1, section 4.2; measured endpoint projections | First Lean target |
| Spectral optimization dynamics | Derive slow modes and exact delayed test-boundary crossing from an actual gradient flow or update recurrence, including convex toy models | Liu et al., arXiv:2205.10343v2, effective embedding dynamics; Žunkovič/Ilievski, arXiv:2210.15435v1, section 3 | Local manuscripts read; source assumptions remain explicit |
| Rule learning versus memorization | State when a reusable rule component wins over an example-specific component under the same training objective | Nanda et al., arXiv:2301.05217v1, sections 4–5; circuit-efficiency literature to investigate | Research pending |
| Geometry of representations | Prove orbit projection identities, scale/bias invariances and counterexamples to symmetry-only success; connect train-fitted probes to held-out decoding | Division common-scaling observer, actual checkpoint features | Six measurements implemented and being checked |
| Gradient coherence and implicit bias | State what batch gradient agreement can and cannot imply; distinguish loss descent, task structure and optimizer-specific bias | Fixed-batch gradients; ordinary AdamW | High agreement at initialization rejects a standalone signal |
| Regularization and norms | Relate an explicitly stated penalty or decay update to competing solutions; reject norm-only success claims | Golechha, arXiv:2405.12755v1, section 3; current grouped norms | Whole-model norm barely changes across the observed transition |
| Phase transitions | Specify an order parameter, control parameter, asymptotic regime and distribution before claiming a thermodynamic transition | Liu effective theory; Žunkovič/Ilievski solvable models | Analogy only for current finite GPTMini; finite-size scaling not established |
| Numerical precision and softmax collapse | Compare exact-real loss gradients with floating-point zeros and prove only the quantization model actually used | Prieto et al., section 3; local CUDA/CPU execution | Research pending; optimizer remains unchanged |
| Actual transformer and AdamW transfer | Identify which premises about the real forward map, token task and optimizer are verified, and which remain unproved | Existing GPTMini List Int semantics; native lab checkpoints | General convergence to a rule is not proved |

## Cycle

1. Read each local manuscript and pin its version, section, definitions,
   hypotheses and quantifiers. Search library lemmas with DT.
2. State the claim as a Lean theorem. Prove a corrected source statement
   when fixable; prove a counterexample when false. Add a satisfying
   example for every hypothesis list. Never use an axiom or move the
   requested result into a definition.
3. Find an observable consequence and a control that can reject it. Run
   the ordinary transformer or read immutable weights; probes and
   ablations never alter training or supply a teacher to the model.
4. Compare with all available seeds and negative controls. Retain failures
   and distinguish a causal signal from a retrospectively selected explanation.
5. Build/audit/index/check Lean changes; run all Python tests for a Python
   change. Commit each checked logical result immediately. Update this
   plan with proved declarations, measurements and remaining gaps.
6. Continue with the next unsolved route or a revised formulation. Do
   not mark the research direction complete merely because a toy
   example, a metric or a long run succeeds.

## Current evidence and immediate work

The original GPTMini seed 1 completed its full 150,000-update budget with
100% held-out answer accuracy. A checkpoint-preserving repeat has the
same canonical train/held-out metrics at every compared observation.
Frozen probes, spectra, neuron profiles, gradients, head/subspace ablations
and endpoint logit changes have been measured around the actual
33,000–36,000 transition. These are associations and named intervention
effects, not a universal detector. Full control/seed budgets remain active
or queued.

Finish the current Python check and results commit, then formalize the
confidence/decision distinction and a genuinely optimized toy trajectory.
Keep the phase-transition hypothesis open until its extra requirements
are checked. New papers may introduce open theorem statements under
AGENTS.md's explicit allowance; existing sorry counts must not increase.
