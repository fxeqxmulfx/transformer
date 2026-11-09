# Checkpoint probability control under full training motion

This continuation addresses the future-probability premise left open by
`ce_metric` at `d1066cfd`. Freeze this protocol before new numerical outcomes.
Use Rust/Kernite and Lean. The user's 2026-11-09 instruction authorizes
recreating the missing training evidence from the committed recipes.

## Mathematical target

For actual softmax logits z and z', a bound on every competitor contrast
`(z'_u-z'_c)-(z_u-z_c) <= D_c` gives

```
p'_u <= 1 / (1 + sum_(c != u) exp(z_c-z_u-D_c)).
```

A uniform D gives the sharp ceiling
`p'_u <= p_u*exp(D)/(1-p_u+p_u*exp(D))`, and, for D>=0, the simpler ceiling
`p'_u <= p_u*exp(D)`. Include the self term correctly and retain every
competitor. Prove a finite-prefix form from cumulative contrast increments.
Compose this ceiling with the existing actual partial-CE momentum result;
do not assume future small probabilities as an additional premise.

Bound logit-contrast motion by both readout and feature motion, including
their mixed term. Full-model motion must include all trainable parameters,
saved moments, class-dependent adaptive scaling, tied embeddings, actual
schedule and explicit execution allowances. A partial-CE nonincrease result
still leaves the full potential perturbation to be bounded separately.

## Reproduction and fixed comparison

Recover the exact source inputs and verify their recorded digests before
running them. Preserve historical protocols and outputs. A fresh run on
different hardware is a reproduction, not a byte-identical restoration of
the missing historical evidence. Record hardware, toolchain, source and
executable fingerprints. Preserve all attempts and errors. Do not fabricate
the missing baseline reports or bypass historical provenance checks.

Recreate all three main-model cohorts: .001 WSD, .01 WSD and early settling;
seeds `[7,11,1,3,5,13,17,19]`; 10000 updates per seed; model, initialization,
data split, sampler, loss, clipping, AdamW and schedules unchanged. Keep the
31 original geometry probes and all update traces. If source recovery needs
a new bootstrap runner, freeze that runner and its provenance checks before
training. Its hardware-specific cap is 7200 seconds per cohort, including
failed attempts and verification, excluding compilation. This is a new
reproduction budget; it does not amend the historical 1800-second protocols.

Use every one of the 144 checkpoint slots at anchors
`[0,1000,4000,6000,8000,9500]`. Construct a ceiling over **all 4704 training
inputs** at each checkpoint, so the bound covers any later training minibatch.
A maximum from one observed minibatch is a separate diagnostic only.
Use horizons `[0,1,2,4,8,16,32,64,128,250,500,1000,4000]`, retaining only
horizons ending by step 10000. Report initialization separately.

Freeze the forecast files and their hashes in a constructor that cannot
read later checkpoints, traces, future gradients or replay outcomes.
A separate verifier reads future evidence. Compare the universal q=1
ceiling, the fixed checkpoint ceiling without motion (explicitly invalid
as a forecast), and the motion-propagated ceiling. Retain isotropic and
checkpoint-fixed feature metrics from `ce_metric`, with their ordinary
energy conditioning cost. Report the actual current probabilities, the
prospective CE condition and the full-update potential change separately.

## Failure policy and checks

Retain the full inventory on missing data, caps, numerical failures and
failed sufficient conditions. No seed or horizon is selected by its result.
Use the existing optimizer allowances, feature cap `128*(1+5e-6)`, partial
gradient allowance `1e-5`, and scaled algebra/closure allowance `1e-8`.
Probability comparisons use absolute allowance `1e-8`; record the maximum
excess independently of the pass flag. These finite checks are not a
universal floating-point theorem. Constructor plus verifier cap is 1800
seconds, excluding compilation and source recreation; capped slots remain.

Run native tests against independent calculations, clippy with warnings
denied and fmt; full Lean build, custom/global axioms, native INDEX and
forbidden scan. Preserve generated evidence at the root outside the
worktree. Commit sources, reproducibility inputs and authored findings;
keep generated reports, plots, logs and binaries ignored.

Success for this step is a verified prospective probability bound with its
useful horizon measured, including zero if it is uninformative. Closing the
overall goal additionally requires useful full-model control and a connection
to geometry and prediction; this step alone does not prove discovery.
