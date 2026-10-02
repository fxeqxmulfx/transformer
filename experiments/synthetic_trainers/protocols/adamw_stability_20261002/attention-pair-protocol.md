# Prospective exploratory softmax/sparsemax pair

The user requested sparsemax after the current native AdamW run and pointed to
the existing Lean formalization. This protocol prepares two fresh runs. Its
scientific manifest is frozen only after the complete 300,000-update reference
has been verified, its actual PNG/PDF figures reviewed, and its result committed.
No scientific normalizer case has started at the time of this preparation.

Sparsemax runs first; the fresh softmax control runs second. Both configurations
are frozen together before either run. The sole configuration difference is
`attention_normalization`. The current continued softmax run is preserved as
reference evidence, including its failed original 150,000-update result; it
does not replace the fresh paired control.

Both fresh cases retain mod-193 division, train fraction 0.25, model/data seeds
0/0, width 128, two layers, four heads, 436,104 parameters, batch 512 with the
original short last batch, learning rate 0.0003, decay 0.1 on all parameters,
native AdamW betas (0.9, 0.98), epsilon 1e-8, warmup 10, and no gradient clipping.
They retain original initialization, QK normalization, learned temperatures,
RoPE, epsilon-regularized XSA, FFN, residuals and tied embeddings. They each
execute all 300,000 updates, with canonical evaluation every 250, immediate
neighbor probes, full tensor diagnostics and a gradient record every update.
There is no early stopping at a target crossing or extension above the user cap.

The unchanged criteria require a pre-target low-held-out memorization plateau,
twenty consecutive joint 99% observations, and joint 99% at every canonical
observation from 250,000 through 300,000. Recovery measurements add episode
counts, durations, right censoring, the final sampled target streak, and
10,000-update windows across the tail and whole post-onset history. Every
support and failure is retained. Leading partial windows remain separate from
nominal-width frequency rates. Finite observations imply no behavior between
evaluations or beyond the budget.

Sparsemax follows the causal simplex projection in
`Transformer.GPTMini.Convex.Basic` and `Attention`: the objective
`sum(a²)/4 - sum(a*score)/2` has the same minimizer as Euclidean projection of
the original scores, without rescaling. The
[Lean and numerical preparation](sparsemax-preparation.md) retains the actual
24-theorem axiom audit and separate floating-point/gradient tests. Convex row
inference provides no joint-training guarantee. Sparse weights do not change
the dense matrix multiplication into a sparse operation.

The manifest pins 43 Python source files, both tagged training sources, nine
relevant Lean sources and the Lean audit receipt, environment, manuscripts,
corpus fingerprints, criteria, instrumentation and the complete reference
archive. Each case checks these fingerprints before entering the original
training loop. Checkpoint resume validates the normalizer before loading and
retains the original native optimizer, RNG, sampling cursor and permutation.
The serial driver never retrains a completed case. The archive worker waits
for the wrapper's completed state, including its peak-memory merge, then
saves and verifies each case without importing Torch. Each complete case can
be reviewed and committed while the second case runs.

The full paired archive copies both individual archives and the frozen
reference, preserving all losses, canonical histories, neighbors, dense
diagnostics, gradient traces, CSV, provenance and PNG/PDF figures. Its verifier
recomputes the comparison and frequency/recovery from those histories, including
rejecting a forged derived result after checksums are regenerated.

The prospective single-pair descriptive rules are an absolute final exhaustive
RHS accuracy gain of at least 0.01, or a speed ratio of at least 1.05 in both
training and wall time to eligible confirmation. Timing eligibility still
requires a complete stable-grokking softmax control and a complete persistent
candidate; rapid persistent candidate generalization without the plateau keeps
its own label. Failed controls retain measured costs and have no eligible
speed ratio. These rules never authorize a repeatable architecture claim.

This user-directed pair is exploratory. The unchanged all-six benchmark gate
(model seeds 4/5/6 crossed with data seeds 2/3), independent architecture
repetitions, and complementary MQAR, copy/parity, lookup/composition and prefix
counting with separate novel-ID and length-transfer scoring remain required.

The [four protocol tests and complete CPU receipt](attention-pair-preparation-validation.json)
retain a real 20+20-update full-width negative pair, twelve invalid manifest
variants, portable verification and all six original PNG and actual PDF reviews.
Together with the previous adapter/trainer checks this is fifteen distinct
tests. The CPU fixtures are pipeline evidence and use short explicitly labeled
criteria; they are not scientific learning or timing results.

After the reference result is committed, freeze and commit the actual scientific
manifest, check the real launcher, and start the serial driver and archive worker:

```bash
.venv/bin/python -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.freeze_attention_pair
.venv/bin/python -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.run_attention_pair --check-only
.venv/bin/python -u -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.run_attention_pair
python3 -u -m experiments.synthetic_trainers.protocols.adamw_stability_20261002.archive_attention_pair --trainer-pid ACTUAL_TRAINER_PID
```

Record the manifest commit and actual process/session IDs before continuing.
Check the processes and state before any restart; preserve all pinned sources
while either scientific trainer or archive worker is active.
