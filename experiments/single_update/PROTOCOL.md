# Actual single-update audit

Frozen 2026-11-08, task #186, before measuring any update. This extends the
saved-buffer diagnostic in task #185. No architecture selection or new
training campaign is authorized by this protocol.

## Cohort and execution

- All four original arms, seeds 7/11, checkpoints 0/5000/10000: 24 independent
  probes. Restore parameters and both AdamW moments in memory. Never write
  into the original checkpoint directories.
- Run the next scheduled minibatch, with the original optimizer, warmup,
  projection constraints and sampler. Extend the sampler to 10001 updates;
  verify its first 10000 updates against the saved sampler hash. The eight
  10000-to-10001 probes are explicitly **post-budget probes**, outside the
  original architecture ranking.
- Use the original Kernite source pinned by SHA256
  `14cff3b553d3afb1f9bf90bb6b50cfd7a879058ec2876f859a198e07a09a83da`
  (rotate90_complex, implementation b70811e). A native build generator checks
  the source digest and appends read-only instrumentation to a generated
  copy. Original forward, backward and optimizer code remain byte-for-byte
  intact. Generated source and reports are ignored.
- Match the original compiled PTX digest. Check metadata, state/history,
  split, initial parameters, sampler and optimizer protocol hashes. Recheck
  input files after the audit. Compare the instrumented next state with an
  ordinary source `train_step` from the same old state: all parameters and
  both moments must match bitwise.
- GPU: RTX 3050 Laptop, 4 GiB; OpenBLAS uses four CPU threads for geometry.
  Runtime cap: 600 seconds excluding compilation. Save completed cases and
  explicit errors incrementally; no failed case is silently dropped.

## Gradient attribution

At each old checkpoint, evaluate the entire 4704-row training gradient
without updating parameters, weighting batches by their actual lengths
(including the 96-row tail). Capture the readout embedding gradient before
the tied encoder adds its contribution. Also capture the full embedding
gradient for the next scheduled minibatch. No validation gradients are used.

Let R be the full-training readout gradient, F the full-training total
embedding gradient, B the next-minibatch total gradient, and M the corrected
first moment **after** that minibatch update. The exact additive comparison is

`M = R + (F-R) + (B-F) + (M-B)`.

The three differences describe encoder, sampling and moment-memory effects
in this declared order. They are additive terms, not independent causal
ablations; retain their cross terms. Use the new coordinate preconditioner D.
For each old training-fitted projector Q, decompose

`QDM = DQR + (QDR-DQR) + QD(F-R) + QD(B-F) + QD(M-B)`.

Report all five energies, all ten pairwise cross terms, total energy and
closure error. Form class contrasts *after* each row's own D is applied;
never assume target and competitor share a preconditioner. Report actual
FP32 parameter movement separately from the FP64 factorization, including
the rounding remainder. No worst-case D-max bound is substituted for these
measured terms.

Validate gradients on the first eight canonical training rows of every
checkpoint: independent scalar FP64 centered differences for two fixed
normalized dense embedding directions (SplitMix64 seeds 313/911), step
1e-5. Absolute-plus-relative tolerance is `2e-5 + 2e-4*abs(reference)`;
retain actual errors. Validate readout gradients independently against
the FP64 softmax derivative on captured GPU normalized features (maximum
error `2e-6 + 2e-4*max_abs(reference)`). Compare GPU logits with the scalar
oracle on those rows (`5e-5*(1+max_abs(reference))`).

## Geometry and margins

Retain sum and cubic-orbit groups, requested ranks 16/32/64, and both splits.
Fit centers and PCA bases only on observed training rows, using the same
normalized readout features and rank threshold as tasks #184/#185. Refit after
the actual update. Record achieved ranks and preserve all groups.

For class contrast w, centered feature displacement x-mu, let
`u=Qw`, `r=Q(x-mu)` and `J=<u,r>`. Track actual signed J and its change,
not only norm products. Resolve the change into decay, the five optimizer
terms, FP32 update rounding, readout-basis movement, centered-feature
movement and feature-basis movement. This identity retains all finite-step
effects; a 5000-step checkpoint difference is not an optimizer step.

The training-affine reconstruction is `s=<w,mu+(I-Q)(x-mu)>`.
Carry the projection defect `e=<w,r>-J`; features already include the final
RMS denominator, so the reconstructed competitor gap is `s+J+e`. Compare it with
the executed GPU gap and retain the measured execution allowance. For each
query report actual margin, reconstructed margin, maximum absolute residual
budget, and the sufficient condition that *every* reconstructed competitor
gap plus its absolute residual/defect/execution allowance is negative.
Ratios to reconstructed margins are defined only when those margins are
positive. Report improvements, regressions, lost/gained correct decisions,
and residual growth alongside margin change; never drop errors or difficult
queries. This is a conditional retrospective diagnostic, not a predictor
that has no access to the learned readout.

FP64 algebra/energy and geometry identity tolerances are
`1e-8*(1+largest relevant magnitude)`; reconstructed-vs-GPU gap tolerance is
`5e-5*(1+largest gap magnitude)`. No positive certificate may misclassify.
Source-inspection correction before any measurement: #184/#185 fit bases
after final RMS, not before it. The initial protocol text said pre-RMS;
the audit uses normalized features to preserve that comparison and checks
their relation to the raw state separately.

Existing actual-Adam and moving-interaction Lean results are the theoretical
reference. Any further convergence theorem must be justified separately.
