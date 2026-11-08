# Forward–Forward loss reformulation: fixed diagnostic

Frozen 2026-11-08, task #192, before numerical outcomes. This is a loss and
local-gradient diagnostic feeding the full training comparison #190. It is
not an end-to-end Forward–Forward trainer or an efficiency claim.

## Candidate

Following Hinton, arXiv:2212.13345 §§2–3.3, retain positive/negative local
learning but change the goodness and loss. Use label-independent features
`h=f(x)` and corrected distance goodness
`G(h,e_c) = (||e_c||² - ||h-e_c||²)/2 = <h,e_c> - ||h||²/2`.
The score is the real inner product (the real part of the Hermitian product
for complex vectors). Goodness differences equal the actual source logit
differences, so neither class-vector norms nor a threshold change inference.
All labels can be scored after one encoder pass. This changes Hinton's
label-conditioned encoder and independent positive/negative binary objective.

At each chosen local boundary use the mean pairwise margin loss
`L = sum_{c != y} softplus(mu + s_c - s_y)/(C-1)`, with **mu=1** and all
**96** competitors for the current task. The positive pair shares its feature
vector with every negative label. Gradients are
`q_c=sigmoid(mu+s_c-s_y)/96`, `q_y=-sum_{c != y} q_c`, and
`dL/dh=sum_c q_c e_c`, `dL/de_c=q_c h`. Batch losses/gradients are averaged.
No negative sampling or hard-negative selection is tuned after measurement.

The implementation exposes a local linear-head loss/gradient with its input
held fixed. Upstream derivatives must not silently cross a declared boundary.
Local derivatives, outer products, prototype construction and optimizer work
still cost computation. The source E is tied between input and readout: a
readout-only gradient is a partial derivative, not the total model gradient.
The full training candidate must specify local losses for every trainable
block and how all contributions to shared parameters are accumulated before
one update. This diagnostic makes no claim to have completed that trainer.

## Fixed checks

1. Rust FP64 tests: distance/dot-product equivalence, complex inner-product
   sign, finite-epsilon RMS norm saturation, stable extreme logits, common
   score-shift invariance, local finite-difference gradients, binary CE
   equivalence at mu=0 and a multiclass gradient that is not collinear with CE.
   Local gradients are tested with fixed input, and a two-block witness
   distinguishes that derivative from the end-to-end derivative.
2. One Kernite kernel computes pairwise loss and logit gradients from existing
   source logits/labels. Test against independent FP64 formulas for every row
   of all 24 original states (four arms, seeds 7/11, steps 0/5000/10000), all
   9409 input pairs, batches up to 512. Match original PTX and checkpoint
   identities; replay saved CE/accuracy. Fixed absolute/relative tolerances:
   loss `2e-5 + 2e-5*abs(reference)`, logit gradient `2e-7 + 2e-5*abs(reference)`.
   Kernel gradient includes division by actual batch size and 96.
3. On the next scheduled training batch, compute the partial readout gradient
   with inputs detached, compare it to CPU FP64 matrix products (tolerance
   `2e-6 + 2e-4*max_abs(reference)`). Take one **FP32-rounded SGD step of 1e-3**
   on E alone. First evaluate the same frozen-feature loss, then re-evaluate
   the actual tied network on the same batch. Retain decreases, increases and
   rounding-to-zero outcomes; do not require a favorable step. This is not an
   AdamW comparison or continuation training run. Restore the original state
   after the probe and verify every input file is unchanged.

Report source CE/accuracy, pairwise loss, gradient errors and the two meanings
of the one-step loss change. Measure no speedup in this diagnostic. Count
successful checks, not only successful descent outcomes. A 600-second cohort
cap excludes compilation and preserves all capped/error cases. Generated
outputs, logs and binaries remain ignored and are preserved for review.
