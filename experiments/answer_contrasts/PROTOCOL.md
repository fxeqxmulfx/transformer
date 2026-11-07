# Combined answer contrasts on frozen cubic checkpoints

Fixed before measurements, 2026-11-08, task #179. No training. Retain all four
arms, seeds 7/11 and steps 0/5000/10000 from the original native cubic cohort.
Preserve `flat_approx/report.json` and its protocol. Reuse the verified source,
split, OpenBLAS and scalar-oracle equations from commit 9328bc83. Check every
checkpoint state/history/split identity and stored correctness count. Compare
batched logits with the scalar oracle on the same fixed 32-input sample.
Limit BLAS to four CPU threads and analysis to five minutes, excluding compilation.

Fit signed indicator probes from the raw residual state plus an intercept,
using training inputs only and the same ridge penalty as flat_approx: 1e-8
times the mean Gram diagonal. Fit all 97 sum indicators and independently
fit the 33 cubic-answer indicators. Keep every coordinate. Correct total mass
to one along a fixed reference: uniform 1/97 over sums and its cubic pushforward
over answers (one zero preimage and three preimages for each nonzero cube).
No clipping, positivity constraint, feature selection or validation fitting.

Pool the 97 predictions by their cubic answer. Fixed-design linear regression
commutes with pooling its targets, and the chosen correction preserves this
identity. Check the independently fitted answer probe against the pooled probe.
An agreement is an implementation check, not two independent discoveries.

Fit two raw-score decoders using training means: one per sum and one per cubic
answer. Center each source score vector across its 97 output classes before
fitting; all class contrasts are unchanged. Every output class competes,
including the 64 labels outside the cubic image. The two decoders remain separate
declared diagnostics. Their ideal features use known sums/answers only when
computing diagnostic targets and margins; they are not substituted into the model.

For each query and competitor, measure the ideal decoder gap, source-alignment
contrast error, and recovered-versus-ideal contrast error after summation.
Compare the following bounds, retaining competitor-specific gaps:

1. Exact magnitude of the combined recovery contrast (retrospective measurement).
2. Sum of absolute coordinate contributions.
3. Half the range of contrast coefficients times L1 representation error.
   Include midpoint times absolute total-mass residual for FP64 rounding.
4. For the quotient decoder, the same coefficient range applied to the unpooled
   97-coordinate L1 error. This isolates cancellation without changing the decoder.

Add measured source-alignment error to each bound and compare with that
competitor's ideal gap. Report a positive budget only if all 96 comparisons
pass, initially granting zero execution error. Also report direct ideal/source
contrast budgets, actual/reconstructed accuracies, L1/L2 errors, coefficient
ranges, denominators, alignment, and best/worst budgets on both splits.
Positive budgets using measured held-out errors are descriptive sufficient
conditions, not a predictive generalization theorem or certified FP32 execution.

As a separate check on alignment, derive the affine residual weights between
the actual tied readout and the probe/decoder composition. Bound each contrast
by `||residual_weights||_2 * ||state||_2 + |residual_bias|` (Cauchy-Schwarz),
and use it in place of measured alignment. These weights use no validation
data. The representation error is still measured per query. Retain failure
of this more conservative condition. Verify the affine expansion directly.

All 24 cases, all failures and both decoder choices remain in the report.
This is a retrospective diagnostic on an already examined validation split.
It is not a new learning cohort, and no source architecture is changed.
