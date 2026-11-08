# Fixed residual learning dynamics diagnostic

Fixed 2026-11-08, task #184, before measurement. No training, optimizer update,
checkpoint modification or new cohort. Keep all 24 original states: four arms,
seeds 7/11, steps 0/5000/10000, and all 16 adjacent intervals. Verify original
state/history/split hashes, scalar-oracle samples and saved correctness counts.
Use four OpenBLAS threads and a five-minute analysis cap excluding compilation.

Freeze features at each checkpoint to calculate the actual partial derivative
of the training CE with respect to the final linear readout:

```
G_c = mean_train (p_i,c - 1[y_i=c]) * h_i.
```

Each h_i retains its actual final RMS denominator. This is a partial readout
gradient: the shared embedding also affects the encoder, and the full training
gradient is not identified with G. Do not infer an AdamW update from this partial
gradient. All probabilities in G come from training rows.

Build centered training-state SVD bases separately for sum classes and cubic
answer orbits. Fix ranks 16,32,64, clamped to the available rank above the
relative singular cutoff 1e-10. The basis uses no validation labels, logits,
probabilities or target errors. Check observed training-label agreement and
the known multiplier/translation routes. Validation labels are used only by
the separate source-accuracy audit. Training and validation residuals are both
reported, with bases fitted on the full training group; training residuals
measure fit, not a self-excluded generalization condition.

For Q = I - B^T B, record mean training CE, ||QG||_F^2 and
2*CE*mean_all_training ||Qh_i||^2. The latter includes all training groups,
not just the answer orbit used to choose B. Evaluate that feature energy via
the full training-state SVD, retaining every singular direction, as a stable
sum of nonnegative squares. Compare against direct training residuals on a
fixed 32-row sample. Retain every discrepancy and failed contraction condition.

For each group and split, record state residual energy of Q(h-mu), readout
contrast residual energy of Q(E_c-E_t), and the mean product of their norms
over all query/competitor pairs. Include actual training-state scale and both
absolute and relative residual measures. These quantities describe parts of
the earlier signed bound; they do not replace its signed prediction or its
orthogonality defect with a new classification certificate.

For every adjacent interval use both old and new training bases. Decompose
readout energy change exactly into (i) changing E with Q frozen and (ii)
changing Q at the new E. Decompose state energy similarly into changing the
normalized displacement h-mu with Q frozen, then changing Q. Also check the
vector identity u_new = u_old + Q_old*(w_new-w_old) + (Q_new-Q_old)*w_new.
These are retrospective finite checkpoint differences, not individual updates
or causal effects of weight decay. Report all interval signs together with CE
changes; never equate CE descent with residual contraction.

Keep the positive fixed-feature Euclidean-GD/decay theorem separate from the
adaptive first-step counterexample and the actual tied-model observations.
The native tests check the CE gradient against central finite differences,
an analytic nullspace-decay witness, a first-Adam-step span-escape witness,
and moving-basis decomposition. Audit tolerance is 1e-8 times (1+relevant
energy scale); FP64 checks do not certify arithmetic.

Keep generated reports and logs local under the new AGENTS.md rule. Commit
source, this protocol and authored findings. Preserve reports and verification
evidence outside the task worktree before removing it; document regeneration.
