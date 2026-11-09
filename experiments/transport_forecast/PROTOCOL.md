# Signed training geometry across an actual update

Task #197, frozen 2026-11-09 before candidate measurements. Connect the
flat-spectrum development's signed training transport (ST1–ST3) to the
Gram/RMS and directional forecasts (FT1–FT3). This is a conditional bridge
for the actual plain complex-product source on `(a+b)^3 mod 97`. It does not
assume nuclear-norm optimality or explain the original `a^3+ab` task.

## Fixed cohort and replay

Use the original plain-product arm, seeds 7 and 11, source hash
`14cff3b553d3afb1f9bf90bb6b50cfd7a879058ec2876f859a198e07a09a83da`,
Kernite revision `a1b6bc2c956e1a5f35960e26b209e1b6c213096c`, original PTX,
data, sampler, full CE backpropagation and coordinate AdamW. Replay each
original seed from its saved initial state to step 10000. Check complete
parameter/moment hashes against the original states at 0, 5000 and 10000.
Do not retune, select a seed, or stop when quality is reached.

Probe the next actual source update at steps
`0,100,250,500,750,1000,1500,2000,3000,4000,5000,7500,10000`.
The step-10000 probe is outside the original training budget. Keep all 26
cases and all 9409 queries, with the original train/validation split. Compare
available replayed metrics to the saved 100-step history. Cap the replay and
diagnostic at 1200 seconds excluding compilation. Retain failures and capped
slots; do not extend the cap after results.

## Information boundary and geometry

For each old state, construct answer-orbit groups and transfer candidate
labels from observed training labels using the existing checked cubic routes.
The constructor accepts operand pairs, old FP64 features, embeddings, and
training index/label pairs. It receives no held-out target labels, future
features, future logits, or future errors. Missing references abstain.
Training queries use the opposite alternating reference fold; validation
queries use all training references in their orbit. Retain actual residuals
and orthogonality defects instead of assuming an exact computed SVD.

Fit centered training-state bases at requested ranks `0,32,128`, clamped to
singular values above `1e-10*s_max`. Rank 32 is the primary compressed bound;
rank 0 is the center baseline, and rank 128 is a reconstruction limit, not
evidence of low-rank discovery. Report achieved ranks and retained energies.
Use the exact source's normalized old features for geometry; bound and audit
the FP32 discrepancy separately using the unchanged task #196 allowances.

For competitor-minus-target readout `w`, old query feature `x`, training
prediction `P` and directional residual bound `R`, ST gives
`<w,x> <= P+R`. Keep the old geometry fixed across this single update. With
known rounded readout `w'`, feature tangent `v`, and a forecast radius `r`,

```
signed_change = <w'-w,x> + <w',v>
new_gap <= P + R + signed_change + ||w'||*r + execution_allowance.
```

This keeps the mixed readout/tangent term. The combined FT1/FT2 radius is
used. Also transport the FT3 intersected interval: if `U` bounds the new
gap, its valid drift upper bound is `U-<w,x>`, so the directional bridge
is `P+R+U-<w,x>`. Report this bridge separately from `U`; do not replace
the primary result by their minimum. Training geometry can add slack, and
the direct forecast is a distinct comparator, not a generalization proof.

Freeze every candidate label and every classwise old and new upper bound
to a hashed binary before any updated-network forward evaluation. Preserve
the original `1e-3` feature, `1e-3` per-logit and scaled `1e-8` strictness/
enclosure allowances. No post-outcome allowance adjustment is permitted.
Prove the signed increment, training-to-forecast composition, class margins,
probability/CE consequences and actual-source adapter in Lean, with explicit
witnesses and no new sorry. Execution allowances remain audited assumptions.

## Outcomes and temporal interpretation

Audit every class upper bound against independent updated FP64 and actual
FP32 source evaluations. Require zero false certificates and no allowance
violations; check old source geometry, old tangents, parameter/moment replay
and all inputs. Record old/new accuracy and CE, old geometry coverage, both
bridge coverages, direct forecast coverage, and query probability floors and
CE ceilings on both complete splits. Retain per-query abstentions, errors
and inconclusive conditions. Charge replay, geometry, forecast, persistence
and audit time separately; this is not an inference speed benchmark.

Use the fixed 50%, 90% and 99% levels for validation accuracy and certified
coverage. Report first crossings only at the declared probe resolution,
including subsequent regressions. A sound correctness certificate cannot
precede the actual correctness it certifies. Test instead whether informative
coverage and positive margin bounds already occur before the trajectory's
first sampled 99% accuracy. A one-update forecast is not a prediction of
eventual convergence or a proof that the optimizer discovers the geometry.
The same original seeds and previously examined split provide a temporal
diagnostic, not fresh-seed confirmation.

Use native Rust/Kernite only. Keep source, this protocol, authored findings
and reproducibility commands committed; generate INDEX with Lean changes.
Keep reports, binaries, logs and generated summaries ignored under the root
experiment's `runs/` and `.kanban/artifacts/197/` before removing the worktree.
