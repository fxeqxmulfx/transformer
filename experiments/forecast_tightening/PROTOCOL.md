# Frozen forecast-tightening comparison

Task #196, 2026-11-09. Freeze before candidate numerical outcomes. Extend
the forecast theory and diagnose inflated bounds; retain ordinary CE/full
backpropagation and source AdamW. No training or optimizer search. Task #189
remains the separate inference-efficiency comparison.

## Controls and information boundary

Use the pinned trainer/PTX/Kernite and all 24 original states: four arms,
seeds 7/11, steps 0/5000/10000. Check original data, sampler, parameters and
moments, replay saved metrics, and require the same next-update state hashes
as task #188. Treat step 10000 as a post-budget probe. Keep all 9409 queries,
both splits, and each split's old-margin bottom decile. Repeated observations
are not independent examples. Hash every input before/after the cohort.

Compute only old values, their directional tangents along the proposed
rounded parameter step, and finite error bounds. Freeze and persist final
centers, radii and outcome intervals before any forward evaluation of the
updated network. Nonlinear nominal values may calculate defects and final
scalar denominators, but must not become the next hidden layer's feature.
Keep all mixed product terms and the source complex 1/sqrt(2) factor.

## Predeclared candidates and ablations

1. **Legacy:** exactly task #188's Frobenius, 2/s RMS and final-feature-ball
   bounds, with the same old tangent center.
2. **Gram only:** replace each linear amplification with the smaller of its
   Frobenius upper bound and sqrt(||M^T M||_F), or sqrt(||M M^T||_F), choosing
   the smaller Gram dimension. Preserve the mixed vector contribution's norm.
   Compute Gram dot products and their Frobenius sum with outward-rounded
   FP64 intervals; no uncertified power iteration or fitted spectral factor.
3. **RMS only:** for nominal x and input radius r, regularized length
   s=sqrt(||x||^2+n*epsilon), use min(2*sqrt(n)/s,
   sqrt(n)/sqrt(s*(s-r))) if r<s. Otherwise retain the legacy bound.
   The new bound follows from the regularized normalization secant identity;
   retain the same nominal nonlinear defect.
4. **Combined:** apply both preceding replacements. Sigmoid, radial gate,
   products and residual rules remain the source forecast rules.
5. **Directional readout:** use the combined ball before the last RMS.
   Bound each readout-contrast numerator directly, then divide its interval
   by the positive RMS-denominator interval. This preserves the known sign
   and direction of each class contrast without transporting a whole feature
   ball through final normalization. Intersect with the combined final-feature
   interval; audit both component intervals as well as their intersection.

All methods retain the same fixed 1e-3 Euclidean executed-feature allowance
and 1e-3 executed-logit allowance per class. For the direct readout, add
||class contrast||*1e-3+2e-3 after real-arithmetic normalization. Keep task
#188's 1e-8 scaled FP64 enclosure/strictness tolerance. Do not fit or widen
allowances after outcomes. These remain audited execution assumptions;
formal results do not certify the complete Rust/BLAS/CUDA implementation.

## Diagnosis and audit

Record per-layer actual versus bounded error after the freeze, each method's
radius distribution, injected nominal/mixed defects and amplification factors.
This diagnoses where scalar bounds inflate; do not describe it as a complete
direction-preserving error representation through every layer.

Prove the new real-arithmetic normalization, Gram and signed quotient/readout
bounds in Lean with explicit nonzero witnesses and no new sorry. Native tests
include orthogonal and rank-one matrices, rectangular/transposed layouts,
near-zero normalization and the fallback boundary, signed/zero numerator
intervals, cancellation and all source arms with nonzero parameter steps.
Independently check old tangents with the existing scalar finite differences.

Audit every feature ball, class-gap interval, margin and CE interval against
updated FP64 and source FP32 outcomes. Require zero false certificates and
zero numerical-allowance violations. Preserve every failure and inconclusive
case. Report full, trained-state and difficult-subset coverage separately,
including all 24 states and all loss-change directions. Compare legacy
coverage with 68040/75280 trained validation observations (90.38%).

## Cost and interpretation

Use Rust/Kernite/native tools only. Cap the cohort at 900 seconds excluding
compilation, checking before each case. Keep all slots, caps and failures.
Time bound preparation (including Gram construction and readout interval
work), source update, persistence and independent audits separately. Report
actual forecast/storage payload. This is a certificate-quality experiment;
it does not establish an inference speedup or use validation to change training.
Higher coverage is a measured outcome, not a prerequisite for retaining a run.
Do not extend the cohort or select favorable cases. A fresh-seed confirmation
or inference benchmark requires its own declared protocol after these results.

Preserve ignored reports, frozen binaries and logs in root runs/artifacts.
Commit source, protocol, generators, authored findings and generated INDEX.md
when Lean sources change; do not commit regenerable experiment outputs.
