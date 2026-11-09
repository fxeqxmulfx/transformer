# Premeasurement extension after task #196

The user identifies trainer commit
`b70811e0e0e7ebdc4a8a2e48044c180332f7c40d` in the `grokking` repository.
Its 15 build-pinned source/configuration files hash to
`14cff3b553d3afb1f9bf90bb6b50cfd7a879058ec2876f859a198e07a09a83da`,
exactly the existing source pin. This records the commit identity; no source,
checkpoint, kernel, bound or measurement rule changes.

2026-11-09, task #189. The original PROTOCOL.md remains the three-method
control. No inference timing or reuse-coverage outcomes have been measured.
The user's next direction after #196 is to reduce forecast cost and test
actual inference savings. Add one fourth method before measurements:
**tight reuse**, using the same old executed feature center as original reuse,
with #196's Gram linear amplification and regularized local RMS secant bound.
This is an inexpensive displacement bound; it computes no tangent GEMMs.

For each RMS with old denominator s, let S=sqrt(128)*s. If r<S,
replace its old 2/s coefficient with
min(2/s, sqrt(128)/sqrt(S*(S-r))); otherwise keep 2/s. Retain the
2*sqrt(128) clipping. For each updated dense matrix use the smaller of the
outward FP64 Frobenius bound and the outward smaller-Gram bound implemented
in forecast_tightening/src/matrix.rs at 60483223. Keep the original Frobenius
norm of D multiplying the old-input norm. All product, gate, residual and
execution allowances remain exactly those of PROTOCOL.md. This fourth
method is not the tangent-based 99.27% correctness certificate from #196;
its label-free agreement coverage is unknown before this cohort.

Preserve all original 24 cases, queries, batch sizes, three warmup/nine measured
rounds and the 900-second cap. Rotate all four methods by round number,
replacing the original three-method rotation only. Check the cap between
cases and rounds; retain incomplete cases and every measured duration.
Keep all partial outcomes if capped; do not extend the budget. Common old
feature/node-stat construction is charged in full to either reuse method.
Time legacy bound preparation and tight bound preparation independently,
including each method's parameter conversion, norms and contrast preparation.
The tight Gram work is charged only to tight reuse. Each uses the same new
weight cache, also charged to its cold setup. No future hidden features enter
either bound. Persist radii, contrast norms and the old-feature digest before
any updated-network forward or approximate-logit evaluation.

Separate CUDA module loading from model-buffer creation by supplying the
already loaded original modules to an allocation-identical constructor.
The original forward, backward and AdamW equations and PTX are unchanged.
The constructor is checked against ordinary saved-state source metrics.
All cache tables and serving input types contain operands, never targets.
Labels enter only independent audit/accuracy reporting. Reject invalid input,
nonfinite logits, and weight/feature-bound caches from another version.

Report both reuse variants' all-query and timing-query agreement coverage,
wrong approximate decisions, fallback counts and final exact agreement.
Audit all pairwise contrast enclosures for each reused feature center, not
only accepted winners. For a model version, frozen bound preparation is once;
prediction acceptance and exact fallback are performed on every request.
No answer memoization, threshold fitting, extra methods or timing selection.
GPU process snapshots include graphics processes; retain contention evidence.
Report descriptive paired times with that limitation, without claiming a
general speedup or exclusive-device measurement.

Cold latency = model allocation/upload + candidate cache preparation + one
median 512-query round. Report additional setup relative to source separately
for break-even: ceil(additional_setup/(source_median-candidate_median)) only
when savings are positive, and also versus cached weights. Cache setup is
never silently amortized away. Shared baseline training buffers remain and
are counted explicitly; cache CPU payload/capacity and GPU deltas are separate
from process/driver peak memory. Hash all inputs and preserve all evidence
under root runs/artifacts. Native source/tests and authored interpretation
are committed; generated reports stay local.
