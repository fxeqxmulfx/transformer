# Main-model AdamW/WSD transfer

Task #200, frozen 2026-11-09 before candidate training. The user explicitly
requests AdamW with WSD from `/home/misha/python_projects/gpt`.

## Source and permitted adaptation

Use the selected main model's `research/main_model/model.json`, at repository
HEAD `6bb9d03a7184cb0a43bfaf240977639213e2cbeb`, and these actual native sources:

| File below `research/main_model/` | SHA256 |
| --- | --- |
| `model.json` | `0af933c3b6ec3e322714cd9654a6ab70b0d6bca193454a8b8dd165635d605e08` |
| `phase_score_native/src/optimizer.rs` | `ceb430d1d5eff338e8826266f20c01870678e7655f0a99f7c62fbeb84b74974b` |
| `phase_score_native/src/optimizer_kernels.rs` | `a5acfb922036ca76a555793d83a225a045ff9911e61ee927b96a19e6e2143258` |
| `wsd_suite_native/src/schedule.rs` | `5853b9f7255428ef8fa7d11b7e98c329dc0129e0990c47430a77eeaef5c22bd4` |

Copy the current FP32 optimizer kernels unchanged, rather than the historical
FP64 norm kernels. Contract tied-matrix gradients before their independent
coordinate L2 norm; clip at one; use separate real/imaginary variances,
betas (0.9,0.95), epsilon 1e-8, weight decay 0.1 and peak LR 0.001. Preserve
the exact FP32 operation order, bias corrections and repaired parameter/moment
ties. The destination model has no learned-frequency parameters, so there
is only the base group; no unused frequency group is invented.

Use the existing 10000-update task budget. Extend only the source schedule's
budget allowlist to admit 10000; its formula and zero-based completed-update
convention stay identical. Warmup is 200 updates, stable until completed step
8000, then cosine decay to 1e-8 at completed step 9999. The first scheduled
learning rate is zero, while moment buffers still update. No off-by-one repair
or retuning is permitted after outcomes.

## Fixed comparison

Compare with task #199's completed constant-rate cohort at `93b64747`.
Use all seeds [7,11,1,3,5,13,17,19], in that order, all 10000 updates and the
same data, model, initialization and sampler. Use the same 31 geometry probes,
21 late probes, ranks [0,32,33,128], training routes and numerical allowances.
No early success stopping, replacement seeds, training search or changed
architecture. A fixed 1800-second cohort cap counts every candidate slot;
retain failures, capped and unrun slots. Pin the baseline report and source
files before training; require all initial states and dataset/order hashes
to match the paired baseline.

This tests the full requested recipe. Beta2, weight decay, clipping, warmup
and terminal decay all change together; an improvement cannot be attributed
to WSD alone. Report every endpoint and late minimum, changes versus its
paired baseline, first 50/90/99 crossings, and class-energy/share trajectories.

## Checks and evidence

Before training, test schedule boundaries and scalar FP32/FP64 update
references, contraction-before-clipping, separate variances and exact ties.
Check actual GPU steps against the FP32 reference at every geometry probe
with a next update; parameter/moment comparison tolerance is 4 f32 ulps,
with exact ties and zero-LR first-step parameters. The global norm must agree
with independent FP64 summation within 2e-6 relative to max(1,norm).
Keep ordinary backward versus instrumented-gradient step comparisons byte
exact. Keep source/checkpoint provenance; candidate trajectories are expected
to diverge from baseline after initialization.

Retain all embedding states and signed energy contributions. Apply the actual
positive clipping scalar to both partial readout and full tied gradients in
the decomposition, with beta2=.95 and decay=.1. Damping of the unused partial
gradient still has the CG1 sign. Keep all adaptive, encoder, memory, rounding
and finite quadratic terms, checking all prefixes with #199's allowances.
Save actual learning rates and clipping norms/scales on every update.

Use an independent verifier for energies, every prefix and FP64 query margins;
require zero false certificates. Preserve all numerical failures and explain
them without outcome-tuned allowances. Run native tests, strict clippy and
formatting. Commit sources, pinned reproducibility inputs, commands and
authored findings; generated evidence stays local under root runs and
`.kanban/artifacts/200/`. Existing Lean CG1–CG6 identities apply with these
explicit scales; no new convergence claim is assumed.
