# Early settling and active-answer reconstruction

Task #202, frozen 2026-11-09 before candidate training or active-basis
measurements. The user asks to close the gap after task #201. Its complete
eight-seed results are known: peak .01 slightly improves final accuracy but
rank-32 coverage stays poor until cooldown. Two interventions are evaluated
separately, with no selection from their outcomes.

## Training intervention

Keep the task #200 main-model FP32 AdamW kernels, beta1=.9, beta2=.95,
epsilon=1e-8, weight decay=.1 and independent-coordinate clipping at 1.
Keep the plain complex-product model, data, initialization, sampler, seeds
[7,11,1,3,5,13,17,19], 10000 updates and all 31 geometry probes. Only the
learning-rate schedule changes from task #201 peak .01:

- completed updates 0 through 199: linear warmup to .01 over 200 updates;
- 200 through 1000: hold .01;
- 1000 through 4000: cosine reduction from .01 to .0001;
- 4000 through 8000: hold .0001, with genuine nonzero training updates;
- 8000 through 9999: cosine reduction from .0001 to 1e-8.

The common endpoints of adjacent intervals agree. Use the existing zero-based
completed-update convention. A new mutually exclusive Cargo feature `settle`
selects this exact schedule. Preserve both completed .001 and .01 cohorts.
Run all eight new seeds sequentially under one fixed 1800-second cap; retain
failed, capped and unrun slots. No replacement, extra training or outcome
tuning. Compare every paired endpoint, 21 late probes (5000–10000), all
50/90/99 crossings, and regressions against both existing cohorts. Primary
training outcomes are final accuracy, final rank-32 coverage and minimum
late rank-32 coverage under the unchanged all-class certificate.

Extend the GPU preflight to each new schedule boundary, while keeping the
4-ulp parameter/moment allowance and 2e-6 norm allowance. Retain all-update
energy accounting, paired initialization hashes, independent FP64 margins,
and all original reconstruction and execution allowances.

## Certificate intervention

Evaluate one additional fixed basis on all 31 saved states of all eight
seeds in each complete cohort: .001, .01 and early settling. Derive the
active class set solely from observed training labels. Center those readout
rows by their own mean, project them into the same centered training-feature
span, and take the same SVD ranks [0,32,33,128]. Keep the original cubic
routes and opposite training folds; query labels, winners and margins cannot
choose the basis or coefficients. An active basis may achieve fewer than
the requested directions; report that rank without relabeling it as a
full all-class span.

Reconstruct every one of the 97 class contrasts with the existing signed
coefficient, orthogonal residual and defect formula. Inactive rival classes
remain in the maximum margin bound: they are never dropped or assigned zero
residual. Keep feature/logit allowances 1e-3/1e-3 and the original strict
positive-margin certification threshold. Freeze bounds before outcome scoring.
The existing arbitrary-basis reconstruction bound applies; this is an
instantiation, not a new convergence theorem or an efficiency claim.

For each state regenerate the original all-class readout margins and require
agreement with the saved reference before interpreting the added method.
Independently evaluate the model in FP64 and check every new margin against
both FP64 and recorded GPU margins; require zero false certificates. Retain
all cases under a fixed 1800-second postprocessing cap, in cohort order .001,
.01, early settling and the original seed/probe order. Native regression
cases must cover distracting inactive directions and an inactive winning
rival. Report counts of incorrect and correct-but-uncertified predictions,
all paired coverage changes and persistence under both bases. Do not change
training based on postprocessing outcomes.

Commit source, this protocol, authored interpretation and exact commands.
Preserve generated runs, bounds, hashes, logs and reports locally under root
`experiments/geometry_gap/runs/` and `.kanban/artifacts/202/`. No Python,
external GPT edits, new axioms, or claims of universal discovery/persistence.
