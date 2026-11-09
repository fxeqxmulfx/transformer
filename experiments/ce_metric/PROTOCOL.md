# Checkpoint-fixed feature metric for CE momentum control

Task #207 continues #206. Freeze before calculating the new diagnostics.
No training change, new seed selection, forecast coverage claim, or Python.

## Proof target

Replace a scalar isotropic reference by positive feature weights r_k shared
by every unused class. Derive from the actual partial CE derivative
`sum_(u,k) r_k*g_U(u,k)^2 <= L_r*<E_U,g_U>`, with
`L_r=clip*q*mean_i sum_k r_k*T(i,k)^2`. The computed unused class means
and the same nonnegative radial product remain. For arbitrary saved m,
`m_R=.9*m+.1*g`, `w_R=w-s*r*m_R`, use

```
V_(r,s)(w,m)=sum_(u,k) ((w-9*s*r*m)^2/r+90*s^2*r*m^2),
delta V=-2*s*<w,g>+19*s^2*sum r*g^2-(171/10)*s^2*sum r*(m-g)^2.
```

Prove nonincrease when s>=0 and 19*s*L_r<=2, with no saved-memory
alignment assumption. Prove weighted and ordinary parameter-energy bounds,
retaining the largest-weight penalty in the latter. Prove a fixed-metric,
fixed-step sequence version and exact signed full-update perturbation
accounting. Neither metric variation nor class-dependent scale differences
may disappear into a definition or an assumed favorable gradient.

## Frozen inventory and weights

Use all 144 existing #206 cases: cohorts lr001/lr01/settle; seeds
[7,11,1,3,5,13,17,19]; anchors [0,1000,4000,6000,8000,9500]. Hash and
validate all source reports and arrays. Reuse their independently checked
original next updates; do not replay a different batch or start training.

Construction is a separate process. It reads only original checkpoint
states, training-label support and the known schedule. Before the verifier
opens #206 reports or next-state arrays, freeze all 144 metric files and
their hashes. For saved time t>0, define
`D_old(c,k)=1/(sqrt(v_old(c,k))/sqrt(1-.95^t)+1e-8)`; at t=0 use 1e8.
Let alpha be the minimum D_old over the whole readout. The feature metric is
`r_k=min_(unused c) D_old(c,k)/alpha`. The isotropic control uses r_k=1.
Both use raw scalar step `s=eta(t)*alpha/(1-.9^(t+1))`. These are entirely
checkpoint quantities, although their eventual full-update errors are not.

Verification checks the frozen source hashes first, then uses the executed
features and independently recomputed FP64 partial CE of the original next
batch. For each metric, retain raw s and the separate diagnostic control
`s_safe=min(s,1/(19*L_r))` when L_r>0. Record current q and L_r. Also record
the prospective conservative condition using q=1, clip<=1 and
`L_global=128*(1+5e-6)*max(r)`; do not call current-q successes forecasts.
The stable control changes the reference and potential only, never training.

## Complete accounting and comparisons

Preserve the real old memory and actual next moments, including all tied
encoder history. For each metric and scalar step, split full-update error
into decay, class-dependent adaptive remainder, tied encoder and execution.
Compute projected diagonal-times-moment before centering. Linear potential
credits add; the quadratic term uses the total perturbation with cross terms.
Record every reference-condition failure and every actual potential increase.
Compare relative potential changes and signed credits divided by the old
potential, as well as raw units; metric changes are not optimizer interventions.
Report max(r) and the corresponding ordinary-energy bound cost. Retain all
144 slots on failure; initialization's 24 zero-LR cases are separate.

No numerical allowance is relaxed: source optimizer checks as in #206,
independent partial-gradient scaled error <=1e-5, feature energy
<=128*(1+5e-6), identities/energy inequalities scaled error <=1e-8. The
fixed construct+verify budget is 600 seconds, excluding compilation/proofs.
The comparison can isolate remaining obstructions but does not establish
full-model future control, discovery, or improved prediction certificates.

## Reproduction and preservation

Use a standalone Rust crate so #206's pinned source remains unchanged.
Run native tests/clippy/fmt, full Lean build, custom/global axiom audits,
native INDEX and forbidden scan. Preserve ignored evidence at root
experiments/ce_metric/runs and .kanban/artifacts/207 before cleanup.
Commit source, this protocol, authored README findings and generated INDEX
with source changes; do not commit generated reports or logs.
