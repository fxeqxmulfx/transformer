# Loss-specific momentum control and its full-model obstruction

Task #206 follows the user's continuation after #205. Freeze this protocol
before measuring the new quantities. The previous checkpoint magnitude cone
adds only three certified hold steps; this study asks whether actual CE
damping can absorb momentum without assuming its discrepancy is small.

## Mathematics

For logits z, prove
`(p_c-p_d)^2 <= q*(z_c-z_d)*(p_c-p_d)` when both probabilities are at most
q. Use the exponential tangent inequality, with every hypothesis explicit.
For unused classes, derive a quantitative bound on the actual partial CE
gradient: `||G_U||^2 <= L*<E_U,G_U>`, where a sufficient L is
`q*mean_i ||T_i||^2`; q bounds every unused probability in the minibatch.
The class projection is the computed within-unused mean subtraction.
A common clipping scalar multiplies L by that scalar. The global q=1
choice permits changing features and minibatches; a smaller measured q is
only a current-state bound unless separately controlled into the future.

For the main beta1=.9 recurrence, define
`m_R=.9*m+.1*g_R`, `w_R=w-s*m_R`, with scalar effective step s, and
`V_s(w,m)=||w-9*s*m||^2+90*s^2*||m||^2`. Derive the exact identity
`V_s(w_R,m_R)-V_s(w,m) =
 -2*s*<w,g_R> + 19*s^2*||g_R||^2
 -(171/10)*s^2*||m-g_R||^2`.
Thus L-cocoercivity and `s>=0,19*s*L<=2` imply nonincrease for arbitrary
initial memory. A fixed-s sequence needs no fixed feature table, provided
its actual CE gradients satisfy a common proved bound. This is a partial
readout dynamics theorem, not the full tied AdamW theorem.

Retain decoupled decay, the coordinate-dependent preconditioner, the tied
encoder gradient, and finite execution residuals as explicit perturbations.
Derive exact potential change from those perturbations, retaining the
square of their total, including cross terms. Do not claim the full-model
perturbations are bounded in advance merely because they can be measured.

## Fixed diagnostic replay

Use all three complete existing cohorts: .001 main WSD, .01 main WSD, and
early settling. Use all eight seeds [7,11,1,3,5,13,17,19] and exactly the six
saved anchors [0,1000,4000,6000,8000,9500]: 144 next-minibatch cases.
Restore each saved state, rebuild the original data/order, and execute its
one original next update with the unchanged main-model kernels. Never
continue a replay into another training step or select a replacement state.
Check each resulting readout against the preserved embedding trace, and
the optimizer against its existing FP32 reference with 4 ulps / 2e-6 norm
allowances. Preserve source hashes, batch identities, outputs and failures.

Independently compute the partial CE gradient and probabilities in FP64
from the executed normalized features and saved readout. Check against
the GPU partial derivative with scaled tolerance 1e-5. Keep the original
scaled 1e-8 energy/identity tolerance. Check normalized feature energies
against 128 with relative allowance 5e-6. No allowance may be enlarged
after observing results. A scaled discrepancy uses 1+max absolute size.

The primary scalar step is s=eta*min(D)/(1-.9^(t+1)), with D obtained
from the actual updated second moment, including epsilon and bias
correction. Also report a separate analytic control
s_safe=min(s,1/(19*L)); if L=0 retain s. This control changes only the
diagnostic reference, not training. Record raw/effective s, q, L, the
stability condition, exact baseline potential change, and actual potential
change for both. The current q and D are not future forecasts.

Use the actual old unused-projected first moment, including its entire
encoder/minibatch history. Split the full update's perturbation of the
isotropic partial-CE reference into decay, adaptive scaling, tied encoder,
and execution residual. Attribute the potential's linear terms to those
four sources; retain one quadratic term of the total perturbation. Report
every signed term, normalized identity error, every baseline-condition
failure, and every actual potential increase. Do not treat the sum of
separate squared contributions as the square of their sum.

Each cohort has a 600-second replay/audit cap, excluding compilation and
Lean work. Retain every declared slot on failure or cap. No new predictive
coverage, final-test selection, training changes or external GPT edits.
The study diagnoses which full-model terms still need future control;
it cannot establish reliable discovery from finite replay.

## Checks and preservation

Use Rust/Kernite, Lean and native tooling only; no Python or subagents.
Run native tests/clippy/fmt, full Lean build, theorem/global axiom audits,
native INDEX and forbidden scan. Preserve generated evidence at the root's
ignored `experiments/ce_lyapunov/runs/` and `.kanban/artifacts/206/`.
Commit source, this frozen protocol, authored findings and reproduction
commands; commit generated INDEX with every src change.
