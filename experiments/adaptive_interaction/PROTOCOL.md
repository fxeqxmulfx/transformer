# Fixed adaptive-forcing and interaction diagnostic

Frozen 2026-11-08 for task #185 before measurement. No training, optimizer
update, model/checkpoint modification, rank selection or new cohort.
Retain all 24 original states (four arms, seeds 7/11, steps 0/5000/10000),
all 16 adjacent intervals, sum/orbit groups, requested ranks 16/32/64
(relative singular cutoff 1e-10), and complete train/validation splits.
Reuse task #184's full-training-group PCA and actual normalized states.
Training residuals measure fit; bases never use validation targets or logits.

Read parameters and both Adam moment buffers from each original state.bin.
Verify state/history/split hashes, case identity, source correctness counts,
scalar-oracle samples and final RMS readout identity. Check the original
protocol/source identity as well: coordinate AdamW, betas .9/.98, epsilon
1e-8, decay 1, learning rate .001 with ten-update warmup. Source equations:
rotate90_complex engine.rs::update, product.rs::moment, ops.rs::adamw,
storage.rs::state_bytes/decode at source hash
14cff3b553d3afb1f9bf90bb6b50cfd7a879058ec2876f859a198e07a09a83da.

At each positive saved step t, form M=m_t/(1-beta1^t),
D=1/(sqrt(v_t)/sqrt(1-beta2^t)+epsilon), coordinatewise on embedding rows.
Evaluate these equations in FP64 on saved FP32 buffers, retaining epsilon;
this is not a bitwise replay of the original FP32 CUDA update.
Step zero has zero moments and no bias-corrected historical direction:
retain its geometry and explicit not-applicable status, never divide by zero.

G is the actual full-training partial final-readout CE derivative at the
saved weights with normalized features held fixed, as in RD1. M contains
earlier minibatch, encoder and momentum effects, evaluated at other parameters.
M-G is only their combined discrepancy; do not label it solely momentum,
encoder gradient, current full gradient, or a prediction of the next update.

For each training residual map Q=I-B^T B, check the exact decomposition

    Q D M = D Q G + (Q D G - D Q G) + Q D (M-G).

Report energies of all three terms and the actual projected saved direction,
their inner-product cross terms/cancellation and the energy upper bound
3*(term1_energy+term2_energy+term3_energy). With dmax=max_abs(D), also check
term1_energy <= dmax^2*2*training_CE*mean_all_training ||Qh||^2,
and the resulting three-term objective bound. Use the full, untruncated
training SVD moment from #184 for the all-training feature energy.
Also retain QDM=DQM+(QDM-DQM) to expose adaptive span mixing of the saved M.
No scalar normalization or diagonal commutation is assumed.

For every group/split, record the signed residual interaction J=<u,r>,
u=Q(E_c-E_t), r=Q(h-mu), for all query/wrong-class pairs: mean absolute J,
mean squared J, signed mean, and mean ||u||*||r||. Retain the numerical
projection defect <E_c-E_t,r>-J; never equate J alone with the previous
signed certificate or certified arithmetic.

For each adjacent interval, with du=u1-u0 and dr=r1-r0, check

    J1-J0 = <du,r0> + <u0,dr> + <du,dr>.

Record each signed and absolute contribution, total interaction change,
and the Cauchy bound ||du||||r0||+||u0||||dr||+||du||||dr|| on its absolute
change. Retain all contraction/growth signs with training CE changes.
These are retrospective 5000-step differences, not one Adam update.
Old/new residuals include both parameter/feature and basis movement.

Use four OpenBLAS threads and a five-minute analysis cap, excluding compile.
Audit tolerance is 1e-8*(1+relevant magnitude); retain discrepancies, failed
contraction and loose bounds. Test stored-moment normalization against an
independent scalar Adam update, mixed-axis commutator escape, isotropic
commutation, actual CE finite differences, and moving-interaction cancellation.
Reports/logs/summaries remain ignored and local; commit source, this protocol,
the summary generator and authored findings. Preserve evidence outside the
temporary worktree before removal. INDEX.md remains generated and committed.
