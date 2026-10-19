# PyTorch / Lean recurrence audit

Audited against the Transformer tree at `470f9b6b` (2026-10-19). The
implementation kernels were extracted from `python/src/lab/infrastructure/optim`
at that commit. The public API removes experiment-specification and GPTMini
dependencies.

This audit compares definitions, not theorem titles. Lean models exact real
arithmetic. Numerical recurrence tests, CUDA checks and historical trajectory
tests do not establish a formal compiler, PyTorch or floating-point refinement.
Convergence results retain their objective, gradient, radius and step-size
hypotheses. An optimizer does not acquire those hypotheses by matching a step.

Coordinate rules skip parameters whose gradient is None, as PyTorch does;
their clocks count gradient-bearing updates for that parameter. The paper's
simultaneous vector recurrence assumes every coordinate has an oracle value
on every update. Missing gradients are not silently identified with zero
gradients in the correspondence. Dash, AMSGradMD, Schedule-Free and the
optimistic method require all participating parameters' gradients.

## Update correspondence

`standard` is an alias of the installed `torch.optim` module: its optimizers
are the very same class objects and kernels. This adds no new Lean coverage
for arbitrary native options or optimizers. The table below concerns the
package's paper recurrences and extensions; PaperAdam/PaperAMSGrad/PaperAdaGrad
are explicit aliases for its raw-moment Adam/AMSGrad/mean-square AdaGrad.

All Lean names below start with `Transformer.`. Paths are relative to `src/`.
The PyTorch implementation is the module in this package.

| Export / module | Lean definition and source file | Correspondence and limits |
|---|---|---|
| SGD / `direction.py` | `AdamBeyond.sgdRule`, `AdamBeyond/Section2_Adam.lean` | Identity metric, zero first-moment coefficient. Extra weight decay is an explicit unprojected extension; for plain SGD it gives x - lr (g + wd x). |
| Adam / `coordinate.py` | `AdamBeyond.adamRule`; `AMSGrad.Setup.step`, `AMSGrad/Section1_AMSGrad.lean` | Raw first and second moments, **no bias correction**. Identity projection. Epsilon after sqrt(v) is a numerical extension of the online rule. |
| AMSGrad / `coordinate.py` | `AMSGrad.amsgradRule`; `AMSGrad.trainingStep`, `AMSGrad/Section4_TrainingModels.lean` | Update m and v first, then elementwise max of the raw v history. Zero buffers; epsilon after sqrt(maximum); no projection. The constant-coefficient training model includes epsilon. |
| AMSGradW / `coordinate.py` | `AMSGradW.trainingStep`, `AMSGradW/Basic.lean` | The old parameter receives independent decay, absent from moments. Constant coefficients match the training model directly. With decays, beta1_t = beta1 c_t and eta_t = lr c_t; eta_t multiplies **both** adaptive displacement and weight decay. |
| AdamX / `coordinate.py` | `AMSGrad.adamXRule`, `AMSGrad/Section5_AdamX.lean` | max(((1-b1_t)/(1-b1_previous))^2 old_max, v). First maximum equals v; multiplying the initial zero maximum is equivalent. Same epsilon/unprojected extension as Adam. |
| AdaGrad / `coordinate.py` | `AdamBeyond.adagradRule`, `AdamBeyond/Section2_Adam.lean` | v_t = ((t-1)v_previous+g^2)/t. Default lr/sqrt(t). Positive epsilon is added to sqrt(mean), so this is not native torch Adagrad's epsilon convention. |
| AdamNC / `coordinate.py` | `AdamBeyond.adamNCRule`, `AdamBeyond/Section5_AdamNC.lean` | beta2_t = 1-1/t, raw first-moment EMA and the same mean squared gradient. The implemented second-moment schedule is this specialization, not every schedule accepted by Lean. |
| RMSProp / `coordinate.py` | `AdamBeyond.adamRule` with beta1=0 | v_t EMA and g/(sqrt(v_t)+eps). There is no separately named complete RMSProp convergence theorem in this tree. |
| AdamW / native torch | `Grokking.AdamW.scalarNativeStep`, `Grokking/AdamW/ScalarRecurrence.lean` | New raw buffers, completed-update clock + 1, both bias corrections, independent decay of the old parameter. Always-updated real coordinates correspond; skipped gradients keep their individual PyTorch clocks. Native fused/foreach rounding is not modeled. |
| AdamWDirection / `coordinate.py` | The same `scalarNativeStep` | Algebraically the same real update, reordered into a direction for stages. Floating-point order can differ from native AdamW. Matrix-only decay is a per-coordinate wd=0 selection. |
| Muon / `matrix.py` | `Muon.muonStep`, `Muon/Section2_Models.lean` | M_new=mu M+g, Nesterov input mu M_new+g, Frobenius normalization, five iterations with (3.4445,-4.775,2.0315), scale 0.2 sqrt(max(rows,cols)), wd=0. Tall-matrix transposition commutes with this odd polynomial. Finite iteration is not the exact polar factor. Auxiliary bias-corrected Adam and matrix selection are benchmark choices. |
| Dash / `matrix.py` | `DASH.leftEma`, `rightEma`, `preconditionedGradient`, `graft`, `DASH/Section2_Models.lean` | Left/right raw EMAs before preconditioning. Frobenius grafting to raw-Adam direction, zero at zero. Blocks and shape buckets are storage choices; vectors are columns. Every parameter must have a gradient. Epsilon regularizes both histories and the graft denominator. |
| AdaFisher / `fisher.py` | `AdaFisher.factorEMA`, `correctedMomentum`, `parameterStep`, `AdaFisher/Section3_Algorithm.lean`; `minMaxDiagonal`, `Section3_MinMax.lean`; `fisherDiagonal`, `Section3_Efficient.lean` | gamma weights fresh factors, constant diagonals normalize to zero, damping is added after the factor product, raw m is corrected at the positive clock. Hooks sum squared inputs and derivatives over batch/positions. Bias-free Linear weights get those factors; other parameters get ones. Hook collection and the identity-factor fallback are benchmark choices, not a proof of Fisher approximation accuracy for arbitrary networks. |
| AMSGradMD / `magnitude.py` | `MagnitudeDirection.amsgradMDProposal`, `MagnitudeDirection/SectionA_AMSGradCallbacks.lean`; `statefulFullProposal`, `SectionA_StatefulAlgorithm.lean` | All three chain-rule gradients use old factors. Independent raw AMSGrad histories update direction and raw row/column gains; project direction to its initial Frobenius sphere, then fuse softplus gains. Unit initial gains. Requires finite nonzero selected matrices. Auxiliary AMSGrad and selected matrix membership are benchmark choices. |
| Guarded / `stages.py` | `Optimization.descentGuard`, `Optimization/Basic.lean` | Check alignment >= sigma ||g||^2 and ||d||^2 <= ||g||^2 jointly over parameters, otherwise use g. Squared norm comparison is equivalent for real nonnegative norms. NaN/overflow behavior has no real counterpart. |
| guarded AMSGradMD / `magnitude.py` | `MagnitudeDirection.checkedProposal`, `SectionA_TrainingModels.lean`; `rebalanceStorage`, `SectionA_TrainingStorage.lean` | Retain proposed optimizer memories on rejection, use true-gradient fallback, halve a matrix step if it would zero storage, repair raw row gains without altering fused weights. Python checks one joint vector, finiteness and each selected matrix's nonzero storage. The API enforces sigma <= 1/2, required by the halved fallback's certificate. Lean's full storage theorem is for a single matrix; it does not certify a general multi-matrix composition or floating-point storage repair. |
| Magma / `stages.py` | `Magma.cosine`, `damping`, `algorithmDisplacement`, `Magma/Section3_Algorithm.lean` | Zero cosine at zero, old damping scalar retained, EMA 0.9/0.1, sigmoid(cos/tau), unnormalized survival mask. Initial damping 0.5 and an auxiliary EMA when the base has none are benchmark choices. Cosine clamp handles roundoff. The fixed seeded mask plan repeats after its configured budget; the normalized 1/p masking analysis does not describe this algorithm. |
| Clipped / `api.py` | `RecurrentGradients.clip`, `RecurrentGradients/Section3_Clipping.lean` | Global gradient clipping happens before moment updates and every stage. PyTorch uses min(1,norm/(||g||+1e-6)); Lean's source rule has no 1e-6. The lab performs clipping in its stepper rather than applying the public wrapper twice. |
| AtomicColumns / `atomic.py` | `GPTMini.Sparsemax.matchingMixture_criterion_convex`, `matchingMixture_gap_certificate`, `matchingHeadPrice_minimum_exists`, `GPTMini/Sparsemax/AtomicMatching*.lean`; paired equivalents in `PairedMatching*.lean` | The active weights live on a simplex over physical heads. Order/binding prices are exact only for the two specified scalar cap-one contexts. Q/K Adam search is local and is never called a global price. Finite capacity, finite corrective iterations, line-search tolerances and numerical QP solves are extensions; the abstract Lean theorem does not verify those solvers. A model/task/head-factory protocol supplies the actual family. |
| ScheduleFreeSGD / `schedule_free.py` | `ScheduleFree.mix`, `query`, `average`, `ScheduleFree/Section2_Averaging.lean`; `schedule_free_sgd`, `Section2_SGD.lean` | Stored base z and evaluation x start at the initial point; x averages z including that initial point. Query y=beta x+(1-beta)z, base moves by -lr g(y). The first new average uses 1/2. The optimal rate theorem additionally specifies lr=D/(G sqrt(T)) and bounded unbiased convex-oracle assumptions. |
| ScheduleFreeAdamW / `schedule_free.py` | `ScheduleFree.adamStep`, `adamRun`, `ScheduleFree/Section4_AdamW.lean` | Zero raw second moment and total; eta=lr sqrt(1-beta2^t) warmup(t); eta^2 weighted average; decay at the **old query** y. Epsilon is added to sqrt(raw_v). Initial total is zero, so the first positive eta gets weight one. No-warmup and zero-rate total-division cases are explicit. No nonconvex convergence theorem is claimed. |
| ScheduleFreeAMSGradW / `schedule_free.py` | `ScheduleFree.AMSGrad.step`, `run`, `ScheduleFree/AMSGrad/Defs.lean`; invariants in `Basic.lean` | Package combination of §4.4 Schedule-Free AdamW and AMSGrad's maximum of raw second moments. The first coefficient controls interpolation; the base proposal uses the gradient directly. Query decay, raw epsilon placement and squared-effective-rate averaging follow ScheduleFreeAdamW. Running maxima are monotone, denominators are positive for positive epsilon and reachable averaging coefficients lie in [0,1]. |
| ScheduleFreeAMSGradMD / `schedule_free_md.py` | `ScheduleFree.AMSGrad.mdStep`, `mdRun`, `mdQueryWeight`, `ScheduleFree/AMSGrad/MagnitudeDirection.lean`; proofs in `MagnitudeInvariants.lean` | Package combination in latent direction/raw-row/raw-column coordinates. The three gradients use the old interpolated factors, and the oracle sees their fused query weight. Project the base direction before averaging; evaluation fuses averaged factors. Each component has independent raw histories, squared-rate totals and rates. Auxiliary tensors follow the W recurrence with zero decay. |
| ClippedOptimistic / `optimistic.py` | `ScheduleFree.optimisticStep`, `optimisticRun`, `ScheduleFree/SectionD_OptimisticBasic.lean`; `optimistic_regret`, `SectionD_Optimistic.lean` | All buffers and parameters start at zero. Clip g-previous using the **old** peak; update the peak and clipped energy; eta=D/sqrt(new_peak^2+new_energy); project auxiliary after the clipped gradient, then project query after g. One joint Euclidean ball over all tensors; safe division reproduces zero prefixes and D=0. Comparator norm <= D is required by the regret theorem. |

## Schedule-Free AMSGrad combinations

These additions combine arXiv:2405.15682v4, Section 4.4, arXiv:1904.03590v4,
Algorithm 1 and Section 6, and for MD arXiv:2606.25971v2, Appendix A,
Algorithm 2. They are local compositions with explicit definitions and
proved recurrence properties. A convergence theorem for the combined
optimizers remains unproved.

For every component, v_new=beta2*v+(1-beta2)*g^2 and
maximum_new=max(maximum,v_new), before the bias-correction factor enters
eta=rate*sqrt(1-beta2^t)*warmup(t). The actual base proposal uses
eta*g/(sqrt(maximum_new)+epsilon). `proposal_eq_callback` proves agreement
with the existing AMSGrad callback at beta-one zero, plus query decay.
The first coefficient supplies interpolation between the average and base.

MD averages the direction and both **raw** gain vectors. All three chain-rule
gradients are computed before any factor advances. `md_query_hasDerivAt`
proves the complete loss directional derivative at the actual fused query,
under its explicit differentiability hypothesis; linear objectives witness
that hypothesis. `softplusDerivative_eq_sigmoid` identifies the derivative
used by PyTorch. Smooth softplus is evaluated with logaddexp rather than a
finite linear threshold.

`md_base_norm` proves the initial-radius projection property for a nonzero
candidate and nonzero effective rate. Zero candidates project to zero.
Zero effective rates freeze the component iterates while histories and
clocks advance, as `zero_rate_iterates` proves. Averages and queries keep
their interpolated direction; the two-step scalar checkpoint has base
direction 2, evaluation direction 0 and training query weight 1.

The Python clocks advance for every participating component on every step;
all gradients must be dense and present. This keeps their clocks aligned
with the common time index in Lean. Checkpoints preserve those clocks,
component totals and train/eval mode. Scratch is discarded on load.

`scripts/ScheduleFreeAMSGradParity.lean` checks three shared checkpoints in
Lean's kernel: a reachable decreasing-variance AMSGradW run, the MD sphere
and average case, and a raw-zero-gain chain-rule update with nonzero gain
rate. The Python suite freezes their expected results and independently
compares the full MD recurrence to autograd, including large signed raw
gains, changing rates, warmup, auxiliary parameters and checkpoints.

## Outer learning-rate schedules

`LearningRateSchedule` and `LinearWarmup` supply explicit per-update rates;
they add no convergence claim. A rate multiplier is linear from zero to one
during warmup, then constant or linearly/cosinusoidally decayed to the chosen
nonnegative floor. Coordinate and matrix direction rules consume that rate
in their existing callbacks. Configured internal coefficient schedules remain
in effect.

For AMSGradMD and ScheduleFreeAMSGradMD, attaching any PyTorch scheduler
introduces the common multiplier `lr / initial_lr` for every independently
configured component rate. This is an explicit extension of the source's
independent-rate interface. It instantiates the same Lean callbacks at the
effective per-step rates; it does not replace their recurrence. Without a
scheduler, gain and auxiliary rates remain independent. If the nominal main
rate is zero, all components freeze while the scheduled main rate remains
zero; a later positive main rate reactivates the configured component rates.
Checked MD retains its true-gradient fallback at the scheduled outer rate.
When all rates that can move a stored tensor are zero, its numerical writes
are masked to avoid projection/rebalance roundoff moving frozen physical
weights. Moments and clocks still advance.

The Python tests compare scheduled MD runs with independent, manually
supplied component rates, including zero prefixes, a guarded direction-rate
override and built-in Schedule-Free warmup. They check checkpoint continuation
through decay, CPU-to-CUDA loading, live rate tensors during CUDA graph replay
and raw allocation events for optimizer and host scheduler steps together.
These runtime and storage properties are outside Lean's real-arithmetic model.

## Inverse-root pipeline

`roots.py` implements `DASH.countedBatchPiInverseFourth` (NDB),
`countedBatchPiCnFour` (CN), and `countedBatchPiPower` (Chebyshev), in
`DASH/Section4_GuardedNdb.lean`, `Section4_GuardedCn.lean` and
`Section4_GuardedChebyshev.lean`. The PI pool consists of one ones vector,
with three iterations. `guarded_scale` uses the smaller Frobenius/absolute
row-sum certificate and only accepts twice the Rayleigh estimate when it
exceeds that certificate. It falls back to twice the certificate, or one
at zero. This corresponds to `DASH.guardedScale` in
`Section3_GuardedScaling.lean`; the additional estimate>0 check is redundant
for a credible estimate on a PSD input.

NDB and CN regularize A+eps I **before** scaling. CN uses c=5^(-1/4) and
the residual recurrence B <- ((5I-B)/4)^4 B, retaining old values when
computing each pair. Chebyshev scales the history and fits
(eps/scale+x)^(-1/4), uses max(samples,degree+1) cosine nodes, half of the
constant coefficient and Clenshaw evaluation. EVD computes the same
regularized inverse fourth root directly, with a tiny-positive eigenvalue
clamp for roundoff. Finite NDB/CN/Chebyshev outputs are approximations,
not an equality to the exact inverse root. CUDA graph capture excludes EVD.

## Corrections found during extraction

At `470f9b6b`, AMSGradW's `lr_decay` scaled only m/sqrt(maximum), leaving
the decay term at the unscaled rate. This disagreed with
`AMSGradW.trainingStep`, whose eta multiplies both. The extracted rule now
scales both terms. A multistep independent test combines a nonconstant
schedule, nonzero weight decay and changing gradients. Constant schedules
and zero-decay historical configurations keep their previous arithmetic.

AMSGradMD's guarded fallback may halve the true gradient to avoid zero
matrix storage. Its alignment can then be only half the gradient's squared
norm. The public constructor and lab specification now require sigma <= 1/2,
matching `checkedDirection_certificate`; the generic direction guard retains
its valid sigma <= 1 range. The boundary and rejection are tested.

## Reproducible checks

- `./run.py optimizer-parity`: kernel-check rational checkpoints from actual
  definitions in `scripts/OptimizerParity.lean` and the new combinations in
  `scripts/ScheduleFreeAMSGradParity.lean`; print the optimistic regret
  theorem's axioms. Their literal values are used in `test_lean_cases.py`.
- `./run.py test`: independent scalar/matrix recurrences, initialization,
  clocks, zero cases, auxiliary selections, checkpoints, all historical
  optimizer trajectories, active matching examples and lab integration.
- CUDA tests: shared-gradient CPU/GPU updates and graph replay reading
  current gradient/rate tensors; skipped explicitly when CUDA is absent.
- Buffered root tests compare float32/64 outputs bit for bit with the original
  allocating recurrences, including reverse scalar division and Chebyshev degrees
  0, 1 and larger. Workspaces change storage lifetimes, not the paper recurrences.
- Allocation tests check raw warmed-step allocator events, released old gradients,
  bounded caches, checkpoint invalidation, retained Fisher graphs and dtype
  promotion across autocast contexts. Mixed-dtype DASH packing retains the
  promotion of the original `torch.cat` implementation.
- `./run.py package`: build a copied standalone directory, install the
  wheel without dependencies into a fresh target, block lab imports, check
  torch-only runtime metadata, and rerun the package tests from another cwd.

No Lean statement is weakened, no new `sorry` or axiom is introduced, and
the existing 39 unrelated unfinished statements are retained.
