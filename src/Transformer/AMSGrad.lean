module

public import Transformer.AMSGrad.Section1_AMSGrad
public import Transformer.AMSGrad.Section1_TheoremA
public import Transformer.AMSGrad.Section2_Prelim
public import Transformer.AMSGrad.Section2_Proj
public import Transformer.AMSGrad.Section3_Step
public import Transformer.AMSGrad.Section3_Issue
public import Transformer.AMSGrad.Section3_Example
public import Transformer.AMSGrad.Section3_Optimal
public import Transformer.AMSGrad.Section4_Lemmas
public import Transformer.AMSGrad.Section4_MainLemma
public import Transformer.AMSGrad.Section4_Telescope
public import Transformer.AMSGrad.Section4_Terms
public import Transformer.AMSGrad.Section4_TheoremAAbel
public import Transformer.AMSGrad.Section4_TheoremAAbelFalse
public import Transformer.AMSGrad.Section4_TheoremAActualAbelFalse
public import Transformer.AMSGrad.Section4_TheoremAReduce
public import Transformer.AMSGrad.Section4_TheoremAGeneral
public import Transformer.AMSGrad.Section4_TheoremAAntitone
public import Transformer.AMSGrad.Section4_TheoremASparse
public import Transformer.AMSGrad.Section4_TheoremABump
public import Transformer.AMSGrad.Section4_TheoremAAlternateStep
public import Transformer.AMSGrad.Section4_TheoremAEnergyBounds
public import Transformer.AMSGrad.Section4_TheoremAEnergyBudget
public import Transformer.AMSGrad.Section4_TheoremAPotential
public import Transformer.AMSGrad.Section4_TheoremAProof
public import Transformer.AMSGrad.Section4_TheoremAFinite
public import Transformer.AMSGrad.Section4_Third
public import Transformer.AMSGrad.Section4_Rate
public import Transformer.AMSGrad.Section4_Theorem
public import Transformer.AMSGrad.Section4_Corollary
public import Transformer.AMSGrad.Section4_OfflineConvergence
public import Transformer.AMSGrad.Section4_OfflineAverage
public import Transformer.AMSGrad.Section4_OfflineAverageConvergence
public import Transformer.AMSGrad.Section4_TrainingModels
public import Transformer.AMSGrad.Section4_TrainingScalarEnergy
public import Transformer.AMSGrad.Section4_TrainingKinetic
public import Transformer.AMSGrad.Section4_TrainingDescent
public import Transformer.AMSGrad.Section4_TrainingGradientBound
public import Transformer.AMSGrad.Section4_TrainingMetricBound
public import Transformer.AMSGrad.Section4_TrainingVelocityLimit
public import Transformer.AMSGrad.Section4_TrainingGradientEnergy
public import Transformer.AMSGrad.Section4_TrainingStationarity
public import Transformer.AMSGrad.Section4_TrainingLossLimit
public import Transformer.AMSGrad.Section4_TrainingMinimum
public import Transformer.AMSGrad.Section4_Counter
public import Transformer.AMSGrad.Section4_CounterRun
public import Transformer.AMSGrad.Section4_CounterRegret
public import Transformer.AMSGrad.Section5_AdamX
public import Transformer.AMSGrad.Section5_Bounds
public import Transformer.AMSGrad.Section5_Theorem
public import Transformer.AMSGrad.Section5_Sums
public import Transformer.AMSGrad.Section5_Corollary

/-!
Formalization of:
  Tran, Le,
  "On the Convergence Proof of AMSGrad and a New Version",
  arXiv:1904.03590v4.

Deviations from the source, each recorded in the docstring of the file that
makes it:
* `γ = β₁/√β₂ < 1` throughout, where the source has `γ ≤ 1` and divides by
  `1 - γ`; `0 < β₂ < 1` and `0 ≤ β_{1,t} ≤ β₁ < 1` are made explicit;
* every regret bound holds for every `x* ∈ F`, not only for the minimizer;
* Lemma 2.3's `Σ_{t≥1} αᵗ = 1/(1-α)` is false and refuted;
* Theorem A is proved with its printed constants using a schedule-free
  projection potential and a Young bound on `g_t - m_{t-1}`
  (`Section4_TheoremAProof`). Its assumptions on losses, steps, and
  first-moment coefficients are needed only for `1 ≤ t ≤ T`; the proof
  extends those data after `T` (`Section4_TheoremAFinite`). The
  original telescoping step is still false, even on an admissible AMSGrad
  run with the best comparator (`not_abel_printed_actual_run`); the earlier
  Abel-based bounds and special cases remain as audits of that route;
* Theorem 4.1's `t₀` is chosen before `T`, not `1 ≤ t₀ ≤ T`;
* Corollary 4.5's `lim R(T)/T = 0` is kept as its upper half only; the lower
  half is false, refuted by a proved counterexample (`not_cor_lower`);
  for a fixed convex objective and feasible minimizer, the added offline
  specialization proves the full average-gap limit and minimum-loss
  convergence of the actual averaged parameters for both source schedules;
* Theorem 5.1's second term carries `(1-β₁)²`, as its proof gives, not `(1-β₁)`;
* Corollaries 5.5 and 5.6 are kept as their upper halves: with `β_{1,t} = 0`
  AdamX is AMSGrad (`adamX_eq_amsgrad`), so `not_cor_lower` refutes them too;
* Corollary 5.6's `β_{1,t} = 1/t` contradicts `β_{1,1} < 1` and is corrected
  to `β_{1,t} = β₁/t`.

Added deterministic training extension of Algorithm 1: a fixed smooth
lower-bounded objective on Euclidean space, constant first-moment
coefficient and step size, no projection or bias correction, and the
`epsilon + sqrt(vhat)` regularizer of §6. With positive `eta`, `epsilon`
and `L`, `0 ≤ beta < 1`, `0 ≤ beta2 ≤ 1`, and `L * eta ≤ epsilon`, the
actual gradient norms tend to zero and the actual losses have a finite
limit. Under strong convexity and an existing stationary point, the
actual last parameters converge to the global minimizer. The proof uses
AMSGrad's nondecreasing metric and derives its upper bound from the
initial loss gap; no direction-replacement guard is added.

Not transcribed, deliberately: the experiments of §6.
-/

@[expose] public section
