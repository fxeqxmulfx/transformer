/-
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

import Transformer.AMSGrad.Section1_AMSGrad
import Transformer.AMSGrad.Section1_TheoremA
import Transformer.AMSGrad.Section2_Prelim
import Transformer.AMSGrad.Section2_Proj
import Transformer.AMSGrad.Section3_Step
import Transformer.AMSGrad.Section3_Issue
import Transformer.AMSGrad.Section3_Example
import Transformer.AMSGrad.Section3_Optimal
import Transformer.AMSGrad.Section4_Lemmas
import Transformer.AMSGrad.Section4_MainLemma
import Transformer.AMSGrad.Section4_Telescope
import Transformer.AMSGrad.Section4_Terms
import Transformer.AMSGrad.Section4_TheoremAAbel
import Transformer.AMSGrad.Section4_TheoremAAbelFalse
import Transformer.AMSGrad.Section4_TheoremAActualAbelFalse
import Transformer.AMSGrad.Section4_TheoremAReduce
import Transformer.AMSGrad.Section4_TheoremAGeneral
import Transformer.AMSGrad.Section4_TheoremAAntitone
import Transformer.AMSGrad.Section4_TheoremASparse
import Transformer.AMSGrad.Section4_TheoremABump
import Transformer.AMSGrad.Section4_TheoremAAlternateStep
import Transformer.AMSGrad.Section4_TheoremAEnergyBounds
import Transformer.AMSGrad.Section4_TheoremAEnergyBudget
import Transformer.AMSGrad.Section4_TheoremAPotential
import Transformer.AMSGrad.Section4_TheoremAProof
import Transformer.AMSGrad.Section4_TheoremAFinite
import Transformer.AMSGrad.Section4_Third
import Transformer.AMSGrad.Section4_Rate
import Transformer.AMSGrad.Section4_Theorem
import Transformer.AMSGrad.Section4_Corollary
import Transformer.AMSGrad.Section4_OfflineConvergence
import Transformer.AMSGrad.Section4_OfflineAverage
import Transformer.AMSGrad.Section4_OfflineAverageConvergence
import Transformer.AMSGrad.Section4_TrainingModels
import Transformer.AMSGrad.Section4_TrainingScalarEnergy
import Transformer.AMSGrad.Section4_TrainingKinetic
import Transformer.AMSGrad.Section4_TrainingDescent
import Transformer.AMSGrad.Section4_TrainingGradientBound
import Transformer.AMSGrad.Section4_TrainingMetricBound
import Transformer.AMSGrad.Section4_TrainingVelocityLimit
import Transformer.AMSGrad.Section4_TrainingGradientEnergy
import Transformer.AMSGrad.Section4_TrainingStationarity
import Transformer.AMSGrad.Section4_TrainingLossLimit
import Transformer.AMSGrad.Section4_TrainingMinimum
import Transformer.AMSGrad.Section4_Counter
import Transformer.AMSGrad.Section4_CounterRun
import Transformer.AMSGrad.Section4_CounterRegret
import Transformer.AMSGrad.Section5_AdamX
import Transformer.AMSGrad.Section5_Bounds
import Transformer.AMSGrad.Section5_Theorem
import Transformer.AMSGrad.Section5_Sums
import Transformer.AMSGrad.Section5_Corollary
