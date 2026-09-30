/-
# AdaFisher: refutation of the printed nonconvex endpoint

arXiv:2405.16397v3, §3.4, Proposition 3.4, Appendix A.2, `eq:gradient_bound`.
The statement allows `βᵗ ≤ β ≤ 1`. At β=1 the run is stationary, even
with deterministic unbiased independent noise, a bounded smooth objective,
and the stated monotonicity of the effective preconditioner.
Algorithm 1's requirement β<1 must also apply to the convergence statement.
-/

import Transformer.AdaFisher.Section3_Algorithm
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds
import Mathlib.Analysis.SpecialFunctions.Pow.Asymptotics
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Probability.Independence.Basic

open Filter Topology MeasureTheory ProbabilityTheory

noncomputable section

namespace Transformer.AdaFisher

/-- The literal RHS in Proposition 3.4, §3.4, `eq:gradient_bound`.
`B` denotes the gradient bound; it need not be identified with damping. -/
def nonconvexRate (L η B : ℝ) (d : ℕ) (C₁ C₂ C₃ C₄ : ℝ) (T : ℕ) : ℝ :=
  L / Real.sqrt T * (C₁ * η ^ 2 * B ^ 2 * (1 + Real.log T) +
    C₂ * d * η + C₃ * d * η ^ 2 + C₄)

/-- For any fixed finite constants the claimed nonconvex rate tends to zero,
Proposition 3.4. This is the paper's asymptotic O(log T / sqrt T) assertion. -/
theorem nonconvexRate_tendsto (L η B : ℝ) (d : ℕ) (C₁ C₂ C₃ C₄ : ℝ) :
    Tendsto (nonconvexRate L η B d C₁ C₂ C₃ C₄) atTop (𝓝 0) := by
  have hlog : Tendsto (fun x : ℝ => Real.log x / Real.sqrt x) atTop (𝓝 0) := by
    simpa only [Real.sqrt_eq_rpow] using
      (isLittleO_log_rpow_atTop (r := 1 / 2) (by norm_num)).tendsto_div_nhds_zero
  have hc := Real.tendsto_sqrt_atTop.const_div_atTop
    (L * (C₁ * η ^ 2 * B ^ 2 + C₂ * d * η + C₃ * d * η ^ 2 + C₄))
  have hl := hlog.const_mul (L * C₁ * η ^ 2 * B ^ 2)
  have hlim := (hl.add hc).comp
    (tendsto_natCast_atTop_atTop : Tendsto (fun t : ℕ => (t : ℝ)) atTop atTop)
  simp only [mul_zero, zero_add] at hlim
  convert hlim using 1
  funext T
  dsimp [nonconvexRate, Function.comp_def]
  ring

/-- A bounded smooth objective witnessing Proposition 3.4's hypotheses. -/
def sineLoss (x : ℝ) : ℝ := (1 / 1000) * Real.sin x

/-- Actual derivative of the counterexample objective, Proposition 3.4. -/
theorem sineLoss_hasDerivAt (x : ℝ) :
    HasDerivAt sineLoss ((1 / 1000) * Real.cos x) x := by
  exact (Real.hasDerivAt_sin x).const_mul (1 / 1000)

/-- The counterexample is lower bounded, has bounded gradient, and its
gradient is 2-Lipschitz, exactly as required in Proposition 3.4. -/
theorem sineLoss_assumptions :
    (∀ x, -(1 / 1000 : ℝ) ≤ sineLoss x) ∧
    (∀ x, |(1 / 1000 : ℝ) * Real.cos x| ≤ 1 / 1000) ∧
    (∀ x y, |(1 / 1000 : ℝ) * Real.cos x - (1 / 1000) * Real.cos y| ≤
      2 * |x - y|) := by
  refine ⟨?_, ?_, ?_⟩
  · intro x
    unfold sineLoss
    nlinarith [Real.neg_one_le_sin x]
  · intro x
    rw [abs_mul]
    norm_num
    nlinarith [Real.abs_cos_le_one x]
  · intro x y
    have hc := Real.lipschitzWith_cos.dist_le_mul x y
    simp only [Real.dist_eq, NNReal.coe_one, one_mul] at hc
    rw [← mul_sub, abs_mul]
    norm_num
    nlinarith [abs_nonneg (x - y)]

/-- Constant zero noise is unbiased and independent on a probability
space, Proposition 3.4, assumption (iii). Thus stochastic noise cannot
explain away the stationary-run counterexample. -/
theorem stationary_noise_assumptions :
    (∫ _ : ℝ, (0 : ℝ) ∂Measure.dirac (0 : ℝ)) = 0 ∧
    IndepFun (fun _ : ℝ => (0 : ℝ)) (fun _ : ℝ => (0 : ℝ))
      (Measure.dirac (0 : ℝ)) := by
  exact ⟨by simp, indepFun_const_left _ _⟩

/-- The constant damped factor satisfies the source's effective-step
monotonicity for ηᵗ=1/sqrt(t), Proposition 3.4. Indices here start at t+1. -/
theorem stationary_preconditioner_monotone :
    Monotone (fun t : ℕ => (1 / 1000 : ℝ) / (1 / Real.sqrt (t + 1 : ℕ))) := by
  intro i j hij
  simp only [one_div, div_inv_eq_mul]
  apply mul_le_mul_of_nonneg_left _ (by norm_num)
  apply Real.sqrt_le_sqrt
  exact_mod_cast Nat.add_le_add_right hij 1

/-- In the actual β=1 run the gradient squared is constantly 10⁻⁶,
Proposition 3.4. The objective/gradient bounds are established above;
F=0.001<2 and βᵗ=β=1 are the printed allowed values. -/
theorem stationary_gradient_squared (t : ℕ) :
    let θ := adaFisherRun (fun t => 1 / Real.sqrt (t + 1 : ℕ)) 0 1
      (fun (_ : ℕ) (_ : Fin 1) => (1 / 1000 : ℝ))
      (fun (_ : ℕ) (_ : Fin 1) => (1 / 1000 : ℝ)) (fun _ => 0)
    ((1 / 1000 : ℝ) * Real.cos (θ t 0)) ^ 2 = (1 / 1000 : ℝ) ^ 2 := by
  dsimp only
  rw [adaFisherRun_one]
  norm_num

/-- Refutation of Proposition 3.4 as quantified over all horizons:
no finite constants independent of T can bound this stationary nonzero
gradient by its printed RHS. The endpoint β=1 must be excluded. -/
theorem nonconvex_rate_counterexample (C₁ C₂ C₃ C₄ : ℝ) :
    ∃ T : ℕ, 0 < T ∧
      nonconvexRate 2 1 (1 / 1000) 1 C₁ C₂ C₃ C₄ T < (1 / 1000 : ℝ) ^ 2 := by
  have hlt := (nonconvexRate_tendsto 2 1 (1 / 1000) 1 C₁ C₂ C₃ C₄).eventually_lt
    tendsto_const_nhds (by norm_num : (0 : ℝ) < (1 / 1000 : ℝ) ^ 2)
  obtain ⟨T, hT, hrate⟩ := (hlt.and (eventually_gt_atTop 0)).exists
  exact ⟨T, hrate, hT⟩

/-- The last inequality in `eq:bound_2`, Appendix A.2, drops a factor η.
Already T=1 with η=1/4000, L=2, F=B=0.001 makes the printed inequality
strictly false. The corrected factor is sqrt(T)*η/L. -/
theorem nonconvex_missing_eta_counterexample :
    (1 / 4000 : ℝ) * ((1 / 1000) ^ 2 / (1 / 1000)) <
      (Real.sqrt 1 / 2) * (1 / 1000) ^ 2 := by
  norm_num

end Transformer.AdaFisher
