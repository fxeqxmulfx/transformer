/-
# Quadratic energy and a nonstationary convergence witness

The quadratic flow checks the nonstationary branch of the finite-length
argument in Appendix D.1 of arXiv:2510.22026v2.
-/

import Transformer.Normalization.LojasiewiczConditional
import Mathlib.Analysis.InnerProductSpace.Calculus
import Mathlib.Analysis.SpecialFunctions.ExpDeriv
import Mathlib.Analysis.Analytic.Constructions
import Mathlib.Analysis.Analytic.Linear

open scoped BigOperators

namespace Transformer.Normalization

/-- Quadratic energy is analytic in every finite dimension; it is a
special case of the energies in Appendix D.1 of arXiv:2510.22026v2. -/
theorem quadratic_energy_analytic {N : ℕ} :
    AnalyticOnNhd ℝ (fun y : EucSpace N => ‖y‖ ^ 2) Set.univ := by
  have h : AnalyticOnNhd ℝ (fun y : EucSpace N => ∑ i : Fin N, y.ofLp i ^ 2) Set.univ :=
    Finset.univ.analyticOnNhd_fun_sum
      (fun i _ => by
        convert ((EuclideanSpace.proj i : EucSpace N →L[ℝ] ℝ).analyticOnNhd
          Set.univ).fun_pow 2 using 1
        ext y
        rfl)
  convert h using 1
  ext y
  exact EuclideanSpace.real_norm_sq_eq y

/-- The quadratic energy has gradient `2x`; it is a basic instance of
the analytic energies in Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem quadratic_energy_gradient {N : ℕ} (y : EucSpace N) :
    gradient (fun z : EucSpace N => ‖z‖ ^ 2) y = (2 : ℝ) • y := by
  apply HasGradientAt.gradient
  rw [hasGradientAt_iff_hasFDerivAt]
  convert (hasStrictFDerivAt_norm_sq y).hasFDerivAt using 1
  ext z
  simp [InnerProductSpace.toDual_apply_apply]

/-- Quadratic energy satisfies the exponent-`1/2` gradient inequality
globally, with constant one, around its critical point zero. This is a
proved instance of the local inequality in Appendix D.1 of
arXiv:2510.22026v2. -/
theorem quadratic_energy_gradient_inequality {N : ℕ} (y : EucSpace N) :
    (‖y‖ ^ 2) ^ (1 / 2 : ℝ) ≤ ‖gradient (fun z : EucSpace N => ‖z‖ ^ 2) y‖ := by
  rw [← Real.sqrt_eq_rpow, Real.sqrt_sq (norm_nonneg _), quadratic_energy_gradient]
  have hscale : ‖(2 : ℝ) • y‖ = 2 * ‖y‖ := by rw [norm_smul]; norm_num
  rw [hscale]
  linarith [norm_nonneg y]

/-- An exponential trajectory solves the identity-modulated flow for
quadratic energy, as in Appendix D.1 of arXiv:2510.22026v2. -/
theorem quadratic_exponential_flow {N : ℕ} (v : EucSpace N) (t : ℝ) :
    HasDerivAt (fun s : ℝ => Real.exp (-2 * s) • v)
      (-gradient (fun z : EucSpace N => ‖z‖ ^ 2) (Real.exp (-2 * t) • v)) t := by
  rw [quadratic_energy_gradient]
  have he : HasDerivAt (fun s : ℝ => Real.exp (-2 * s))
      (-2 * Real.exp (-2 * t)) t := by
    convert ((hasDerivAt_id t).const_mul (-2)).exp using 1
    · ext s
      congr 1
    · simp only [id_eq, mul_one]
      ring
  convert he.smul_const v using 1
  simp [mul_smul, neg_smul]

/-- A unit initial state exists for the nonstationary quadratic flow
used to witness Appendix D.1 of arXiv:2510.22026v2. -/
example : ∃ v : EucSpace 1, ‖v‖ = 1 := by
  exact ⟨EuclideanSpace.single (0 : Fin 1) 1, by simp⟩

/-- A quadratic exponential trajectory has strictly positive energy,
an exponent-`1/2` gradient inequality and the full nonstationary
convergence hypotheses of Appendix D.1 of arXiv:2510.22026v2. -/
example (v : EucSpace 1) (hv : ‖v‖ = 1) :
    ∃ z : EucSpace 1, Filter.Tendsto (fun t : ℝ => Real.exp (-2 * t) • v)
      Filter.atTop (nhds z) := by
  let E : EucSpace 1 → ℝ := fun z => ‖z‖ ^ 2
  let x : ℝ → EucSpace 1 := fun t => Real.exp (-2 * t) • v
  have hxnorm t : ‖x t‖ = Real.exp (-2 * t) := by
    simp [x, norm_smul, Real.norm_eq_abs, abs_of_pos (Real.exp_pos _), hv]
  apply modulated_converges_of_strict_gradient_inequality E x
    (fun _ => ContinuousLinearMap.id ℝ (EucSpace 1)) 0 0 1 1 (1 / 2)
    zero_le_one zero_lt_one (by norm_num)
  · intro t _
    exact (hasStrictFDerivAt_norm_sq (x t)).hasFDerivAt.differentiableAt
  · intro t _
    exact quadratic_exponential_flow v t
  · intro t _
    simp only [ContinuousLinearMap.id_apply, one_mul, real_inner_self_eq_norm_sq]
    nlinarith
  · intro t _
    dsimp [E]
    rw [hxnorm]
    positivity
  · intro t _
    dsimp [E]
    rw [sub_zero, ← Real.sqrt_eq_rpow, Real.sqrt_sq (norm_nonneg _),
      quadratic_energy_gradient]
    have hscale : ‖(2 : ℝ) • x t‖ = 2 * ‖x t‖ := by
      rw [norm_smul]
      norm_num
    rw [one_mul, hscale]
    linarith [norm_nonneg (x t)]
  · have ht : Filter.Tendsto (fun t : ℝ => 2 * t) Filter.atTop Filter.atTop :=
      Filter.tendsto_id.const_mul_atTop (by norm_num)
    have he := Real.tendsto_exp_neg_atTop_nhds_zero.comp ht
    have hn : Filter.Tendsto (fun t : ℝ => ‖x t‖) Filter.atTop (nhds 0) := by
      convert he using 1
      ext t
      rw [hxnorm]
      congr 1
      ring
    change Filter.Tendsto (fun t => ‖x t‖ ^ 2) Filter.atTop (nhds 0)
    convert hn.pow 2 using 1
    norm_num

end Transformer.Normalization
