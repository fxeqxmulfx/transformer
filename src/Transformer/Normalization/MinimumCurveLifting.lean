/-
# Energy-level comparison from one-sided curves of gradient minima

The one-sided output of analytic curve selection supplies the energy-level
lifting needed by the comparison-curve proof of the local gradient
inequality in Appendix D.1 of arXiv:2510.22026v2.
-/

import Transformer.Normalization.AnalyticCurveEnergyLevels
import Transformer.Normalization.GradientMinimumCluster

open Filter Set

namespace Transformer.Normalization

/-- A one-sided analytic curve of energy-fiber gradient minima compares
the gradient at every nearby energy level of its sign. Arbitrarily small
parameters suffice. This proves the lifting step after analytic curve
selection for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem minimum_curve_lifts_energy_side {N : ℕ} (E : EucSpace N → ℝ)
    (z : EucSpace N) (K : Set (EucSpace N)) (hK : K ∈ nhds z)
    (hE : ContinuousAt E z) (sign : ℝ) (hsign : sign ≠ 0)
    (gamma : ℝ → EucSpace N) (hgamma : AnalyticAt ℝ gamma 0)
    (hEg : AnalyticAt ℝ E (gamma 0)) (hgE : E (gamma 0) = E z)
    (epsilon : ℝ) (he : 0 < epsilon)
    (hmin : ∀ t ∈ Ioo 0 epsilon,
      gamma t ∈ energyFiberGradientMinima E K {c | 0 < sign * (c - E z)}) :
    ∀ delta : ℝ, 0 < delta → ∀ᶠ y in nhds z,
      0 < sign * (E y - E z) → ∃ t : ℝ, |t| < delta ∧
        E (gamma t) = E y ∧ ‖gradient E (gamma t)‖ ≤ ‖gradient E y‖ := by
  let f : ℝ → ℝ := fun t => sign * (E (gamma t) - E z)
  have hf : AnalyticAt ℝ f 0 :=
    analyticAt_const.mul ((hEg.comp hgamma).sub analyticAt_const)
  have hf0 : f 0 = 0 := by simp only [f, hgE, sub_self, mul_zero]
  have hfpos : ∀ t ∈ Ioo 0 epsilon, 0 < f t := fun t ht => (hmin t ht).2.1
  intro delta hd
  obtain ⟨eta, heta, hcover⟩ := analytic_curve_covers_positive_levels f hf hf0
    epsilon he hfpos delta hd
  have hgap : ∀ᶠ y in nhds z, sign * (E y - E z) < eta := by
    have hc : ContinuousAt (fun y => sign * (E y - E z)) z :=
      (hE.sub continuousAt_const).const_mul sign
    exact hc.eventually (eventually_lt_nhds (by simpa only [sub_self, mul_zero] using heta))
  have hnear : ∀ᶠ y in nhds z, y ∈ K := hK
  filter_upwards [hgap, hnear] with y hygap hyK
  intro hypos
  obtain ⟨t, ht, hte, htc⟩ := hcover _ ⟨hypos, hygap⟩
  have henergy : E (gamma t) = E y := by
    have heq := mul_left_cancel₀ hsign htc
    exact sub_left_inj.mp heq
  refine ⟨t, by simpa only [abs_of_pos ht.1] using ht.2, henergy, ?_⟩
  exact (hmin t ⟨ht.1, hte⟩).2.2 y hyK henergy.symm

/-- For the squared norm in dimension one, the coordinate line is a
curve of positive-energy gradient minima: gradient norm depends only on
energy. All one-sided lifting hypotheses are jointly satisfiable.
Auxiliary example for Appendix D.1 of arXiv:2510.22026v2. -/
example : ∀ delta : ℝ, 0 < delta → ∀ᶠ y in nhds (0 : EucSpace 1),
    0 < ‖y‖ ^ 2 → ∃ t : ℝ, |t| < delta ∧
      ‖t • (PiLp.single 2 (0 : Fin 1) 1 : EucSpace 1)‖ ^ 2 = ‖y‖ ^ 2 ∧
      ‖gradient (fun x : EucSpace 1 => ‖x‖ ^ 2)
        (t • (PiLp.single 2 (0 : Fin 1) 1 : EucSpace 1))‖ ≤
          ‖gradient (fun x : EucSpace 1 => ‖x‖ ^ 2) y‖ := by
  let v : EucSpace 1 := PiLp.single 2 (0 : Fin 1) 1
  let gamma : ℝ → EucSpace 1 := fun t => t • v
  have hg (t : ℝ) : ‖gamma t‖ = |t| := by simp [gamma, v, norm_smul]
  have hmin : ∀ t ∈ Ioo (0 : ℝ) 1,
      gamma t ∈ energyFiberGradientMinima (fun x : EucSpace 1 => ‖x‖ ^ 2)
        univ {c | 0 < (1 : ℝ) * (c - ‖(0 : EucSpace 1)‖ ^ 2)} := by
    intro t ht
    refine ⟨mem_univ _, ?_, ?_⟩
    · simp only [norm_zero, zero_pow (by norm_num : (2 : ℕ) ≠ 0), sub_zero,
        one_mul, hg, abs_of_pos ht.1]
      exact sq_pos_of_pos ht.1
    · intro y hy hyE
      have hyG : ‖y‖ = ‖gamma t‖ := by
        nlinarith [norm_nonneg y, norm_nonneg (gamma t)]
      simp only [quadratic_energy_gradient, norm_smul, Real.norm_of_nonneg (by norm_num :
        (0 : ℝ) ≤ 2), hyG, le_refl]
  simpa only [gamma, v, norm_zero, zero_pow (by norm_num : (2 : ℕ) ≠ 0), sub_zero,
    one_mul] using minimum_curve_lifts_energy_side
      (fun x : EucSpace 1 => ‖x‖ ^ 2) 0 univ univ_mem
      (quadratic_energy_analytic 0 (mem_univ _)).continuousAt 1 one_ne_zero gamma
      (analyticAt_id.fun_smul analyticAt_const)
      (quadratic_energy_analytic (gamma 0) (mem_univ _))
      (by simp [gamma]) 1 zero_lt_one hmin

end Transformer.Normalization
