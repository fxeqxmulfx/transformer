/-
# The analytic gradient inequality in one real variable

The local analytic input of Appendix D.1, `lem: loj`, in
arXiv:2510.22026v2 follows from the finite order of an analytic zero
in one variable, including degenerate critical points.
-/

import Mathlib.Analysis.Analytic.Order
import Mathlib.Analysis.SpecialFunctions.Pow.Continuity

open Filter Set

namespace Transformer.Normalization

/-- The local analytic gradient inequality used in Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2, in one real variable. If the energy
is not locally constant and its zero after shifting by `f z` has order
`n + 1`, its derivative has order `n`, and the exponent is `n / (n + 1)`.
Criticality guarantees `n > 0`; no nondegeneracy assumption is needed. -/
theorem scalar_analytic_gradient_inequality (f : ℝ → ℝ) (z : ℝ)
    (hf : AnalyticAt ℝ f z) (hcritical : deriv f z = 0) :
    ∃ alpha k : ℝ, ∃ V : Set ℝ,
      0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ z ∈ V ∧
        ∀ y ∈ V, |f y - f z| ^ alpha ≤ k * |deriv f y| := by
  let F : ℝ → ℝ := fun y => f y - f z
  have hF : AnalyticAt ℝ F z := hf.sub analyticAt_const
  by_cases hzero : ∀ᶠ y in nhds z, F y = 0
  · obtain ⟨V, hV, hVo, hzV⟩ := eventually_nhds_iff.1 hzero
    refine ⟨1 / 2, 1, V, by norm_num, by norm_num, zero_lt_one, hVo, hzV, ?_⟩
    intro y hy
    have he : f y - f z = 0 := hV y hy
    rw [he]
    simp
  obtain ⟨m, g, hg, hg0, hfactor⟩ := hF.exists_eventuallyEq_pow_smul_nonzero_iff.2 hzero
  have hm : m ≠ 0 := by
    intro hm
    have he := hfactor.self_of_nhds
    simp [hm, F] at he
    exact hg0 he.symm
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hm
  have horder : analyticOrderAt F z = (n + 1 : ℕ) :=
    hF.analyticOrderAt_eq_natCast.2 ⟨g, hg, hg0, hfactor⟩
  have hderivF : deriv F = deriv f := by
    ext y
    simp [F, deriv_sub_const]
  have horderD : analyticOrderAt (deriv f) z = (n : ℕ) := by
    rw [← hderivF]
    apply analyticOrderAt_deriv_of_pos hF
    simpa only [Nat.cast_add, Nat.cast_one] using horder
  have hn : n ≠ 0 := by
    intro hn
    have hne := hf.deriv.analyticOrderAt_ne_zero.2 hcritical
    exact hne (by simpa [hn] using horderD)
  have hnR : (0 : ℝ) < n := by exact_mod_cast Nat.pos_of_ne_zero hn
  obtain ⟨h, hh, hh0, hfactorD⟩ := hf.deriv.analyticOrderAt_eq_natCast.1 horderD
  let alpha : ℝ := n / (n + 1)
  let A : ℝ := |g z| + 1
  let b : ℝ := |h z| / 2
  have ha : 0 < alpha := div_pos hnR (by positivity)
  have ha1 : alpha < 1 := (div_lt_one (by positivity)).2 (by linarith)
  have hA : 0 < A := by dsimp [A]; positivity
  have hb : 0 < b := half_pos (abs_pos.2 hh0)
  have hgBound : ∀ᶠ y in nhds z, |g y| ≤ A := by
    have hlt : |g z| < A := by dsimp [A]; linarith
    exact ((hg.continuousAt.abs).eventually (eventually_lt_nhds hlt)).mono
      (fun y hy => hy.le)
  have hhBound : ∀ᶠ y in nhds z, b ≤ |h y| := by
    have hlt : b < |h z| := half_lt_self (abs_pos.2 hh0)
    exact ((hh.continuousAt.abs).eventually (eventually_gt_nhds hlt)).mono
      (fun y hy => hy.le)
  have hevent : ∀ᶠ y in nhds z,
      |f y - f z| ^ alpha ≤ (A ^ alpha / b) * |deriv f y| := by
    filter_upwards [hfactor, hfactorD, hgBound, hhBound] with y hy hyD hyG hyH
    have he : f y - f z = (y - z) ^ (n + 1) * g y := hy
    have heD : deriv f y = (y - z) ^ n * h y := hyD
    have hexp : (n + 1 : ℝ) * alpha = n := by
      dsimp [alpha]
      field_simp
    have hpow : (|y - z| ^ (n + 1)) ^ alpha = |y - z| ^ n := by
      rw [← Real.rpow_natCast, ← Real.rpow_mul (abs_nonneg _),
        Nat.cast_add, Nat.cast_one, hexp, Real.rpow_natCast]
    rw [he, heD, abs_mul, abs_pow, abs_mul, abs_pow,
      Real.mul_rpow (by positivity) (abs_nonneg _), hpow]
    have hG := Real.rpow_le_rpow (abs_nonneg (g y)) hyG ha.le
    have hH : A ^ alpha ≤ (A ^ alpha / b) * |h y| := by
      rw [div_mul_eq_mul_div]
      apply (le_div_iff₀ hb).2
      nlinarith [Real.rpow_nonneg hA.le alpha]
    calc
      |y - z| ^ n * |g y| ^ alpha ≤ |y - z| ^ n * A ^ alpha :=
        mul_le_mul_of_nonneg_left hG (by positivity)
      _ ≤ |y - z| ^ n * ((A ^ alpha / b) * |h y|) :=
        mul_le_mul_of_nonneg_left hH (by positivity)
      _ = A ^ alpha / b * (|y - z| ^ n * |h y|) := by ring
  obtain ⟨V, hV, hVo, hzV⟩ := eventually_nhds_iff.1 hevent
  exact ⟨alpha, A ^ alpha / b, V, ha, ha1,
    div_pos (Real.rpow_pos_of_pos hA alpha) hb, hVo, hzV, hV⟩

/-- The scalar gradient inequality holds at every analytic base point.
At a regular point the derivative is bounded away from zero; at a critical
point the finite-order argument above applies. Auxiliary scalar input to
the analytic-curve route for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem scalar_analytic_gradient_inequality_at_any_point (f : ℝ → ℝ) (z : ℝ)
    (hf : AnalyticAt ℝ f z) :
    ∃ alpha k : ℝ, ∃ V : Set ℝ,
      0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ z ∈ V ∧
        ∀ y ∈ V, |f y - f z| ^ alpha ≤ k * |deriv f y| := by
  by_cases hcritical : deriv f z = 0
  · exact scalar_analytic_gradient_inequality f z hf hcritical
  let b : ℝ := |deriv f z| / 2
  have hb : 0 < b := half_pos (abs_pos.mpr hcritical)
  have hsmall : ∀ᶠ y in nhds z, |f y - f z| < 1 := by
    simpa only [Real.dist_eq] using
      (Metric.tendsto_nhds.mp hf.continuousAt.tendsto) 1 zero_lt_one
  have hbound : ∀ᶠ y in nhds z, b ≤ |deriv f y| :=
    (hf.deriv.continuousAt.abs.eventually
      (eventually_gt_nhds (half_lt_self (abs_pos.mpr hcritical)))).mono
        (fun _ hy => hy.le)
  have hevent : ∀ᶠ y in nhds z,
      |f y - f z| ^ (1 / 2 : ℝ) ≤ (1 / b) * |deriv f y| := by
    filter_upwards [hsmall, hbound] with y hyE hyD
    calc
      |f y - f z| ^ (1 / 2 : ℝ) ≤ 1 :=
        Real.rpow_le_one (abs_nonneg _) hyE.le (by norm_num)
      _ ≤ (1 / b) * |deriv f y| := by
        rw [div_mul_eq_mul_div, one_mul]
        exact (le_div_iff₀ hb).mpr (by simpa only [one_mul] using hyD)
  obtain ⟨V, hV, hVo, hzV⟩ := eventually_nhds_iff.mp hevent
  exact ⟨1 / 2, 1 / b, V, by norm_num, by norm_num, div_pos zero_lt_one hb,
    hVo, hzV, hV⟩

/-- The regular branch is satisfiable for the identity energy at zero;
auxiliary scalar example for Appendix D.1 of arXiv:2510.22026v2. -/
example : ∃ alpha k : ℝ, ∃ V : Set ℝ,
    0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ (0 : ℝ) ∈ V ∧
      ∀ y ∈ V, |y| ^ alpha ≤ k := by
  simpa only [deriv_id, id_eq, sub_zero, abs_one, mul_one] using
    scalar_analytic_gradient_inequality_at_any_point id 0 analyticAt_id

/-- The scalar analytic hypotheses of Appendix D.1 of arXiv:2510.22026v2
hold at the degenerate critical point of `f(y) = y^4`. Applying the
inequality here exercises the finite-order branch of the proof. -/
example : ∃ alpha k : ℝ, ∃ V : Set ℝ,
    0 < alpha ∧ alpha < 1 ∧ 0 < k ∧ IsOpen V ∧ (0 : ℝ) ∈ V ∧
      ∀ y ∈ V, |y ^ 4| ^ alpha ≤ k * |4 * y ^ 3| := by
  have hd (y : ℝ) : deriv (fun t : ℝ => t ^ 4) y = 4 * y ^ 3 := by
    simp
  simpa only [hd, zero_pow (by norm_num : (4 : ℕ) ≠ 0), sub_zero] using
    scalar_analytic_gradient_inequality (fun y => y ^ 4) 0
      (analyticAt_id.fun_pow 4) (by rw [hd]; norm_num)

end Transformer.Normalization
