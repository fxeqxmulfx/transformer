import Transformer.Modes.Section5_PtBddBoxBounds

/-
# The number of modes of a Gaussian KDE — the continuous-density claim is false

`lem: pt.bdd` of arXiv:2412.09080v3, §4.1, claims a continuous joint density
for every `β > 0` and every `n ≥ 5`. Its proof in §5.5 uses the Fourier decay
estimate already refuted for `β > 2` in `Section5_PtBddDecayFalse.lean`.
Here the density conclusion itself is refuted, without any Fourier argument.

For `β > 2`, all samples in `[t-z-1,t-z]` put the normalized pair in a
rectangle of radius `c exp(-a z²)`, where `a = (β+2)/4` and `c > 0` is fixed.
That event has probability at least `(2π)^{-n/2} exp(-n(z+1-t)²/2)`.
A density bounded near the origin would bound this by `C exp(-2a z²)`.
Their ratio has positive quadratic exponent when `n < β + 2`, a contradiction.

In particular `n = 5`, `β = 10` satisfies all the source's numerical
hypotheses and gives no continuous density, at any observation point `t`.
`Section4_KacRiceAppl.lean` records the resulting counterexamples to the
source's universal application claims.

Source: arXiv:2412.09080v3, §4.1, `lem: pt.bdd`; §5.5, its proposed proof.
-/

open Real MeasureTheory ProbabilityTheory
open scoped ENNReal

namespace Transformer.Modes

/-- Small rectangles do not have the quadratic mass bound required by
`lem: pt.bdd`: the Gaussian tail event has a slower decay rate than their area.
Source: arXiv:2412.09080v3, §4.1, `lem: pt.bdd`; §5.5. -/
theorem not_quadratic_box_mass_bound {n : ℕ} (hn : 0 < n) {β : ℝ}
    (hβ : 2 < β) (hnβ : (n : ℝ) < β + 2) (t : ℝ) :
    ¬ ∃ C : ℝ, ∀ r : ℝ, 0 ≤ r → r ≤ 1 →
      (gaussianSample n).real (sumGG' n β t ⁻¹' centeredBox r) ≤ C * r ^ 2 := by
  rintro ⟨C, hmass⟩
  have hnR : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  let K : ℝ := 2 * β / (β - 2)
  have hK : 0 < K := div_pos (by linarith) (by linarith)
  let a : ℝ := (β + 2) / 4
  have ha : 0 < a := by dsimp [a]; linarith
  let c : ℝ := (Real.sqrt n)⁻¹ * (n * K)
  have hc : 0 < c := mul_pos (inv_pos.mpr (Real.sqrt_pos.mpr hnR)) (mul_pos hnR hK)
  let Q : ℝ := Real.sqrt (2 * π)
  have hQ : 0 < Q := Real.sqrt_pos.mpr (by positivity)
  let M : ℝ := Q ^ n * (C * c ^ 2)
  let κ : ℝ := 2 * a - n / 2
  have hκ : 0 < κ := by dsimp [κ, a]; linarith
  obtain ⟨z, hz1, hzL, hzE⟩ := exists_far_exp_gt hκ (n * (1 - t))
    (n * (1 - t) ^ 2 / 2) M (max t (c / a))
  have hzt : t ≤ z := (le_max_left _ _).trans hzL
  have hzc : c / a ≤ z := (le_max_right _ _).trans hzL
  have hcz : c ≤ a * z ^ 2 := by
    have hlinear : c ≤ a * z := by rwa [div_le_iff₀ ha, mul_comm] at hzc
    have hsq : z ≤ z ^ 2 := by nlinarith
    exact hlinear.trans (mul_le_mul_of_nonneg_left hsq ha.le)
  let r : ℝ := c * Real.exp (-a * z ^ 2)
  have hr : 0 < r := mul_pos hc (Real.exp_pos _)
  have hr1 : r ≤ 1 := by
    dsimp [r]
    rw [show -a * z ^ 2 = -(a * z ^ 2) by ring, Real.exp_neg,
      mul_inv_le_iff₀ (Real.exp_pos _), one_mul]
    linarith [Real.add_one_le_exp (a * z ^ 2)]
  have hlower : Q⁻¹ ^ n * Real.exp (-n * (z + 1 - t) ^ 2 / 2) ≤
      (gaussianSample n).real (sumGG' n β t ⁻¹' centeredBox r) := by
    dsimp [Q, r, c, K, a]
    exact gaussianSample_box_mass_lower hβ n t hz1 hzt
  have hupper := hmass r hr.le hr1
  have hcompare := hlower.trans hupper
  rw [inv_pow, inv_mul_le_iff₀ (pow_pos hQ n)] at hcompare
  have hscaled := mul_le_mul_of_nonneg_right hcompare (Real.exp_pos (2 * a * z ^ 2)).le
  have hcancel : Real.exp (-a * z ^ 2) ^ 2 * Real.exp (2 * a * z ^ 2) = 1 := by
    rw [pow_two, ← Real.exp_add, ← Real.exp_add]
    ring_nf
    exact Real.exp_zero
  have hleft : Real.exp (-n * (z + 1 - t) ^ 2 / 2) * Real.exp (2 * a * z ^ 2) =
      Real.exp (κ * z ^ 2 - n * (1 - t) * z - n * (1 - t) ^ 2 / 2) := by
    rw [← Real.exp_add]
    congr 1
    dsimp [κ]
    ring
  have hright : (Q ^ n * (C * r ^ 2)) * Real.exp (2 * a * z ^ 2) = M := by
    dsimp [r, M]
    rw [mul_pow]
    linear_combination (Q ^ n * (C * c ^ 2)) * hcancel
  rw [hleft, hright] at hscaled
  exact (not_lt_of_ge hscaled) hzE

/-- All the numerical assumptions hold at the mass-bound counterexample. -/
example : 0 < (5 : ℕ) ∧ (2 : ℝ) < 10 ∧ (5 : ℝ) < 10 + 2 := by norm_num

/-- **Counterexample to the density conclusion of `lem: pt.bdd`.**
For `β > 2` and `0 < n < β + 2`, no density is continuous even on the
unit rectangle. Such continuity would give the refuted quadratic mass bound.
Source: arXiv:2412.09080v3, §4.1, `lem: pt.bdd`; §5.5. -/
theorem not_continuous_density_sumGG' {n : ℕ} (hn : 0 < n) {β : ℝ}
    (hβ : 2 < β) (hnβ : (n : ℝ) < β + 2) (t : ℝ) :
    ¬ ∃ q : ℝ × ℝ → ℝ, ContinuousOn q (centeredBox 1) ∧
      Measure.map (sumGG' n β t) (gaussianSample n) =
        volume.withDensity fun z => ENNReal.ofReal (q z) := by
  rintro ⟨q, hq, hqP⟩
  obtain ⟨C, _, hmass⟩ := continuous_density_box_le hq hqP
  apply not_quadratic_box_mass_bound hn hβ hnβ t
  refine ⟨C, fun r hr hr1 => ?_⟩
  simpa only [measureReal_def, Measure.map_apply (measurable_sumGG' n β t)
    (measurableSet_centeredBox r)] using hmass r hr hr1

/-- The continuity obstruction's numerical assumptions hold simultaneously. -/
example : 0 < (5 : ℕ) ∧ (2 : ℝ) < 10 ∧ (5 : ℝ) < 10 + 2 := by norm_num

/-- **The bounded-density assertion of `lem: pt.bdd` is also false.**
Continuity is unnecessary: any globally bounded density would give the same
quadratic mass estimate. Source: arXiv:2412.09080v3, §4.1 and §5.5. -/
theorem not_bounded_density_sumGG' {n : ℕ} (hn : 0 < n) {β : ℝ}
    (hβ : 2 < β) (hnβ : (n : ℝ) < β + 2) (t : ℝ) :
    ¬ ∃ (q : ℝ × ℝ → ℝ) (B : ℝ),
      Measure.map (sumGG' n β t) (gaussianSample n) =
        volume.withDensity (fun z => ENNReal.ofReal (q z)) ∧ ∀ z, q z ≤ B := by
  rintro ⟨q, B, hP, hq⟩
  apply not_quadratic_box_mass_bound hn hβ hnβ t
  refine ⟨4 * max B 0, fun r hr _ => ?_⟩
  have hmass := density_box_le_of_bound (le_max_right B 0) hr
    (fun z _ => (hq z).trans (le_max_left B 0)) hP
  simpa only [measureReal_def, Measure.map_apply (measurable_sumGG' n β t)
    (measurableSet_centeredBox r)] using hmass

/-- The bounded-density obstruction has the same numerical witnesses. -/
example : 0 < (5 : ℕ) ∧ (2 : ℝ) < 10 ∧ (5 : ℝ) < 10 + 2 := by norm_num

/-- Increasing the sample count does not remove the obstruction when the
precision parameter `β` also grows: fifty samples and `β = 100` still satisfy
the counterexample inequalities. -/
example : 0 < (50 : ℕ) ∧ (2 : ℝ) < 100 ∧ (50 : ℝ) < 100 + 2 := by
  norm_num

/-- **A concrete counterexample to `lem: pt.bdd`.** At `n = 5`, `β = 10`,
no observation point has the claimed continuous joint density.
Source: arXiv:2412.09080v3, §4.1, `lem: pt.bdd`; §5.5. -/
theorem not_continuous_density_five_ten (t : ℝ) :
    ¬ ∃ q : ℝ × ℝ → ℝ, Continuous q ∧
      Measure.map (sumGG' 5 10 t) (gaussianSample 5) =
        volume.withDensity fun z => ENNReal.ofReal (q z) := by
  rintro ⟨q, hq, hP⟩
  exact not_continuous_density_sumGG' (by norm_num) (by norm_num) (by norm_num) t
    ⟨q, hq.continuousOn, hP⟩

/-- **The density counterexample for the field as defined in `eq:Fn`.**
The normalization in `sumGG'` agrees exactly with `(F_n(t), F_n'(t))`;
passing to the field therefore changes neither its law nor the density
claim. The hypotheses give an entire range of counterexamples, including
values of `n` arbitrarily larger than five.

Source: arXiv:2412.09080v3, §4.1, `lem: pt.bdd`, and §2.2, the display
after `eq: Gt`. The source's assertion is refuted for `β > 2`, `n < β + 2`. -/
theorem not_continuous_density_fieldF {n : ℕ} (hn : 0 < n) {β : ℝ}
    (hβ : 2 < β) (hnβ : (n : ℝ) < β + 2) (t : ℝ) :
    ¬ ∃ q : ℝ × ℝ → ℝ, Continuous q ∧
      Measure.map (fun X => (fieldF β X t, deriv (fieldF β X) t)) (gaussianSample n) =
        volume.withDensity fun z => ENNReal.ofReal (q z) := by
  rintro ⟨q, hq, hP⟩
  have hpair : (fun X : Fin n → ℝ => (fieldF β X t, deriv (fieldF β X) t)) =
      sumGG' n β t := by
    funext X
    simpa only [sumGG', one_div] using fieldF_pair_eq β X t
  rw [hpair] at hP
  exact not_continuous_density_sumGG' hn hβ hnβ t ⟨q, hq.continuousOn, hP⟩

/-- The field's density refutation has simultaneous numerical witnesses. -/
example : 0 < (5 : ℕ) ∧ (2 : ℝ) < 10 ∧ (5 : ℝ) < 10 + 2 := by norm_num

/-- The same counterexample in the original field notation, at the center
of every window `T`. Its numerical assumptions are discharged explicitly. -/
example : ¬ ∃ q : ℝ × ℝ → ℝ, Continuous q ∧
    Measure.map (fun X => (fieldF 10 X 0, deriv (fieldF 10 X) 0)) (gaussianSample 5) =
      volume.withDensity fun z => ENNReal.ofReal (q z) :=
  not_continuous_density_fieldF (by norm_num) (by norm_num) (by norm_num) 0

end Transformer.Modes
