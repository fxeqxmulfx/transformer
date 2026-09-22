import Transformer.Modes.Section3_ScaledSumMGF

/-
# The number of modes of a Gaussian KDE — the third moment of a normalized sum

The moment identity of arXiv:2412.09080v3, §3.1, rests on one scaling law: a
third moment of `S_n = n^{-1/2} Σ Yᵢ` is `n^{-1/2}` times the corresponding
third moment of `Y`, provided `Y` is centred.  Here it is proved for every
linear form at once,

  `𝔼 (s (S_n)₁ + t (S_n)₂)³ = n^{-1/2} 𝔼 (s Y₁ + t Y₂)³`,

which is enough: `s` and `t` are free, and the four mixed third moments are
recovered from four values of the cubic form in `Section3_ScaledSumMoments.lean`.

Two steps.  `integral_pi_sum_cube` is the probabilistic one: `𝔼 (Σ g(Yᵢ))³ =
n 𝔼 g(Y)³` for centred `g`, read off the third derivative at `0` of
`(𝔼 e^{θ g})^n` — three of the four terms of that derivative carry a factor
`𝔼 g(Y) = 0`.  The rest is the bookkeeping `n · (n^{-1/2})³ = n^{-1/2}` and a
change of measure along `scaledSum`.

Source: arXiv:2412.09080v3, §3.1, the display after `eq:psi`.
-/

open Real MeasureTheory Filter
open scoped Topology

namespace Transformer
namespace Modes

variable {μ : Measure (ℝ × ℝ)} {ε : ℝ} {n : ℕ}

/-- `0 < n^{-1/2} ≤ 1` for `n ≥ 1`: the normalization never widens the interval
on which exponential moments are finite. -/
theorem inv_sqrt_natCast_mem (hn : 1 ≤ n) : 0 < (Real.sqrt n)⁻¹ ∧ (Real.sqrt n)⁻¹ ≤ 1 := by
  have h1 : (1 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
  have hs : (1 : ℝ) ≤ Real.sqrt n := by
    rw [show (1 : ℝ) = Real.sqrt 1 by simp]
    exact Real.sqrt_le_sqrt h1
  exact ⟨inv_pos.2 (lt_of_lt_of_le one_pos hs), inv_le_one_of_one_le₀ hs⟩

/-- `n · (n^{-1/2})³ = n^{-1/2}`: the third cumulant of a normalized sum of `n`
summands carries exactly one factor `n^{-1/2}`.  arXiv:2412.09080v3, §3.1. -/
theorem natCast_mul_inv_sqrt_cube (hn : 1 ≤ n) :
    (n : ℝ) * ((Real.sqrt n)⁻¹) ^ 3 = (Real.sqrt n)⁻¹ := by
  have hn0 : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.2 (by omega)
  have hsq : Real.sqrt n * Real.sqrt n = (n : ℝ) := Real.mul_self_sqrt (by positivity)
  have h3 : ((Real.sqrt n)⁻¹) ^ 3 = (Real.sqrt n)⁻¹ * (Real.sqrt n * Real.sqrt n)⁻¹ := by
    rw [mul_inv]; ring
  rw [h3, hsq]
  field_simp

/-- **The third moment of a sum of independent centred copies is `n` times the
third moment of one**, `𝔼 (Σᵢ g(Yᵢ))³ = n 𝔼 g(Y)³`.

This is the additivity of the third cumulant, read off the third derivative at
`0` of `(𝔼 e^{θ g})^n`: of the four terms that derivative has, three carry a
factor `𝔼 g(Y)`, which vanishes.

Source: arXiv:2412.09080v3, §3.1, the display after `eq:psi`. -/
theorem integral_pi_sum_cube [IsProbabilityMeasure μ] {g : ℝ × ℝ → ℝ} {E : ℝ}
    (hgm : Measurable g) (hgE : HasScalarExpMomentsOn μ g E) (hE0 : 0 < E)
    (hg0 : ∫ x, g x ∂μ = 0) (n : ℕ) :
    ∫ X : Fin n → ℝ × ℝ, (∑ i, g (X i)) ^ 3 ∂(Measure.pi fun _ : Fin n => μ)
      = (n : ℝ) * ∫ x, g x ^ 3 ∂μ := by
  have hVm : Measurable fun X : Fin n → ℝ × ℝ => ∑ i, g (X i) := by fun_prop
  rw [← iteratedDeriv_three_scalarMoment hVm (hasScalarExpMomentsOn_pi_sum hgE n) hE0,
    show scalarMoment (Measure.pi fun _ : Fin n => μ) (fun X => ∑ i, g (X i)) 0
      = fun θ => scalarMoment μ g 0 θ ^ n from funext fun θ => scalarMoment_pi_sum n θ]
  have hball : Metric.ball (0 : ℝ) E ∈ 𝓝 (0 : ℝ) := Metric.ball_mem_nhds 0 hE0
  have habs : ∀ θ ∈ Metric.ball (0 : ℝ) E, |θ| < E := fun θ hθ => by
    simpa [Real.dist_eq] using hθ
  rw [iteratedDeriv_three_pow n
    (by filter_upwards [hball] with θ hθ using hasDerivAt_scalarMoment hgm hgE (habs θ hθ) 0)
    (by filter_upwards [hball] with θ hθ using hasDerivAt_scalarMoment hgm hgE (habs θ hθ) 1)
    (hasDerivAt_scalarMoment hgm hgE (by simpa using hE0) 2)
    (scalarMoment_zero_zero μ g) (by rw [scalarMoment_zero]; simpa using hg0),
    scalarMoment_zero]

/-- The hypotheses of `integral_pi_sum_cube` are satisfiable. -/
example : Measurable (fun x : ℝ × ℝ => 1 * x.1 + 0 * x.2) ∧
    HasScalarExpMomentsOn stdGauss2 (fun x => 1 * x.1 + 0 * x.2)
      (1 / (1 + |(1 : ℝ)| + |(0 : ℝ)|)) ∧
    (0 : ℝ) < 1 / (1 + |(1 : ℝ)| + |(0 : ℝ)|) ∧
    ∫ x, (1 * x.1 + 0 * x.2) ∂stdGauss2 = 0 :=
  ⟨by fun_prop, hasScalarExpMomentsOn_linear hasExpMomentsOn_stdGauss2 1 0, by norm_num,
    by simpa using isStandardized_stdGauss2.mean_fst⟩

/-- **A third moment of a linear form scales by `n^{-1/2}` under normalization**,
`𝔼 (s (S_n)₁ + t (S_n)₂)³ = n^{-1/2} 𝔼 (s Y₁ + t Y₂)³`, for a centred law `μ`
with exponential moments.

Source: arXiv:2412.09080v3, §3.1, the display after `eq:psi`, where it is the
factor `n^{-1/2}` the source omits (see `Section3_Cumulants.lean`). -/
theorem integral_map_scaledSum_dot_cube [IsProbabilityMeasure μ] (hE : HasExpMomentsOn μ ε)
    (hε : 0 < ε) (hfst : ∫ x, x.1 ∂μ = 0) (hsnd : ∫ x, x.2 ∂μ = 0) (hn : 1 ≤ n) (s t : ℝ) :
    ∫ x, (s * x.1 + t * x.2) ^ 3 ∂((Measure.pi fun _ : Fin n => μ).map (scaledSum n))
      = (Real.sqrt n)⁻¹ * ∫ x, (s * x.1 + t * x.2) ^ 3 ∂μ := by
  have hfi : Integrable (fun x : ℝ × ℝ => x.1) μ := by
    simpa using integrable_pow_mul_pow hE hε 1 0
  have hsi : Integrable (fun x : ℝ × ℝ => x.2) μ := by
    simpa using integrable_pow_mul_pow hE hε 0 1
  have hg0 : ∫ x, ((Real.sqrt n)⁻¹ * s * x.1 + (Real.sqrt n)⁻¹ * t * x.2) ∂μ = 0 := by
    rw [integral_add (hfi.const_mul _) (hsi.const_mul _), integral_const_mul, integral_const_mul,
      hfst, hsnd]
    ring
  have key := integral_pi_sum_cube (μ := μ) (by fun_prop)
    (hasScalarExpMomentsOn_linear hE ((Real.sqrt n)⁻¹ * s) ((Real.sqrt n)⁻¹ * t))
    (div_pos hε (by positivity)) hg0 n
  rw [integral_map (measurable_scaledSum n).aemeasurable
    ((by fun_prop : Continuous fun x : ℝ × ℝ => (s * x.1 + t * x.2) ^ 3).aestronglyMeasurable)]
  have hL : ∀ X : Fin n → ℝ × ℝ,
      (s * (scaledSum n X).1 + t * (scaledSum n X).2) ^ 3
        = (∑ i, ((Real.sqrt n)⁻¹ * s * (X i).1 + (Real.sqrt n)⁻¹ * t * (X i).2)) ^ 3 := by
    intro X
    rw [scaledSum_dot n s t X]
    exact congrArg (· ^ 3) (Finset.sum_congr rfl fun i _ => by ring)
  have hR : ∫ x, ((Real.sqrt n)⁻¹ * s * x.1 + (Real.sqrt n)⁻¹ * t * x.2) ^ 3 ∂μ
      = ((Real.sqrt n)⁻¹) ^ 3 * ∫ x, (s * x.1 + t * x.2) ^ 3 ∂μ := by
    rw [← integral_const_mul]
    exact integral_congr_ae (Eventually.of_forall fun x => by ring)
  simp only [hL]
  rw [key, hR, ← mul_assoc, natCast_mul_inv_sqrt_cube hn]

/-- The hypotheses of `integral_map_scaledSum_dot_cube` are satisfiable. -/
example : HasExpMomentsOn stdGauss2 1 ∧ (0 : ℝ) < 1 ∧ ∫ x, x.1 ∂stdGauss2 = 0 ∧
    ∫ x, x.2 ∂stdGauss2 = 0 ∧ 1 ≤ 1 :=
  ⟨hasExpMomentsOn_stdGauss2, one_pos, isStandardized_stdGauss2.mean_fst,
    isStandardized_stdGauss2.mean_snd, le_rfl⟩

/-- **The law of the normalized sum inherits the exponential moments of the
summands**, on the same interval, because `n^{-1/2} ≤ 1` for `n ≥ 1`.

Source: arXiv:2412.09080v3, §3.1. -/
theorem hasExpMomentsOn_map_scaledSum [IsProbabilityMeasure μ] (hE : HasExpMomentsOn μ ε)
    (hn : 1 ≤ n) :
    HasExpMomentsOn ((Measure.pi fun _ : Fin n => μ).map (scaledSum n)) ε := by
  obtain ⟨hc0, hc1⟩ := inv_sqrt_natCast_mem hn
  intro u v hu hv
  have hshrink : ∀ r : ℝ, |r| < ε → |(Real.sqrt n)⁻¹ * r| < ε := by
    intro r hr
    rw [abs_mul, abs_of_pos hc0]
    exact lt_of_le_of_lt (mul_le_of_le_one_left (abs_nonneg r) hc1) hr
  have hprod : Integrable (fun X : Fin n → ℝ × ℝ =>
      ∏ i, exp ((Real.sqrt n)⁻¹ * u * (X i).1 + (Real.sqrt n)⁻¹ * v * (X i).2))
      (Measure.pi fun _ : Fin n => μ) :=
    Integrable.fintype_prod
      (f := fun _ (x : ℝ × ℝ) => exp ((Real.sqrt n)⁻¹ * u * x.1 + (Real.sqrt n)⁻¹ * v * x.2))
      fun _ => hE _ _ (hshrink u hu) (hshrink v hv)
  rw [integrable_map_measure
    ((by fun_prop : Continuous fun x : ℝ × ℝ => exp (u * x.1 + v * x.2)).aestronglyMeasurable)
    (measurable_scaledSum n).aemeasurable]
  refine hprod.congr (Eventually.of_forall fun X => ?_)
  show (∏ i, exp ((Real.sqrt n)⁻¹ * u * (X i).1 + (Real.sqrt n)⁻¹ * v * (X i).2))
      = exp (u * (scaledSum n X).1 + v * (scaledSum n X).2)
  rw [scaledSum_dot n u v X, Real.exp_sum]
  exact Finset.prod_congr rfl fun i _ => congrArg exp (by ring)

/-- The hypotheses of `hasExpMomentsOn_map_scaledSum` are satisfiable. -/
example : HasExpMomentsOn stdGauss2 1 ∧ 1 ≤ 1 := ⟨hasExpMomentsOn_stdGauss2, le_rfl⟩

end Modes
end Transformer
