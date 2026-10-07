/-
# Gaussian vertices: conditional tail estimates

The proof of `prop: d.to.infty` in §4 of arXiv:2508.09628v1 compares
`⟨Bx,x⟩` with `⟨Bx,y⟩` for independent standard Gaussian vectors. Conditional
on `x`, the second expression has mean zero and variance `‖Bx‖²`.
Chebyshev's inequality therefore bounds a failed comparison outside a fixed
ball by `‖B‖² / (λ_min² R²)`. The radius can tend to infinity after the
dimension tends to infinity; no quantitative chi-square concentration is
needed for this qualitative conclusion.

The source's `B ≻ 0` includes symmetry. Quadratic upper and lower bounds
alone do not control a skew-symmetric part, so symmetry is retained in the
operator-norm estimate below.
-/

import Transformer.FrankWolfe.Section4_Polytope
import Mathlib.Analysis.InnerProductSpace.Rayleigh
import Mathlib.Probability.Distributions.Gaussian.Multivariate
import Mathlib.Probability.Distributions.Gaussian.Fernique
import Mathlib.Probability.Moments.Variance

open Real MeasureTheory ProbabilityTheory

namespace Transformer.FrankWolfe

/-- A symmetric positive quadratic form with upper bound `M‖x‖²` has
operator norm at most `|M|`; arXiv:2508.09628v1, §4, `prop: d.to.infty`.
Symmetry is part of the source's positive definiteness. -/
theorem norm_le_of_symmetric_quadratic_bounds {d : ℕ} (B : ParamMatrix d)
    (hB : B.toLinearMap.IsSymmetric) (M : ℝ)
    (hlower : ∀ x, 0 ≤ inner (𝕜 := ℝ) (B x) x)
    (hupper : ∀ x, inner (𝕜 := ℝ) (B x) x ≤ M * ‖x‖ ^ 2) :
    ‖B‖ ≤ |M| := by
  rw [B.norm_eq_iSup_rayleighQuotient hB]
  apply ciSup_le
  intro x
  change |inner (𝕜 := ℝ) (B x) x / ‖x‖ ^ 2| ≤ |M|
  rw [abs_of_nonneg (div_nonneg (hlower x) (sq_nonneg _))]
  by_cases hx : x = 0
  · simp [hx]
  · rw [div_le_iff₀ (sq_pos_of_pos (norm_pos_iff.mpr hx))]
    exact (hupper x).trans
      (mul_le_mul_of_nonneg_right (le_abs_self M) (sq_nonneg _))

/-- The quadratic-bound hypotheses hold for the identity in every dimension;
arXiv:2508.09628v1, §4, `prop: d.to.infty`. -/
example (d : ℕ) :
    (ContinuousLinearMap.id ℝ (EucSpace d)).toLinearMap.IsSymmetric ∧
    (∀ x : EucSpace d, 0 ≤ inner (𝕜 := ℝ) x x) ∧
    (∀ x : EucSpace d, inner (𝕜 := ℝ) x x ≤ (1 : ℝ) * ‖x‖ ^ 2) := by
  refine ⟨fun _ _ => rfl, fun _ => real_inner_self_nonneg, ?_⟩
  intro x
  rw [real_inner_self_eq_norm_sq, one_mul]

/-- The one-dimensional Gaussian tail used after conditioning on a vertex;
arXiv:2508.09628v1, §4, proof of `prop: d.to.infty`. -/
theorem stdGaussian_inner_tail {d : ℕ} (z : EucSpace d) {c : ℝ} (hc : 0 < c) :
    (stdGaussian (EucSpace d)) {y | c ≤ inner (𝕜 := ℝ) z y} ≤
      ENNReal.ofReal (‖z‖ ^ 2 / c ^ 2) := by
  have hLp : MemLp (fun y : EucSpace d => inner (𝕜 := ℝ) z y) 2
      (stdGaussian (EucSpace d)) :=
    IsGaussian.memLp_two_id.continuousLinearMap_comp (innerSL ℝ z)
  have h := meas_ge_le_variance_div_sq hLp hc
  have hzero : (∫ y, inner (𝕜 := ℝ) z y ∂stdGaussian (EucSpace d)) = 0 := by
    simpa using integral_strongDual_stdGaussian (innerSL ℝ z)
  have hvar : variance (fun y => inner (𝕜 := ℝ) z y)
      (stdGaussian (EucSpace d)) = ‖z‖ ^ 2 := by
    simpa using variance_dual_stdGaussian (innerSL ℝ z)
  rw [hzero, hvar] at h
  exact (measure_mono fun y hy =>
    show c ≤ |inner (𝕜 := ℝ) z y - 0| from by
      rw [sub_zero]; exact hy.trans (le_abs_self _)).trans h

/-- The tail hypothesis is met at `c = 1` in any dimension;
arXiv:2508.09628v1, §4, proof of `prop: d.to.infty`. -/
example : (0 : ℝ) < 1 := one_pos

/-- Outside a ball, the conditional probability of losing one's cell is
uniformly small; arXiv:2508.09628v1, §4, `prop: d.to.infty`.
The norm bound here is derived from the source's symmetric quadratic bounds
by `norm_le_of_symmetric_quadratic_bounds`. -/
theorem gaussian_cell_section_le {d : ℕ} (B : ParamMatrix d) {l M R : ℝ}
    (hl : 0 < l) (hR : 0 < R) (hB : ‖B‖ ≤ M)
    (hlower : ∀ x, l * ‖x‖ ^ 2 ≤ inner (𝕜 := ℝ) (B x) x)
    (x : EucSpace d) (hx : R < ‖x‖) :
    (stdGaussian (EucSpace d)) {y | inner (𝕜 := ℝ) (B x) x ≤
        inner (𝕜 := ℝ) (B x) y} ≤ ENNReal.ofReal (M ^ 2 / (l * R) ^ 2) := by
  have hn : 0 < ‖x‖ := hR.trans hx
  have hc : 0 < l * R * ‖x‖ := by positivity
  have hnorm : ‖B x‖ ≤ M * ‖x‖ :=
    (B.le_opNorm x).trans (mul_le_mul_of_nonneg_right hB (norm_nonneg _))
  refine (measure_mono fun y hy =>
    show l * R * ‖x‖ ≤ inner (𝕜 := ℝ) (B x) y from ?_).trans
      ((stdGaussian_inner_tail (B x) hc).trans (ENNReal.ofReal_le_ofReal ?_))
  · calc l * R * ‖x‖ ≤ l * ‖x‖ ^ 2 := by nlinarith
      _ ≤ inner (𝕜 := ℝ) (B x) x := hlower x
      _ ≤ inner (𝕜 := ℝ) (B x) y := hy
  · rw [div_le_iff₀ (sq_pos_of_pos hc)]
    have hsq : ‖B x‖ ^ 2 ≤ (M * ‖x‖) ^ 2 := by
      nlinarith [norm_nonneg (B x)]
    calc ‖B x‖ ^ 2 ≤ (M * ‖x‖) ^ 2 := hsq
      _ = M ^ 2 / (l * R) ^ 2 * (l * R * ‖x‖) ^ 2 := by
        field_simp

/-- The section-bound hypotheses hold for `B = I₁`, `l = M = R = 1`
and `x = 2e₀`; arXiv:2508.09628v1, §4, `prop: d.to.infty`. -/
example :
    (0 : ℝ) < 1 ∧
    ‖ContinuousLinearMap.id ℝ (EucSpace 1)‖ ≤ 1 ∧
    (∀ x : EucSpace 1, (1 : ℝ) * ‖x‖ ^ 2 ≤ inner (𝕜 := ℝ) x x) ∧
    (1 : ℝ) < ‖EuclideanSpace.single (0 : Fin 1) (2 : ℝ)‖ := by
  refine ⟨one_pos, ContinuousLinearMap.norm_id_le, ?_, by simp⟩
  intro x
  rw [real_inner_self_eq_norm_sq, one_mul]

/-- Integrating the conditional estimate gives a dimension-free error
outside a fixed ball; arXiv:2508.09628v1, §4, `prop: d.to.infty`.
The mass of the ball is treated separately in `Section4_GaussianBalls`. -/
theorem gaussian_cell_pair_le {d : ℕ} (B : ParamMatrix d) {l M R : ℝ}
    (hl : 0 < l) (hR : 0 < R) (hB : ‖B‖ ≤ M)
    (hlower : ∀ x, l * ‖x‖ ^ 2 ≤ inner (𝕜 := ℝ) (B x) x) :
    ((stdGaussian (EucSpace d)).prod (stdGaussian (EucSpace d)))
        {p | inner (𝕜 := ℝ) (B p.1) p.1 ≤ inner (𝕜 := ℝ) (B p.1) p.2} ≤
      (stdGaussian (EucSpace d)) {x | ‖x‖ ≤ R} +
        ENNReal.ofReal (M ^ 2 / (l * R) ^ 2) := by
  classical
  have hs : MeasurableSet {x : EucSpace d | ‖x‖ ≤ R} := by
    exact measurableSet_le (by fun_prop) measurable_const
  have hp : MeasurableSet {p : EucSpace d × EucSpace d |
      inner (𝕜 := ℝ) (B p.1) p.1 ≤ inner (𝕜 := ℝ) (B p.1) p.2} :=
    measurableSet_le (by fun_prop) (by fun_prop)
  rw [Measure.prod_apply hp]
  calc
    _ ≤ ∫⁻ x : EucSpace d,
        ({x : EucSpace d | ‖x‖ ≤ R}.indicator 1 x +
          ENNReal.ofReal (M ^ 2 / (l * R) ^ 2)) ∂stdGaussian (EucSpace d) := by
      apply lintegral_mono
      intro x
      by_cases hx : ‖x‖ ≤ R
      · simp only [Set.indicator, Set.mem_ofPred_eq, hx, ↓reduceIte, Pi.one_apply]
        exact prob_le_one.trans (le_add_right le_rfl)
      · simp only [Set.indicator, Set.mem_ofPred_eq, hx, ↓reduceIte, zero_add]
        exact gaussian_cell_section_le B hl hR hB hlower x (lt_of_not_ge hx)
    _ = _ := by
      rw [lintegral_add_right _ measurable_const,
        lintegral_indicator_one hs, lintegral_const, measure_univ, mul_one]

/-- The pair estimate applies to the identity, with `l = M = R = 1`;
arXiv:2508.09628v1, §4, `prop: d.to.infty`. -/
example (d : ℕ) :
    (0 : ℝ) < 1 ∧ ‖ContinuousLinearMap.id ℝ (EucSpace d)‖ ≤ 1 ∧
    (∀ x : EucSpace d, (1 : ℝ) * ‖x‖ ^ 2 ≤ inner (𝕜 := ℝ) x x) := by
  refine ⟨one_pos, ContinuousLinearMap.norm_id_le, ?_⟩
  intro x
  rw [real_inner_self_eq_norm_sq, one_mul]

end Transformer.FrankWolfe
