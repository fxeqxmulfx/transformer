/-
# Homogenized Transformers — Gaussian moments of projected attention fields

The mixed second moment in the drift of the overlap `R_ij` is the trace of
two tangent projections, multiplied by the softmax-barycenter correlation:

  `E⟨G_i,G_j⟩ = σ_V² (d - 2 + R_ij²) s_μ(x_i,x_j)`.

The finite-dimensional trace identity is proved below by summing over the
standard basis.  Integrability follows from the bounded attention averages
and the Gaussian mixed-moment bound, rather than from a formal manipulation
of Bochner integrals.

Source: arXiv:2604.01978v1, proof of `lem:Ito_formula`,
`eq:trace_proj_proj_clean` and `eq:Dij_explicit_clean`.
-/

import Transformer.Homogenized.GaussianVariance
import Transformer.Homogenized.GaussianKernel

open scoped BigOperators NNReal
open Real MeasureTheory

namespace Transformer.Homogenized

/-- The inner product of two projected vectors, expanded in the standard
basis.  Source: arXiv:2604.01978v1, proof of `lem:Ito_formula`. -/
private theorem inner_proj_eq_sum {d : ℕ} (x y a b : EucSpace d) :
    inner (𝕜 := ℝ) (proj d x a) (proj d y b) =
      ∑ k : Fin d,
        inner (𝕜 := ℝ) (proj d x (EuclideanSpace.single k (1 : ℝ))) a *
          inner (𝕜 := ℝ) (proj d y (EuclideanSpace.single k (1 : ℝ))) b := by
  have hcoord (z w : EucSpace d) (k : Fin d) :
      inner (𝕜 := ℝ) (proj d z (EuclideanSpace.single k (1 : ℝ))) w = (proj d z w) k := by
    rw [real_inner_proj_left, EuclideanSpace.inner_single_left]
    simp
  simp_rw [hcoord]
  rw [EuclideanSpace.inner_eq_star_dotProduct]
  simp [dotProduct, mul_comm]

/-- `Tr(Proj_x Proj_y) = d - 2 + ⟨x,y⟩²` for unit vectors: the trace identity
`eq:trace_proj_proj_clean` used by `lem:Ito_formula`.

Source: arXiv:2604.01978v1, `eq:trace_proj_proj_clean`. -/
theorem sum_proj_proj {d : ℕ} (x y : EucSpace d) (hx : ‖x‖ = 1) (hy : ‖y‖ = 1) :
    (∑ k : Fin d, inner (𝕜 := ℝ)
      (proj d x (EuclideanSpace.single k (1 : ℝ)))
      (proj d y (EuclideanSpace.single k (1 : ℝ)))) =
      (d : ℝ) - 2 + inner (𝕜 := ℝ) x y ^ 2 := by
  simp only [proj, inner_sub_left, inner_sub_right, real_inner_smul_left,
    real_inner_smul_right]
  have hcoord (v : EucSpace d) (k : Fin d) :
      inner (𝕜 := ℝ) v (EuclideanSpace.single k (1 : ℝ)) = v k := by
    rw [real_inner_comm, EuclideanSpace.inner_single_left]
    simp
  simp only [hcoord, EuclideanSpace.inner_single_left, PiLp.single_apply,
    ite_true, map_one, one_mul]
  have hxsum : (∑ k : Fin d, x k * x k) = 1 := by
    simpa [pow_two, hx] using (EuclideanSpace.real_norm_sq_eq x).symm
  have hysum : (∑ k : Fin d, y k * y k) = 1 := by
    simpa [pow_two, hy] using (EuclideanSpace.real_norm_sq_eq y).symm
  have hxy : (∑ k : Fin d, y k * x k) = inner (𝕜 := ℝ) x y := by
    simp [EuclideanSpace.inner_eq_star_dotProduct, dotProduct, mul_comm]
  have hterm : ∀ k : Fin d,
      1 - x k * x k - (y k * y k - y k * (x k * inner (𝕜 := ℝ) x y)) =
        1 - x k * x k - y k * y k + (y k * x k) * inner (𝕜 := ℝ) x y := by
    intro k; ring
  simp only [hterm, Finset.sum_add_distrib, Finset.sum_sub_distrib,
    Finset.sum_const, Finset.card_fin, nsmul_eq_mul, mul_one, ← Finset.sum_mul]
  rw [hxsum, hysum, hxy]
  ring

/-- Under (G), the projected fluctuation is the projected Gaussian value map
applied to the softmax barycenter.

Source: arXiv:2604.01978v1, proof of `lem:Ito_formula`. -/
private theorem Gfield_eq_proj_value {d n : ℕ} (β : ℝ) {σV σA : ℝ≥0}
    {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    (x : Idx n → EucSpace d) (θ : HeadParam d) (i : Idx n) :
    Gfield β ρ x θ i =
      proj d (x i) (valueMap θ (softBary β θ.2 (empMeasure x) (x i))) := by
  rw [Gfield, fluct, meanField_gaussian β hρ x (x i), sub_zero,
    attnField_self_eq_valueMap, softBary_empMeasure]

/-- The scalar mixed Gaussian value moment is integrable at an empirical
measure of unit tokens.

Source: arXiv:2604.01978v1, proof of `lem:Ito_formula`. -/
private theorem integrable_value_cross_empMeasure {d n : ℕ} (β : ℝ) {σV σA : ℝ≥0}
    {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    {x : Idx n → EucSpace d} (hx : ∀ k, ‖x k‖ = 1)
    (i j : Idx n) (u v : EucSpace d) :
    Integrable (fun θ => inner (𝕜 := ℝ) u
        (valueMap θ (softBary β θ.2 (empMeasure x) (x i))) *
      inner (𝕜 := ℝ) v
        (valueMap θ (softBary β θ.2 (empMeasure x) (x j)))) ρ := by
  simpa only [valueMap, softBary_empMeasure] using
    (integrable_gaussian_value_cross hρ
      (measurable_sum_attnProb_smul β x i)
      (measurable_sum_attnProb_smul β x j)
      (fun A => norm_sum_attnProb_smul_le β A hx i)
      (fun A => norm_sum_attnProb_smul_le β A hx j) u v)

/-- The projected mixed second moment in `lem:Ito_formula`, with
`Tr(Proj_{x_i} Proj_{x_j}) = d - 2 + R_ij²`.

Source: arXiv:2604.01978v1, proof of `lem:Ito_formula`. -/
theorem integral_inner_Gfield_cross_gaussian {d n : ℕ} (β : ℝ) {σV σA : ℝ≥0}
    {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    {x : Idx n → EucSpace d} (hx : ∀ k, ‖x k‖ = 1)
    (i j : Idx n) :
    ∫ θ, inner (𝕜 := ℝ) (Gfield β ρ x θ i) (Gfield β ρ x θ j) ∂ρ =
      (σV : ℝ) ^ 2 * ((d : ℝ) - 2 + inner (𝕜 := ℝ) (x i) (x j) ^ 2) *
        baryCorr β ρ (empMeasure x) (x i) (x j) := by
  have hpt (θ : HeadParam d) :
      inner (𝕜 := ℝ) (Gfield β ρ x θ i) (Gfield β ρ x θ j) =
        ∑ k : Fin d,
          inner (𝕜 := ℝ) (proj d (x i) (EuclideanSpace.single k (1 : ℝ)))
            (valueMap θ (softBary β θ.2 (empMeasure x) (x i))) *
          inner (𝕜 := ℝ) (proj d (x j) (EuclideanSpace.single k (1 : ℝ)))
            (valueMap θ (softBary β θ.2 (empMeasure x) (x j))) := by
    rw [Gfield_eq_proj_value β hρ x θ i, Gfield_eq_proj_value β hρ x θ j]
    exact inner_proj_eq_sum _ _ _ _
  have hint (k : Fin d) : Integrable (fun θ =>
      inner (𝕜 := ℝ) (proj d (x i) (EuclideanSpace.single k (1 : ℝ)))
        (valueMap θ (softBary β θ.2 (empMeasure x) (x i))) *
      inner (𝕜 := ℝ) (proj d (x j) (EuclideanSpace.single k (1 : ℝ)))
        (valueMap θ (softBary β θ.2 (empMeasure x) (x j)))) ρ :=
    integrable_value_cross_empMeasure β hρ hx i j _ _
  rw [integral_congr_ae (Filter.Eventually.of_forall hpt)]
  rw [integral_finsetSum _ (fun k _ => hint k)]
  simp_rw [gaussian_value_cross_empMeasure β hρ hx i j]
  rw [← Finset.sum_mul]
  rw [← Finset.mul_sum, sum_proj_proj (x i) (x j) (hx i) (hx j)]

/-- Integrability required to split the overlap generator's cross term.
Source: arXiv:2604.01978v1, proof of `lem:Ito_formula`. -/
theorem integrable_inner_Gfield_cross_gaussian {d n : ℕ} (β : ℝ) {σV σA : ℝ≥0}
    {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    {x : Idx n → EucSpace d} (hx : ∀ k, ‖x k‖ = 1)
    (i j : Idx n) :
    Integrable (fun θ => inner (𝕜 := ℝ) (Gfield β ρ x θ i) (Gfield β ρ x θ j)) ρ := by
  have hpt (θ : HeadParam d) :
      inner (𝕜 := ℝ) (Gfield β ρ x θ i) (Gfield β ρ x θ j) =
        ∑ k : Fin d,
          inner (𝕜 := ℝ) (proj d (x i) (EuclideanSpace.single k (1 : ℝ)))
            (valueMap θ (softBary β θ.2 (empMeasure x) (x i))) *
          inner (𝕜 := ℝ) (proj d (x j) (EuclideanSpace.single k (1 : ℝ)))
            (valueMap θ (softBary β θ.2 (empMeasure x) (x j))) := by
    rw [Gfield_eq_proj_value β hρ x θ i, Gfield_eq_proj_value β hρ x θ j]
    exact inner_proj_eq_sum _ _ _ _
  simp_rw [hpt]
  exact integrable_finsetSum _ (fun k _ =>
    integrable_value_cross_empMeasure β hρ hx i j _ _)

/-- The diagonal moment from consequence (i) of (G), expressed using the
paper's softmax-barycenter statistic `s_μ(x_i)`.

Source: arXiv:2604.01978v1, proof of `lem:Ito_formula`. -/
theorem integral_norm_Gfield_sq_baryCorr_gaussian {d n : ℕ} (β : ℝ) {σV σA : ℝ≥0}
    {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    {x : Idx n → EucSpace d} (hx : ∀ k, ‖x k‖ = 1) (i : Idx n) :
    ∫ θ, ‖Gfield β ρ x θ i‖ ^ 2 ∂ρ =
      (σV : ℝ) ^ 2 * ((d : ℝ) - 1) * baryCorr β ρ (empMeasure x) (x i) (x i) := by
  rw [integral_norm_Gfield_sq_gaussian β hρ hx i]
  have h : (∫ θ, ‖∑ k, attnProb β θ.2 x i k • x k‖ ^ 2 ∂ρ) =
      baryCorr β ρ (empMeasure x) (x i) (x i) := by
    rw [baryCorr_self]
    simp only [softBary_empMeasure]
  rw [h]

/-- The hypotheses of the Gaussian overlap moment lemmas are satisfiable at
the zero Gaussian law with one unit token. -/
example (d : ℕ) :
    IsGaussianHeadLaw (d + 1) 0 0 (Measure.dirac (0 : HeadParam (d + 1))) ∧
      ∀ _k : Idx 1, ‖((basePoint d : SSphere (d + 1)) : EucSpace (d + 1))‖ = 1 :=
  ⟨isGaussianHeadLaw_dirac_zero _, fun _ => by simp [basePoint, PiLp.norm_single]⟩

end Transformer.Homogenized
