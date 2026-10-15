/-
# Uniform bounds for the actual Gaussian overlap noise

ArXiv:2604.01978v1, §7, the bracket estimate following
eq:Gronwall_E_selfcontained.
The mutual projection identity factors the original noise density
as σV²(1-R²)(P+Q-2RS). Actual barycenter bounds and the original
unit overlap then give the universal bound (5/2)σV².

At the source's standard value variance this implies 3/d, and
integrating the actual density gives 3T/d. The source's displayed
4T/d equality after bounding is replaced by a proved upper bound.
The proof keeps the original head law and original token indices;
no noise-moment or variance bound is assumed.
-/

import Transformer.Homogenized.Section7_OverlapNoiseMoments

open MeasureTheory Set
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- The two original tangent projections have inner product -R(1-R²).
Source: arXiv:2604.01978v1, §7, the overlap bracket calculation. -/
theorem inner_mutual_tangent_proj {d : ℕ} (x y : EucSpace d) (hx : ‖x‖ = 1) (hy : ‖y‖ = 1) :
    inner (𝕜 := ℝ) (proj d x y) (proj d y x) =
      -inner (𝕜 := ℝ) x y * (1 - inner (𝕜 := ℝ) x y ^ 2) := by
  simp only [proj, inner_sub_left, inner_sub_right, real_inner_smul_left, real_inner_smul_right,
    real_inner_self_eq_norm_sq, hx, hy, one_pow, real_inner_comm x y]
  ring

/-- The exact projection polynomial has universal upper bound 5/2.
Source: arXiv:2604.01978v1, §7, the bounding step after the bracket calculation. -/
theorem overlap_noise_polynomial_le_five_halves (r P Q S : ℝ) (hr : |r| ≤ 1) (hP : P ≤ 1) (hQ : Q ≤ 1)
    (hS : |S| ≤ 1) : (1 - r ^ 2) * (P + Q - 2 * r * S) ≤ 5 / 2 := by
  have hr2 := (sq_le_one_iff_abs_le_one r).2 hr
  have hc : -r * S ≤ |r| := by
    calc
      _ ≤ |-r * S| := le_abs_self _
      _ = |r| * |S| := by rw [abs_mul, abs_neg]
      _ ≤ |r| := mul_le_of_le_one_right (abs_nonneg _) hS
  have hi : P + Q - 2 * r * S ≤ 2 + 2 * |r| := by linarith
  calc
    _ ≤ (1 - r ^ 2) * (2 + 2 * |r|) :=
      mul_le_mul_of_nonneg_left hi (sub_nonneg.mpr hr2)
    _ = 2 * (1 - |r| ^ 2) * (1 + |r|) := by rw [sq_abs]; ring
    _ ≤ 5 / 2 := by nlinarith [sq_nonneg (|r| - 1 / 2), pow_nonneg (abs_nonneg r) 3]

/-- The actual original noise density factors through the original overlap
and barycenter correlations. Source: arXiv:2604.01978v1, §7,
the bracket calculation after eq:Gronwall_E_selfcontained. -/
theorem integral_overlapNoiseIntegrand_sq_factor {d n : ℕ} (β : ℝ) {σV σA : ℝ≥0}
    {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) (i j : Idx n) :
    (∫ θ, overlapNoiseIntegrand β ρ X i j θ ^ 2 ∂ρ) =
      (σV : ℝ) ^ 2 * (1 - inner (𝕜 := ℝ) (X i) (X j) ^ 2) *
        (baryCorr β ρ (empMeasure X) (X i) (X i) + baryCorr β ρ (empMeasure X) (X j) (X j) -
          2 * inner (𝕜 := ℝ) (X i) (X j) * baryCorr β ρ (empMeasure X) (X i) (X j)) := by
  rw [integral_overlapNoiseIntegrand_sq_gaussian β hρ X hX,
    inner_mutual_tangent_proj (X i) (X j) (hX i) (hX j)]
  simp only [norm_proj_sq (hX i), norm_proj_sq (hX j), hX i, hX j, one_pow,
    real_inner_comm (X i) (X j)]
  ring

/-- The actual noise density is bounded by (5/2)σV².
Source: arXiv:2604.01978v1, §7, the bracket estimate. This sharpens the
source's coefficient four by retaining the actual mutual projections. -/
theorem integral_overlapNoiseIntegrand_sq_le {d n : ℕ} (β : ℝ) {σV σA : ℝ≥0}
    {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) (i j : Idx n) :
    (∫ θ, overlapNoiseIntegrand β ρ X i j θ ^ 2 ∂ρ) ≤ (5 / 2 : ℝ) * (σV : ℝ) ^ 2 := by
  have : IsProbabilityMeasure ρ := isProbabilityMeasure_of_isGaussianHeadLaw hρ
  have hr : |inner (𝕜 := ℝ) (X i) (X j)| ≤ 1 := by
    simpa [hX i, hX j] using abs_real_inner_le_norm (X i) (X j)
  have hP := (le_abs_self _).trans (abs_baryCorr_empMeasure_le_one β ρ X hX i i)
  have hQ := (le_abs_self _).trans (abs_baryCorr_empMeasure_le_one β ρ X hX j j)
  have hS := abs_baryCorr_empMeasure_le_one β ρ X hX i j
  have hp := overlap_noise_polynomial_le_five_halves _ _ _ _ hr hP hQ hS
  rw [integral_overlapNoiseIntegrand_sq_factor β hρ X hX]
  calc
    _ = (σV : ℝ) ^ 2 * ((1 - inner (𝕜 := ℝ) (X i) (X j) ^ 2) *
        (baryCorr β ρ (empMeasure X) (X i) (X i) + baryCorr β ρ (empMeasure X) (X j) (X j) -
          2 * inner (𝕜 := ℝ) (X i) (X j) * baryCorr β ρ (empMeasure X) (X i) (X j))) := by ring
    _ ≤ (σV : ℝ) ^ 2 * (5 / 2) := mul_le_mul_of_nonneg_left hp (sq_nonneg _)
    _ = _ := by ring

/-- Standard scaling gives the actual noise density bound 3/d.
Source: arXiv:2604.01978v1, §7, the bracket estimate after
eq:Gronwall_E_selfcontained, with the source's σV²=1/d specialization. -/
theorem integral_overlapNoiseIntegrand_sq_le_three_div_dim {d n : ℕ} (β : ℝ) {σV σA : ℝ≥0}
    {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ))
    (X : Idx n → EucSpace d) (hX : ∀ k, ‖X k‖ = 1) (i j : Idx n) :
    (∫ θ, overlapNoiseIntegrand β ρ X i j θ ^ 2 ∂ρ) ≤ 3 / (d : ℝ) := by
  have h := integral_overlapNoiseIntegrand_sq_le β hρ X hX i j
  rw [hσ] at h
  apply h.trans
  calc
    _ = (5 / 2 : ℝ) / (d : ℝ) := by ring
    _ ≤ 3 / (d : ℝ) := div_le_div_of_nonneg_right (by norm_num) (Nat.cast_nonneg d)

/-- Integrating the actual noise density gives the upper bound 3T/d.
Source: arXiv:2604.01978v1, §7, the bracket estimate. The source displays
4T/d as an equality after bounding; the corrected assertion here is
an upper bound, with an improved constant and actual time integrability. -/
theorem integral_overlapNoise_energy_le {d n : ℕ} (β : ℝ) {σV σA : ℝ≥0}
    {ρ : Measure (HeadParam d)} (hρ : IsGaussianHeadLaw d σV σA ρ)
    (hσ : (σV : ℝ) ^ 2 = 1 / (d : ℝ))
    (X : ℝ → Idx n → EucSpace d) (T : ℝ) (hT : 0 ≤ T)
    (hX : ∀ s ∈ Icc (0 : ℝ) T, ∀ k, ‖X s k‖ = 1) (i j : Idx n)
    (hInt : IntervalIntegrable (fun s => ∫ θ, overlapNoiseIntegrand β ρ (X s) i j θ ^ 2 ∂ρ) volume 0 T) :
    (∫ s in (0 : ℝ)..T, ∫ θ, overlapNoiseIntegrand β ρ (X s) i j θ ^ 2 ∂ρ) ≤ 3 * T / (d : ℝ) := by
  calc
    _ ≤ ∫ _ in (0 : ℝ)..T, (3 / (d : ℝ)) :=
      intervalIntegral.integral_mono_on hT hInt intervalIntegrable_const
        (fun s hs => integral_overlapNoiseIntegrand_sq_le_three_div_dim β hρ hσ (X s) (hX s hs) i j)
    _ = 3 * T / (d : ℝ) := by rw [intervalIntegral.integral_const, sub_zero, smul_eq_mul]; ring

/-- The orthonormal two-token configuration with uniform attention has
exact actual noise density σV². Source: arXiv:2604.01978v1, §7,
the Gaussian bracket calculation, specialized to zero attention variance. -/
theorem integral_overlapNoiseIntegrand_sq_basis_zero_attention (β : ℝ) (σV : ℝ≥0) :
    (∫ θ, overlapNoiseIntegrand β (gaussHeadLaw 2 σV 0)
      (fun k : Idx 2 => (EuclideanSpace.single k 1 : EucSpace 2)) 0 1 θ ^ 2 ∂(gaussHeadLaw 2 σV 0)) =
        (σV : ℝ) ^ 2 := by
  let X : Idx 2 → EucSpace 2 := fun k => EuclideanSpace.single k 1
  have hX : ∀ k, ‖X k‖ = 1 := fun k => by simp [X]
  have hb (k : Idx 2) : baryCorr β (gaussHeadLaw 2 σV 0) (empMeasure X) (X k) (X k) = 1 / 2 := by
    rw [baryCorr, integral_snd_gaussHeadLaw_zero 2 σV
      (fun A => inner (𝕜 := ℝ) (softBary β A (empMeasure X) (X k)) (softBary β A (empMeasure X) (X k)))]
    norm_num [softBary_empMeasure, attnProb, attnWeight, qkMap, X, Fin.sum_univ_two,
      sum_inner, inner_sum, real_inner_smul_left, real_inner_smul_right,
      EuclideanSpace.inner_single_left, PiLp.single_apply]
    rw [EuclideanSpace.real_norm_sq_eq]
    norm_num [Fin.sum_univ_two, PiLp.single_apply]
  change (∫ θ, overlapNoiseIntegrand β (gaussHeadLaw 2 σV 0) X 0 1 θ ^ 2 ∂(gaussHeadLaw 2 σV 0)) = _
  rw [integral_overlapNoiseIntegrand_sq_factor β (isGaussianHeadLaw_gaussHeadLaw 2 σV 0) X hX, hb 0, hb 1]
  norm_num [X, EuclideanSpace.inner_single_left, PiLp.single_apply]

/-- Standard positive value variance, original Gaussian heads and unit basis
tokens have density 1/2 and genuine time integrability. -/
example : ∃ (σV : ℝ≥0) (X : Idx 2 → EucSpace 2),
    (σV : ℝ) ^ 2 = 1 / (2 : ℝ) ∧ IsGaussianHeadLaw 2 σV 0 (gaussHeadLaw 2 σV 0) ∧
    (∀ k, ‖X k‖ = 1) ∧ |inner (𝕜 := ℝ) (X 0) (X 1)| ≤ 1 ∧
    (∫ θ, overlapNoiseIntegrand 1 (gaussHeadLaw 2 σV 0) X 0 1 θ ^ 2 ∂(gaussHeadLaw 2 σV 0)) = 1 / 2 ∧
    IntervalIntegrable (fun _ : ℝ =>
      ∫ θ, overlapNoiseIntegrand 1 (gaussHeadLaw 2 σV 0) X 0 1 θ ^ 2 ∂(gaussHeadLaw 2 σV 0)) volume 0 1 := by
  let X : Idx 2 → EucSpace 2 := fun k => EuclideanSpace.single k 1
  refine ⟨stdSigmaV 2, X, stdSigmaV_sq 2, isGaussianHeadLaw_gaussHeadLaw 2 _ 0,
    fun k => by simp [X], ?_, ?_, intervalIntegrable_const⟩
  · norm_num [X, EuclideanSpace.inner_single_left, PiLp.single_apply]
  · rw [show X = (fun k : Idx 2 => (EuclideanSpace.single k 1 : EucSpace 2)) from rfl,
      integral_overlapNoiseIntegrand_sq_basis_zero_attention, stdSigmaV_sq]
    norm_num

/-- These nonzero correlation data satisfy the scalar bound premises. -/
example : |(0 : ℝ)| ≤ 1 ∧ (1 / 2 : ℝ) ≤ 1 ∧ |(1 / 2 : ℝ)| ≤ 1 := by norm_num
end Transformer.Homogenized
