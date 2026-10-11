/-
# The actual attention scores are governed by the leading eigencoordinate

ArXiv:2305.05465v6, §9, `e:majorizeweights`, in the proof of
`l:3hyperplanes11`. The attention form is allowed to be nonsymmetric
and indefinite away from the leading line. Its leading coefficient
`c₁₁ = ⟨Q φ, K φ⟩` is positive under `d:good`.

The proved invariant splitting bounds both normalized complementary
vectors by a common exponentially decaying factor. Expanding the
bilinear form leaves only two cross terms and a quadratic error.
Thus the normalized actual score differs from `c₁₁ f(z_i) f(z_j)`
by an exponentially small error, without an eigenbasis assumption.

The final equality rewrites the original attention weights as a
softmax of these normalized scores at inverse temperature `exp(2λt)`.
No positive attention square root is introduced for this setting.
-/

import Transformer.Clusters.Section9_LeadingProjection
import Transformer.Clusters.Section10_ScoreError

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

variable {d n : ℕ}

/-- An arbitrary bilinear attention form has exponentially small
normalized error from its leading projection. Source: arXiv:2305.05465v6,
§9, `e:majorizeweights`, and `r:notdiagonalizable`. -/
theorem RescaledDynamics.exists_bound_normalized_bilinear_error
    (Q K V proj : ParamMatrix d) (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hV : IsParanormalOn V F G lam) (hproj : IsProjOnto proj F G)
    (Z : ℝ → Idx n → EucSpace d) (hZ : RescaledDynamics Q K V Z) :
    ∃ δ D : ℝ, 0 < δ ∧ 0 < D ∧ ∀ t : ℝ, 0 ≤ t → ∀ i j : Idx n,
      |inner (𝕜 := ℝ) (Q (Real.exp (-lam * t) • expTime V t (Z t i)))
          (K (Real.exp (-lam * t) • expTime V t (Z t j))) -
        inner (𝕜 := ℝ) (Q (proj (Z t i))) (K (proj (Z t j)))| ≤
        D * Real.exp (-δ * t) := by
  obtain ⟨δ, R, C, hδ, hR, hC, hleading, herror⟩ :=
    hZ.exists_bound_attention_normalization Q K V proj 1 F G lam hV hproj Z
  let M := ‖Q‖ + ‖K‖ + 1
  have hM : 0 < M := by dsimp only [M]; positivity
  have hQM : ‖Q‖ ≤ M := by dsimp only [M]; linarith [norm_nonneg K]
  have hKM : ‖K‖ ≤ M := by dsimp only [M]; linarith [norm_nonneg Q]
  refine ⟨δ, 2 * (M * R) * (M * C) + (M * C) ^ 2, hδ, by positivity, ?_⟩
  intro t ht i j
  let N : Idx n → EucSpace d := fun k => Real.exp (-lam * t) • expTime V t (Z t k)
  have hlead (A : ParamMatrix d) (hA : ‖A‖ ≤ M) (k : Idx n) :
      ‖A (proj (Z t k))‖ ≤ M * R := by
    have hbound : ‖proj (Z t k)‖ ≤ R := by
      simpa only [one_apply_eq_self] using hleading t ht k
    exact (A.le_opNorm _).trans (mul_le_mul hA hbound (norm_nonneg _) hM.le)
  have herr (A : ParamMatrix d) (hA : ‖A‖ ≤ M) (k : Idx n) :
      ‖A (N k) - A (proj (Z t k))‖ ≤ (M * C) * Real.exp (-δ * t) := by
    have hbound : ‖N k - proj (Z t k)‖ ≤ C * Real.exp (-δ * t) := by
      simpa only [one_apply_eq_self] using herror t ht k
    rw [← map_sub]
    exact (A.le_opNorm _).trans ((mul_le_mul hA hbound (norm_nonneg _) hM.le).trans_eq
      (by ring))
  have hsum (x y : EucSpace d) : x + (y - x) = y := by abel
  simpa only [hsum] using abs_inner_add_error_le_of_bounds
    (Q (proj (Z t i))) (K (proj (Z t j)))
    (Q (N i) - Q (proj (Z t i))) (K (N j) - K (proj (Z t j)))
    (M * R) (M * C) (Real.exp (-δ * t)) (by positivity) (by positivity)
    (Real.exp_pos _).le (Real.exp_le_one_iff.mpr (by nlinarith))
    (hlead Q hQM i) (hlead K hKM j) (herr Q hQM i) (herr K hKM j)

example : IsParanormalOn (1 : ParamMatrix 1) ⊤ ⊥ 1 ∧
    IsProjOnto (1 : ParamMatrix 1) ⊤ ⊥ ∧
    RescaledDynamics (1 : ParamMatrix 1) 1 1
      (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) :=
  ⟨(isGoodTripleMulti_one 1).2, isProjOnto_one 1, rescaledDynamics_one_const 1 1 _⟩

/-- **The normalized leading score estimate for the original good-triple
hypotheses.** Source: arXiv:2305.05465v6, §9, `e:majorizeweights`.
The strict spectral gap gives some `δ > 0`, including nondiagonalizable
complements. No global positivity or symmetry of `QᵀK` is assumed. -/
theorem RescaledDynamics.exists_bound_simple_leading_score_error
    (Q K V : ParamMatrix d) (lam : ℝ) (φ : EucSpace d)
    (G : Submodule ℝ (EucSpace d)) (hgood : IsGoodTripleWith Q K V lam φ G)
    (f : EucSpace d →L[ℝ] ℝ) (hf : IsEigenFunctional V f lam) (hfφ : f φ = 1)
    (Z : ℝ → Idx n → EucSpace d) (hZ : RescaledDynamics Q K V Z) :
    ∃ δ D : ℝ, 0 < δ ∧ 0 < D ∧ ∀ t : ℝ, 0 ≤ t → ∀ i j : Idx n,
      |inner (𝕜 := ℝ) (Q (Real.exp (-lam * t) • expTime V t (Z t i)))
          (K (Real.exp (-lam * t) • expTime V t (Z t j))) -
        inner (𝕜 := ℝ) (Q φ) (K φ) * f (Z t i) * f (Z t j)| ≤
        D * Real.exp (-δ * t) := by
  obtain ⟨δ, D, hδ, hD, hbound⟩ := hZ.exists_bound_normalized_bilinear_error Q K V
    (f.smulRight φ) (Submodule.span ℝ {φ}) G lam (hgood.isParanormalOn Q K V lam φ G)
    (hgood.isProjOnto_smulRight Q K V lam φ G f hf hfφ) Z
  refine ⟨δ, D, hδ, hD, fun t ht i j => ?_⟩
  have hleading : inner (𝕜 := ℝ) (Q ((f.smulRight φ) (Z t i)))
      (K ((f.smulRight φ) (Z t j))) = inner (𝕜 := ℝ) (Q φ) (K φ) * f (Z t i) * f (Z t j) := by
    simp only [ContinuousLinearMap.smulRight_apply, map_smul,
      real_inner_smul_left, real_inner_smul_right]
    ring
  simpa only [hleading] using hbound t ht i j

example : IsGoodTripleWith (1 : ParamMatrix 1) 1 1 1
    (EuclideanSpace.single 0 (1 : ℝ)) ⊥ ∧
    IsEigenFunctional (1 : ParamMatrix 1) (EuclideanSpace.proj (𝕜 := ℝ) 0) 1 ∧
    (EuclideanSpace.proj (𝕜 := ℝ) 0) (EuclideanSpace.single 0 (1 : ℝ)) = 1 ∧
    RescaledDynamics (1 : ParamMatrix 1) 1 1
      (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) :=
  ⟨isGoodTripleWith_one, isEigenFunctional_one _, by simp, rescaledDynamics_one_const 1 1 _⟩

/-- **The raw leading score estimate, with a strict gap below `2λ`.**
Source: arXiv:2305.05465v6, §9, `e:majorizeweights` and
`r:notdiagonalizable`. Some `δ > 0` replaces the exact second-eigenvalue
exponent, as permitted by the source for a nondiagonalizable complement. -/
theorem RescaledDynamics.exists_bound_simple_score_error
    (Q K V : ParamMatrix d) (lam : ℝ) (φ : EucSpace d)
    (G : Submodule ℝ (EucSpace d)) (hgood : IsGoodTripleWith Q K V lam φ G)
    (f : EucSpace d →L[ℝ] ℝ) (hf : IsEigenFunctional V f lam) (hfφ : f φ = 1)
    (Z : ℝ → Idx n → EucSpace d) (hZ : RescaledDynamics Q K V Z) :
    ∃ δ D : ℝ, 0 < δ ∧ 0 < D ∧ ∀ t : ℝ, 0 ≤ t → ∀ i j : Idx n,
      |inner (𝕜 := ℝ) (Q (expTime V t (Z t i))) (K (expTime V t (Z t j))) -
        inner (𝕜 := ℝ) (Q φ) (K φ) * Real.exp (2 * lam * t) * f (Z t i) * f (Z t j)| ≤
        D * Real.exp ((2 * lam - δ) * t) := by
  obtain ⟨δ, D, hδ, hD, hbound⟩ :=
    hZ.exists_bound_simple_leading_score_error Q K V lam φ G hgood f hf hfφ Z
  refine ⟨δ, D, hδ, hD, fun t ht i j => ?_⟩
  have hexp : Real.exp (2 * lam * t) * Real.exp (-lam * t) * Real.exp (-lam * t) = 1 := by
    rw [← Real.exp_add, ← Real.exp_add]
    have hz : 2 * lam * t + -lam * t + -lam * t = 0 := by ring
    rw [hz, Real.exp_zero]
  have hscale : inner (𝕜 := ℝ) (Q (expTime V t (Z t i))) (K (expTime V t (Z t j))) -
        inner (𝕜 := ℝ) (Q φ) (K φ) * Real.exp (2 * lam * t) * f (Z t i) * f (Z t j) =
      Real.exp (2 * lam * t) *
        (inner (𝕜 := ℝ) (Q (Real.exp (-lam * t) • expTime V t (Z t i)))
          (K (Real.exp (-lam * t) • expTime V t (Z t j))) -
          inner (𝕜 := ℝ) (Q φ) (K φ) * f (Z t i) * f (Z t j)) := by
    simp only [map_smul, real_inner_smul_left, real_inner_smul_right]
    calc
      _ = (Real.exp (2 * lam * t) * Real.exp (-lam * t) * Real.exp (-lam * t)) *
          inner (𝕜 := ℝ) (Q (expTime V t (Z t i))) (K (expTime V t (Z t j))) -
          inner (𝕜 := ℝ) (Q φ) (K φ) * Real.exp (2 * lam * t) * f (Z t i) * f (Z t j) := by
        rw [hexp, one_mul]
      _ = _ := by ring
  rw [hscale, abs_mul, abs_of_pos (Real.exp_pos _)]
  calc
    _ ≤ Real.exp (2 * lam * t) * (D * Real.exp (-δ * t)) :=
      mul_le_mul_of_nonneg_left (hbound t ht i j) (Real.exp_pos _).le
    _ = D * (Real.exp (2 * lam * t) * Real.exp (-δ * t)) := by ring
    _ = D * Real.exp ((2 * lam - δ) * t) := by rw [← Real.exp_add]; congr 2; ring

example : IsGoodTripleWith (1 : ParamMatrix 1) 1 1 1
    (EuclideanSpace.single 0 (1 : ℝ)) ⊥ ∧
    IsEigenFunctional (1 : ParamMatrix 1) (EuclideanSpace.proj (𝕜 := ℝ) 0) 1 ∧
    (EuclideanSpace.proj (𝕜 := ℝ) 0) (EuclideanSpace.single 0 (1 : ℝ)) = 1 ∧
    RescaledDynamics (1 : ParamMatrix 1) 1 1
      (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) :=
  ⟨isGoodTripleWith_one, isEigenFunctional_one _, by simp, rescaledDynamics_one_const 1 1 _⟩

/-- The original attention weights are exactly the softmax of the
normalized bilinear scores at temperature factor `exp(2λt)`.
Source: arXiv:2305.05465v6, §9, `w_j` in `eq:phistarvar`. -/
theorem attentionMatrix_eq_normalized_bilinear_softmax (Q K : ParamMatrix d)
    (X : Idx n → EucSpace d) (lam t : ℝ) (i j : Idx n) :
    attentionMatrix Q K X i j = Perspective.softmaxWeight
      (fun k => Real.exp (2 * lam * t) * inner (𝕜 := ℝ)
        (Q (Real.exp (-lam * t) • X i)) (K (Real.exp (-lam * t) • X k))) j := by
  unfold attentionMatrix
  apply congrArg (fun w : Idx n → ℝ => Perspective.softmaxWeight w j)
  funext k
  rw [map_smul, map_smul, real_inner_smul_left, real_inner_smul_right]
  have hexp : Real.exp (2 * lam * t) * Real.exp (-lam * t) * Real.exp (-lam * t) = 1 := by
    rw [← Real.exp_add, ← Real.exp_add]
    have hz : 2 * lam * t + -lam * t + -lam * t = 0 := by ring
    rw [hz, Real.exp_zero]
  calc
    _ = (Real.exp (2 * lam * t) * Real.exp (-lam * t) * Real.exp (-lam * t)) *
        inner (𝕜 := ℝ) (Q (X i)) (K (X k)) := by rw [hexp, one_mul]
    _ = _ := by ring

end Transformer.Clusters
