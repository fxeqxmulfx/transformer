module

public import Transformer.Clusters.Section10_ComplementaryGrowth
public import Transformer.Clusters.Section10_LeadingFlow
public import Transformer.Clusters.Section10_ProjectedHullLimit

/-!
# Exponentially small error in the normalized leading scores

ArXiv:2305.05465v6, §10, the score decomposition in Step 2' of
`t:multiplicity`. The leading vectors entering the attention form are
`A π_F z_i`, in that order. Their difference from the normalized actual
unrescaled vectors is exponentially small by the complementary growth
bound and the exact exponential splitting.

The manuscript instead writes `π_F A z_i` in its auxiliary estimate;
`Section10_RemainderFalse` gives a counterexample to that remainder bound.
Here the order is corrected from the original attention scores themselves.
No commutation of the attention root and the projection is assumed.
-/

@[expose] public section

open scoped BigOperators
open Real Filter Topology

namespace Transformer.Clusters

variable {d n : ℕ}

/-- Expanding the inner product bounds its perturbation by the three
cross terms. Source: arXiv:2305.05465v6, §10, the attention exponent
decomposition in Step 2', with the leading term in the corrected order. -/
theorem abs_inner_add_error_le (u v e f : EucSpace d) :
    |inner (𝕜 := ℝ) (u + e) (v + f) - inner (𝕜 := ℝ) u v| ≤
      ‖u‖ * ‖f‖ + ‖e‖ * ‖v‖ + ‖e‖ * ‖f‖ := by
  have hexpand : inner (𝕜 := ℝ) (u + e) (v + f) - inner (𝕜 := ℝ) u v =
      inner (𝕜 := ℝ) u f + inner (𝕜 := ℝ) e v + inner (𝕜 := ℝ) e f := by
    simp only [inner_add_left, inner_add_right]
    ring
  rw [hexpand]
  exact (abs_add_three _ _ _).trans
    (add_le_add (add_le_add (abs_real_inner_le_norm u f) (abs_real_inner_le_norm e v))
      (abs_real_inner_le_norm e f))

/-- Bounded leading vectors and small complementary vectors give a
uniform score error proportional to their decay factor.
Source: arXiv:2305.05465v6, §10, Step 2', corrected leading score. -/
theorem abs_inner_add_error_le_of_bounds (u v e f : EucSpace d) (R C ε : ℝ)
    (hR : 0 ≤ R) (hC : 0 ≤ C) (hε : 0 ≤ ε) (hεone : ε ≤ 1)
    (hu : ‖u‖ ≤ R) (hv : ‖v‖ ≤ R) (he : ‖e‖ ≤ C * ε) (hf : ‖f‖ ≤ C * ε) :
    |inner (𝕜 := ℝ) (u + e) (v + f) - inner (𝕜 := ℝ) u v| ≤
      (2 * R * C + C ^ 2) * ε := by
  have hεsq : ε ^ 2 ≤ ε := by nlinarith
  have hlast : (C * ε) * (C * ε) ≤ C ^ 2 * ε := by
    calc
      (C * ε) * (C * ε) = C ^ 2 * ε ^ 2 := by ring
      _ ≤ C ^ 2 * ε := mul_le_mul_of_nonneg_left hεsq (sq_nonneg C)
  calc
    _ ≤ ‖u‖ * ‖f‖ + ‖e‖ * ‖v‖ + ‖e‖ * ‖f‖ := abs_inner_add_error_le u v e f
    _ ≤ R * (C * ε) + (C * ε) * R + (C * ε) * (C * ε) := by gcongr
    _ ≤ R * (C * ε) + (C * ε) * R + C ^ 2 * ε := add_le_add le_rfl hlast
    _ = (2 * R * C + C ^ 2) * ε := by ring

/-- All scalar and vector bounds hold simultaneously for zero vectors
and a unit decay factor. Source: arXiv:2305.05465v6, §10, Step 2'. -/
example : (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 1 ∧
    ‖(0 : EucSpace d)‖ ≤ (1 : ℝ) ∧ ‖(0 : EucSpace d)‖ ≤ (1 : ℝ) ∧
    ‖(0 : EucSpace d)‖ ≤ (1 : ℝ) * 1 ∧ ‖(0 : EucSpace d)‖ ≤ (1 : ℝ) * 1 := by
  simp only [norm_zero, one_mul, le_refl, zero_le_one, and_self]

/-- The attention map applied to the leading tokens stays bounded,
and its difference from the normalized actual tokens decays exponentially.
Source: arXiv:2305.05465v6, §10, the attention exponent decomposition.
Both estimates come from the actual ODE and the paranormal splitting. -/
theorem RescaledDynamics.exists_bound_attention_normalization
    (Q K V proj A : ParamMatrix d) (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hV : IsParanormalOn V F G lam) (hproj : IsProjOnto proj F G)
    (Z : ℝ → Idx n → EucSpace d) (hZ : RescaledDynamics Q K V Z) :
    ∃ δ R C : ℝ, 0 < δ ∧ 0 < R ∧ 0 < C ∧
      (∀ t : ℝ, 0 ≤ t → ∀ i : Idx n, ‖A (proj (Z t i))‖ ≤ R) ∧
      ∀ t : ℝ, 0 ≤ t → ∀ i : Idx n,
        ‖A (Real.exp (-lam * t) • expTime V t (Z t i)) - A (proj (Z t i))‖ ≤
          C * Real.exp (-δ * t) := by
  obtain ⟨M, hM, hleading⟩ := hZ.exists_bound_projected Q K V proj F G lam hV hproj Z
  obtain ⟨δ, C, hδ, hC, herror⟩ :=
    hZ.exists_bound_normalized_complementary Q K V proj F G lam hV hproj Z
  refine ⟨δ, ‖A‖ * M + 1, ‖A‖ * C + 1, hδ, by positivity, by positivity, ?_, ?_⟩
  · intro t ht i
    exact (A.le_opNorm _).trans ((mul_le_mul_of_nonneg_left (hleading i t ht)
      (norm_nonneg A)).trans (by linarith))
  · intro t ht i
    have hnorm : ‖Real.exp (-lam * t) • expTime V t (Z t i) - proj (Z t i)‖ ≤
        C * Real.exp (-δ * t) := by
      rw [hproj.norm_normalized_expTime_sub proj V F G lam hV]
      exact herror t ht i
    rw [← map_sub]
    calc
      _ ≤ ‖A‖ * ‖Real.exp (-lam * t) • expTime V t (Z t i) - proj (Z t i)‖ :=
        A.le_opNorm _
      _ ≤ ‖A‖ * (C * Real.exp (-δ * t)) :=
        mul_le_mul_of_nonneg_left hnorm (norm_nonneg A)
      _ ≤ (‖A‖ * C + 1) * Real.exp (-δ * t) := by nlinarith [Real.exp_pos (-δ * t)]

/-- Every dynamical input holds for coincident nonzero tokens, including
the actual ODE. Source: arXiv:2305.05465v6, §3, `e:Rres`, and §10. -/
example : IsParanormalOn (1 : ParamMatrix 1) ⊤ ⊥ 1 ∧
    IsProjOnto (1 : ParamMatrix 1) ⊤ ⊥ ∧
    RescaledDynamics (1 : ParamMatrix 1) 1 1
      (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) :=
  ⟨(isGoodTripleMulti_one 1).2, isProjOnto_one 1, rescaledDynamics_one_const 1 1 _⟩

/-- The normalized actual vectors entering the attention root are
uniformly bounded. Source: arXiv:2305.05465v6, §10, Step 2'; this bound
uses the true leading part and the decaying complementary part together. -/
theorem RescaledDynamics.exists_bound_normalized_attention
    (Q K V proj A : ParamMatrix d) (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hV : IsParanormalOn V F G lam) (hproj : IsProjOnto proj F G)
    (Z : ℝ → Idx n → EucSpace d) (hZ : RescaledDynamics Q K V Z) :
    ∃ M : ℝ, 0 < M ∧ ∀ t : ℝ, 0 ≤ t → ∀ i : Idx n,
      ‖A (Real.exp (-lam * t) • expTime V t (Z t i))‖ ≤ M := by
  obtain ⟨δ, R, C, hδ, hR, hC, hleading, herror⟩ :=
    hZ.exists_bound_attention_normalization Q K V proj A F G lam hV hproj Z
  refine ⟨R + C, add_pos hR hC, fun t ht i => ?_⟩
  have hε : Real.exp (-δ * t) ≤ 1 :=
    Real.exp_le_one_iff.mpr (mul_nonpos_of_nonpos_of_nonneg (neg_nonpos.mpr hδ.le) ht)
  have htriangle : ‖A (Real.exp (-lam * t) • expTime V t (Z t i))‖ ≤
      ‖A (proj (Z t i))‖ +
        ‖A (Real.exp (-lam * t) • expTime V t (Z t i)) - A (proj (Z t i))‖ := by
    have hsum : A (proj (Z t i)) +
        (A (Real.exp (-lam * t) • expTime V t (Z t i)) - A (proj (Z t i))) =
        A (Real.exp (-lam * t) • expTime V t (Z t i)) := by abel
    simpa only [hsum] using norm_add_le (A (proj (Z t i)))
      (A (Real.exp (-lam * t) • expTime V t (Z t i)) - A (proj (Z t i)))
  have hsmall : C * Real.exp (-δ * t) ≤ C := by
    simpa only [mul_one] using mul_le_mul_of_nonneg_left hε hC.le
  exact htriangle.trans (add_le_add (hleading t ht i) ((herror t ht i).trans hsmall))

/-- The normalized-attention bound has an actual nonzero-token instance.
Source: arXiv:2305.05465v6, §3, `e:Rres`, and §10, Step 2'. -/
example : IsParanormalOn (1 : ParamMatrix 1) ⊤ ⊥ 1 ∧
    IsProjOnto (1 : ParamMatrix 1) ⊤ ⊥ ∧
    RescaledDynamics (1 : ParamMatrix 1) 1 1
      (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) :=
  ⟨(isGoodTripleMulti_one 1).2, isProjOnto_one 1, rescaledDynamics_one_const 1 1 _⟩

/-- **The normalized actual attention form has exponentially small error
from the leading form in the correct operator order.**
Source: arXiv:2305.05465v6, §10, Step 2'. The source uses `π_F A`;
the expansion of its original scores gives `A π_F` instead. This corrected
estimate is derived without assuming the false projected-root remainder. -/
theorem RescaledDynamics.exists_bound_normalized_score_error
    (Q K V proj A : ParamMatrix d) (F G : Submodule ℝ (EucSpace d)) (lam : ℝ)
    (hV : IsParanormalOn V F G lam) (hproj : IsProjOnto proj F G)
    (Z : ℝ → Idx n → EucSpace d) (hZ : RescaledDynamics Q K V Z) :
    ∃ δ D : ℝ, 0 < δ ∧ 0 < D ∧ ∀ t : ℝ, 0 ≤ t → ∀ i j : Idx n,
      |inner (𝕜 := ℝ) (A (Real.exp (-lam * t) • expTime V t (Z t i)))
          (A (Real.exp (-lam * t) • expTime V t (Z t j))) -
        inner (𝕜 := ℝ) (A (proj (Z t i))) (A (proj (Z t j)))| ≤
        D * Real.exp (-δ * t) := by
  obtain ⟨δ, R, C, hδ, hR, hC, hleading, herror⟩ :=
    hZ.exists_bound_attention_normalization Q K V proj A F G lam hV hproj Z
  refine ⟨δ, 2 * R * C + C ^ 2, hδ, by positivity, fun t ht i j => ?_⟩
  have hεone : Real.exp (-δ * t) ≤ 1 :=
    Real.exp_le_one_iff.mpr (mul_nonpos_of_nonpos_of_nonneg (neg_nonpos.mpr hδ.le) ht)
  have hsum (x y : EucSpace d) : x + (y - x) = y := by abel
  simpa only [hsum] using abs_inner_add_error_le_of_bounds
    (A (proj (Z t i))) (A (proj (Z t j)))
    (A (Real.exp (-lam * t) • expTime V t (Z t i)) - A (proj (Z t i)))
    (A (Real.exp (-lam * t) • expTime V t (Z t j)) - A (proj (Z t j)))
    R C (Real.exp (-δ * t)) hR.le hC.le (Real.exp_pos _).le hεone
    (hleading t ht i) (hleading t ht j) (herror t ht i) (herror t ht j)

/-- The full score-error inputs hold for an actual stationary rescaled
configuration. Source: arXiv:2305.05465v6, §3, `e:Rres`, and §10, Step 2'. -/
example : IsParanormalOn (1 : ParamMatrix 1) ⊤ ⊥ 1 ∧
    IsProjOnto (1 : ParamMatrix 1) ⊤ ⊥ ∧
    RescaledDynamics (1 : ParamMatrix 1) 1 1
      (fun _ (_ : Idx 2) => (basePoint 0 : EucSpace 1)) :=
  ⟨(isGoodTripleMulti_one 1).2, isProjOnto_one 1, rescaledDynamics_one_const 1 1 _⟩

end Transformer.Clusters
