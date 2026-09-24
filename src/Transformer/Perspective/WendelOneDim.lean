/-
# Wendel's formula in dimension one

The one-dimensional unit sphere has two points of equal probability.
The hemisphere event is the union of the two constant configurations.

Source: arXiv:2312.10794v5, §6.1, `r:wendel` (Wendel 1962).
-/

import Transformer.Perspective.WendelOneDimBasic

open MeasureTheory

namespace Transformer.Perspective

/-- A one-dimensional sample lies in an open hemisphere precisely when all
points have the same sign. Source: arXiv:2312.10794v5, §6.1, `r:wendel`. -/
theorem hemisphere_one_iff (n : ℕ) (X : SphereTuple 1 n) :
    (∃ w : SSphere 1, ∀ i : Idx n,
        0 < inner (𝕜 := ℝ) ((X i : EucSpace 1)) ((w : EucSpace 1))) ↔
      X = (fun _ => eOne) ∨ X = (fun _ => eNeg) := by
  constructor
  · rintro ⟨w, hw⟩
    rcases sphere_one_cases w with rfl | rfl
    · left
      funext i
      rcases sphere_one_cases (X i) with hi | hi
      · exact hi
      · have hpos := hw i
        rw [hi, inner_eNeg_eOne] at hpos
        norm_num at hpos
    · right
      funext i
      rcases sphere_one_cases (X i) with hi | hi
      · have hpos := hw i
        rw [hi, inner_eOne_eNeg] at hpos
        norm_num at hpos
      · exact hi
  · rintro (h | h)
    · refine ⟨eOne, fun i => ?_⟩
      rw [h, inner_eOne_eOne]
      norm_num
    · refine ⟨eNeg, fun i => ?_⟩
      rw [h, inner_eNeg_eNeg]
      norm_num

/-- **Theorem `r:wendel` in dimension one.** For every `n ≥ 1`, the
probability of a common open hemisphere is `2^{-(n-1)}`.

Source: arXiv:2312.10794v5, §6.1, `r:wendel` (Wendel 1962). -/
theorem wendel_one_dim (n : ℕ) (hn : 1 ≤ n) :
    ∀ P : Measure (SphereTuple 1 n), UniformTuple 1 n P →
      (P { X₀ : SphereTuple 1 n | ∃ w : SSphere 1, ∀ i : Idx n,
            0 < inner (𝕜 := ℝ) ((X₀ i : EucSpace 1)) ((w : EucSpace 1)) }).toReal
        = (∑ k ∈ Finset.range 1, ((n - 1).choose k : ℝ)) / 2 ^ (n - 1) := by
  rintro P ⟨σ, hσprob, hσinv, rfl⟩
  let _ : IsProbabilityMeasure σ := hσprob
  let fOne : SphereTuple 1 n := fun _ => eOne
  let fNeg : SphereTuple 1 n := fun _ => eNeg
  have hne : fOne ≠ fNeg := by
    intro h
    exact eOne_ne_eNeg (congrFun h ⟨0, hn⟩)
  have hevent :
      { X₀ : SphereTuple 1 n | ∃ w : SSphere 1, ∀ i : Idx n,
          0 < inner (𝕜 := ℝ) ((X₀ i : EucSpace 1)) ((w : EucSpace 1)) } =
        {fOne} ∪ {fNeg} := by
    ext X
    simpa only [Set.mem_ofPred_eq, Set.mem_union, Set.mem_singleton_iff, fOne, fNeg] using
      (hemisphere_one_iff n X)
  have hmeasure :
      (Measure.pi fun _ : Idx n => σ)
          { X₀ : SphereTuple 1 n | ∃ w : SSphere 1, ∀ i : Idx n,
              0 < inner (𝕜 := ℝ) ((X₀ i : EucSpace 1)) ((w : EucSpace 1)) } =
        (Measure.pi fun _ : Idx n => σ) {fOne} +
          (Measure.pi fun _ : Idx n => σ) {fNeg} := by
    rw [hevent]
    exact measure_union (by simp [hne]) (measurableSet_singleton fNeg)
  have hhalf := sphere_one_atom_half σ hσinv
  have hprod (x : SSphere 1) (hx : (σ {x}).toReal = 1 / 2) :
      ((Measure.pi fun _ : Idx n => σ) {fun _ => x}).toReal = (1 / 2 : ℝ) ^ n := by
    rw [Measure.pi_singleton]
    simp only [Finset.prod_const, Finset.card_univ, Fintype.card_fin,
      ENNReal.toReal_pow, hx]
  have hfin1 : (Measure.pi fun _ : Idx n => σ) {fOne} ≠ ⊤ := measure_ne_top _ _
  have hfin2 : (Measure.pi fun _ : Idx n => σ) {fNeg} ≠ ⊤ := measure_ne_top _ _
  rw [hmeasure, ENNReal.toReal_add hfin1 hfin2]
  change ((Measure.pi fun _ : Idx n => σ) {fun _ => eOne}).toReal +
      ((Measure.pi fun _ : Idx n => σ) {fun _ => eNeg}).toReal = _
  rw [hprod eOne hhalf.1, hprod eNeg hhalf.2]
  simp only [Finset.range_one, Finset.sum_singleton, Nat.choose_zero_right, Nat.cast_one]
  obtain ⟨m, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (by omega : n ≠ 0)
  simp only [Nat.succ_sub_one, pow_succ]
  rw [one_div_pow]
  field_simp
  ring

/-- The dimension and sample-size hypotheses of `wendel_one_dim` are
satisfiable with `n = 1`. -/
example : 1 ≤ 1 := le_rfl

end Transformer.Perspective
