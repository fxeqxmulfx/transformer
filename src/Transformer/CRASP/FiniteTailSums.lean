/-
# A stabilized distance sum is a prefix count plus a finite window correction

arXiv:2506.16055v3, Appendix F, `thm:rtfr_to_TLCly`.
The stable tail contributes a weighted count of all source states. Only the
finitely many smaller distances require reading previous-position states.
-/

import Transformer.CRASP.AlibiTables

namespace Transformer.CRASP

universe u
variable {α : Type u}

/-- Reverse the distance index inside a finite prefix (Appendix F). -/
theorem sum_prefix_reverse (i : ℕ) (f : ℕ → ℤ) :
    (∑ j ∈ Finset.Icc 0 i, f j) = ∑ δ ∈ Finset.Icc 0 i, f (i - δ) := by
  refine Finset.sum_bij (fun j _ => i - j) ?_ ?_ ?_ ?_
  · intro j hj
    exact Finset.mem_Icc.mpr ⟨Nat.zero_le _, Nat.sub_le _ _⟩
  · intro j hj t ht heq
    have hjb := Finset.mem_Icc.mp hj
    have htb := Finset.mem_Icc.mp ht
    omega
  · intro δ hδ
    have hδb := Finset.mem_Icc.mp hδ
    refine ⟨i - δ, Finset.mem_Icc.mpr ⟨Nat.zero_le _, Nat.sub_le _ _⟩, ?_⟩
    omega
  · intro j hj
    have hjb := Finset.mem_Icc.mp hj
    congr 1
    omega

/-- Stable coefficients give a full prefix sum plus only a finite correction.
Source: arXiv:2506.16055v3, Appendix F, `thm:rtfr_to_TLCly`, with the
rounding-before-summing convention of Appendix B.1 preserved. -/
theorem sum_stable_tail (f : ℕ → α → ℤ) (B : α → ℤ) (Δ : ℕ)
    (hB : ∀ δ, Δ ≤ δ → ∀ a, f δ a = B a) (q : ℕ → α) (i : ℕ) :
    (∑ j ∈ Finset.Icc 0 i, f (i - j) (q j)) =
      B (q 0) + (∑ j ∈ Finset.Icc 1 i, B (q j)) +
        ∑ δ ∈ Finset.range Δ, if δ ≤ i then
          f δ (q (i - δ)) - B (q (i - δ)) else 0 := by
  let g := fun δ => f δ (q (i - δ)) - B (q (i - δ))
  have hsplit : (∑ j ∈ Finset.Icc 0 i, f (i - j) (q j)) =
      (∑ j ∈ Finset.Icc 0 i, B (q j)) +
        ∑ j ∈ Finset.Icc 0 i, (f (i - j) (q j) - B (q j)) := by
    rw [Finset.sum_sub_distrib]
    ring
  have hrev : (∑ j ∈ Finset.Icc 0 i, (f (i - j) (q j) - B (q j))) =
      ∑ δ ∈ Finset.Icc 0 i, g δ := by
    rw [sum_prefix_reverse i (fun j => f (i - j) (q j) - B (q j))]
    apply Finset.sum_congr rfl
    intro δ hδ
    have hb := Finset.mem_Icc.mp hδ
    simp only [g, show i - (i - δ) = δ by omega]
  let S := (Finset.range Δ).filter (fun δ => δ ≤ i)
  have hS : S ⊆ Finset.Icc 0 i := by
    intro δ hδ
    exact Finset.mem_Icc.mpr ⟨Nat.zero_le _, (Finset.mem_filter.mp hδ).2⟩
  have hz : ∀ δ ∈ Finset.Icc 0 i, δ ∉ S → g δ = 0 := by
    intro δ hδ hn
    have hδb := Finset.mem_Icc.mp hδ
    have hd : Δ ≤ δ := by
      by_contra hc
      apply hn
      exact Finset.mem_filter.mpr ⟨Finset.mem_range.mpr (by omega), hδb.2⟩
    simp [g, hB δ hd]
  have hw : (∑ δ ∈ Finset.Icc 0 i, g δ) =
      ∑ δ ∈ Finset.range Δ, if δ ≤ i then g δ else 0 := by
    rw [← Finset.sum_subset hS hz]
    exact Finset.sum_filter _ _
  have hprefix : (∑ j ∈ Finset.Icc 0 i, B (q j)) =
      B (q 0) + ∑ j ∈ Finset.Icc 1 i, B (q j) := by
    have hs : Finset.Icc 0 i = insert 0 (Finset.Icc 1 i) := by
      ext j
      simp only [Finset.mem_Icc, Finset.mem_insert]
      omega
    rw [hs, Finset.sum_insert (by simp)]
  rw [hsplit, hrev, hw, hprefix]

/-- The stable-tail hypothesis has a nonzero constant witness (Appendix F). -/
example : ∀ δ : ℕ, 1 ≤ δ → ∀ a : Bool,
    (fun (_ : ℕ) (b : Bool) => if b then (3 : ℤ) else -2) δ a =
      (if a then (3 : ℤ) else -2) := fun _ _ _ => rfl

end Transformer.CRASP
