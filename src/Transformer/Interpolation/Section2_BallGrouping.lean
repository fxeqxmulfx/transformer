module

public import Mathlib.Algebra.BigOperators.Fin
public import Mathlib.Basic.ENNReal.Operations

/-!
# Grouping small masses into prescribed weights

ArXiv:2411.04551v3, §2.2, proof of `prop: compression`, Step 1: the
support is partitioned "into `M` pieces" with `μ_0^i(𝓒_k^i) = α_k^i`, and
Step 2 packs each piece with balls. Here the order is reversed: finitely
many small masses, each at most `η`, whose total exceeds the total weight
up to `η`, are grouped so that every weight is reached by its group up to
`η`. The first weight takes a greedy subfamily whose mass stays below it
and reaches it up to `η`; the remaining weights are grouped among the
remaining masses.
-/

@[expose] public section

open scoped ENNReal NNReal

namespace Transformer.Interpolation

/-- **Greedy subfamily.** Masses at most `η` whose total reaches `a` up
to `η` contain a subfamily of total at most `a` that still reaches `a` up
to `η`. Source auxiliary: arXiv:2411.04551v3, §2.2, proof of
`prop: compression`, Step 1, the piece of mass `α_k^i`. -/
theorem exists_subset_sum_le {ι : Type*} (S : Finset ι) (w : ι → ℝ≥0∞) (η a : ℝ≥0∞)
    (hw : ∀ p ∈ S, w p ≤ η) (ha : a ≤ ∑ p ∈ S, w p + η) :
    ∃ T ⊆ S, a ≤ ∑ p ∈ T, w p + η ∧ ∑ p ∈ T, w p ≤ a := by
  classical
  induction S using Finset.induction_on with
  | empty => exact ⟨∅, subset_rfl, ha, by rw [Finset.sum_empty]; exact zero_le⟩
  | @insert q S hq ih =>
    by_cases h : a ≤ ∑ p ∈ S, w p + η
    · obtain ⟨T, hT, h1, h2⟩ := ih (fun p hp => hw p (Finset.mem_insert_of_mem hp)) h
      exact ⟨T, hT.trans (Finset.subset_insert q S), h1, h2⟩
    · refine ⟨insert q S, subset_rfl, ha, ?_⟩
      rw [Finset.sum_insert hq, add_comm]
      exact (add_le_add le_rfl (hw q (Finset.mem_insert_self q S))).trans (not_le.mp h).le

/-- **Grouping.** Masses at most `η` whose total reaches the total of
`M + 1` weights up to `η` are grouped so that each weight is reached by
the mass of its group up to `η`. Source auxiliary: arXiv:2411.04551v3,
§2.2, proof of `prop: compression`, Steps 1–2, the pieces `𝓒_k^i` of
mass `α_k^i` and their packings, up to `η` per piece. -/
theorem exists_grouping {ι : Type*} [DecidableEq ι] (M : ℕ) (S : Finset ι)
    (w : ι → ℝ≥0∞) (η : ℝ≥0∞) (hw : ∀ p ∈ S, w p ≤ η) (α : Fin (M + 1) → ℝ≥0)
    (hsum : ∑ k, (α k : ℝ≥0∞) ≤ ∑ p ∈ S, w p + η) :
    ∃ g : ι → Fin (M + 1), ∀ k, (α k : ℝ≥0∞) ≤ ∑ p ∈ S with g p = k, w p + η := by
  induction M generalizing S with
  | zero =>
    refine ⟨fun _ => 0, fun k => ?_⟩
    rw [Fin.fin_one_eq_zero k, Finset.filter_true_of_mem fun _ _ => rfl]
    rw [Fin.sum_univ_succ, Fin.sum_univ_zero, add_zero] at hsum
    exact hsum
  | succ M ih =>
    have hα0 : (α 0 : ℝ≥0∞) ≤ ∑ p ∈ S, w p + η := by
      refine le_trans ?_ hsum
      rw [Fin.sum_univ_succ]
      exact le_self_add
    obtain ⟨T, hTS, hT1, hT2⟩ := exists_subset_sum_le S w η (α 0) hw hα0
    have hrest : ∑ k : Fin (M + 1), (α k.succ : ℝ≥0∞) ≤ ∑ p ∈ S \ T, w p + η := by
      refine ENNReal.le_of_add_le_add_left (ENNReal.coe_ne_top : (α 0 : ℝ≥0∞) ≠ ∞) ?_
      calc (α 0 : ℝ≥0∞) + ∑ k : Fin (M + 1), (α k.succ : ℝ≥0∞) = ∑ k, (α k : ℝ≥0∞) :=
            (Fin.sum_univ_succ fun k => (α k : ℝ≥0∞)).symm
        _ ≤ ∑ p ∈ S, w p + η := hsum
        _ = ∑ p ∈ T, w p + (∑ p ∈ S \ T, w p + η) := by
            rw [← Finset.sum_sdiff hTS, add_comm (∑ p ∈ S \ T, w p), add_assoc]
        _ ≤ α 0 + (∑ p ∈ S \ T, w p + η) := add_le_add hT2 le_rfl
    obtain ⟨g', hg'⟩ := ih (S \ T) (fun p hp => hw p (Finset.sdiff_subset hp))
      (fun k => α k.succ) hrest
    refine ⟨fun p => if p ∈ T then 0 else (g' p).succ, fun k => ?_⟩
    refine Fin.cases ?_ (fun k => ?_) k
    · have hT : (S.filter fun p => (if p ∈ T then 0 else (g' p).succ) = 0) = T := by
        ext p
        simp only [Finset.mem_filter]
        constructor
        · rintro ⟨-, h⟩
          by_contra hpT
          rw [ite_eq_right hpT] at h
          exact Fin.succ_ne_zero _ h
        · intro hpT
          exact ⟨hTS hpT, by rw [ite_eq_left hpT]⟩
      rw [hT]
      exact hT1
    · have hk : (S.filter fun p => (if p ∈ T then 0 else (g' p).succ) = k.succ) =
          (S \ T).filter fun p => g' p = k := by
        ext p
        simp only [Finset.mem_filter, Finset.mem_sdiff]
        constructor
        · rintro ⟨hp, h⟩
          by_cases hpT : p ∈ T
          · rw [ite_eq_left hpT] at h
            exact absurd h.symm (Fin.succ_ne_zero k)
          · rw [ite_eq_right hpT] at h
            exact ⟨⟨hp, hpT⟩, Fin.succ_injective _ h⟩
        · rintro ⟨⟨hp, hpT⟩, h⟩
          exact ⟨hp, by rw [ite_eq_right hpT, h]⟩
      rw [hk]
      exact hg' k

/-- The hypotheses of `exists_subset_sum_le` and `exists_grouping` are
satisfiable: one mass `1`, slack `1`, and one weight `1`. -/
example : (∀ p ∈ ({()} : Finset Unit), (fun _ => (1 : ℝ≥0∞)) p ≤ 1) ∧
    ∑ _k : Fin 1, ((1 : ℝ≥0) : ℝ≥0∞) ≤ ∑ p ∈ ({()} : Finset Unit), (fun _ => (1 : ℝ≥0∞)) p + 1 :=
  ⟨fun _ _ => le_rfl, by simp⟩

end Transformer.Interpolation
