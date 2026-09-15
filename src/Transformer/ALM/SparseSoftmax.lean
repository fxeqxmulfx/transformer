/-
# Truncating the softmax to the retained keys

Hardmax is not fundamental to the fast path.  A head that scores only the top
`k` keys and softmaxes over those is what a geometric retriever would actually
feed a real attention layer, and the question is what that truncation costs.

This file answers exactly that and nothing else.  `sparseWeight` is the
softmax restricted to a retained set `T`, renormalized there and zero outside;
`sparse_total_variation` computes the ℓ¹ distance between it and the full
softmax — twice the mass the full softmax puts outside `T`, exactly, not up to
a constant — and `sparse_softmax_output_close` turns that into the bound on
the two heads' *outputs*, the only thing a downstream layer sees.
`softmax_mass_outside_le` prices the mass by a score gap, so that with the
integer gap of `Transformer.ALM.Lattice` the whole error is
`2 (n - k) e^{-β} C`.

**What is not here.**  The nested convex hulls that retrieve the top `k` in
`O(k + log n)` are a different data structure from the single hull of
`Transformer.ALM.Hull`, and nothing in this file bounds the cost of producing
`T`.  `T` is an arbitrary retained set, supplied by whatever retrieves it; the
theorems say what the answer is worth once it is in hand, and say nothing
about how long it took.

The generic step `dist_weighted_sum_sub_le` is the companion of
`dist_weighted_sum_le`: where that one bounds the distance from a convex
combination to one of its points, this one bounds the distance between two
convex combinations by their ℓ¹ distance.  Neither mentions softmax.

Source: Percepta, *Can LLMs Be Computers?* (2026-03-11), the `k`-sparse
softmax remark; the head is `transformer_vm/attention/hull2d_cht.h`,
lines 203-215.
-/

import Transformer.ALM.SoftmaxValue

open scoped BigOperators

namespace Transformer
namespace ALM

variable {n : ℕ}

/-! ### The truncated head -/

/-- The `k`-sparse softmax weights: the softmax of the scores over the retained
set `T`, renormalized on `T` and zero off it.  `T.card` is the `k`. -/
noncomputable def sparseWeight (β : ℝ) (s : Fin n → ℝ) (T : Finset (Fin n))
    (j : Fin n) : ℝ :=
  if j ∈ T then Real.exp (β * s j) / ∑ l ∈ T, Real.exp (β * s l) else 0

/-- The truncated weights are nonnegative. -/
theorem sparseWeight_nonneg (β : ℝ) (s : Fin n → ℝ) (T : Finset (Fin n)) (j : Fin n) :
    0 ≤ sparseWeight β s T j := by
  unfold sparseWeight
  split
  · exact div_nonneg (Real.exp_pos _).le (Finset.sum_nonneg fun _ _ => (Real.exp_pos _).le)
  · exact le_rfl

/-- And they sum to one, so the truncated head is still an average of the
stored values. -/
theorem sparseWeight_sum (β : ℝ) (s : Fin n → ℝ) (T : Finset (Fin n)) (hT : T.Nonempty) :
    ∑ j, sparseWeight β s T j = 1 := by
  have hpos : 0 < ∑ l ∈ T, Real.exp (β * s l) :=
    Finset.sum_pos (fun _ _ => Real.exp_pos _) hT
  rw [← Finset.sum_add_sum_compl T]
  have hval : ∀ j ∈ T, sparseWeight β s T j
      = Real.exp (β * s j) / ∑ l ∈ T, Real.exp (β * s l) := fun j hj => ite_eq_left hj
  have h1 : ∑ j ∈ T, sparseWeight β s T j = 1 := by
    rw [Finset.sum_congr rfl hval, ← Finset.sum_div, div_self hpos.ne']
  have h2 : ∑ j ∈ Tᶜ, sparseWeight β s T j = 0 :=
    Finset.sum_eq_zero fun j hj => by
      simp only [sparseWeight, ite_eq_right (Finset.mem_compl.mp hj)]
  rw [h1, h2, add_zero]

/-! ### Two convex combinations, at their ℓ¹ distance -/

/-- **The ℓ¹ distance of the weights is the distance of the outputs.**  Two
convex combinations of the same values differ by at most their ℓ¹ weight
distance times the spread of the values around any point.  Nothing about
softmax, scores or retrieval enters; this is the companion of
`dist_weighted_sum_le`, which is the case where one combination is a point
mass. -/
theorem dist_weighted_sum_sub_le {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (w w' : Fin n → ℝ) (hsum : ∑ j, w j = 1) (hsum' : ∑ j, w' j = 1)
    (V : Fin n → E) (p : E) (C : ℝ) (hC : ∀ j, ‖V j - p‖ ≤ C) :
    ‖(∑ j, w j • V j) - ∑ j, w' j • V j‖ ≤ (∑ j, |w j - w' j|) * C := by
  have hzero : ∑ j, (w j - w' j) = 0 := by
    rw [Finset.sum_sub_distrib, hsum, hsum', sub_self]
  have hrw : (∑ j, w j • V j) - ∑ j, w' j • V j = ∑ j, (w j - w' j) • (V j - p) := by
    have hexp : ∑ j, (w j - w' j) • (V j - p)
        = (∑ j, (w j - w' j) • V j) - (∑ j, (w j - w' j)) • p := by
      rw [Finset.sum_smul, ← Finset.sum_sub_distrib]
      exact Finset.sum_congr rfl fun j _ => smul_sub _ _ _
    rw [hexp, hzero, zero_smul, sub_zero, Finset.sum_congr rfl
      (fun j _ => sub_smul (w j) (w' j) (V j)), Finset.sum_sub_distrib]
  rw [hrw]
  calc ‖∑ j, (w j - w' j) • (V j - p)‖
      ≤ ∑ j, ‖(w j - w' j) • (V j - p)‖ := norm_sum_le _ _
    _ = ∑ j, |w j - w' j| * ‖V j - p‖ := by
        refine Finset.sum_congr rfl fun j _ => ?_
        rw [norm_smul, Real.norm_eq_abs]
    _ ≤ ∑ j, |w j - w' j| * C :=
        Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (hC j) (abs_nonneg _)
    _ = (∑ j, |w j - w' j|) * C := by rw [Finset.sum_mul]

/-! ### What truncation costs -/

/-- **Truncation moves exactly twice the discarded mass.**  Restricting the
softmax to `T` and renormalizing changes the weight vector by precisely
`2 · (mass outside T)` in ℓ¹: the discarded mass is lost off `T`, and the same
amount is added back on `T` by the renormalization. -/
theorem sparse_total_variation (β : ℝ) (s : Fin n → ℝ) (T : Finset (Fin n))
    (hT : T.Nonempty) :
    ∑ j, |Real.exp (β * s j) / (∑ l, Real.exp (β * s l)) - sparseWeight β s T j|
      = 2 * ∑ j ∈ Tᶜ, Real.exp (β * s j) / ∑ l, Real.exp (β * s l) := by
  obtain ⟨i₀, hi₀⟩ := hT
  have hZTpos : 0 < ∑ l ∈ T, Real.exp (β * s l) :=
    Finset.sum_pos (fun _ _ => Real.exp_pos _) ⟨i₀, hi₀⟩
  have hZpos : 0 < ∑ l, Real.exp (β * s l) :=
    Finset.sum_pos (fun _ _ => Real.exp_pos _) ⟨i₀, Finset.mem_univ i₀⟩
  have hle : ∑ l ∈ T, Real.exp (β * s l) ≤ ∑ l, Real.exp (β * s l) :=
    Finset.sum_le_sum_of_subset_of_nonneg (Finset.subset_univ T)
      fun _ _ _ => (Real.exp_pos _).le
  have hTin : ∀ j ∈ T,
      |Real.exp (β * s j) / (∑ l, Real.exp (β * s l)) - sparseWeight β s T j|
        = sparseWeight β s T j - Real.exp (β * s j) / ∑ l, Real.exp (β * s l) := by
    intro j hj
    have hb : Real.exp (β * s j) / (∑ l, Real.exp (β * s l)) ≤ sparseWeight β s T j := by
      simp only [sparseWeight, ite_eq_left hj]
      exact div_le_div_of_nonneg_left (Real.exp_pos _).le hZTpos hle
    rw [abs_sub_comm, abs_of_nonneg (by linarith)]
  have hTout : ∀ j ∈ Tᶜ,
      |Real.exp (β * s j) / (∑ l, Real.exp (β * s l)) - sparseWeight β s T j|
        = Real.exp (β * s j) / ∑ l, Real.exp (β * s l) := by
    intro j hj
    simp only [sparseWeight, ite_eq_right (Finset.mem_compl.mp hj), sub_zero]
    exact abs_of_nonneg (div_nonneg (Real.exp_pos _).le hZpos.le)
  have hval : ∀ j ∈ T, sparseWeight β s T j
      = Real.exp (β * s j) / ∑ l ∈ T, Real.exp (β * s l) := fun j hj => ite_eq_left hj
  have hone : ∑ j ∈ T, sparseWeight β s T j = 1 := by
    rw [Finset.sum_congr rfl hval, ← Finset.sum_div, div_self hZTpos.ne']
  have hsplit : (∑ j ∈ T, Real.exp (β * s j) / ∑ l, Real.exp (β * s l))
      + ∑ j ∈ Tᶜ, Real.exp (β * s j) / ∑ l, Real.exp (β * s l) = 1 := by
    rw [Finset.sum_add_sum_compl, ← Finset.sum_div, div_self hZpos.ne']
  rw [← Finset.sum_add_sum_compl T, Finset.sum_congr rfl hTin,
    Finset.sum_congr rfl hTout, Finset.sum_sub_distrib, hone]
  linarith

/-- **And so the truncated head is close to the full head.**  The distance
between the two outputs is at most twice the discarded mass times the spread
of the stored values.  `T` is arbitrary: the bound holds for any retained set,
however it was found. -/
theorem sparse_softmax_output_close {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (β : ℝ) (s : Fin n → ℝ) (T : Finset (Fin n)) (hT : T.Nonempty)
    (V : Fin n → E) (p : E) (C : ℝ) (hC : ∀ j, ‖V j - p‖ ≤ C) :
    ‖(∑ j, (Real.exp (β * s j) / ∑ l, Real.exp (β * s l)) • V j)
        - ∑ j, sparseWeight β s T j • V j‖
      ≤ 2 * (∑ j ∈ Tᶜ, Real.exp (β * s j) / ∑ l, Real.exp (β * s l)) * C := by
  obtain ⟨i₀, hi₀⟩ := hT
  have : Nonempty (Fin n) := ⟨i₀⟩
  have h := dist_weighted_sum_sub_le
    (fun j => Real.exp (β * s j) / ∑ l, Real.exp (β * s l)) (sparseWeight β s T)
    (softmax_weight_sum β s) (sparseWeight_sum β s T ⟨i₀, hi₀⟩) V p C hC
  rwa [sparse_total_variation β s T ⟨i₀, hi₀⟩] at h

/-- **Priced by a score gap.**  If every discarded key scores at least `δ`
below some key, the discarded mass is at most `(n - k) e^{-βδ}` — the same
estimate as `softmax_winner_ge`, over the complement of the retained set
instead of over the complement of the winner. -/
theorem softmax_mass_outside_le (β : ℝ) (hβ : 0 ≤ β) (s : Fin n → ℝ)
    (T : Finset (Fin n)) (i₀ : Fin n) (δ : ℝ) (hgap : ∀ j ∉ T, s j + δ ≤ s i₀) :
    ∑ j ∈ Tᶜ, Real.exp (β * s j) / ∑ l, Real.exp (β * s l)
      ≤ ((n : ℝ) - T.card) * Real.exp (-(β * δ)) := by
  have hterm : ∀ j ∈ Tᶜ, Real.exp (β * s j) / (∑ l, Real.exp (β * s l))
      ≤ Real.exp (-(β * δ)) := by
    intro j hj
    have hZge : Real.exp (β * s i₀) ≤ ∑ l, Real.exp (β * s l) :=
      Finset.single_le_sum (f := fun l => Real.exp (β * s l))
        (fun l _ => (Real.exp_pos _).le) (Finset.mem_univ i₀)
    have h0 : β * (s j + δ) ≤ β * s i₀ :=
      mul_le_mul_of_nonneg_left (hgap j (Finset.mem_compl.mp hj)) hβ
    rw [mul_add] at h0
    calc Real.exp (β * s j) / (∑ l, Real.exp (β * s l))
        ≤ Real.exp (β * s j) / Real.exp (β * s i₀) :=
          div_le_div_of_nonneg_left (Real.exp_pos _).le (Real.exp_pos _) hZge
      _ = Real.exp (β * s j - β * s i₀) := (Real.exp_sub _ _).symm
      _ ≤ Real.exp (-(β * δ)) := Real.exp_le_exp.mpr (by linarith)
  have hcard : ((Tᶜ).card : ℝ) = (n : ℝ) - T.card := by
    rw [Finset.card_compl, Fintype.card_fin]
    have hle : T.card ≤ n := by simpa using Finset.card_le_univ T
    push_cast [Nat.cast_sub hle]
    ring
  calc ∑ j ∈ Tᶜ, Real.exp (β * s j) / ∑ l, Real.exp (β * s l)
      ≤ (Tᶜ).card • Real.exp (-(β * δ)) := Finset.sum_le_card_nsmul _ _ _ hterm
    _ = ((n : ℝ) - T.card) * Real.exp (-(β * δ)) := by rw [nsmul_eq_mul, hcard]

/-- **The whole truncation error.**  Retain any set whose discarded keys are
`δ` below the best score: the truncated head's output is within
`2 (n - k) e^{-βδ} C` of the full head's.  With the unit integer gap of
`Transformer.ALM.Lattice` this is `2 (n - k) e^{-β} C`. -/
theorem sparse_softmax_output_close_of_gap {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] (β : ℝ) (hβ : 0 ≤ β) (s : Fin n → ℝ) (T : Finset (Fin n))
    (i₀ : Fin n) (hi₀ : i₀ ∈ T) (δ : ℝ) (hgap : ∀ j ∉ T, s j + δ ≤ s i₀)
    (V : Fin n → E) (p : E) (C : ℝ) (hC : ∀ j, ‖V j - p‖ ≤ C) :
    ‖(∑ j, (Real.exp (β * s j) / ∑ l, Real.exp (β * s l)) • V j)
        - ∑ j, sparseWeight β s T j • V j‖
      ≤ 2 * (((n : ℝ) - T.card) * Real.exp (-(β * δ))) * C := by
  have hC0 : 0 ≤ C := le_trans (norm_nonneg _) (hC i₀)
  refine le_trans (sparse_softmax_output_close β s T ⟨i₀, hi₀⟩ V p C hC) ?_
  have hmass := softmax_mass_outside_le β hβ s T i₀ δ hgap
  exact mul_le_mul_of_nonneg_right
    (mul_le_mul_of_nonneg_left hmass (by norm_num)) hC0

/-- The hypotheses are satisfiable, and not by a degenerate retained set: two
keys scoring `1` and `0`, the better one retained, `β = 1` and `δ = 1`, which
is the integer gap the lookup head enjoys. -/
example :
    (0 : ℝ) ≤ 1 ∧ ({0} : Finset (Fin 2)).Nonempty ∧ (0 : Fin 2) ∈ ({0} : Finset (Fin 2)) ∧
      (∀ j ∉ ({0} : Finset (Fin 2)),
        (fun i : Fin 2 => if i = 0 then (1 : ℝ) else 0) j + 1
          ≤ (fun i : Fin 2 => if i = 0 then (1 : ℝ) else 0) 0) ∧
      ∀ j : Fin 2, ‖(fun _ : Fin 2 => (0 : ℝ)) j - (0 : ℝ)‖ ≤ 0 := by
  refine ⟨by norm_num, ⟨0, by simp⟩, by simp, fun j hj => ?_, fun j => by simp⟩
  have hj0 : j ≠ 0 := by simpa using hj
  simp [hj0]

end ALM
end Transformer
