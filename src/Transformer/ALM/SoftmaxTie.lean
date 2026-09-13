/-
# The head and the machine agree where the machine has to choose

`Transformer.ALM.SoftmaxValue` says what the softmax head returns when one key
wins: the value stored at it, up to `(n-1)e^{-β}·C`.  Its hypothesis is a
strict winner, and `Transformer.ALM.TieHyperplane` says exactly where that
hypothesis fails — on the bisecting hyperplanes, which in dimension one is the
single query `2q = k₁ + k₂`.  There the machine does something: `HullHalf::query`
merges the two tied lines and `resolve` averages them
(`transformer_vm/attention/hull2d_cht.h`, lines 70-83, 276-306), and
`Transformer.ALM.HullCost` proves that is what it hands back.

So the one place where the tie-break mode is observable at all was the one
place with no statement about the head.  The two halves of the answer — the
machine's and the head's — were proved on complementary sets and never
compared on either.

They agree, and for a reason that needs no geometry: softmax gives equal
weight to equal scores.  `dist_weighted_sum_le_of_level` is that observation
at full strength — for any convex combination whose weights are constant on a
set `T` carrying mass `1 - ε`, the output sits within `ε·C` of the *centroid*
of `T`, whatever `T` is and however large.  It subsumes
`dist_weighted_sum_le`, which is the case `T = {i₀}`.
`softmax_head_resolves_average` is then the two-key instance, stated against
the aggregate the walk actually builds: on the tie locus the running head
returns what `TieBreak::AVERAGE` returns.
-/

import Transformer.ALM.SoftmaxValue
import Transformer.ALM.HullResolve

open scoped BigOperators

namespace Transformer
namespace ALM

variable {n : ℕ}

/-! ### Mass spread evenly over a set lands on its centroid -/

/-- **Concentration on a level set.**  If the weights of a convex combination
are constant on `T` and `T` carries all but `ε` of the mass, the output is
within `ε·C` of the centroid of the values on `T`, where `C` bounds their
spread around it.  Nothing here is about softmax, scores or ties; the case
`T = {i₀}` is `dist_weighted_sum_le`. -/
theorem dist_weighted_sum_le_of_level {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (w : Fin n → ℝ) (hw0 : ∀ j, 0 ≤ w j) (hsum : ∑ j, w j = 1)
    (V : Fin n → E) (T : Finset (Fin n)) (hT : T.Nonempty) (a : ℝ) (ha : ∀ i ∈ T, w i = a)
    (ε C : ℝ) (hmass : 1 - ε ≤ ∑ j ∈ T, w j)
    (hC : ∀ j, ‖V j - (T.card : ℝ)⁻¹ • ∑ i ∈ T, V i‖ ≤ C) :
    ‖(∑ j, w j • V j) - (T.card : ℝ)⁻¹ • ∑ i ∈ T, V i‖ ≤ ε * C := by
  set A : E := (T.card : ℝ)⁻¹ • ∑ i ∈ T, V i with hA
  obtain ⟨i₁, hi₁⟩ := hT
  have hC0 : 0 ≤ C := le_trans (norm_nonneg _) (hC i₁)
  have hcard : (T.card : ℝ) ≠ 0 :=
    Nat.cast_ne_zero.mpr (Finset.card_ne_zero_of_mem hi₁)
  -- the values on `T` are centred at `A`
  have hzero : ∑ i ∈ T, (V i - A) = 0 := by
    rw [Finset.sum_sub_distrib, Finset.sum_const, ← Nat.cast_smul_eq_nsmul ℝ, hA, smul_smul,
      mul_inv_cancel₀ hcard, one_smul, sub_self]
  -- and the whole combination is a combination of displacements from `A`
  have hrw : (∑ j, w j • V j) - A = ∑ j, w j • (V j - A) := by
    simp only [smul_sub]
    rw [Finset.sum_sub_distrib, ← Finset.sum_smul, hsum, one_smul]
  have hsplit : ∑ j ∈ T, w j + ∑ j ∈ Tᶜ, w j = 1 := by
    rw [Finset.sum_add_sum_compl, hsum]
  have hTzero : ∑ j ∈ T, w j • (V j - A) = 0 := by
    rw [Finset.sum_congr rfl fun i hi => by rw [ha i hi], ← Finset.smul_sum, hzero, smul_zero]
  have houter : ∑ j, w j • (V j - A) = ∑ j ∈ Tᶜ, w j • (V j - A) := by
    rw [← Finset.sum_add_sum_compl T fun j => w j • (V j - A), hTzero, zero_add]
  rw [hrw, houter]
  calc ‖∑ j ∈ Tᶜ, w j • (V j - A)‖
      ≤ ∑ j ∈ Tᶜ, ‖w j • (V j - A)‖ := norm_sum_le _ _
    _ = ∑ j ∈ Tᶜ, w j * ‖V j - A‖ := by
        refine Finset.sum_congr rfl fun j _ => ?_
        rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (hw0 j)]
    _ ≤ ∑ j ∈ Tᶜ, w j * C :=
        Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (hC j) (hw0 j)
    _ = (∑ j ∈ Tᶜ, w j) * C := by rw [Finset.sum_mul]
    _ ≤ ε * C := mul_le_mul_of_nonneg_right (by linarith) hC0

/-- The hypotheses are satisfiable, and not only for a singleton `T`: the
uniform combination on `Fin 2` is constant on all of `T = univ`, which then
carries the whole mass. -/
example : (∀ j : Fin 2, (0 : ℝ) ≤ ![1 / 2, 1 / 2] j) ∧
    (∑ j : Fin 2, (![1 / 2, 1 / 2] : Fin 2 → ℝ) j) = 1 ∧
    ∀ i ∈ (Finset.univ : Finset (Fin 2)), (![1 / 2, 1 / 2] : Fin 2 → ℝ) i = 1 / 2 := by
  refine ⟨fun j => ?_, by simp [Fin.sum_univ_two]; norm_num, fun i _ => ?_⟩
  · fin_cases j <;> norm_num
  · fin_cases i <;> norm_num

/-! ### Softmax weights are constant on a level set of the score -/

/-- Equal scores get equal softmax weight — the whole reason the head cannot
prefer one tied key over the other. -/
theorem softmax_weight_eq_of_score_eq (β : ℝ) (s : Fin n → ℝ) {i j : Fin n} (h : s i = s j) :
    Real.exp (β * s i) / ∑ k, Real.exp (β * s k)
      = Real.exp (β * s j) / ∑ k, Real.exp (β * s k) := by
  rw [h]

/-- **The head returns the mean over a tied set.**  On a level set of the
score carrying mass `1 - ε`, the softmax output is within `ε·C` of the
centroid of the values there — it splits its mass evenly over the tie, having
nothing to tell the tied keys apart with. -/
theorem softmax_output_close_level {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [Nonempty (Fin n)] (β : ℝ) (s : Fin n → ℝ) (V : Fin n → E)
    (T : Finset (Fin n)) (hT : T.Nonempty) (σ : ℝ) (hlevel : ∀ i ∈ T, s i = σ)
    (ε C : ℝ) (hmass : 1 - ε ≤ ∑ j ∈ T, Real.exp (β * s j) / ∑ k, Real.exp (β * s k))
    (hC : ∀ j, ‖V j - (T.card : ℝ)⁻¹ • ∑ i ∈ T, V i‖ ≤ C) :
    ‖(∑ j, (Real.exp (β * s j) / ∑ k, Real.exp (β * s k)) • V j)
        - (T.card : ℝ)⁻¹ • ∑ i ∈ T, V i‖ ≤ ε * C :=
  dist_weighted_sum_le_of_level _ (softmax_weight_nonneg β s) (softmax_weight_sum β s) V T hT
    (Real.exp (β * σ) / ∑ k, Real.exp (β * s k)) (fun i hi => by rw [hlevel i hi]) ε C hmass hC

/-! ### And that is exactly what `resolve` hands back -/

/-- **The head implements `TieBreak::AVERAGE`.**  Where two stored keys tie,
the walk of `HullHalf::query` merges their aggregates and `resolve` returns
the componentwise mean of the two payloads
(`Transformer.ALM.HullResolve.scanCombined_resolveAverage`); the softmax head,
weighting the two equally because their scores are equal, returns the same
vector up to `ε·C`.  So the machine's tie-break is not a convention the head
has to be reconciled with — it is what the head already does. -/
theorem softmax_head_resolves_average [Nonempty (Fin n)] (β : ℝ) (s : Fin n → ℝ)
    (V : Fin n → ℝ × ℝ) (b c : Fin n) (hbc : b ≠ c) (σ : ℝ) (hb : s b = σ) (hc : s c = σ)
    (M : ℕ → Meta) (p r : ℕ) (sp sr : ℤ) (hsp : 0 ≤ sp) (hsr : 0 ≤ sr)
    (hMp : M p = Meta.empty.add (V b) sp) (hMr : M r = Meta.empty.add (V c) sr)
    (ε C : ℝ)
    (hmass : 1 - ε ≤ Real.exp (β * s b) / ∑ k, Real.exp (β * s k)
      + Real.exp (β * s c) / ∑ k, Real.exp (β * s k))
    (hC : ∀ j, ‖V j - (((V b).1 + (V c).1) / 2, ((V b).2 + (V c).2) / 2)‖ ≤ C) :
    ‖(∑ j, (Real.exp (β * s j) / ∑ k, Real.exp (β * s k)) • V j)
        - (scanCombined M p r).resolveAverage‖ ≤ ε * C := by
  have hpair : ({b, c} : Finset (Fin n)).card = 2 := Finset.card_pair hbc
  have hcentroid : ((({b, c} : Finset (Fin n)).card : ℝ))⁻¹ • ∑ i ∈ ({b, c} : Finset (Fin n)), V i
      = (((V b).1 + (V c).1) / 2, ((V b).2 + (V c).2) / 2) := by
    rw [hpair, Finset.sum_pair hbc]
    refine Prod.ext ?_ ?_ <;> simp [Prod.smul_def] <;> ring
  rw [scanCombined_resolveAverage M p r (V b) (V c) sp sr hsp hsr hMp hMr, ← hcentroid]
  refine softmax_output_close_level β s V {b, c} ⟨b, by simp⟩ σ ?_ ε C ?_ ?_
  · intro i hi
    rcases Finset.mem_insert.mp hi with rfl | hi
    · exact hb
    · rw [Finset.mem_singleton.mp hi]; exact hc
  · rwa [Finset.sum_pair hbc]
  · rwa [hcentroid]

/-- The hypotheses are satisfiable: two distinct keys carrying the same
payload tie at every query, and their spread around the mean is zero. -/
example : (0 : Fin 2) ≠ 1 ∧
    ∀ j : Fin 2, ‖(fun _ : Fin 2 => ((0 : ℝ), (0 : ℝ))) j
      - ((((0 : ℝ), (0 : ℝ)).1 + ((0 : ℝ), (0 : ℝ)).1) / 2,
         (((0 : ℝ), (0 : ℝ)).2 + ((0 : ℝ), (0 : ℝ)).2) / 2)‖ ≤ 0 := by
  refine ⟨by decide, fun j => ?_⟩
  norm_num

end ALM
end Transformer
