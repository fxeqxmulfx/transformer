/-
# Softmax associative recall with a logarithmic score scale

Arora et al., arXiv:2312.04927v1, §4, Proposition `prop: attention-ar`,
and Appendix Proposition `prop: app-attention`.  The appendix proves its
exact vector result for attention *without* softmax.  This companion result
gives the correct finite-temperature statement: a genuine softmax head
places more than half its mass on the uniquely matching key at scale
`log (2m)`, where `m` is the number of candidate keys.  Thus one-hot values
are decoded exactly by argmax, although the output vector itself retains
nonzero softmax leakage.
-/

import Transformer.Zoology.Appendix_Attention
import Transformer.ALM.SoftmaxValue

open scoped BigOperators

namespace Transformer.Zoology

/-- Equality scores produced by one-hot query-key dot products.
Source: Appendix Proposition `prop: app-attention`, second layer. -/
def equalityScores {m c : ℕ} (keys : Fin m → Fin c)
    (q : Fin c) (j : Fin m) : ℝ :=
  if keys j = q then 1 else 0

/-- The genuine softmax weight of a candidate key at score scale `β`.
Source: §2 attention equation and Appendix Proposition `prop: app-attention`,
with softmax retained. -/
noncomputable def recallWeight {m c : ℕ} (keys : Fin m → Fin c)
    (q : Fin c) (β : ℝ) (j : Fin m) : ℝ :=
  Real.exp (β * equalityScores keys q j) /
    ∑ k, Real.exp (β * equalityScores keys q k)

/-- At logarithmic score scale, a unique exact match gets strictly more than
half the attention mass.  This is the softmax correction of Appendix
Proposition `prop: app-attention`: the paper's exact vector equality uses
unnormalized `QKᵀV`, not finite-temperature softmax. -/
theorem matching_weight_majority {m c : ℕ} (hm : 0 < m)
    (keys : Fin m → Fin c) (hkeys : Function.Injective keys)
    (q : Fin c) (j₀ : Fin m) (hmatch : keys j₀ = q) :
    (1 / 2 : ℝ) < recallWeight keys q (Real.log (2 * (m : ℝ))) j₀ := by
  let β : ℝ := Real.log (2 * (m : ℝ))
  have hmreal : (1 : ℝ) ≤ (m : ℝ) := by exact_mod_cast hm
  have hpos : (0 : ℝ) < 2 * (m : ℝ) := by positivity
  have hβ : 0 ≤ β := Real.log_nonneg (by nlinarith)
  have hgap : ∀ j : Fin m, j ≠ j₀ →
      equalityScores keys q j + 1 ≤ equalityScores keys q j₀ := by
    intro j hj
    have hne : keys j ≠ q := by
      intro h
      exact hj (hkeys (h.trans hmatch.symm))
    simp [equalityScores, hne, hmatch]
  have hw := Transformer.ALM.softmax_winner_ge β hβ
    (equalityScores keys q) j₀ 1 hgap
  have hexp : Real.exp (-(β * 1)) = 1 / (2 * (m : ℝ)) := by
    simp [β, Real.exp_neg, Real.exp_log hpos]
  rw [hexp] at hw
  change (1 / 2 : ℝ) <
    Real.exp (β * equalityScores keys q j₀) /
      ∑ k, Real.exp (β * equalityScores keys q k)
  have hbound : (1 / 2 : ℝ) < 1 - ((m : ℝ) - 1) * (1 / (2 * (m : ℝ))) := by
    have hlt : ((m : ℝ) - 1) / (2 * (m : ℝ)) < 1 / 2 := by
      rw [div_lt_div_iff₀ hpos (by norm_num : (0 : ℝ) < 2)]
      nlinarith
    have hrewrite : ((m : ℝ) - 1) * (1 / (2 * (m : ℝ))) =
        ((m : ℝ) - 1) / (2 * (m : ℝ)) := by ring
    rw [hrewrite]
    linarith
  exact lt_of_lt_of_le hbound hw

/-- Output coordinate for one-hot values under a genuine softmax head.
Source: §2 attention equation, in the appendix's paired-key setting. -/
noncomputable def recallCoordinate {m c : ℕ} (keys values : Fin m → Fin c)
    (q : Fin c) (β : ℝ) (v : Fin c) : ℝ :=
  ∑ j, recallWeight keys q β j * oneHot (values j) v

/-- A strict majority weight gives the correct one-hot value coordinate
strictly above one half.  Source: Appendix Proposition `prop: app-attention`,
softmax correction for a unique exact match. -/
theorem recall_correct_coordinate {m c : ℕ} (hm : 0 < m)
    (keys values : Fin m → Fin c) (hkeys : Function.Injective keys)
    (q : Fin c) (j₀ : Fin m) (hmatch : keys j₀ = q) :
    (1 / 2 : ℝ) <
      recallCoordinate keys values q (Real.log (2 * (m : ℝ))) (values j₀) := by
  classical
  let β : ℝ := Real.log (2 * (m : ℝ))
  have hw := matching_weight_majority hm keys hkeys q j₀ hmatch
  have hnonneg : ∀ j : Fin m,
      0 ≤ recallWeight keys q β j * oneHot (values j) (values j₀) := by
    intro j
    exact mul_nonneg
      (Transformer.ALM.softmax_weight_nonneg β (equalityScores keys q) j)
      (by unfold oneHot; split_ifs <;> norm_num)
  have hsplit := Finset.sum_erase_add Finset.univ
    (fun j => recallWeight keys q β j * oneHot (values j) (values j₀))
    (Finset.mem_univ j₀)
  have htail : 0 ≤ ∑ j ∈ Finset.univ.erase j₀,
      recallWeight keys q β j * oneHot (values j) (values j₀) :=
    Finset.sum_nonneg (fun j _ => hnonneg j)
  have hterm : recallWeight keys q β j₀ * oneHot (values j₀) (values j₀) =
      recallWeight keys q β j₀ := by simp [oneHot]
  rw [hterm] at hsplit
  unfold recallCoordinate
  change (1 / 2 : ℝ) <
    ∑ j, recallWeight keys q β j * oneHot (values j) (values j₀)
  have hmain : (1 / 2 : ℝ) < recallWeight keys q β j₀ := hw
  linarith

/-- Every other value coordinate is strictly below one half.  The result
still holds if many nonmatching keys carry the same distractor value.
Source: Appendix Proposition `prop: app-attention`, softmax correction. -/
theorem recall_other_coordinate {m c : ℕ} (hm : 0 < m)
    (keys values : Fin m → Fin c) (hkeys : Function.Injective keys)
    (q : Fin c) (j₀ : Fin m) (hmatch : keys j₀ = q)
    (v : Fin c) (hv : v ≠ values j₀) :
    recallCoordinate keys values q (Real.log (2 * (m : ℝ))) v < (1 / 2 : ℝ) := by
  classical
  let β : ℝ := Real.log (2 * (m : ℝ))
  have hnonempty : Nonempty (Fin m) := ⟨j₀⟩
  have hw := matching_weight_majority hm keys hkeys q j₀ hmatch
  have hsplit := Finset.sum_erase_add Finset.univ
    (fun j => recallWeight keys q β j * oneHot (values j) v)
    (Finset.mem_univ j₀)
  have hterm : recallWeight keys q β j₀ * oneHot (values j₀) v = 0 := by
    simp [oneHot, hv]
  rw [hterm, add_zero] at hsplit
  have htail : (∑ j ∈ Finset.univ.erase j₀,
      recallWeight keys q β j * oneHot (values j) v) ≤
      ∑ j ∈ Finset.univ.erase j₀, recallWeight keys q β j := by
    apply Finset.sum_le_sum
    intro j _
    have hone : oneHot (values j) v ≤ 1 := by
      unfold oneHot
      split_ifs <;> norm_num
    simpa only [recallWeight, mul_one] using
      mul_le_mul_of_nonneg_left hone
        (Transformer.ALM.softmax_weight_nonneg β (equalityScores keys q) j)
  have hsum : ∑ j : Fin m, recallWeight keys q β j = 1 :=
    @Transformer.ALM.softmax_weight_sum m hnonempty β (equalityScores keys q)
  have hsplitw := Finset.sum_erase_add Finset.univ
    (recallWeight keys q β) (Finset.mem_univ j₀)
  unfold recallCoordinate
  linarith

/-- The correct token is the unique argmax of the softmax output, for any
number of distinct one-hot keys and any assignment of one-hot values.
This is exact *discrete* recall; the real output vector is not exactly one-hot
at finite temperature.  Source: Appendix Proposition `prop: app-attention`,
with genuine softmax and logarithmic score scaling. -/
theorem recall_argmax_correct {m c : ℕ} (hm : 0 < m)
    (keys values : Fin m → Fin c) (hkeys : Function.Injective keys)
    (q : Fin c) (j₀ : Fin m) (hmatch : keys j₀ = q)
    (v : Fin c) (hv : v ≠ values j₀) :
    recallCoordinate keys values q (Real.log (2 * (m : ℝ))) v <
      recallCoordinate keys values q (Real.log (2 * (m : ℝ))) (values j₀) := by
  exact lt_trans (recall_other_coordinate hm keys values hkeys q j₀ hmatch v hv)
    (recall_correct_coordinate hm keys values hkeys q j₀ hmatch)

/-- The score-gap hypothesis is realized by distinct one-hot keys.
Source: Appendix Proposition `prop: app-attention`, one-hot encoding. -/
example : Function.Injective (id : Fin 2 → Fin 2) ∧
    (id : Fin 2 → Fin 2) 0 = (0 : Fin 2) := by
  constructor
  · intro i j h
    exact h
  · rfl

end Transformer.Zoology
