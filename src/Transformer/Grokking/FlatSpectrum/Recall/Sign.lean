module

public import Mathlib.LinearAlgebra.Dimension.Constructions
public import Transformer.Grokking.FlatSpectrum.Recall.Defs

/-!
# Recall decoded by a sign or an argmax

Source: `review/flat_spectrum.md`, *Associative recall has a flat spectrum* (added), item AR5.
The squared error of item AR4 bounds a regression. A classifier is correct when a sign or an
argmax is right, whatever the margin. The count of keys survives.

Let `F q π = g q + ∑_{r < d} a_r(q) b_r(π)` have cut width `d` on dictionaries with two values,
and let the sign of `F q π` tell the value of `q`: `F q π > 0` exactly when `π q` is the second
value. Then `m ≤ d` (`HasCutWidth.card_le_of_sign`). Proof: if `m > d`, the `m` vectors
`a(q) ∈ ℝ^d` satisfy `∑_q λ_q a(q) = 0` with `λ ≠ 0`, so `∑_q λ_q F q π = ∑_q λ_q g q` for every
`π`. The dictionary with `π q` the second value exactly when `λ_q > 0` makes every term
`λ_q F q π ≥ 0`, and the one with `π q` the second value exactly when `λ_q < 0` makes every term
`≤ 0`; one of the two sums is strict at a key with `λ_q ≠ 0`.

For logits `L v q π` over any set of values whose differences have cut width `d`, a correct
argmax on the dictionaries with values in `{v₀, v₁}` is the sign of `L v₁ − L v₀`, so again
`m ≤ d` (`card_le_of_argmax`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

variable {κ ν : Type*} [Fintype κ]

/-- Cut width does not grow when the context is the image of another one. -/
theorem HasCutWidth.comp_context {Q C C' : Type*} {d : ℕ} {F : Q → C → ℝ}
    (hF : HasCutWidth d F) (ι : C' → C) : HasCutWidth d fun q x => F q (ι x) := by
  obtain ⟨g, a, b, hF⟩ := hF
  exact ⟨g, a, fun x => b (ι x), fun q x => hF q (ι x)⟩

example := (show HasCutWidth 1 fun (q : Fin 2) (x : ℝ) => (q : ℝ) * x from
  ⟨fun _ => 0, fun q _ => q, fun x _ => x, fun q x => by simp⟩).comp_context fun x : ℝ => 2 * x

/-- **Recall decoded by a sign takes cut width at least the number of keys**
(`review/flat_spectrum.md`, AR5): if a readout of cut width `d` is positive exactly when the
queried key is bound to `true`, for every dictionary with values in `Bool`, then `m ≤ d`. -/
theorem HasCutWidth.card_le_of_sign {d : ℕ} {F : κ → (κ → Bool) → ℝ} (hF : HasCutWidth d F)
    (h : ∀ π q, 0 < F q π ↔ π q = true) : Fintype.card κ ≤ d := by
  obtain ⟨g, a, b, hF⟩ := hF
  by_contra hlt
  have hdep : ¬ LinearIndependent ℝ a := fun hli => hlt (by
    simpa using hli.fintype_card_le_finrank)
  obtain ⟨lam, hsum, q₀, hq₀⟩ := Fintype.not_linearIndependent_iff.1 hdep
  have hG (π : κ → Bool) : ∑ q, lam q * F q π = ∑ q, lam q * g q := by
    have h0 : ∑ q, lam q * ∑ r, a q r * b π r = 0 := by
      calc ∑ q, lam q * ∑ r, a q r * b π r = ∑ r, (∑ q, lam q • a q) r * b π r := by
            simp only [Finset.sum_apply, Pi.smul_apply, smul_eq_mul, Finset.mul_sum,
              Finset.sum_mul]
            rw [Finset.sum_comm]
            exact Finset.sum_congr rfl fun r _ => Finset.sum_congr rfl fun q _ => by ring
        _ = 0 := by rw [hsum]; simp
    simp only [hF, mul_add, Finset.sum_add_distrib, h0, add_zero]
  have t1 (q : κ) : 0 ≤ lam q * F q fun k => decide (0 < lam k) := by
    rcases lt_or_ge 0 (lam q) with hl | hl
    · exact (mul_pos hl ((h _ q).2 (decide_eq_true hl))).le
    · exact mul_nonneg_of_nonpos_of_nonpos hl (not_lt.1 fun hF' =>
        absurd (of_decide_eq_true ((h _ q).1 hF')) (not_lt.2 hl))
  have t2 (q : κ) : lam q * F q (fun k => decide (lam k < 0)) ≤ 0 := by
    rcases lt_or_ge (lam q) 0 with hl | hl
    · exact (mul_neg_of_neg_of_pos hl ((h _ q).2 (decide_eq_true hl))).le
    · exact mul_nonpos_of_nonneg_of_nonpos hl (not_lt.1 fun hF' =>
        absurd (of_decide_eq_true ((h _ q).1 hF')) (not_lt.2 hl))
  have h1 : 0 ≤ ∑ q, lam q * g q := hG _ ▸ Finset.sum_nonneg fun q _ => t1 q
  have h2 : ∑ q, lam q * g q ≤ 0 := hG _ ▸ Finset.sum_nonpos fun q _ => t2 q
  rcases lt_or_gt_of_ne hq₀ with hl | hl
  · have h3 : ∑ q, lam q * F q (fun k => decide (lam k < 0)) < 0 :=
      Finset.sum_neg' (fun q _ => t2 q) ⟨q₀, Finset.mem_univ _,
        mul_neg_of_neg_of_pos hl ((h _ q₀).2 (decide_eq_true hl))⟩
    rw [hG] at h3
    exact absurd h1 (not_le.2 h3)
  · have h3 : 0 < ∑ q, lam q * F q fun k => decide (0 < lam k) :=
      Finset.sum_pos' (fun q _ => t1 q) ⟨q₀, Finset.mem_univ _,
        mul_pos hl ((h _ q₀).2 (decide_eq_true hl))⟩
    rw [hG] at h3
    exact absurd h2 (not_le.2 h3)

/-- Two keys read by one-hot features of width two: the sign of `±1` tells the bound value. -/
example : Fintype.card (Fin 2) ≤ 2 :=
  HasCutWidth.card_le_of_sign (F := fun q π => if π q then 1 else -1)
    ⟨fun _ => 0, fun q r => if r = q then 1 else 0, fun π r => if π r then 1 else -1,
      fun q π => by simp only [zero_add, ite_mul, one_mul, zero_mul, Finset.sum_ite_eq',
        Finset.mem_univ, ↓reduceIte]⟩
    fun π q => by cases π q <;> simp

/-- **Recall decoded by an argmax takes cut width at least the number of keys**
(`review/flat_spectrum.md`, AR5): if the difference of the logits of two values has cut width
`d` and the largest logit is always that of the bound value, then `m ≤ d`. -/
theorem card_le_of_argmax {d : ℕ} {L : ν → κ → (κ → ν) → ℝ} {v₀ v₁ : ν} (hv : v₀ ≠ v₁)
    (hL : HasCutWidth d fun q π => L v₁ q π - L v₀ q π)
    (h : ∀ π q v, v ≠ π q → L v q π < L (π q) q π) : Fintype.card κ ≤ d := by
  refine (hL.comp_context fun (β : κ → Bool) q => if β q then v₁ else v₀).card_le_of_sign
    fun β q => ?_
  cases hβ : β q
  · have h1 := h (fun q => if β q then v₁ else v₀) q v₁ (by simp [hβ, hv.symm])
    simp only [hβ, Bool.false_eq_true, ↓reduceIte] at h1
    simp only [Bool.false_eq_true, iff_false, not_lt, sub_nonpos]
    exact h1.le
  · have h1 := h (fun q => if β q then v₁ else v₀) q v₀ (by simp [hβ, hv])
    simp only [hβ, ↓reduceIte] at h1
    simp only [iff_true, sub_pos]
    exact h1

/-- Two keys, two values and one-hot logits of width two. -/
example : Fintype.card (Fin 2) ≤ 2 :=
  card_le_of_argmax (ν := Bool) (L := fun v q π => if π q = v then 1 else 0)
    (v₀ := false) (v₁ := true) (by decide)
    ⟨fun _ => 0, fun q r => if r = q then 1 else 0, fun π r => if π r then 1 else -1,
      fun q π => by
        simp only [zero_add, ite_mul, one_mul, zero_mul, Finset.sum_ite_eq', Finset.mem_univ,
          ↓reduceIte]
        cases π q <;> simp⟩
    fun π q v hv => by simp [hv.symm]

end Transformer.Grokking.FlatSpectrum.Recall
