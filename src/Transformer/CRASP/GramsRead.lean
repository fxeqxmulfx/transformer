module

public import Transformer.CRASP.GramsLayer

/-!
# Positional logits read the last tokens: the attention of the reading layers

In no paper; the construction is `Transformer.CRASP.Grams`, for the rounded transformers of
arXiv:2506.16055v3 (Appendix B.1, `def:transformer`, Equation `eq:att`), with a logit `L` of the
positions and the dot product (`Transformer.CRASP.LTfr`).

At precision `p ≥ 3`, with `b < 2^(p-1)` and `ReadsPrev L b`, the dot product of a reading layer
is `b` between differing grams and `0` between equal ones (`dot_read`), so attention keeps `i`
and, if its gram differs, `i - 1` (`sum_read`), and the average of the values `2` at their
coordinates is the sum of the two one-hot grams in either case (`val_attention_read`).
`Transformer.CRASP.GramsCorrect` adds the residual and the feed-forward map (`layer_read`).
-/

@[expose] public section

namespace Transformer.CRASP.Grams

universe u v
variable {σ : Type u} {Γ : Type v} {n p d k : ℕ}

/-- The values `-1, 0, 1, 2, 3` are exact at precision `p ≥ 3`. -/
theorem val_round_small (hp : 3 ≤ p) {x y : ℝ} (hxy : x = y)
    (hy : y = -1 ∨ y = 0 ∨ y = 1 ∨ y = 2 ∨ y = 3) : (Fx.round p 0 x).val = y := by
  subst hxy
  rcases hy with rfl | rfl | rfl | rfl | rfl
  · exact val_round_int hp (-1) (by norm_num) (by norm_num) (by norm_num)
  · exact val_round_int hp 0 (by norm_num) (by norm_num) (by norm_num)
  · exact val_round_int hp 1 (by norm_num) (by norm_num) (by norm_num)
  · exact val_round_int hp 2 (by norm_num) (by norm_num) (by norm_num)
  · exact val_round_int hp 3 (by norm_num) (by norm_num) (by norm_num)

/-- The hypotheses of `val_round_small` hold for `1 + 1` and `2`. -/
example : (Fx.round 3 0 (1 + 1)).val = 2 := val_round_small le_rfl (by norm_num) (by norm_num)

variable [DecidableEq σ]

theorem val_onehot (hp : 3 ≤ p) (c c' : Gram σ n) :
    (onehot (p := p) c c').val = if c' = c then 1 else 0 := by
  unfold onehot
  split_ifs
  · exact val_round_small hp rfl (by norm_num)
  · exact val_round_small hp rfl (by norm_num)

/-- The hypothesis of `val_onehot` holds at precision `3`. -/
example (c : Gram Bool 0) : (onehot (p := 3) c c).val = if c = c then 1 else 0 :=
  val_onehot le_rfl c c

variable [Fintype σ] (L : ℕ → ℕ → ℝ → ℝ) (b : ℕ) (M : RTfr (Option Γ) p 0 d k)
  (κ : Gram σ n → Γ)

/-- The dot product of a reading layer: `b` if the grams differ, `0` if not. -/
theorem dot_read (hp : 3 ≤ p) (hb : b < 2 ^ (p - 1)) {ℓ : ℕ} (hℓ : ℓ < n) (c c' : Gram σ n) :
    ∑ t, ((model L b M κ).WQ ℓ (enc M κ c) t).val * ((model L b M κ).WK ℓ (enc M κ c') t).val =
      if c' = c then 0 else (b : ℝ) := by
  have hq : (model L b M κ).WQ ℓ (enc M κ c) =
      mk 0 fun c'' => Fx.round p 0 (b * (1 - (onehot (p := p) c c'').val)) := by
    simp [model, hℓ, enc]
  have hk : (model L b M κ).WK ℓ (enc M κ c') = mk 0 (onehot c') := by
    simp [model, hℓ, enc]
  rw [hq, hk, sum_width]
  simp only [mk_castAdd, mk_natAdd, Pi.zero_apply, Fx.val_zero, Finset.sum_const_zero,
    zero_add, val_onehot hp, mul_ite, mul_one, mul_zero, Finset.sum_ite_eq', Finset.mem_univ,
    ite_true]
  split_ifs
  · simp
  · rw [sub_zero, mul_one, val_round_natCast hb]

/-- The hypotheses of `dot_read` hold for ALiBi of slope `1` at precision `3`, in the first of
its reading layers. -/
example (M : RTfr (Option (Gram Bool 1)) 3 0 0 0) (c c' : Gram Bool 1) :=
  dot_read (alibiLogit (1 : ℕ)) 1 M id le_rfl (by norm_num) (ℓ := 0) (by norm_num) c c'

/-- The logit of a reading layer: `L` of `b` if the grams differ, of `0` if not. -/
theorem logit_read (hp : 3 ≤ p) (hb : b < 2 ^ (p - 1)) {ℓ : ℕ} (hℓ : ℓ < n) (c : ℕ → Gram σ n)
    (i j : ℕ) :
    (model L b M κ).logit i j (∑ t, ((model L b M κ).WQ ℓ (enc M κ (c i)) t).val *
      ((model L b M κ).WK ℓ (enc M κ (c j)) t).val) = L i j (if c j = c i then 0 else (b : ℝ)) := by
  rw [dot_read L b M κ hp hb hℓ]
  rfl

example (M : RTfr (Option (Gram Bool 1)) 3 0 0 0) (c : ℕ → Gram Bool 1) :=
  logit_read (alibiLogit (1 : ℕ)) 1 M id le_rfl (by norm_num) (ℓ := 0) (by norm_num) c 1 0

theorem WV_read {ℓ : ℕ} (hℓ : ℓ < n) (c : Gram σ n) :
    (model L b M κ).WV ℓ (enc M κ c) =
      mk 0 fun c'' => Fx.round p 0 (2 * (onehot (p := p) c c'').val) := by
  simp [model, hℓ, enc]

example (M : RTfr (Option (Gram Bool 1)) 3 0 0 0) (c : Gram Bool 1) :=
  WV_read (alibiLogit (1 : ℕ)) 1 M id (ℓ := 0) (by norm_num) c

/-- The value of a reading layer: `2` at the coordinate of the gram. -/
theorem val_WV_read (hp : 3 ≤ p) {ℓ : ℕ} (hℓ : ℓ < n) (c c' : Gram σ n) :
    ((model L b M κ).WV ℓ (enc M κ c) (Fin.natAdd d (idx c'))).val =
      if c' = c then 2 else 0 := by
  rw [WV_read L b M κ hℓ, mk_natAdd, val_onehot hp]
  split_ifs
  · exact val_round_small hp (by norm_num) (by norm_num)
  · exact val_round_small hp (by norm_num) (by norm_num)

example (M : RTfr (Option (Gram Bool 1)) 3 0 0 0) (c c' : Gram Bool 1) :=
  val_WV_read (alibiLogit (1 : ℕ)) 1 M id le_rfl (ℓ := 0) (by norm_num) c c'

/-- The values of a reading layer lie in `{0, 2}`, and `round` keeps them. -/
theorem bound_WV_read (hp : 3 ≤ p) {ℓ : ℕ} (hℓ : ℓ < n) (c : Gram σ n)
    (t : Fin (width σ n d)) :
    0 ≤ ((model L b M κ).WV ℓ (enc M κ c) t).val ∧
      ((model L b M κ).WV ℓ (enc M κ c) t).val ≤ 2 ∧
      (Fx.round p 0 ((model L b M κ).WV ℓ (enc M κ c) t).val).val =
        ((model L b M κ).WV ℓ (enc M κ c) t).val := by
  refine ⟨?_, ?_, by rw [Fx.round_val]⟩ <;>
  · induction t using Fin.addCases with
    | left t => rw [WV_read L b M κ hℓ, mk_castAdd]; simp
    | right t =>
        obtain ⟨c', rfl⟩ := idx.surjective t
        rw [val_WV_read L b M κ hp hℓ]
        split_ifs <;> norm_num

example (M : RTfr (Option (Gram Bool 1)) 3 0 0 0) (c : Gram Bool 1) (t : Fin (width Bool 1 0)) :=
  bound_WV_read (alibiLogit (1 : ℕ)) 1 M id le_rfl (ℓ := 0) (by norm_num) c t

/-- **The attention of a reading layer** at the coordinate of `c'`: `1` for each of the grams at
`i` and `i - 1` that is `c'`. -/
theorem val_attention_read (hp : 3 ≤ p) (hb : b < 2 ^ (p - 1)) (hL : ReadsPrev L b) {ℓ : ℕ}
    (hℓ : ℓ < n) (c : ℕ → Gram σ n) {N : ℕ} (i : Fin N) (c' : Gram σ n) :
    ((model L b M κ).attention ℓ (fun j => enc M κ (c j.val)) i (Fin.natAdd d (idx c'))).val =
      (if c' = c i.val then 1 else 0) + (if c' = c (i.val - 1) then 1 else 0) := by
  let z : ℕ → ℝ := fun j => L i.val j (if c j = c i.val then 0 else (b : ℝ))
  have hself : z i.val = 0 := by simp only [z, ite_true]; exact hL.self _
  have hprev : 0 < i.val → c (i.val - 1) ≠ c i.val → z (i.val - 1) = 0 := fun h hc => by
    simp only [z, ite_eq_right hc]; exact hL.prev _ h
  have hsame : 0 < i.val → c (i.val - 1) = c i.val → z (i.val - 1) ≤ -1 := fun h hc => by
    simp only [z, ite_eq_left hc]; exact hL.prevSame _ h
  have hfar : ∀ j, j + 2 ≤ i.val → z j ≤ -1 := fun j hj => by
    simp only [z]; split_ifs
    · exact (hL.far _ _ hj).1
    · exact (hL.far _ _ hj).2
  have hD := sum_read (p := p) c i z (fun _ => (1 : ℝ)) (fun _ =>
    ⟨zero_le_one, by norm_num, val_round_small hp rfl (by norm_num)⟩) hself hprev hsame hfar
  simp only [mul_one] at hD
  have hN := sum_read (p := p) c i z
    (fun j => ((model L b M κ).WV ℓ (enc M κ (c j.val)) (Fin.natAdd d (idx c'))).val)
    (fun j => bound_WV_read L b M κ hp hℓ _ _) hself hprev hsame hfar
  simp only [LTfr.attention, logit_read L b M κ hp hb hℓ c]
  rw [hD, hN]
  by_cases hc : 0 < i.val ∧ c (i.val - 1) ≠ c i.val
  · simp only [dite_eq_left hc, val_WV_read L b M κ hp hℓ]
    rw [ite_eq_right (by norm_num : ¬ ((1 : ℝ) + 1 = 0))]
    exact val_round_small hp (by split_ifs <;> norm_num) (by split_ifs <;> norm_num)
  · have hci : c (i.val - 1) = c i.val := by
      by_cases hi : i.val = 0
      · rw [hi]
      · by_contra hne
        exact hc ⟨by omega, hne⟩
    simp only [dite_eq_right hc, add_zero, val_WV_read L b M κ hp hℓ]
    rw [ite_eq_right (by norm_num : ¬ ((1 : ℝ) = 0)), hci]
    exact val_round_small hp (by split_ifs <;> norm_num) (by split_ifs <;> norm_num)

example (M : RTfr (Option (Gram Bool 1)) 3 0 0 0) (c : ℕ → Gram Bool 1) (i : Fin 2)
    (c' : Gram Bool 1) :=
  val_attention_read (alibiLogit (1 : ℕ)) 1 M id le_rfl (by norm_num) (readsPrev_alibi le_rfl)
    (ℓ := 0) (by norm_num) c i c'

end Transformer.CRASP.Grams
