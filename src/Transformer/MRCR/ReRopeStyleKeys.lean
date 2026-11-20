module

public import Transformer.MRCR.ReRopeSlots
public import Transformer.MRCR.PhraseKeyStyle

/-!
# The keys of MRCR-V2 read off four rounded ReRoPE slots

In no paper. Source of the task: MRCR-V2 of arXiv:2507.06261v6, line 75 of
`8_appendix/1_evaluation_additional_details.tex`, keys of depth three spelled by words as in
`Transformer.MRCR.PhraseKeyStyle`: an ask is `fmt f`, `about`, `topic t`, `style σ`. The position
`x` carries the key of the ask that ends just before it, `rowKey w x`, and `none` if no ask does
(`rowKey_eq_some`).

Layer 1 of `Transformer.MRCR.ReRopeStyleRecall` reads that key. Four slot heads of
`Transformer.MRCR.ReRopeSlots` at the distances `4, 3, 2, 1 < W`, rounded by `clampUnit`, return
the tokens there exactly at every length (`Transformer.MRCR.clampUnit_slotHead`), and the first
token while `x` is smaller. One ReLU unit per key, `max(0, Σ slots − 3)`, fires iff the slots
spell its ask (`relu_slots_eq`, `keyUnit_eq`). Before the position `4` the slots of the distances
`4` and `3` both read the first token, which is not both `fmt f` and `about`, so no unit fires.
With the unit `1 − Σ units` of `none` the units form the one-hot vector of `rowKey w x`
(`keyVec_eq`), and their dot product at two positions is `1` iff the keys agree
(`sum_keyVec_mul`): the raw score of the counting head `Transformer.MRCR.countHead`.
-/

@[expose] public section

namespace Transformer.MRCR

open Finset

variable {K F N S V R : ℕ}

/-- The key that four tokens spell: `some (f, t, σ)` iff they are `fmt f`, `about`, `topic t`,
`style σ`. -/
def gramKey : Option (StyleTok F N S V R) → Option (StyleTok F N S V R) →
    Option (StyleTok F N S V R) → Option (StyleTok F N S V R) → Option (Fin F × Fin N × Fin S)
  | some (.fmt f), some .about, some (.topic t), some (.style σ) => some (f, t, σ)
  | _, _, _, _ => none

theorem gramKey_eq_some {a b c d : Option (StyleTok F N S V R)} {f : Fin F} {t : Fin N}
    {σ : Fin S} : gramKey a b c d = some (f, t, σ) ↔
      a = some (.fmt f) ∧ b = some .about ∧ c = some (.topic t) ∧ d = some (.style σ) := by
  constructor
  · intro h
    unfold gramKey at h
    split at h
    · simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, rfl, rfl⟩ := h
      exact ⟨rfl, rfl, rfl, rfl⟩
    · exact absurd h (by simp)
  · rintro ⟨rfl, rfl, rfl, rfl⟩
    rfl

/-- **The key of a position**: the key of the ask whose style word stands just before it. -/
def rowKey (w : List (StyleTok F N S V R)) (x : ℕ) : Option (Fin F × Fin N × Fin S) :=
  gramKey w[x - 4]? w[x - 3]? w[x - 2]? w[x - 1]?

/-- The position `x` carries the key `(f, t, σ)` iff an ask "`f` about `t` in the style `σ`"
starts at the index `x - 4`. -/
theorem rowKey_eq_some {w : List (StyleTok F N S V R)} {x : ℕ} {f : Fin F} {t : Fin N}
    {σ : Fin S} : rowKey w x = some (f, t, σ) ↔ 4 ≤ x ∧ StyleAskAt w f t σ (x - 4) := by
  rw [rowKey, gramKey_eq_some]
  constructor
  · rintro ⟨h4, h3, h2, h1⟩
    by_cases hx : 4 ≤ x
    · refine ⟨hx, h4, ?_, ?_, ?_⟩
      · rwa [show x - 4 + 1 = x - 3 by omega]
      · rwa [show x - 4 + 2 = x - 2 by omega]
      · rwa [show x - 4 + 3 = x - 1 by omega]
    · rw [show x - 4 = x - 3 by omega, h3] at h4
      exact absurd h4 (by simp)
  · rintro ⟨hx, h4, h3, h2, h1⟩
    refine ⟨h4, ?_, ?_, ?_⟩
    · rwa [show x - 3 = x - 4 + 1 by omega]
    · rwa [show x - 2 = x - 4 + 2 by omega]
    · rwa [show x - 1 = x - 4 + 3 by omega]

/-- **One ReLU unit reads a key off exact slots**: on the slots `[w[x − k]? = a]` the unit
`max(0, Σ − 3)` of the key `(f, t, σ)` is `1` iff the position `x` carries that key, else `0`. -/
theorem relu_slots_eq (w : List (StyleTok F N S V R)) (x : ℕ) (f : Fin F) (t : Fin N)
    (σ : Fin S) :
    max 0 ((if w[x - 4]? = some (.fmt f) then 1 else 0) +
        (if w[x - 3]? = some .about then 1 else 0) +
        (if w[x - 2]? = some (.topic t) then 1 else 0) +
        (if w[x - 1]? = some (.style σ) then 1 else 0) - 3 : ℝ) =
      if rowKey w x = some (f, t, σ) then 1 else 0 := by
  simp only [rowKey, gramKey_eq_some]
  split_ifs <;> first | (exfalso; tauto) | norm_num [max_def]

/-- **The ReLU unit of the key `(f, t, σ)`** on the four rounded slots of the position `x`. -/
noncomputable def keyUnit (s A : ℝ) (θ : Fin K → ℝ) (m : Fin K) (W : ℕ)
    (w : List (StyleTok F N S V R)) (x : ℕ) (f : Fin F) (t : Fin N) (σ : Fin S) : ℝ :=
  max 0 (clampUnit (slotHead s A θ m W w x 4 (.fmt f)) +
    clampUnit (slotHead s A θ m W w x 3 .about) + clampUnit (slotHead s A θ m W w x 2 (.topic t)) +
    clampUnit (slotHead s A θ m W w x 1 (.style σ)) - 3)

/-- **Layer 1 reads the keys exactly at every length**: for `s A (1 − cos θ_m) > 4`, a window
`W > 4` and `θ_m W ≤ π` the unit of `(f, t, σ)` at the position `x` is `1` iff `x` carries that
key, and `0` otherwise. -/
theorem keyUnit_eq {s A : ℝ} (hA : 0 ≤ A) (hs : 0 ≤ s) {θ : Fin K → ℝ} {m : Fin K}
    (hθ : 0 < θ m) {W : ℕ} (hW : 4 < W) (hθW : θ m * W ≤ Real.pi)
    (hsg : 4 < s * (A * (1 - Real.cos (θ m)))) (w : List (StyleTok F N S V R)) (x : ℕ)
    (f : Fin F) (t : Fin N) (σ : Fin S) :
    keyUnit s A θ m W w x f t σ = if rowKey w x = some (f, t, σ) then 1 else 0 := by
  rw [keyUnit, clampUnit_slotHead hA hs hθ hW hθW hsg,
    clampUnit_slotHead hA hs hθ (by omega : 3 < W) hθW hsg,
    clampUnit_slotHead hA hs hθ (by omega : 2 < W) hθW hsg,
    clampUnit_slotHead hA hs hθ (by omega : 1 < W) hθW hsg]
  exact relu_slots_eq w x f t σ

/-- The units of the position `x` as a vector over the keys, with the unit `1 − Σ units` of
`none`. -/
noncomputable def keyVec (s A : ℝ) (θ : Fin K → ℝ) (m : Fin K) (W : ℕ)
    (w : List (StyleTok F N S V R)) (x : ℕ) : Option (Fin F × Fin N × Fin S) → ℝ
  | some k => keyUnit s A θ m W w x k.1 k.2.1 k.2.2
  | none => 1 - ∑ k : Fin F × Fin N × Fin S, keyUnit s A θ m W w x k.1 k.2.1 k.2.2

section Vec

variable {s A : ℝ} (hA : 0 ≤ A) (hs : 0 ≤ s) {θ : Fin K → ℝ} {m : Fin K} (hθ : 0 < θ m) {W : ℕ}
  (hW : 4 < W) (hθW : θ m * W ≤ Real.pi) (hsg : 4 < s * (A * (1 - Real.cos (θ m))))
  (w : List (StyleTok F N S V R))
include hA hs hθ hW hθW hsg

/-- **The units are the one-hot vector of the key** of the position `x`. -/
theorem keyVec_eq (x : ℕ) (o : Option (Fin F × Fin N × Fin S)) :
    keyVec s A θ m W w x o = if rowKey w x = o then 1 else 0 := by
  cases o with
  | some k => exact keyUnit_eq hA hs hθ hW hθW hsg w x k.1 k.2.1 k.2.2
  | none =>
    simp only [keyVec, keyUnit_eq hA hs hθ hW hθW hsg]
    rcases rowKey w x with _ | k
    · simp
    · simp [Finset.sum_ite_eq]

/-- **The raw score of the counting head**: the dot product of the units of two positions is `1`
iff they carry the same key, and `0` otherwise. -/
theorem sum_keyVec_mul (x y : ℕ) :
    ∑ o, keyVec s A θ m W w x o * keyVec s A θ m W w y o =
      if rowKey w x = rowKey w y then 1 else 0 := by
  simp only [keyVec_eq hA hs hθ hW hθW hsg, ite_mul, one_mul, zero_mul, Finset.sum_ite_eq,
    mem_univ, ite_true]
  rcases eq_or_ne (rowKey w x) (rowKey w y) with h | h
  · simp [h]
  · simp [h, Ne.symm h]

end Vec

/-- The hypotheses of `keyUnit_eq` hold at `s = 40`, `A = 1`, `θ = π/6` and the window `6`, where
`40 (1 − √3/2) > 4`. -/
example (w : List (StyleTok 1 1 1 1 8)) (x : ℕ) :=
  keyUnit_eq (s := 40) (A := 1) zero_le_one (by norm_num) (θ := fun _ : Fin 1 => Real.pi / 6)
    (m := 0) (by positivity) (W := 6) (by norm_num) (by push_cast; linarith)
    (by rw [Real.cos_pi_div_six]; nlinarith [Real.sq_sqrt (show (0 : ℝ) ≤ 3 by norm_num),
      Real.sqrt_nonneg 3]) w x 0 0 0

/-- The same hypotheses for `keyVec_eq` and `sum_keyVec_mul`. -/
example (w : List (StyleTok 1 1 1 1 8)) (x y : ℕ) :=
  sum_keyVec_mul (s := 40) (A := 1) zero_le_one (by norm_num) (θ := fun _ : Fin 1 => Real.pi / 6)
    (m := 0) (by positivity) (W := 6) (by norm_num) (by push_cast; linarith)
    (by rw [Real.cos_pi_div_six]; nlinarith [Real.sq_sqrt (show (0 : ℝ) ≤ 3 by norm_num),
      Real.sqrt_nonneg 3]) w x y

example (w : List (StyleTok 1 1 1 1 8)) (x : ℕ) :=
  keyVec_eq (s := 40) (A := 1) zero_le_one (by norm_num) (θ := fun _ : Fin 1 => Real.pi / 6)
    (m := 0) (by positivity) (W := 6) (by norm_num) (by push_cast; linarith)
    (by rw [Real.cos_pi_div_six]; nlinarith [Real.sq_sqrt (show (0 : ℝ) ≤ 3 by norm_num),
      Real.sqrt_nonneg 3]) w x none

end Transformer.MRCR
