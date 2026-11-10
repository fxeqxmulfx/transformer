module

public import Mathlib.Tactic.LinearCombination
public import Mathlib.Tactic.NormNum
public import Transformer.Grokking.FlatSpectrum.BlockCut.Basic
public import Transformer.Grokking.FlatSpectrum.Recall.Defs

/-!
# Five numbers across a cut decide every `L_k` by a sign

Source: `review/flat_spectrum.md`, *The block languages across a cut* (added), item BL3, for
the languages `L_k` of arXiv:2506.16055v3, §2.4, and its acceptance by a sign, Appendix
`app:transformers`: "We say that `T` accepts `w` if `T(w) > 0`."

Cut a word into a prefix `u` and a suffix `v`. By BL1 and BL2, `u ++ v ∈ L_k` exactly when `u`
starts with `a` and `ρ = blocks u + shift v − k + ℓ(u) · slope v` is `0`. With `A(u) = [u starts
with a]`, the number `A (3/2 − ρ²) − 1` is positive exactly then, `ρ` being an integer. It is a
polynomial of degree two in the numbers `blocks u` and `ℓ(u)` of the prefix, and `ℓ² = ℓ`:

    A (3/2 − ρ²) − 1 = −1 + (3/2 − G²) A − A P² − (H² + 2 G H) A ℓ − 2 G A P − 2 H A P ℓ,

`P = blocks u`, `G = shift v − k`, `H = slope v`. So the five numbers `A, A P², A ℓ, A P, A P ℓ`
of the prefix cross the cut (`prefixFeature`), the suffix combines them linearly
(`suffixCoeff`), and the sign of the readout decides `L_k` for every `k`
(`hasCutWidth_blockReadout`, `blockReadout_pos_iff`, `exists_hasCutWidth_altPlus_sign`). The
empty prefix has the five numbers `1, 0, 3/4, 1/4, 1/8`, for which the readout is
`1/2 − (blocks v − k)²` on a suffix that starts with `a` and `G (1 − G) ≤ 0` on one that starts
with `b`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.BlockCut

open Transformer.CRASP Recall

/-- `[u starts with a]` as a number. -/
noncomputable def headA (u : List Bool) : ℝ := if u.head? = some false then 1 else 0

/-- **The five numbers of a prefix that cross the cut** (`review/flat_spectrum.md`, BL3):
`A, A P², A ℓ, A P, A P ℓ` for `A = [u starts with a]`, `P = blocks u`, `ℓ = [u ends with b]`. -/
noncomputable def prefixFeature (u : List Bool) : Fin 5 → ℝ :=
  if u = [] then ![1, 0, 3 / 4, 1 / 4, 1 / 8]
  else ![headA u, headA u * (blocks u : ℝ) ^ 2, headA u * lastB u, headA u * blocks u,
    headA u * blocks u * lastB u]

/-- **The coefficients of a suffix for `L_k`** (`review/flat_spectrum.md`, BL3), with
`G = shift v − k` and `H = slope v`. -/
noncomputable def suffixCoeff (k : ℕ) (v : List Bool) : Fin 5 → ℝ :=
  ![3 / 2 - (shift v - k) ^ 2, -1, -(slope v ^ 2 + 2 * (shift v - k) * slope v),
    -2 * (shift v - k), -2 * slope v]

/-- The readout of the suffix `v` after the prefix `u` for `L_k`. -/
noncomputable def blockReadout (k : ℕ) (v u : List Bool) : ℝ :=
  -1 + ∑ r, suffixCoeff k v r * prefixFeature u r

/-- **The readout has cut width five** (`review/flat_spectrum.md`, BL3). -/
theorem hasCutWidth_blockReadout (k : ℕ) : HasCutWidth 5 (blockReadout k) :=
  ⟨fun _ => -1, suffixCoeff k, prefixFeature, fun _ _ => rfl⟩

/-- Different natural numbers differ by at least one. -/
lemma one_le_sq_sub {n k : ℕ} (h : n ≠ k) : (1 : ℝ) ≤ ((n : ℝ) - k) ^ 2 := by
  have h2 : (1 : ℤ) ≤ ((n : ℤ) - k) ^ 2 := by
    have h0 : ((n : ℤ) - k) ^ 2 ≠ 0 := pow_ne_zero 2 (sub_ne_zero.2 (by exact_mod_cast h))
    have := sq_nonneg ((n : ℤ) - k)
    omega
  exact_mod_cast h2

example : (1 : ℝ) ≤ (((3 : ℕ) : ℝ) - (5 : ℕ)) ^ 2 := one_le_sq_sub (by decide)

/-- An integer `G` has `G (1 − G) ≤ 0`. -/
lemma sub_mul_one_sub_nonpos (n k : ℕ) : ((n : ℝ) - k) * (1 - ((n : ℝ) - k)) ≤ 0 := by
  rcases Nat.lt_or_ge k n with h | h
  · have h1 : (k : ℝ) + 1 ≤ n := by exact_mod_cast h
    nlinarith [mul_nonneg (sub_nonneg.2 h1) (show (0 : ℝ) ≤ (n : ℝ) - k by linarith)]
  · have h1 : (n : ℝ) ≤ k := by exact_mod_cast h
    nlinarith [mul_nonneg (sub_nonneg.2 h1) (show (0 : ℝ) ≤ 1 - ((n : ℝ) - k) by linarith)]

/-- After a prefix that is not empty the readout is `A (3/2 − (blocks (u ++ v) − k)²) − 1`. -/
lemma blockReadout_of_ne_nil (k : ℕ) {u : List Bool} (hu : u ≠ []) (v : List Bool) :
    blockReadout k v u = headA u * (3 / 2 - ((blocks (u ++ v) : ℝ) - k) ^ 2) - 1 := by
  have hl : lastB u * lastB u = lastB u := by
    unfold lastB
    split_ifs <;> simp
  rw [cast_blocks_append hu, blockReadout, Fin.sum_univ_five, prefixFeature, ite_eq_right hu]
  simp only [suffixCoeff, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val]
  linear_combination (headA u * slope v ^ 2) * hl

/-- **After a prefix that is not empty the sign of the readout decides `L_k`**. -/
lemma blockReadout_pos_iff_of_ne_nil (k : ℕ) {u : List Bool} (hu : u ≠ []) (v : List Bool) :
    0 < blockReadout k v u ↔ u ++ v ∈ altPlus false k := by
  rw [blockReadout_of_ne_nil k hu, mem_altPlus_iff (by simp [hu]),
    List.head?_append_of_ne_nil _ hu]
  by_cases hA : u.head? = some false
  · rw [headA, ite_eq_left hA]
    constructor
    · intro h
      refine ⟨hA, ?_⟩
      by_contra hne
      linarith [one_le_sq_sub hne]
    · rintro ⟨-, hb⟩
      rw [hb]
      norm_num
  · rw [headA, ite_eq_right hA]
    exact ⟨fun h => by norm_num at h, fun h => absurd h.1 hA⟩

/-- **After the empty prefix the sign of the readout decides `L_k`**. -/
lemma blockReadout_nil_pos_iff (k : ℕ) (v : List Bool) :
    0 < blockReadout k v [] ↔ v ∈ altPlus false k := by
  rw [blockReadout, Fin.sum_univ_five, prefixFeature, ite_eq_left rfl]
  simp only [suffixCoeff, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val]
  by_cases hv : v = []
  · subst hv
    simp only [shift, slope, ite_true]
    cases k with
    | zero => exact ⟨fun _ => rfl, fun _ => by norm_num⟩
    | succ k =>
      refine ⟨fun h => ?_, fun h => by simpa using head?_of_mem_altPlus h⟩
      have hk : (1 : ℝ) ≤ ((k + 1 : ℕ) : ℝ) := by exact_mod_cast Nat.succ_pos k
      nlinarith [mul_nonneg (sub_nonneg.2 hk) (show (0 : ℝ) ≤ ((k + 1 : ℕ) : ℝ) by linarith)]
  · rw [mem_altPlus_iff hv]
    obtain ⟨y, v, rfl⟩ := List.exists_cons_of_ne_nil hv
    simp only [shift, slope, hv, ite_false, List.head?_cons]
    cases y
    · simp only [Option.some.injEq, Bool.false_eq_true, ite_false, true_and]
      constructor
      · intro h
        by_contra hne
        nlinarith [one_le_sq_sub hne]
      · intro hb
        rw [hb]
        norm_num
    · simp only [ite_true, Option.some.injEq, Bool.true_eq_false, false_and, iff_false, not_lt]
      nlinarith [sub_mul_one_sub_nonpos (blocks (true :: v)) k]

/-- **The sign of five numbers across the cut decides `L_k`** (`review/flat_spectrum.md`,
BL3): the readout is positive exactly when the word `u ++ v` is in `L_k`. -/
theorem blockReadout_pos_iff (k : ℕ) (u v : List Bool) :
    0 < blockReadout k v u ↔ u ++ v ∈ altPlus false k := by
  by_cases hu : u = []
  · subst hu
    exact blockReadout_nil_pos_iff k v
  · exact blockReadout_pos_iff_of_ne_nil k hu v

/-- **Every `L_k` has cut width five by a sign** (`review/flat_spectrum.md`, BL3): for every
`k` a readout of cut width `5` is positive exactly on the words of `L_k`, at every cut. -/
theorem exists_hasCutWidth_altPlus_sign (k : ℕ) :
    ∃ F : List Bool → List Bool → ℝ, HasCutWidth 5 F ∧
      ∀ u v, 0 < F v u ↔ u ++ v ∈ altPlus false k :=
  ⟨blockReadout k, hasCutWidth_blockReadout k, blockReadout_pos_iff k⟩

/-- `ab · ba` is in `L_3`, and the readout for `k = 3` is positive on it. -/
example : 0 < blockReadout 3 [true, false] [false, true] :=
  (blockReadout_pos_iff 3 _ _).2 ((mem_altPlus_iff (by simp)).2 ⟨rfl, by decide⟩)

example := blockReadout_of_ne_nil 3 (u := [false, true]) (by simp) [true, false]

example := blockReadout_pos_iff_of_ne_nil 3 (u := [false, true]) (by simp) [true, false]

end Transformer.Grokking.FlatSpectrum.BlockCut
