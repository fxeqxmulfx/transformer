module

public import Mathlib.InformationTheory.Hamming
public import Transformer.MRCR.NearMissThreshold

/-!
# Keys of several parts: near misses triple the scale at depth three

Source: MRCR-V2 (8-needle) of arXiv:2507.06261v6, Appendix, line 75 of
`8_appendix/1_evaluation_additional_details.tex`: keys of depth 3, a topic, a format and a style;
the scale `s log n` of SSMax, arXiv:2501.19399v1, §2.2.

Let the raw score add `w` for each of the `d` parts on which a key agrees with the query
(`partScore`). A key that differs in `t` parts trails the queried key by `t w`
(`partScore_add`).

* Keys that differ from the queried key in every part trail by `d w`, so `s d w > 1` recalls at
  every length (`partScore_recall_of_far`).
* Near misses differ in one part and trail by `w`. Over distinct keys `s w > 1` recalls at every
  length whatever their number (`partScore_recall`); with `s w < 1` and a number of them linear in
  the length, the indicator dictionary is answered with an error that tends to `2`
  (`partScore_nearMiss_error`).

At `d = 3`, near misses in a number linear in the length raise the threshold from `s w = 1/3`
to `s w = 1`. Additive scores are a model: the report does not say how a trained model scores
the parts.
-/

@[expose] public section

namespace Transformer.MRCR

open Finset Filter Transformer.Grokking.FlatSpectrum.Recall

/-- **Additive raw scores over `d` parts** (MRCR-V2: a topic, a format and a style): `w` for each
part on which the key `k` agrees with the query `q`. -/
def partScore {A : Type*} [DecidableEq A] {d : ℕ} (w : ℝ) (q k : Fin d → A) : ℝ :=
  w * ((d : ℝ) - hammingDist q k)

/-- A key that differs from the query in `t` parts trails the query's own key by `t w`. -/
theorem partScore_add {A : Type*} [DecidableEq A] {d : ℕ} (w : ℝ) (q k : Fin d → A) :
    partScore w q k + w * hammingDist q k = partScore w q q := by
  simp only [partScore, hammingDist_self, Nat.cast_zero, sub_zero]
  ring

/-- Keys `0, 1, 2, …` written in each of `d` parts, over `ℕ`: two of them differ in every part. -/
def constKey (d m : ℕ) (k : Fin (m + 1)) : Fin d → ℕ := fun _ => k.val

theorem hammingDist_constKey (d m : ℕ) (q k : Fin (m + 1)) :
    hammingDist (constKey d m q) (constKey d m k) = if q = k then 0 else d := by
  split_ifs with h
  · subst h
    exact hammingDist_self _
  · have hp : ∀ i ∈ (univ : Finset (Fin d)), constKey d m q i ≠ constKey d m k i :=
      fun _ _ => Fin.val_ne_iff.2 h
    rw [hammingDist, filter_true_of_mem hp, card_univ, Fintype.card_fin]

theorem constKey_injective (m : ℕ) : Function.Injective (constKey 1 m) :=
  fun _ _ h => Fin.ext (congrFun h 0)

/-- **Above `s w = 1` part scores recall at every length**, whatever the near misses: over
distinct keys `key m : Fin (m + 1) → (Fin d → A)` with the raw score
`partScore w (key m q) (key m k)`, `s w > 1` recalls values bounded by `C` to every accuracy
`ε > 0` from some number of keys on. -/
theorem partScore_recall {A ν : Type*} [DecidableEq A] {d : ℕ} {s w C ε : ℝ} (hs : 0 ≤ s)
    (hsw : 1 < s * w) (key : (m : ℕ) → Fin (m + 1) → Fin d → A)
    (hkey : ∀ m, Function.Injective (key m)) {φ : ν → ℝ} (hφ : ∀ v, |φ v| ≤ C) (hε : 0 < ε) :
    ∃ M : ℕ, ∀ m ≥ M, ∀ (q : Fin (m + 1)) (π : Fin (m + 1) → ν),
      |logScaleAttention s (fun k => partScore w (key m q) (key m k)) (fun k => φ (π k)) -
        φ (π q)| ≤ ε := by
  have hw : 0 < w := by
    by_contra h
    nlinarith [mul_nonpos_of_nonneg_of_nonpos hs (not_lt.1 h)]
  refine exists_forall_logScaleAttention_recall hs hsw _ (fun m q k hk => ?_) hφ hε
  have h1 : (1 : ℝ) ≤ hammingDist (key m q) (key m k) := by
    exact_mod_cast hammingDist_pos.2 fun h => hk ((hkey m) h).symm
  rw [← partScore_add w (key m q) (key m k)]
  nlinarith

/-- Keys of one part over `ℕ` at `s = 1`, `w = 2`. -/
example := partScore_recall (ν := Bool) (s := 1) (w := 2) (ε := 1 / 10) zero_le_one
  (by norm_num) (constKey 1) constKey_injective (φ := fun b => if b then 1 else -1) (C := 1)
  (fun b => by cases b <;> simp) (by norm_num)

/-- **Below `s w = 1` near misses fade part scores**: with the raw score
`partScore w (key m q) (key m k)` and `s w < 1`, if from some number of keys on at least
`c (m + 1)` keys, `c > 0`, differ from the queried key in one part, attention at the scale
`s log (m + 1)` answers the indicator dictionary with an error above every `ε < 2` from some
number of keys on. -/
theorem partScore_nearMiss_error {A : Type*} [DecidableEq A] {d : ℕ} {s w c ε : ℝ}
    (hs : 0 ≤ s) (hsw : s * w < 1) (hc : 0 < c) (hε : ε < 2)
    (key : (m : ℕ) → Fin (m + 1) → Fin d → A)
    (hN : ∀ᶠ m : ℕ in atTop, ∀ q, c * ((m : ℝ) + 1) ≤
      (univ.filter fun k => hammingDist (key m q) (key m k) = 1).card) :
    ∃ M : ℕ, ∀ m ≥ M, ∀ q : Fin (m + 1),
      ε < 1 - logScaleAttention s (fun k => partScore w (key m q) (key m k))
        fun k => if k = q then 1 else -1 := by
  refine exists_lt_logScaleAttention_nearMiss_error (g₁ := w) (α := 1) hs hc (by linarith) hε
    (fun m q k => partScore w (key m q) (key m k))
    (fun m q => univ.filter fun k => hammingDist (key m q) (key m k) = 1)
    (hN.mono fun m hm q => ?_) fun m q k hk => ?_
  · rw [Real.rpow_one, erase_eq_of_notMem (by simp)]
    exact hm q
  · rw [← partScore_add w (key m q) (key m k), (mem_filter.1 hk).2]
    simp

/-- Keys of one part over `ℕ` at `s = 1`, `w = 1/2`: every other key is a near miss. -/
example := partScore_nearMiss_error (s := 1) (w := 1 / 2) (c := 1 / 2) (ε := 1) zero_le_one
  (by norm_num) (by norm_num) (by norm_num) (constKey 1)
  (eventually_atTop.2 ⟨1, fun m hm q => by
    have hf : (univ.filter fun k => hammingDist (constKey 1 m q) (constKey 1 m k) = 1) =
        univ.erase q := by
      ext k
      by_cases h : q = k
      · subst h
        simp [hammingDist_constKey]
      · simp [hammingDist_constKey, h, Ne.symm h]
    rw [hf, card_erase_of_mem (mem_univ q), card_univ, Fintype.card_fin, Nat.add_sub_cancel]
    have : (1 : ℝ) ≤ m := by exact_mod_cast hm
    linarith⟩)

/-- **Without near misses the whole gap counts**: if every other key differs from the queried
key in all `d` parts, `s d w > 1` recalls at every length. For `d = 3` that is a third of the
scale that near misses ask for (`partScore_nearMiss_error`). -/
theorem partScore_recall_of_far {A ν : Type*} [DecidableEq A] {d : ℕ} {s w C ε : ℝ}
    (hs : 0 ≤ s) (hsw : 1 < s * (d * w)) (key : (m : ℕ) → Fin (m + 1) → Fin d → A)
    (hfar : ∀ m q k, k ≠ q → hammingDist (key m q) (key m k) = d) {φ : ν → ℝ}
    (hφ : ∀ v, |φ v| ≤ C) (hε : 0 < ε) :
    ∃ M : ℕ, ∀ m ≥ M, ∀ (q : Fin (m + 1)) (π : Fin (m + 1) → ν),
      |logScaleAttention s (fun k => partScore w (key m q) (key m k)) (fun k => φ (π k)) -
        φ (π q)| ≤ ε := by
  refine exists_forall_logScaleAttention_recall hs hsw _ (fun m q k hk => ?_) hφ hε
  rw [← partScore_add w (key m q) (key m k), hfar m q k hk]
  exact le_of_eq (by ring)

/-- Keys of three equal parts over `ℕ`, which differ in every part, at `s = 1` and `w = 1/2`:
`s d w = 3/2 > 1` while `s w = 1/2 < 1`. -/
example := partScore_recall_of_far (ν := Bool) (d := 3) (s := 1) (w := 1 / 2) (ε := 1 / 10)
  zero_le_one (by norm_num) (constKey 3)
  (fun m q k hk => by simp [hammingDist_constKey, Ne.symm hk])
  (φ := fun b => if b then 1 else -1) (C := 1) (fun b => by cases b <;> simp) (by norm_num)


end Transformer.MRCR
