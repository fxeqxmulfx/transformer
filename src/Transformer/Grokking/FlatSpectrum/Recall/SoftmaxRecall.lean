module

public import Transformer.Grokking.FlatSpectrum.Recall.Softmax

/-!
# Softmax attention recalls every number of bindings at dimension two

Source: `review/flat_spectrum.md`, *Softmax attention escapes the cut* (added), item SA2, for
the attention of arXiv:2506.16055v3, §2.1, and for the second half of Claim 1 of Arora et al.,
arXiv:2312.04927v1, §4: "attention models can solve it with near-constant dimensionality".

Let the token `j` carry the key `j` and the value `φ(π j)` of a dictionary `π` of `m` keys, and
project to dimension two: the query `q` to `β (q, 1)`, the key `j` to `(j, −j²/2)`. The score
`β (q j − j²/2) = β (q² − (q − j)²)/2` is largest at `j = q`, by `β/2`. So the attention is
within `2 B (m − 1) e^{−β/2}` of `φ(π q)` (`softmaxAttention_recall_error`), and within
`2 B m e^{−β/2}` when the query's own token is attended too
(`softmaxAttention_self_recall_error`). At dimension two the error is below every `ε > 0` for
every `m` (`exists_softmaxAttention_recall`), and the sign of the attention is the bound value
(`exists_softmaxAttention_recall_sign`). Without the exponential the same scores have cut
width two, and no readout of cut width `d < m` recalls (`Recall.Width`, `Recall.Attention`).
The scale `β` grows with `m`; `Recall.SoftmaxScale` shows that it must.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Finset

/-- The projection of the query `q` to dimension two: `β (q, 1)`. -/
noncomputable def queryProj (β : ℝ) {m : ℕ} (q : Fin m) : Fin 2 → ℝ := ![β * q, β]

/-- The projection of the key `j` to dimension two: `(j, −j²/2)`. -/
noncomputable def keyProj {m : ℕ} (j : Fin m) : Fin 2 → ℝ := ![j, -(j : ℝ) ^ 2 / 2]

/-- The score of the key `j` for the query `q` is `β (q² − (q − j)²)/2`, largest at `j = q`. -/
lemma sum_queryProj_mul_keyProj (β : ℝ) {m : ℕ} (q j : Fin m) :
    ∑ r, queryProj β q r * keyProj j r = β * ((q : ℝ) ^ 2 - ((q : ℝ) - j) ^ 2) / 2 := by
  simp only [queryProj, keyProj, Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one]
  ring

/-- The queried key leads the score of every other key by `β/2`. -/
lemma sum_queryProj_mul_keyProj_add_le {β : ℝ} (hβ : 0 ≤ β) {m : ℕ} {q j : Fin m}
    (hj : j ≠ q) :
    ∑ r, queryProj β q r * keyProj j r + β / 2 ≤ ∑ r, queryProj β q r * keyProj q r := by
  simp only [sum_queryProj_mul_keyProj]
  have h1 : (1 : ℝ) ≤ ((q : ℝ) - j) ^ 2 := by
    have hne : (q : ℤ) ≠ (j : ℤ) := fun h => hj (Fin.ext (by exact_mod_cast h.symm))
    have h2 : (1 : ℤ) ≤ ((q : ℤ) - j) ^ 2 := by
      have := sq_nonneg ((q : ℤ) - j)
      have h0 : ((q : ℤ) - j) ^ 2 ≠ 0 := pow_ne_zero 2 (sub_ne_zero.2 hne)
      omega
    exact_mod_cast h2
  nlinarith [mul_le_mul_of_nonneg_left h1 hβ]

example := sum_queryProj_mul_keyProj_add_le (β := 1) zero_le_one (m := 2) (q := 0) (j := 1)
  (by decide)

/-- **Softmax attention at dimension two on recall** (`review/flat_spectrum.md`, SA2): with the
key `j` and the value `φ(π j)` in the token `j`, the attention is within
`2 B (m − 1) e^{−β/2}` of `φ(π q)`. -/
theorem softmaxAttention_recall_error {ν : Type*} {m : ℕ} {β B : ℝ} (hβ : 0 ≤ β) {φ : ν → ℝ}
    (hφ : ∀ v, |φ v| ≤ B) (q : Fin m) (π : Fin m → ν) :
    |softmaxAttention (fun j => ∑ r, queryProj β q r * keyProj j r) (fun j => φ (π j)) -
      φ (π q)| ≤ 2 * B * (m - 1) * Real.exp (-(β / 2)) := by
  simpa only [Fintype.card_fin] using abs_softmaxAttention_sub_le
    (fun j hj => sum_queryProj_mul_keyProj_add_le hβ hj) fun j => hφ (π j)

/-- The key projection of the query's own token: `(0, −1/2)`, the score `−β/2`. -/
noncomputable def selfProj : Fin 2 → ℝ := ![0, -1 / 2]

/-- **The same with the query's own token among the attended ones** (`review/flat_spectrum.md`,
SA2), as under the future mask of arXiv:2506.16055v3: whatever value `u` that token carries,
the attention over the `m + 1` tokens is within `2 B m e^{−β/2}` of `φ(π q)`. -/
theorem softmaxAttention_self_recall_error {ν : Type*} {m : ℕ} {β B : ℝ} (hβ : 0 ≤ β)
    {φ : ν → ℝ} (hφ : ∀ v, |φ v| ≤ B) {u : ℝ} (hu : |u| ≤ B) (q : Fin m) (π : Fin m → ν) :
    |softmaxAttention
        (fun o : Option (Fin m) => ∑ r, queryProj β q r * o.elim selfProj keyProj r)
        (fun o => o.elim u fun j => φ (π j)) - φ (π q)| ≤ 2 * B * m * Real.exp (-(β / 2)) := by
  have h := abs_softmaxAttention_sub_le
    (s := fun o : Option (Fin m) => ∑ r, queryProj β q r * o.elim selfProj keyProj r)
    (v := fun o : Option (Fin m) => o.elim u fun j => φ (π j)) (j₀ := some q) (g := β / 2)
    (B := B) (fun o ho => ?_) fun o => ?_
  · simpa using h
  · cases o with
    | none =>
      simp only [Option.elim_none, Option.elim_some, sum_queryProj_mul_keyProj]
      simp only [queryProj, selfProj, Fin.sum_univ_two, Matrix.cons_val_zero,
        Matrix.cons_val_one]
      nlinarith [mul_nonneg hβ (sq_nonneg (q : ℝ))]
    | some j => exact sum_queryProj_mul_keyProj_add_le hβ fun h => ho (congrArg some h)
  · cases o with
    | none => exact hu
    | some j => exact hφ (π j)

/-- **At dimension two softmax attention recalls every number of bindings to every accuracy**
(`review/flat_spectrum.md`, SA2). -/
theorem exists_softmaxAttention_recall {ν : Type*} [Finite ν] (m : ℕ) (φ : ν → ℝ) {ε : ℝ}
    (hε : 0 < ε) :
    ∃ (qf kf : Fin m → Fin 2 → ℝ), ∀ (q : Fin m) (π : Fin m → ν),
      |softmaxAttention (fun j => ∑ r, qf q r * kf j r) (fun j => φ (π j)) - φ (π q)| ≤ ε := by
  cases nonempty_fintype ν
  set B : ℝ := ∑ v, |φ v|
  have hB (v : ν) : |φ v| ≤ B :=
    single_le_sum (f := fun v => |φ v|) (fun v _ => abs_nonneg _) (mem_univ v)
  have hB0 : 0 ≤ B := sum_nonneg fun v _ => abs_nonneg _
  set β : ℝ := 4 * B * m / ε + 2
  have hβ : 0 < β := by positivity
  refine ⟨queryProj β, keyProj, fun q π => (softmaxAttention_recall_error hβ.le hB q π).trans ?_⟩
  have hexp : Real.exp (-(β / 2)) ≤ 2 / β := by
    rw [Real.exp_neg, inv_le_comm₀ (Real.exp_pos _) (by positivity), inv_div]
    linarith [Real.add_one_le_exp (β / 2)]
  have hm : (m : ℝ) - 1 ≤ m := by linarith
  calc 2 * B * (m - 1) * Real.exp (-(β / 2)) ≤ 2 * B * m * (2 / β) := by
        refine mul_le_mul (mul_le_mul_of_nonneg_left hm (by positivity)) hexp
          (Real.exp_pos _).le (by positivity)
    _ ≤ ε := by
        rw [show 2 * B * m * (2 / β) = 4 * B * m / β by ring, div_le_iff₀ hβ]
        have h1 : 4 * B * m / ε * ε = 4 * B * m := div_mul_cancel₀ _ hε.ne'
        nlinarith [mul_pos hε (show (0 : ℝ) < 2 by norm_num)]

/-- **At dimension two the sign of softmax attention recalls every number of bindings**
(`review/flat_spectrum.md`, SA2): values `±1`, positive exactly at the keys bound to `true`. -/
theorem exists_softmaxAttention_recall_sign (m : ℕ) :
    ∃ (qf kf : Fin m → Fin 2 → ℝ), ∀ (π : Fin m → Bool) (q : Fin m),
      0 < softmaxAttention (fun j => ∑ r, qf q r * kf j r) (fun j => if π j then 1 else -1) ↔
        π q = true := by
  obtain ⟨qf, kf, h⟩ := exists_softmaxAttention_recall m (fun b : Bool => if b then (1 : ℝ) else -1)
    (ε := 1 / 2) (by norm_num)
  refine ⟨qf, kf, fun π q => ?_⟩
  have h1 := abs_le.1 (h q π)
  cases hq : π q <;> simp only [hq, Bool.false_eq_true, ite_false, ite_true] at h1 <;>
    simp only [Bool.false_eq_true, iff_false, not_lt, iff_true] <;> linarith [h1.1, h1.2]

/-- Three keys with values `±1`, to accuracy `1/10`. -/
example := exists_softmaxAttention_recall 3 (fun b : Bool => if b then (1 : ℝ) else -1)
  (ε := 1 / 10) (by norm_num)

example (q : Fin 3) (π : Fin 3 → Bool) :=
  softmaxAttention_recall_error (β := 4) (B := 1) (by norm_num)
    (φ := fun b : Bool => if b then (1 : ℝ) else -1) (fun b => by cases b <;> simp) q π

example (q : Fin 3) (π : Fin 3 → Bool) :=
  softmaxAttention_self_recall_error (β := 4) (B := 1) (by norm_num)
    (φ := fun b : Bool => if b then (1 : ℝ) else -1) (fun b => by cases b <;> simp) (u := 0)
    (by simp) q π

end Transformer.Grokking.FlatSpectrum.Recall
