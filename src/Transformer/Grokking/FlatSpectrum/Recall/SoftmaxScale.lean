module

public import Transformer.Grokking.FlatSpectrum.Recall.Softmax
public import Transformer.Grokking.FlatSpectrum.Recall.Width

/-!
# Softmax attention pays for recall in the scale of its scores

Source: `review/flat_spectrum.md`, *Softmax attention escapes the cut* (added), item SA3, for
the attention of arXiv:2506.16055v3, §2.1, on the task of arXiv:2312.04927v1, §3.

A token `(k, π k)` for each of the `m` keys; its score `s q k (π k)` is any function of the
query, of the key of the token and of its value, bounded by `B`; its value is `φ(π k)`. Write
`ρ = (m − 1) / (e^{2B} + m − 1)`. Changing `π q` does not change the average `a` of the other
tokens, so the errors at the values `v` of `π q` are at least `ρ |φ(v) − a|` with one `a`
(`exists_forall_mul_abs_sub_le_softmaxAttention`, from `Recall.Softmax`). Summing, with
`c` values and `μ` the mean of `φ`,

    m c^{m−1} ρ² ∑_v (φ(v) − μ)² ≤ ∑_{q, π} (φ(π q) − att)²   (`mul_le_sum_softmaxAttention_sq`),

and recall of values `±1` to accuracy `ε` has `(m − 1)(1 − ε) ≤ ε e^{2B}`
(`sub_one_mul_le_of_softmaxAttention_recall`): the scores grow as `½ log (m / ε)`. One-hot keys
with the scores `±β/2` reach the error `2 (m − 1) e^{−β}` (`softmaxAttention_oneHot_recall_error`),
so the bound is sharp up to the factor `2` of `ε`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Finset

variable {κ ν : Type*} [Fintype κ] [DecidableEq κ] [Fintype ν]

/-- Summing over a dictionary and a new value of the key `q` counts every dictionary once for
each value. -/
lemma sum_sum_update (q : κ) (g : (κ → ν) → ℝ) :
    ∑ π : κ → ν, ∑ v, g (Function.update π q v) = Fintype.card ν * ∑ π, g π := by
  have hinv : Function.Involutive fun x : (κ → ν) × ν => (Function.update x.1 q x.2, x.1 q) :=
    fun x => by simp
  calc ∑ π : κ → ν, ∑ v, g (Function.update π q v)
      = ∑ x : (κ → ν) × ν, g (Function.update x.1 q x.2) :=
        (Fintype.sum_prod_type' fun (π : κ → ν) v => g (Function.update π q v)).symm
    _ = ∑ x : (κ → ν) × ν, g x.1 :=
        Fintype.sum_bijective _ hinv.bijective _ (fun x => g x.1) fun _ => rfl
    _ = Fintype.card ν * ∑ π, g π := by
        rw [Fintype.sum_prod_type]
        simp only [sum_const, card_univ, nsmul_eq_mul, mul_sum]

example (g : (Fin 2 → Fin 2) → ℝ) := sum_sum_update 0 g

omit [Fintype ν] in
/-- **Recall by softmax attention with bounded scores** (`review/flat_spectrum.md`, SA3): a
token `(k, π k)` for each key, a score `s q k (π k)` bounded by `B`, the value `φ(π k)`. Some
`a` that does not depend on `π q` has every value `v` of `π q` answered with an error of at
least `(m − 1) / (e^{2B} + m − 1) · |φ(v) − a|`. -/
theorem exists_forall_mul_abs_sub_le_softmaxAttention {B : ℝ} {s : κ → κ → ν → ℝ}
    (hs : ∀ q k v, |s q k v| ≤ B) (φ : ν → ℝ) (q : κ) (π : κ → ν) :
    ∃ a : ℝ, ∀ v, ((Fintype.card κ : ℝ) - 1) / (Real.exp (2 * B) + Fintype.card κ - 1) *
      |φ v - a| ≤ |φ v - softmaxAttention (fun k => s q k (Function.update π q v k))
        fun k => φ (Function.update π q v k)| := by
  refine ⟨(∑ k ∈ univ.erase q, Real.exp (s q k (π k)) * φ (π k)) /
    ∑ k ∈ univ.erase q, Real.exp (s q k (π k)), fun v => ?_⟩
  have h := mul_abs_sub_le_abs_sub_softmaxAttention
    (s := fun k => s q k (Function.update π q v k)) (v := fun k => φ (Function.update π q v k))
    (fun k => hs q k _) q
  have h1 : ∑ k ∈ univ.erase q, Real.exp (s q k (Function.update π q v k)) *
      φ (Function.update π q v k) = ∑ k ∈ univ.erase q, Real.exp (s q k (π k)) * φ (π k) :=
    sum_congr rfl fun k hk => by rw [Function.update_of_ne (ne_of_mem_erase hk)]
  have h2 : ∑ k ∈ univ.erase q, Real.exp (s q k (Function.update π q v k)) =
      ∑ k ∈ univ.erase q, Real.exp (s q k (π k)) :=
    sum_congr rfl fun k hk => by rw [Function.update_of_ne (ne_of_mem_erase hk)]
  rwa [h1, h2, Function.update_self] at h

/-- Uniform attention over three keys: scores `0`, bounded by `0`. -/
example (π : Fin 3 → Bool) := exists_forall_mul_abs_sub_le_softmaxAttention (B := 0)
  (s := fun _ _ _ => 0) (fun _ _ _ => by simp) (fun b => if b then (1 : ℝ) else -1) 0 π


/-- **The error of softmax attention with bounded scores on recall**
(`review/flat_spectrum.md`, SA3): with `μ` the mean of `φ`, the squared error over all queries
and dictionaries is at least `m c^{m−1} ρ² ∑_v (φ(v) − μ)²`, `ρ = (m − 1) / (e^{2B} + m − 1)`. -/
theorem mul_le_sum_softmaxAttention_sq {B : ℝ} {s : κ → κ → ν → ℝ}
    (hs : ∀ q k v, |s q k v| ≤ B) {φ : ν → ℝ} {μ : ℝ} (hμ : ∑ v, (φ v - μ) = 0) :
    (Fintype.card κ : ℝ) * ((Fintype.card ν : ℝ) ^ (Fintype.card κ - 1) *
      ∑ v, (φ v - μ) ^ 2) *
        (((Fintype.card κ : ℝ) - 1) / (Real.exp (2 * B) + Fintype.card κ - 1)) ^ 2 ≤
      ∑ q, ∑ π : κ → ν,
        (φ (π q) - softmaxAttention (fun k => s q k (π k)) fun k => φ (π k)) ^ 2 := by
  set ρ := ((Fintype.card κ : ℝ) - 1) / (Real.exp (2 * B) + Fintype.card κ - 1)
  set S := ∑ v, (φ v - μ) ^ 2 with hS
  have hrow (q : κ) : (Fintype.card ν : ℝ) ^ (Fintype.card κ - 1) * S * ρ ^ 2 ≤
      ∑ π : κ → ν,
        (φ (π q) - softmaxAttention (fun k => s q k (π k)) fun k => φ (π k)) ^ 2 := by
    have hm : (1 : ℝ) ≤ Fintype.card κ := by exact_mod_cast Fintype.card_pos_iff.2 ⟨q⟩
    have hρ0 : 0 ≤ ρ := div_nonneg (sub_nonneg.2 hm) (by linarith [Real.exp_pos (2 * B)])
    rcases Nat.eq_zero_or_pos (Fintype.card ν) with hc | hc
    · have := Fintype.card_eq_zero_iff.1 hc
      rw [show S = 0 by simp [hS], mul_zero, zero_mul]
      exact sum_nonneg fun _ _ => sq_nonneg _
    have hπ (π : κ → ν) : S * ρ ^ 2 ≤ ∑ v, (φ (Function.update π q v q) -
        softmaxAttention (fun k => s q k (Function.update π q v k))
          fun k => φ (Function.update π q v k)) ^ 2 := by
      obtain ⟨a, ha⟩ := exists_forall_mul_abs_sub_le_softmaxAttention hs φ q π
      calc S * ρ ^ 2 ≤ (∑ v, (φ v - μ - (a - μ)) ^ 2) * ρ ^ 2 :=
            mul_le_mul_of_nonneg_right (sum_sq_le_sum_sub_sq hμ (a - μ)) (sq_nonneg _)
        _ = ∑ v, (ρ * |φ v - a|) ^ 2 := by
            rw [sum_mul]
            exact sum_congr rfl fun v _ => by rw [mul_pow, sq_abs]; ring
        _ ≤ _ := sum_le_sum fun v _ => by
            rw [Function.update_self]
            exact (pow_le_pow_left₀ (mul_nonneg hρ0 (abs_nonneg _)) (ha v) 2).trans_eq
              (sq_abs _)
    have hsum := sum_sum_update q fun π : κ → ν =>
      (φ (π q) - softmaxAttention (fun k => s q k (π k)) fun k => φ (π k)) ^ 2
    have hle : (Fintype.card ν : ℝ) * ((Fintype.card ν : ℝ) ^ (Fintype.card κ - 1) *
        (S * ρ ^ 2)) ≤ Fintype.card ν * ∑ π : κ → ν,
          (φ (π q) - softmaxAttention (fun k => s q k (π k)) fun k => φ (π k)) ^ 2 := by
      rw [← hsum, ← mul_assoc, ← pow_succ', Nat.sub_add_cancel (Fintype.card_pos_iff.2 ⟨q⟩)]
      calc (Fintype.card ν : ℝ) ^ Fintype.card κ * (S * ρ ^ 2)
          = ∑ _π : κ → ν, S * ρ ^ 2 := by
            rw [sum_const, card_univ, Fintype.card_fun, nsmul_eq_mul, Nat.cast_pow]
        _ ≤ _ := sum_le_sum fun π _ => hπ π
    exact (mul_assoc _ _ _).le.trans (le_of_mul_le_mul_left hle (Nat.cast_pos.2 hc))
  calc (Fintype.card κ : ℝ) * ((Fintype.card ν : ℝ) ^ (Fintype.card κ - 1) * S) * ρ ^ 2
      = ∑ _q : κ, (Fintype.card ν : ℝ) ^ (Fintype.card κ - 1) * S * ρ ^ 2 := by
        rw [sum_const, card_univ, nsmul_eq_mul]
        ring
    _ ≤ _ := sum_le_sum fun q _ => hrow q

/-- Uniform attention over three keys with values `±1`. -/
example := mul_le_sum_softmaxAttention_sq (κ := Fin 3) (B := 0) (s := fun _ _ _ => 0)
  (fun _ _ _ => by simp) (φ := fun b : Bool => if b then (1 : ℝ) else -1) (μ := 0) (by simp)

omit [Fintype ν] in
/-- **The scores of softmax attention grow with the number of keys**
(`review/flat_spectrum.md`, SA3): recall of values `±1` to accuracy `ε` with scores bounded by
`B` has `(m − 1)(1 − ε) ≤ ε e^{2B}`. -/
theorem sub_one_mul_le_of_softmaxAttention_recall [Nonempty κ] {B ε : ℝ}
    {s : κ → κ → Bool → ℝ} (hs : ∀ q k v, |s q k v| ≤ B)
    (h : ∀ q (π : κ → Bool), |(if π q then (1 : ℝ) else -1) -
      softmaxAttention (fun k => s q k (π k)) fun k => if π k then (1 : ℝ) else -1| ≤ ε) :
    ((Fintype.card κ : ℝ) - 1) * (1 - ε) ≤ ε * Real.exp (2 * B) := by
  obtain ⟨q⟩ := ‹Nonempty κ›
  have hm : (1 : ℝ) ≤ Fintype.card κ := by exact_mod_cast Fintype.card_pos_iff.2 ⟨q⟩
  have hden : 0 < Real.exp (2 * B) + Fintype.card κ - 1 := by linarith [Real.exp_pos (2 * B)]
  obtain ⟨a, ha⟩ := exists_forall_mul_abs_sub_le_softmaxAttention hs
    (fun b : Bool => if b then (1 : ℝ) else -1) q fun _ => true
  have key (v : Bool) : ((Fintype.card κ : ℝ) - 1) / (Real.exp (2 * B) + Fintype.card κ - 1) *
      |(if v then (1 : ℝ) else -1) - a| ≤ ε := by
    have h' := h q (Function.update (fun _ => true) q v)
    rw [Function.update_self] at h'
    exact (ha v).trans h'
  have h1 := key true
  have h2 := key false
  simp only [ite_true, Bool.false_eq_true, ite_false] at h1 h2
  have h3 : (2 : ℝ) ≤ |1 - a| + |-1 - a| := by
    linarith [le_abs_self (1 - a), neg_le_abs (-1 - a)]
  have hρ : ((Fintype.card κ : ℝ) - 1) / (Real.exp (2 * B) + Fintype.card κ - 1) ≤ ε := by
    have h4 := mul_le_mul_of_nonneg_left h3 (div_nonneg (sub_nonneg.2 hm) hden.le)
    linarith
  rw [div_le_iff₀ hden] at hρ
  linarith

/-- One key: its token is all that the attention sees, and scores `0` recall it exactly. -/
example : ((Fintype.card (Fin 1) : ℝ) - 1) * (1 - 0) ≤ 0 * Real.exp (2 * 0) :=
  sub_one_mul_le_of_softmaxAttention_recall (s := fun _ _ _ => 0) (fun _ _ _ => by simp)
    fun q π => by
      obtain rfl : q = 0 := Subsingleton.elim _ _
      simp [softmaxAttention]

omit [Fintype ν] in
/-- **One-hot keys reach the scale of the bound** (`review/flat_spectrum.md`, SA3): with the
score `β/2` at the queried key and `−β/2` elsewhere, softmax attention is within
`2 B (m − 1) e^{−β}` of `φ(π q)`. -/
theorem softmaxAttention_oneHot_recall_error {β B : ℝ} {φ : ν → ℝ} (hφ : ∀ v, |φ v| ≤ B)
    (q : κ) (π : κ → ν) :
    |softmaxAttention (fun k => if k = q then β / 2 else -(β / 2)) (fun k => φ (π k)) -
      φ (π q)| ≤ 2 * B * (Fintype.card κ - 1) * Real.exp (-β) :=
  abs_softmaxAttention_sub_le (fun k hk => by rw [ite_eq_right hk, ite_eq_left rfl]; linarith)
    fun k => hφ (π k)

example (q : Fin 3) (π : Fin 3 → Bool) := softmaxAttention_oneHot_recall_error (β := 4)
  (B := 1) (φ := fun b : Bool => if b then (1 : ℝ) else -1) (fun b => by cases b <;> simp) q π

end Transformer.Grokking.FlatSpectrum.Recall
