module

public import Transformer.Grokking.FlatSpectrum.Recall.LogScale

/-!
# Two layers recall a key and its value from separate tokens

Source: `review/flat_spectrum.md`, *Softmax attention escapes the cut* (added), item SA5, for
the attention scale `β_n = γ log n` of arXiv:2512.01868v4, §6 (*Long context Transformers*), on
the task of arXiv:2312.04927v1, §3, whose context lists every key and its value as two tokens.

SA2 and SA5 put a binding in one token. Here the context is `k₀ v₀ k₁ v₁ … k_m v_m`: the token
`(i, false)` carries the key `i` and the token `(i, true)` its value, `n = 2 (m + 1)` tokens.
Two layers of attention at the scale `s log n` recall it, as an induction head does.

* Layer 1 (`prevTokenHead`) attends at every token to the token before it, read cyclically,
  with the positional raw score `1` there and `0` elsewhere, and copies the key channel: the
  one-hot key at a key token, zero at a value token. The value token of `i` receives the key
  `i`; a key token receives zero from the value token before it.
* Layer 2 (`inductionHead`) attends at the query `q` with the raw score `⟨e_q, a_t⟩`, the
  coordinate `q` of the key channel `a_t` copied to the token `t`, and reads the value channel.

The error of layer 1 enters the raw scores of layer 2 (`abs_logScaleAttention_comp_sub_le`). If
layer 1 leads at the partner `p t` of every token `t` by `g₁`, it is within
`δ = 2 H (n − 1) n^{−s₁ g₁}` of the match `h (p t)`, for matches bounded by `H`. If the partner of
the target leads the partners of the other tokens by `g₂`, layer 2 leads by `g₂ − 2δ`, so it is
within `2 C (n − 1) n^{−s₂ (g₂ − 2δ)}` of the value of the target. The scale `s₂ log n`
multiplies the error `δ` of the copy, and `δ` falls like `n^{1 − s₁ g₁}`. The induction head
has `g₁ = g₂ = H = 1` (`abs_inductionHead_sub_le`), so with `s₁ > 1` and `s₂ > 1` it recalls at
every large number of keys to every accuracy (`exists_forall_inductionHead_recall`): a binding
split over two tokens is recalled under the condition `s g > 1` of SA5, at both layers. That
the condition is needed at layer 1 is not proved here.

The proof of arXiv:2312.04927v1, Appendix, *The Multiple-Query Associative Recall Problem*,
`prop: app-attention`, uses two layers of attention without softmax and moves the value to the
key token instead. The cyclic reading stands for a first token without a key. Causal attention,
whose scale at a token is `s log` of the length of its prefix, is not treated.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Recall

open Filter Topology

/-- **Two layers at the scale `s log n`** (`review/flat_spectrum.md`, SA5): layer 1 leads at the
partner `p t` of every token `t` by `g₁`, and the match `h`, bounded by `H`, is at the partner of
`t₀` ahead of the partners of the other tokens by `g₂`. Then layer 2, whose raw scores are the
averages of `h` by layer 1, is within `2 C (n − 1) n^{−s₂ (g₂ − 2δ)}` of the value of `t₀`, for
values bounded by `C`, where `δ = 2 H (n − 1) n^{−s₁ g₁}` bounds the error of layer 1. -/
theorem abs_logScaleAttention_comp_sub_le {T : Type*} [Fintype T] [DecidableEq T]
    {s₁ s₂ g₁ g₂ H C : ℝ} (hs₁ : 0 ≤ s₁) (hs₂ : 0 ≤ s₂) {z : T → T → ℝ} {p : T → T}
    (hz : ∀ t t', t' ≠ p t → z t t' + g₁ ≤ z t (p t)) {h : T → ℝ} (hh : ∀ t, |h t| ≤ H)
    {t₀ : T} (hg : ∀ t, t ≠ t₀ → h (p t) + g₂ ≤ h (p t₀)) {V : T → ℝ} (hV : ∀ t, |V t| ≤ C) :
    |logScaleAttention s₂ (fun t => logScaleAttention s₁ (z t) h) V - V t₀| ≤
      2 * C * (Fintype.card T - 1) * (Fintype.card T : ℝ) ^ (-(s₂ * (g₂ -
        2 * (2 * H * (Fintype.card T - 1) * (Fintype.card T : ℝ) ^ (-(s₁ * g₁)))))) := by
  have h1 (t : T) := abs_le.1 (abs_logScaleAttention_sub_le hs₁ (hz t) hh)
  refine abs_logScaleAttention_sub_le hs₂ (fun t ht => ?_) hV
  linarith [hg t ht, (h1 t).1, (h1 t).2, (h1 t₀).1, (h1 t₀).2]

/-- Two tokens, each attending to the other, with the matches `0` and `1`. -/
example := abs_logScaleAttention_comp_sub_le (T := Fin 2) (s₁ := 1) (s₂ := 1) (g₁ := 1) (g₂ := 1)
  (H := 1) (C := 1) zero_le_one zero_le_one (z := fun t t' => if t' = t + 1 then 1 else 0)
  (p := fun t => t + 1) (fun t t' ht => by simp [ht]) (h := ![0, 1])
  (fun t => by fin_cases t <;> simp) (t₀ := 0) (fun t ht => by fin_cases t <;> simp_all)
  (V := ![1, -1]) (fun t => by fin_cases t <;> simp)

/-- The token before `t` in the context `k₀ v₀ k₁ v₁ … k_m v_m`, read cyclically: the key token
of `i` before its value token, the value token of `i − 1` before the key token of `i`, and the
last value token before the first key token. -/
def prevToken {m : ℕ} : Fin (m + 1) × Bool → Fin (m + 1) × Bool
  | (i, true) => (i, false)
  | (i, false) => (i - 1, true)

/-- The key token of `q` comes before the value token of `q` and before no other token. -/
lemma prevToken_eq_iff {m : ℕ} (t : Fin (m + 1) × Bool) (q : Fin (m + 1)) :
    prevToken t = (q, false) ↔ t = (q, true) := by
  obtain ⟨i, _ | _⟩ := t <;> simp [prevToken]

/-- The key channel of a token: the one-hot vector of its key at a key token, zero at a value
token. -/
noncomputable def keyChannel {m : ℕ} (t : Fin (m + 1) × Bool) (c : Fin (m + 1)) : ℝ :=
  if t = (c, false) then 1 else 0

/-- **The previous-token head**, layer 1 (`review/flat_spectrum.md`, SA5): at the token `t`,
attention at the scale `s log n` with the positional raw score `1` at the token before `t` and
`0` elsewhere, copying the key channel. -/
noncomputable def prevTokenHead (s : ℝ) {m : ℕ} (t : Fin (m + 1) × Bool) (c : Fin (m + 1)) :
    ℝ :=
  logScaleAttention s (fun t' => if t' = prevToken t then 1 else 0) fun t' => keyChannel t' c

/-- **The induction head**, layer 2 on layer 1 (`review/flat_spectrum.md`, SA5): at the query
`q`, attention at the scale `s₂ log n` with the raw score `⟨e_q, a_t⟩`, the coordinate `q` of
the key channel `a_t` that layer 1 copied to the token `t`, reading the value channel: `φ (π i)`
at the value token of `i`, zero at a key token. -/
noncomputable def inductionHead (s₁ s₂ : ℝ) {ν : Type*} (φ : ν → ℝ) {m : ℕ}
    (π : Fin (m + 1) → ν) (q : Fin (m + 1)) : ℝ :=
  logScaleAttention s₂ (fun t => prevTokenHead s₁ t q) fun t => if t.2 then φ (π t.1) else 0

/-- **The induction head recalls** (`review/flat_spectrum.md`, SA5): over `m + 1` keys and
`n = 2m + 2` tokens it is within `2 C (n − 1) n^{−s₂ (1 − 2δ)}` of the value of the query, for
values bounded by `C`, where `δ = 2 (n − 1) n^{−s₁}` bounds the error of the copy. -/
theorem abs_inductionHead_sub_le {ν : Type*} {s₁ s₂ C : ℝ} (hs₁ : 0 ≤ s₁) (hs₂ : 0 ≤ s₂)
    {φ : ν → ℝ} (hφ : ∀ v, |φ v| ≤ C) {m : ℕ} (π : Fin (m + 1) → ν) (q : Fin (m + 1)) :
    |inductionHead s₁ s₂ φ π q - φ (π q)| ≤ 2 * C * (2 * m + 1) * (2 * m + 2 : ℝ) ^
      (-(s₂ * (1 - 2 * (2 * (2 * m + 1) * (2 * m + 2 : ℝ) ^ (-s₁))))) := by
  have hC : 0 ≤ C := (abs_nonneg _).trans (hφ (π q))
  have hz (t t' : Fin (m + 1) × Bool) (ht : t' ≠ prevToken t) :
      (if t' = prevToken t then (1 : ℝ) else 0) + 1 ≤
        if prevToken t = prevToken t then 1 else 0 := by
    simp [ht]
  have hh (t : Fin (m + 1) × Bool) : |keyChannel t q| ≤ 1 := by
    unfold keyChannel
    split_ifs <;> simp
  have hg (t : Fin (m + 1) × Bool) (ht : t ≠ (q, true)) :
      keyChannel (prevToken t) q + 1 ≤ keyChannel (prevToken (q, true)) q := by
    simp [keyChannel, prevToken_eq_iff, ht]
  have hV (t : Fin (m + 1) × Bool) : |if t.2 then φ (π t.1) else 0| ≤ C := by
    split_ifs
    · exact hφ _
    · simpa using hC
  have h := abs_logScaleAttention_comp_sub_le (z := fun t t' => if t' = prevToken t then 1 else 0)
    (h := fun t' => keyChannel t' q) (V := fun t => if t.2 then φ (π t.1) else 0) hs₁ hs₂ hz hh
    hg hV
  have hcard : (Fintype.card (Fin (m + 1) × Bool) : ℝ) = 2 * m + 2 := by
    simp only [Fintype.card_prod, Fintype.card_fin, Fintype.card_bool]
    push_cast
    ring
  rw [hcard, show (2 * m + 2 - 1 : ℝ) = 2 * m + 1 by ring, mul_one, mul_one] at h
  exact h

/-- Four keys, values `±1`, at `s₁ = s₂ = 2`. -/
example := abs_inductionHead_sub_le (s₁ := 2) (s₂ := 2) (by norm_num) (by norm_num)
  (φ := fun b : Bool => if b then (1 : ℝ) else -1) (C := 1) (fun b => by cases b <;> simp)
  (fun _ : Fin 4 => true) 0

/-- **Above the threshold two layers recall at every length** (`review/flat_spectrum.md`, SA5):
with `s₁ > 1` and `s₂ > 1`, the induction head recalls values bounded by `C` to every accuracy
`ε > 0` from some number of keys on. -/
theorem exists_forall_inductionHead_recall {ν : Type*} {s₁ s₂ C : ℝ} (hs₁ : 1 < s₁)
    (hs₂ : 1 < s₂) {φ : ν → ℝ} (hφ : ∀ v, |φ v| ≤ C) {ε : ℝ} (hε : 0 < ε) :
    ∃ M : ℕ, ∀ m ≥ M, ∀ (q : Fin (m + 1)) (π : Fin (m + 1) → ν),
      |inductionHead s₁ s₂ φ π q - φ (π q)| ≤ ε := by
  set σ := (s₂ + 1) / 2 with hσ
  have hx : Tendsto (fun m : ℕ => 2 * (m : ℝ) + 2) atTop atTop :=
    tendsto_atTop_add_const_right _ 2 (tendsto_natCast_atTop_atTop.const_mul_atTop two_pos)
  have h1 := ((tendsto_rpow_neg_atTop (by linarith : 0 < s₁ - 1)).comp hx).const_mul 4
  have h2 := ((tendsto_rpow_neg_atTop (by linarith : 0 < σ - 1)).comp hx).const_mul (2 * C)
  rw [mul_zero] at h1 h2
  have hη : 0 < 1 - σ / s₂ := by
    rw [sub_pos, div_lt_one (by linarith)]
    linarith
  obtain ⟨M₁, hM₁⟩ := eventually_atTop.1 (h1.eventually (ge_mem_nhds hη))
  obtain ⟨M₂, hM₂⟩ := eventually_atTop.1 (h2.eventually (ge_mem_nhds hε))
  refine ⟨max M₁ M₂, fun m hm q π => ?_⟩
  have hD := hM₁ m ((le_max_left _ _).trans hm)
  have hE := hM₂ m ((le_max_right _ _).trans hm)
  simp only [Function.comp_apply] at hD hE
  have hx0 : (0 : ℝ) < 2 * m + 2 := by positivity
  have hx1 : (1 : ℝ) ≤ 2 * m + 2 := by linarith [(Nat.cast_nonneg m : (0 : ℝ) ≤ m)]
  have hC : 0 ≤ C := (abs_nonneg _).trans (hφ (π q))
  have hpow (a : ℝ) : (2 * (m : ℝ) + 2) ^ (-(a - 1)) = (2 * m + 2) * (2 * m + 2) ^ (-a) := by
    rw [show -(a - 1) = 1 + -a by ring, Real.rpow_add hx0, Real.rpow_one]
  rw [hpow] at hD hE
  have hr := Real.rpow_nonneg hx0.le (-s₁)
  have hδ : s₂ * (2 * (2 * (2 * m + 1) * (2 * m + 2 : ℝ) ^ (-s₁))) ≤ s₂ - σ := by
    have h := mul_le_mul_of_nonneg_left hD (by linarith : (0 : ℝ) ≤ s₂)
    rw [mul_sub, mul_one, mul_div_cancel₀ _ (by linarith : s₂ ≠ 0)] at h
    nlinarith
  have hexp : -(s₂ * (1 - 2 * (2 * (2 * m + 1) * (2 * m + 2 : ℝ) ^ (-s₁)))) ≤ -σ := by
    linarith
  calc |inductionHead s₁ s₂ φ π q - φ (π q)| ≤ _ :=
        abs_inductionHead_sub_le (by linarith) (by linarith) hφ π q
    _ ≤ 2 * C * (2 * m + 2) * (2 * m + 2 : ℝ) ^ (-σ) :=
        mul_le_mul (mul_le_mul_of_nonneg_left (by linarith) (by linarith))
          (Real.rpow_le_rpow_of_exponent_le hx1 hexp) (Real.rpow_nonneg hx0.le _)
          (by positivity)
    _ ≤ ε := by linarith

/-- At `s₁ = s₂ = 2`, values `±1` to accuracy `1/10` from some number of keys on. -/
example := exists_forall_inductionHead_recall (s₁ := 2) (s₂ := 2) (by norm_num) (by norm_num)
  (φ := fun b : Bool => if b then (1 : ℝ) else -1) (C := 1) (fun b => by cases b <;> simp)
  (ε := 1 / 10) (by norm_num)

end Transformer.Grokking.FlatSpectrum.Recall
