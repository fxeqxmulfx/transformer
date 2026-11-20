module

public import Transformer.MRCR.ReRopeStyleRecall
public import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# Su's scale in the three real-softmax layers

In no paper. Su runs ReRoPE with the scale `max(1, log_T n)`, `T` the training length and `n` the
number of tokens the query sees: in the `log n†` runs each query is multiplied by it at test time
(kexue.fm/archives/9708, section 实验, trained at `T = 512`; `Transformer.RoPE.ReRoPE.logScale`).
Over `n ≥ 2` keys that is SSMax, arXiv:2501.19399v1, §2.2, with the parameter
`max(1, log_T n) / log n`, `suScale T (n − 1)` (`logScaleAttention_mul_suScale`). A slot head at
`suScale T x` is Su's head at every position `x`, the first one too, which sees one token
(`slotHead_suScale`). From the training length on Su's scale is SSMax with `s = 1 / log T`
(`logScaleAttention_div_log`).

Each head of `Transformer.MRCR.ReRopeStyleRecall` takes for `n` the number of keys it sees: the
`x + 1` tokens up to `x` in layer 1, those and the BOS in layer 2, and in layer 3 the BOS and the
tokens before the query, as many as the query sees. Since `suScale T x ≥ 1 / log T`, with Su's
scale in every head the three layers answer the query of depth three once
`A (1 − cos θ_m) > 4 log T` and the weight `G₃` of layer 3 exceeds `log T`: some weight `G₂` of
layer 2 makes them return the answer to every accuracy from some length on
(`exists_forall_styleLookup_su`). Below the training length Su's scale is `1`, and there the plain
softmax at the weights `A` and `G₂` reads the keys exactly and counts within the accuracy that
layer 3 needs.
-/

@[expose] public section

namespace Transformer.MRCR

open Finset Transformer.Grokking.FlatSpectrum.Recall
open Transformer.RoPE.ReRoPE (logScale)

variable {K F N S V R : ℕ}

/-- **Su's scale as a parameter of SSMax**: at the position `x`, which sees `x + 1` tokens,
`max(1, log_T (x + 1)) / log (x + 1)`; at the position `0`, where `log 1 = 0` and the one token
seen takes the whole weight at every scale, the lower bound `1 / log T` of the other positions. -/
noncomputable def suScale (T : ℝ) (x : ℕ) : ℝ :=
  max (1 / Real.log T) (logScale T ((x : ℝ) + 1) / Real.log ((x : ℝ) + 1))

/-- `suScale T x` is at least `1 / log T`. -/
theorem one_div_le_suScale (T : ℝ) (x : ℕ) : 1 / Real.log T ≤ suScale T x :=
  le_max_left _ _

/-- From the position `1` on, `suScale T x` is the quotient `max(1, log_T (x + 1)) / log (x + 1)`.
-/
theorem suScale_eq {T : ℝ} {x : ℕ} (hx : 0 < x) :
    suScale T x = logScale T ((x : ℝ) + 1) / Real.log ((x : ℝ) + 1) := by
  have hx1 : (1 : ℝ) ≤ x := by exact_mod_cast hx
  have hl : 0 < Real.log ((x : ℝ) + 1) := Real.log_pos (by linarith)
  refine max_eq_right ?_
  rw [le_div_iff₀ hl, one_div_mul_eq_div, Real.log_div_log]
  exact le_max_right _ _

/-- The hypothesis of `suScale_eq` holds at the position `1`, at Su's training length `512`. -/
example := suScale_eq (T := 512) (x := 1) one_pos

/-- **Su's scale is SSMax at the parameter `suScale`**: over `x + 1 ≥ 2` keys, attention at the
scale `G suScale T x · log n` is attention with the logit multiplied by `G max(1, log_T (x + 1))`.
-/
theorem logScaleAttention_mul_suScale {ι : Type*} [Fintype ι] (T G : ℝ) {x : ℕ} (hx : 0 < x)
    (hn : Fintype.card ι = x + 1) (z v : ι → ℝ) :
    logScaleAttention (G * suScale T x) z v =
      softmaxAttention (fun j => G * logScale T ((x : ℝ) + 1) * z j) v := by
  have hx1 : (1 : ℝ) ≤ x := by exact_mod_cast hx
  have hl : Real.log ((x : ℝ) + 1) ≠ 0 := (Real.log_pos (by linarith)).ne'
  rw [logScaleAttention, hn, suScale_eq hx]
  push_cast
  congr 1
  funext j
  field_simp

/-- The hypotheses of `logScaleAttention_mul_suScale` hold over the BOS and two positions. -/
example := logScaleAttention_mul_suScale (ι := Option (Fin 2)) 512 1 (x := 2) two_pos rfl
  (fun _ => 0) fun _ => 0

/-- **From the training length on Su's scale is SSMax with `s = 1 / log T`**: over `n ≥ T` keys,
attention at the scale `(G / log T) log n` is attention with the logit multiplied by
`G max(1, log_T n)`. -/
theorem logScaleAttention_div_log {ι : Type*} [Fintype ι] {T : ℝ} (hT : 1 < T) (G : ℝ)
    (hn : T ≤ Fintype.card ι) (z v : ι → ℝ) :
    logScaleAttention (G / Real.log T) z v =
      softmaxAttention (fun j => G * logScale T (Fintype.card ι) * z j) v := by
  have hlogT := Real.log_pos hT
  have h1 : 1 ≤ Real.logb T (Fintype.card ι) := by
    rw [Real.logb, one_le_div hlogT]
    exact Real.log_le_log (by linarith) hn
  rw [logScaleAttention, logScale, max_eq_right h1, Real.logb]
  congr 1
  funext j
  field_simp

/-- The hypotheses of `logScaleAttention_div_log` hold over `512` keys at Su's training length
`512`. -/
example := logScaleAttention_div_log (ι := Fin 512) (T := 512) (by norm_num) 1 (by simp)
  (fun _ => 0) fun _ => 0

/-- **Layer 1 at `suScale` is Su's head**: the slot head at the scale `suScale T x` is attention
with the ReRoPE logit multiplied by `max(1, log_T (x + 1))`, at every position. -/
theorem slotHead_suScale (T A : ℝ) (θ : Fin K → ℝ) (m : Fin K) (W : ℕ) {α : Type*}
    [DecidableEq α] (w : List α) (x k : ℕ) (a : α) :
    slotHead (suScale T x) A θ m W w x k a =
      softmaxAttention (fun j : Fin (x + 1) => logScale T ((x : ℝ) + 1) * ropeScore θ
        (Pi.single m (A * Real.cos (θ m * k))) (Pi.single m (A * Real.sin (θ m * k)))
        (min (x - j) W : ℕ)) fun j => if w[(j : ℕ)]? = some a then 1 else 0 := by
  rcases Nat.eq_zero_or_pos x with rfl | hx
  · simp [slotHead, logScaleAttention, softmaxAttention]
    split_ifs
    · exact (div_self (Real.exp_pos _).ne').symm
    · exact (zero_div _).symm
  · rw [slotHead, ← one_mul (suScale T x), logScaleAttention_mul_suScale T 1 hx
      (Fintype.card_fin _)]
    simp only [one_mul]

/-- **Three real-softmax layers with ReRoPE answer asks with a style under Su's scale** (MRCR-V2,
arXiv:2507.06261v6; Su, kexue.fm/archives/9708): with every head at Su's scale for the keys it
sees, layer 1 at `suScale T x` (`slotHead_suScale`), layer 2 at the weight `G₂`
(`logScaleAttention_mul_suScale`) and layer 3 at the weight `G₃` over the `n > T` keys of the
query (`logScaleAttention_div_log`), for `A (1 − cos θ_m) > 4 log T` and `G₃ > log T` some
`G₂ > 0` makes layer 3 return `φ (val v)`, `v` the answer, to every accuracy `ε > 0` for all
conversations of more than `M ≥ T` tokens, for every value read `φ` bounded by `C`. -/
theorem exists_forall_styleLookup_su {T : ℝ} (hT : 1 < T) {A : ℝ} (hA : 0 ≤ A) {θ : Fin K → ℝ}
    {m : Fin K} (hθ : 0 < θ m) {W : ℕ} (hW : 4 < W) (hθW : θ m * W ≤ Real.pi)
    (hg : 4 * Real.log T < A * (1 - Real.cos (θ m))) {G₃ C : ℝ} (hG₃ : Real.log T < G₃)
    {φ : Option (StyleTok F N S V R) → ℝ} (hφ : ∀ o, |φ o| ≤ C) :
    ∃ G₂ : ℝ, 0 < G₂ ∧ ∀ ε : ℝ, 0 < ε → ∃ M : ℕ, T ≤ M ∧ ∀ (w : List (StyleTok F N S V R))
      (f : Fin F) (t : Fin N) (σ : Fin S) (r : Fin R) (v : Fin V), M < w.length →
      w.getLast? = some (.query f t σ r) → StyleAnswer w v →
      |styleLookup (suScale T) A θ m W (fun x => G₂ * suScale T (x + 1)) (G₃ / Real.log T) φ w
        f t σ r - φ (some (.val v))| ≤ ε := by
  have hlogT := Real.log_pos hT
  have hg0 : 0 ≤ A * (1 - Real.cos (θ m)) := by linarith
  have hs (x : ℕ) : 0 ≤ suScale T x := (one_div_pos.2 hlogT).le.trans (one_div_le_suScale T x)
  have hsg (x : ℕ) : 4 < suScale T x * (A * (1 - Real.cos (θ m))) :=
    calc (4 : ℝ) < 1 / Real.log T * (A * (1 - Real.cos (θ m))) := by
          rw [one_div_mul_eq_div, lt_div_iff₀ hlogT]
          linarith
      _ ≤ suScale T x * (A * (1 - Real.cos (θ m))) :=
          mul_le_mul_of_nonneg_right (one_div_le_suScale T x) hg0
  obtain ⟨b, hb, h⟩ := exists_forall_styleLookup (F := F) (N := N) (S := S) (V := V) (R := R) hA
    hs hθ hW hθW hsg (by rwa [one_lt_div hlogT] : 1 < G₃ / Real.log T) hφ
  refine ⟨b * Real.log T, by positivity, fun ε hε => ?_⟩
  obtain ⟨M, hM⟩ := h ε hε
  refine ⟨max M ⌈T⌉₊, (Nat.le_ceil T).trans (by exact_mod_cast le_max_right _ _),
    fun w f t σ r v hw => hM _ (fun x => ?_) w f t σ r v (lt_of_le_of_lt (le_max_left _ _) hw)⟩
  calc b = b * Real.log T * (1 / Real.log T) := by field_simp
    _ ≤ b * Real.log T * suScale T (x + 1) :=
        mul_le_mul_of_nonneg_left (one_div_le_suScale T _) (by positivity)

/-- Su's training length `512`, eight needles: `A = 200`, `θ = π/6` and the window `6`, where
`200 (1 − √3/2) ≈ 26.79 > 4 log 512 ≈ 24.95`, and `G₃ = 7 > log 512 ≈ 6.24`. -/
example := exists_forall_styleLookup_su (F := 1) (N := 1) (S := 1) (V := 2) (R := 8) (T := 512)
  (by norm_num) (A := 200) (by norm_num) (θ := fun _ : Fin 1 => Real.pi / 6) (m := 0)
  (by positivity) (W := 6) (by norm_num) (by push_cast; linarith)
  (by
    have h2 := Real.log_two_lt_d9
    have h3 : Real.sqrt 3 < 1.75 := by rw [Real.sqrt_lt' (by norm_num)]; norm_num
    rw [show (512 : ℝ) = 2 ^ 9 by norm_num, Real.log_pow, Real.cos_pi_div_six]
    push_cast
    linarith)
  (G₃ := 7) (C := 1)
  (by
    have h2 := Real.log_two_lt_d9
    rw [show (512 : ℝ) = 2 ^ 9 by norm_num, Real.log_pow]
    push_cast
    linarith)
  (φ := fun o => if o = some (.val 0) then 1 else 0) fun o => by split_ifs <;> simp

end Transformer.MRCR
