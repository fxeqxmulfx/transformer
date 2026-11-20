module

public import Transformer.MRCR.PhraseKeyReRope
public import Transformer.MRCR.ReRopeStyleRecall
public import Transformer.RoPE.LeRoPE.Section3_LearnedFrequencies
public import Mathlib.Analysis.Complex.Trigonometric
public import Mathlib.Analysis.Real.Pi.Bounds
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Bounds

/-!
# LeRoPE with ReRoPE's window answers spelled asks

In no paper. LeRoPE (Karypis et al., arXiv:2607.10134v1, §3.1, Eq. (5)) learns the frequencies
`θ̂_m = e^{α_m} θ_m`, one scale `α_m` per band, shared by every head and layer. With the distance
unclamped no learned frequencies read the token before (`Transformer.MRCR.not_readsPrev_lerope`).
Clamping the distance at a window `W`, as ReRoPE does (`Transformer.MRCR.reropeLogit`), repairs
LeRoPE as it repairs RoPE: the constructions of `Transformer.MRCR.PhraseKeyReRope` and
`Transformer.MRCR.ReRopeStyleRecall` need one band, of any frequency up to about `π / W`.

* `exists_lerope_rerope_phraseAnswer`, `exists_ssmax_lerope_rerope_phraseAnswer`: five rounded
  layers answer spelled asks at every length once one band has `θ̂_m (W + 2) ≤ π`, the logit
  divided by `c d`, alone or under SSMax.
* `learnedFrequency_mul_le_pi_iff`: a band has it exactly for `α_m ≤ log(π / ((W + 2) θ_m))`, so
  LeRoPE reaches it from every initial frequency.
* `exists_forall_styleLookup_lerope`: three real-softmax layers under SSMax answer asks with a
  style at every length once one band has `θ̂_m W ≤ π` and `s A (1 − cos θ̂_m) > 4`.

LeRoPE's dominant band, the one band whose contribution to the logits is large and nearly
purely positional (§6), turns `θ̂ L` between `2.714` and `3.086` radians over the training length
`L` in the six runs of Table 10 (Appendix D.1), less than `π`. So in the rounded transformer it
serves every window `W ≤ L − 2`: the examples take the 2.52B run, `e^α = 0.2009` at `L = 2048`,
on `θ_17 = 10^{−2.125} ≈ 0.0075`, the frequency of band 17 for the head dimension `64` and the
base `10⁴` (§4). With the real softmax it serves badly. Since `1 − cos θ ≤ θ² / 2`, the slot
heads need `s A > 8 / θ̂²` (`eight_lt_mul_sq`). For a band with `θ̂ L ≤ π` that exceeds
`8 L² / π²` (`eight_mul_sq_div_lt`), about `3.4 · 10⁶` at `L = 2048`. A fast band near `π / W`
needs far less: `A = 100` at `θ = π/6`.
-/

@[expose] public section

namespace Transformer.MRCR

open Transformer.CRASP Transformer.CRASP.Grams Transformer.RoPE.LeRoPE

variable {K F N S V R : ℕ}

/-- **LeRoPE with ReRoPE's window answers spelled asks at every length**: five rounded layers,
the logit divided by `c d`, at every window `W ≥ 2`, once one band of the learned frequencies
`e^{α_m} θ_m` (arXiv:2607.10134v1, Eq. (5)) has `e^{α_m} θ_m (W + 2) ≤ π`. Unclamped, no
learned frequencies do (`not_readsPrev_lerope`). -/
theorem exists_lerope_rerope_phraseAnswer (c : ℕ → ℝ) (hc : ∀ d, 0 < d → 0 < c d)
    {ω α : Fin K → ℝ} {m : Fin K} (hω : 0 < ω m) {W : ℕ} (hW : 2 ≤ W)
    (hθW : learnedFrequency (ω m) (α m) * (W + 2) ≤ Real.pi) (v : Fin V) :
    ∃ (p d : ℕ) (b : Fin K → ℝ) (T : LTfr (Option (PhraseTok F N V R)) p 0 d 5),
      T.logit = reropeLogit (c d) (fun l => learnedFrequency (ω l) (α l)) 0 b W ∧
      T.Recognizes {w | PhraseAnswer w v} :=
  exists_rerope_phraseAnswer c hc (θ := fun l => learnedFrequency (ω l) (α l))
    (learned_frequency_pos hω (α m)) hW hθW v

/-- The dominant band of Table 10's 2.52B run at `L = 2048`, `e^α = 0.2009` on
`θ_17 ≈ 0.0075`, serves the window `2046`; the divisor is `√d`. -/
example := exists_lerope_rerope_phraseAnswer (F := 1) (N := 1) (R := 1) (fun d => Real.sqrt d)
  (fun _ hd => Real.sqrt_pos.mpr (by exact_mod_cast hd)) (ω := fun _ : Fin 1 => 0.0075)
  (α := fun _ => Real.log 0.2009) (m := 0) (by norm_num) (W := 2046) (by norm_num)
  (by
    rw [learnedFrequency, Real.exp_log (by norm_num)]
    push_cast
    linarith [Real.pi_gt_d2])
  (0 : Fin 1)

/-- **The same under SSMax** with `s log 2 ≥ 1` (arXiv:2501.19399v1, `eq:ssmax`), scaling the
whole logit. -/
theorem exists_ssmax_lerope_rerope_phraseAnswer (c : ℕ → ℝ) (hc : ∀ d, 0 < d → 0 < c d)
    {s : ℝ} (hs : 1 ≤ s * Real.log 2) {ω α : Fin K → ℝ} {m : Fin K} (hω : 0 < ω m) {W : ℕ}
    (hW : 2 ≤ W) (hθW : learnedFrequency (ω m) (α m) * (W + 2) ≤ Real.pi) (v : Fin V) :
    ∃ (p d : ℕ) (b : Fin K → ℝ) (T : LTfr (Option (PhraseTok F N V R)) p 0 d 5),
      T.logit = ssmaxLogit s (reropeLogit (c d) (fun l => learnedFrequency (ω l) (α l)) 0 b W) ∧
      T.Recognizes {w | PhraseAnswer w v} :=
  exists_ssmax_rerope_phraseAnswer c hc hs (θ := fun l => learnedFrequency (ω l) (α l))
    (learned_frequency_pos hω (α m)) hW hθW v

/-- The hypotheses of `exists_ssmax_lerope_rerope_phraseAnswer` hold for `s = 2`, the divisor
`d` and the same band and window. -/
example := exists_ssmax_lerope_rerope_phraseAnswer (F := 1) (N := 1) (R := 1)
  (fun d => (d : ℝ)) (fun _ hd => by exact_mod_cast hd) (s := 2)
  (by have := Real.log_two_gt_d9; linarith) (ω := fun _ : Fin 1 => 0.0075)
  (α := fun _ => Real.log 0.2009) (m := 0) (by norm_num) (W := 2046) (by norm_num)
  (by
    rw [learnedFrequency, Real.exp_log (by norm_num)]
    push_cast
    linarith [Real.pi_gt_d2])
  (0 : Fin 1)

/-- **LeRoPE reaches the band from every initial frequency**: `e^α θ (W + 2) ≤ π` exactly when
`α ≤ log(π / ((W + 2) θ))`, a half-line of scales. -/
theorem learnedFrequency_mul_le_pi_iff {ω : ℝ} (hω : 0 < ω) (α : ℝ) (W : ℕ) :
    learnedFrequency ω α * (W + 2) ≤ Real.pi ↔
      α ≤ Real.log (Real.pi / ((W + 2) * ω)) := by
  rw [Real.le_log_iff_exp_le (by positivity), le_div_iff₀ (by positivity), learnedFrequency,
    mul_assoc, mul_comm ω]

/-- The hypothesis of `learnedFrequency_mul_le_pi_iff` holds for `θ = 0.0075`. -/
example := learnedFrequency_mul_le_pi_iff (by norm_num : (0 : ℝ) < 0.0075) 0 2046

/-- **Three real-softmax layers with LeRoPE and ReRoPE's window answer asks with a style at
every length** (`exists_forall_styleLookup`), once one band of the learned frequencies has
`θ̂_m W ≤ π` and the slot heads' scale and amplitude have `s A (1 − cos θ̂_m) > 4`. -/
theorem exists_forall_styleLookup_lerope {s : ℕ → ℝ} {A : ℝ} (hA : 0 ≤ A)
    (hs : ∀ x, 0 ≤ s x) {ω α : Fin K → ℝ} {m : Fin K} (hω : 0 < ω m) {W : ℕ} (hW : 4 < W)
    (hθW : learnedFrequency (ω m) (α m) * W ≤ Real.pi)
    (hsg : ∀ x, 4 < s x * (A * (1 - Real.cos (learnedFrequency (ω m) (α m)))))
    {s₂ C : ℝ} (hs₂ : 1 < s₂) {φ : Option (StyleTok F N S V R) → ℝ} (hφ : ∀ o, |φ o| ≤ C) :
    ∃ b : ℝ, 1 ≤ b ∧ ∀ ε : ℝ, 0 < ε → ∃ M : ℕ, ∀ s₁ : ℕ → ℝ, (∀ x, b ≤ s₁ x) →
      ∀ (w : List (StyleTok F N S V R)) (f : Fin F) (t : Fin N) (σ : Fin S) (r : Fin R)
      (v : Fin V), M < w.length → w.getLast? = some (.query f t σ r) → StyleAnswer w v →
      |styleLookup s A (fun l => learnedFrequency (ω l) (α l)) m W s₁ s₂ φ w f t σ r -
        φ (some (.val v))| ≤ ε :=
  exists_forall_styleLookup hA hs (θ := fun l => learnedFrequency (ω l) (α l))
    (learned_frequency_pos hω (α m)) hW hθW hsg hs₂ hφ

/-- The hypotheses of `exists_forall_styleLookup_lerope` hold for the band `θ = π/6` left at
`α = 0`, the window `6`, `s = 1`, `A = 100`, `s₂ = 2` and `φ = 0`. -/
example := exists_forall_styleLookup_lerope (F := 1) (N := 1) (S := 1) (V := 1) (R := 1)
  (s := fun _ => 1) (A := 100) (by norm_num) (fun _ => zero_le_one)
  (ω := fun _ : Fin 1 => Real.pi / 6) (α := fun _ => 0) (m := 0) (by positivity) (W := 6)
  (by norm_num)
  (by rw [learned_frequency_init]; push_cast; linarith)
  (fun _ => by
    rw [learned_frequency_init, Real.cos_pi_div_six]
    nlinarith [Real.sq_sqrt (show (0 : ℝ) ≤ 3 by norm_num), Real.sqrt_nonneg 3])
  (s₂ := 2) (C := 0) (by norm_num) (φ := fun _ => 0) (fun _ => by simp)

/-- **Slow bands need large amplitudes**: since `1 − cos θ ≤ θ² / 2`, a slot head with
`s A (1 − cos θ) > 4` has `s A θ² > 8`. -/
theorem eight_lt_mul_sq {s A θ : ℝ} (hsA : 0 ≤ s * A) (h : 4 < s * (A * (1 - Real.cos θ))) :
    8 < s * A * θ ^ 2 := by
  have hc : 1 - Real.cos θ ≤ θ ^ 2 / 2 := by linarith [Real.one_sub_sq_div_two_le_cos (x := θ)]
  nlinarith [mul_le_mul_of_nonneg_left hc hsA]

/-- The hypotheses of `eight_lt_mul_sq` hold for `θ = π/6`, `s = 1` and `A = 100`. -/
example := eight_lt_mul_sq (s := 1) (A := 100) (θ := Real.pi / 6) (by norm_num)
  (by
    rw [Real.cos_pi_div_six]
    nlinarith [Real.sq_sqrt (show (0 : ℝ) ≤ 3 by norm_num), Real.sqrt_nonneg 3])

/-- **A band with `θ L ≤ π` needs `s A > 8 L² / π²`**, as LeRoPE's dominant band does at its
training length `L` (arXiv:2607.10134v1, Table 10). -/
theorem eight_mul_sq_div_lt {s A θ L : ℝ} (hsA : 0 ≤ s * A) (hθ : 0 ≤ θ) (hL : 0 < L)
    (hθL : θ * L ≤ Real.pi) (h : 4 < s * (A * (1 - Real.cos θ))) :
    8 * L ^ 2 / Real.pi ^ 2 < s * A := by
  have h1 : 8 * L ^ 2 < s * A * θ ^ 2 * L ^ 2 :=
    mul_lt_mul_of_pos_right (eight_lt_mul_sq hsA h) (by positivity)
  have h2 : s * A * (θ * L) ^ 2 ≤ s * A * Real.pi ^ 2 :=
    mul_le_mul_of_nonneg_left (pow_le_pow_left₀ (by positivity) hθL 2) hsA
  rw [div_lt_iff₀ (by positivity)]
  linarith [show s * A * θ ^ 2 * L ^ 2 = s * A * (θ * L) ^ 2 by ring]

/-- The hypotheses of `eight_mul_sq_div_lt` hold for the dominant band of Table 10's 2.52B run,
`θ̂ = 0.001507` at `L = 2048`, with `s = 1` and `A = 4 · 10⁶`. -/
example := eight_mul_sq_div_lt (s := 1) (A := 4000000) (θ := 0.001507) (L := 2048)
  (by norm_num) (by norm_num) (by norm_num) (by linarith [Real.pi_gt_d2])
  (by
    have hb := Real.cos_bound (x := 0.001507) (by rw [abs_of_pos (by norm_num)]; norm_num)
    rw [abs_of_pos (by norm_num : (0 : ℝ) < 0.001507)] at hb
    nlinarith [(abs_le.mp hb).2])

end Transformer.MRCR
