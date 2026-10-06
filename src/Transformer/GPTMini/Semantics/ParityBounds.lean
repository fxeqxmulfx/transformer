import Transformer.GPTMini.Semantics.ParityCoordinates

/-!
# Uniform finite-weight bounds for every legal parity context

Source: Basis Parity's one-through-sixteen-bit grammar at cbafbe9,
ParityInputs' actual ONE/BOS residual formula, and GPTMini RMSNorm at
f11b6e2. Bounds deliberately leave slack: every relevant context has at
most nineteen tokens, the ONE and BOS coordinates are at most eight,
and the actual FFN prenorm scale is at least one half. All epsilons in
(0, 1/64] are covered, including the reference's RMS epsilon 1e-5.

The finite FFN weights 1024 and 131072 then suffice for a strict parity
label margin and an EOS score above every parity competitor. These are
ordinary model weights, not an optimization or convexity assertion.
All scalar bounds below are consequences of the raw-input grammar.
-/

namespace Transformer.GPTMini.Semantics

open TokenInterface

/-- The actual active-token RMS multiplier is between four and eight.
Source: countScale's exact unit-embedding formula, with an explicit epsilon range. -/
theorem countScale_bounds (eps : ℝ) (hzero : 0 ≤ eps) (hsmall : eps ≤ 1 / 64) :
    4 ≤ countScale eps ∧ countScale eps ≤ 8 := by
  have hlow := Real.sqrt_le_sqrt (by linarith : (1 : ℝ) ≤ 1 + 64 * eps)
  have hhigh := Real.sqrt_le_sqrt (by linarith : 1 + 64 * eps ≤ (4 : ℝ))
  simp only [Real.sqrt_one] at hlow
  have hhigh' : Real.sqrt (1 + 64 * eps) ≤ 2 := by
    simpa only [show Real.sqrt (4 : ℝ) = 2 from by norm_num] using hhigh
  have hd : 0 < Real.sqrt (1 + 64 * eps) := by linarith
  norm_num only [countScale]
  change 4 ≤ 8 / Real.sqrt (1 + 64 * eps) ∧ 8 / Real.sqrt (1 + 64 * eps) ≤ 8
  constructor
  · rw [le_div_iff₀ hd]
    linarith [hhigh']
  · rw [div_le_iff₀ hd]
    linarith

example : (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 / 64 := by norm_num

/-- Exact raw count coordinates are bounded without using an externally supplied count representation.
Source: the proved raw ONE/BOS formula; t is the actual positive context length and c its ONE count. -/
theorem parity_coordinate_bounds (eps : ℝ) (hzero : 0 ≤ eps) (hsmall : eps ≤ 1 / 64)
    (t c : ℕ) (ht : 1 ≤ t) (htop : t ≤ 19) (hc : c ≤ t) :
    0 ≤ (c : ℝ) / t * countScale eps ∧
      (c : ℝ) / t * countScale eps ≤ 8 ∧
      4 / 19 ≤ (1 : ℝ) / t * countScale eps ∧
      (1 : ℝ) / t * countScale eps ≤ 8 := by
  have hs := countScale_bounds eps hzero hsmall
  have ht1 : (1 : ℝ) ≤ t := by exact_mod_cast ht
  have ht19 : (t : ℝ) ≤ 19 := by exact_mod_cast htop
  have hct : (c : ℝ) ≤ t := by exact_mod_cast hc
  have htp : (0 : ℝ) < t := by linarith
  have hratio : (c : ℝ) / t ≤ 1 := (div_le_iff₀ htp).mpr (by linarith)
  have hn : 0 ≤ (c : ℝ) / t := by positivity
  have hs0 : 0 ≤ countScale eps := by linarith [hs.1]
  have hprod := mul_le_mul_of_nonneg hratio hs.2 hn (by norm_num : (0 : ℝ) ≤ 8)
  have hb : (1 : ℝ) / t * countScale eps = countScale eps / t := by ring
  rw [hb]
  refine ⟨mul_nonneg hn hs0, by simpa using hprod, ?_, ?_⟩
  · rw [le_div_iff₀ htp]
    nlinarith [hs.1]
  · rw [div_le_iff₀ htp]
    nlinarith [hs.2]

example : (0 : ℝ) ≤ 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 / 64 ∧
    1 ≤ (19 : ℕ) ∧ (19 : ℕ) ≤ 19 ∧ (16 : ℕ) ≤ 19 := by norm_num

/-- The true first-FFN normalization cannot collapse these bounded states below scale one half.
Source: RMSNorm's exact coefficient, not a separately assumed positive readout scale. -/
theorem parity_rms_lower (eps : ℝ) (heps : 0 < eps) (hsmall : eps ≤ 1 / 64)
    (x : EucSpace 64) (hx : ‖x‖ ^ 2 ≤ 130) :
    1 / 2 ≤ Real.sqrt 64 / Real.sqrt (‖x‖ ^ 2 + 64 * eps) := by
  have hd : 0 < Real.sqrt (‖x‖ ^ 2 + 64 * eps) := by positivity
  have hh := Real.sqrt_le_sqrt (by nlinarith : ‖x‖ ^ 2 + 64 * eps ≤ (256 : ℝ))
  norm_num at hh ⊢
  rw [le_div_iff₀ hd]
  linarith

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 / 64 ∧
    ‖phaseUnit‖ ^ 2 ≤ 130 := by norm_num [phaseUnit, PiLp.norm_single]

/-- Every legal prompt or supplied-answer state has the required actual squared-norm bound.
Source: the exact orthogonal-coordinate norm; the phase and label residual codes have unit magnitude. -/
theorem parity_state_norm_bound (p a b q : ℝ) (hp : p ^ 2 = 1) (hq : q ^ 2 ≤ 1)
    (ha : 0 ≤ a ∧ a ≤ 8) (hb : 0 ≤ b ∧ b ≤ 8) :
    ‖decoderState p a b q 0‖ ^ 2 ≤ 130 := by
  rw [decoderState_norm_sq]
  have haa := (sq_le_sq₀ ha.1 (by norm_num : (0 : ℝ) ≤ 8)).mpr ha.2
  have hbb := (sq_le_sq₀ hb.1 (by norm_num : (0 : ℝ) ≤ 8)).mpr hb.2
  nlinarith

example : (1 : ℝ) ^ 2 = 1 ∧ (0 : ℝ) ^ 2 ≤ 1 ∧
    (0 ≤ (2 : ℝ) ∧ (2 : ℝ) ≤ 8) ∧ (0 ≤ (1 : ℝ) ∧ (1 : ℝ) ≤ 8) := by norm_num

/-- A finite original FFN weight gives a uniform label signal strictly above every input-code competitor.
Source: the exact homogeneous parity decoder and the preceding raw BOS/RMS lower bounds. -/
theorem parity_signal_margin (r b : ℝ) (hr : 1 / 2 ≤ r) (hb : 4 / 19 ≤ b) :
    9 < 1024 * (r * b) ^ 2 := by
  have hr0 : 0 ≤ r := by linarith
  have hb0 : 0 ≤ b := by linarith
  have hl : (2 / 19 : ℝ) ≤ r * b := by nlinarith
  have hsq := (sq_le_sq₀ (by norm_num : (0 : ℝ) ≤ 2 / 19)
    (mul_nonneg hr0 hb0)).mpr hl
  nlinarith

example : (1 / 2 : ℝ) ≤ 1 ∧ (4 / 19 : ℝ) ≤ 1 := by norm_num

/-- The finite EOS weight beats both tied-label scores and all input-code scores after an answer.
Source: the actual simultaneous completion unit and bounded BOS coordinate, using the same RMS factor. -/
theorem parity_eos_margin (r b : ℝ) (hr : 1 / 2 ≤ r) (hb : 0 ≤ b ∧ b ≤ 8) :
    2 + 1024 * (r * b) ^ 2 < 131072 * r ^ 2 ∧ 8 < 131072 * r ^ 2 := by
  have hr0 : 0 ≤ r := by linarith
  have hrr := (sq_le_sq₀ (by norm_num : (0 : ℝ) ≤ 1 / 2) hr0).mpr hr
  have hbb := (sq_le_sq₀ hb.1 (by norm_num : (0 : ℝ) ≤ 8)).mpr hb.2
  have hscaled := mul_le_mul_of_nonneg_left hbb (sq_nonneg r)
  constructor <;> nlinarith [sq_nonneg (r * b)]

example : (1 / 2 : ℝ) ≤ 1 ∧ (0 ≤ (1 : ℝ) ∧ (1 : ℝ) ≤ 8) := by norm_num

/-- Every raw bit word in the benchmark supplies the margin bounds in either supervised phase.
Source: the actual prompt/answer lengths len+2 and len+3, count_le_length and the exact residual state. -/
theorem parity_raw_state_bounds (eps : ℝ) (heps : 0 < eps) (hsmall : eps ≤ 1 / 64)
    (p q : ℝ) (hp : p ^ 2 = 1) (hq : q ^ 2 ≤ 1)
    (bits : List Bool) (hlen : bits.length ≤ 16) (extra : ℕ)
    (hextra : 2 ≤ extra ∧ extra ≤ 3) :
    let a := (bits.count true : ℝ) / (bits.length + extra : ℕ) * countScale eps
    let b := (1 : ℝ) / (bits.length + extra : ℕ) * countScale eps
    let x := decoderState p a b q 0
    a ≤ 8 ∧ 4 / 19 ≤ b ∧ b ≤ 8 ∧
      1 / 2 ≤ Real.sqrt 64 / Real.sqrt (‖x‖ ^ 2 + 64 * eps) := by
  dsimp only
  have hc : bits.count true ≤ bits.length + extra := by
    have hh : bits.count true ≤ bits.length := List.count_le_length
    omega
  have hcoordinates := parity_coordinate_bounds eps heps.le hsmall
    (bits.length + extra) (bits.count true) (by omega) (by omega) hc
  refine ⟨hcoordinates.2.1, hcoordinates.2.2.1, hcoordinates.2.2.2, ?_⟩
  apply parity_rms_lower eps heps hsmall
  apply parity_state_norm_bound p _ _ q hp hq
  · exact ⟨hcoordinates.1, hcoordinates.2.1⟩
  · exact ⟨by linarith [hcoordinates.2.2.1], hcoordinates.2.2.2⟩

example : (0 : ℝ) < 1 / 100000 ∧ (1 / 100000 : ℝ) ≤ 1 / 64 ∧
    (-1 : ℝ) ^ 2 = 1 ∧ (countSign 0) ^ 2 ≤ 1 ∧ [false].length ≤ 16 ∧
    (2 ≤ (3 : ℕ) ∧ (3 : ℕ) ≤ 3) := by norm_num [countSign]

end Transformer.GPTMini.Semantics
