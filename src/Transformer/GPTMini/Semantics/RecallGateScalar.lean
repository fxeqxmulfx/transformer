import Transformer.GPTMini.Semantics.CountSpline
import Transformer.GPTMini.Semantics.RecallRawMarker

/-!
# A finite table cutoff using the original quadratic activation

Source: MQAR's fixed table of P adjacent pairs at cbafbe9 and the
bias-free ReLU2_FFN at f11b6e2. The actual BOS marker equals 1/(i+1).
The midpoint between the last table value and the next position is
therefore a fixed linear threshold with an explicit positive margin.

Two ordinary quadratic hinges compute a gated product on separated
positive/negative regions. Eight copied-key coordinates need sixteen
units, within the original width 256. Positive RMS scaling preserves
the gate and multiplies its output by the square of that scale.
These scalar computations do not insert a semantic indicator into
the model: the following matrix realization will use its real residual
coordinates. Raw-prefix parsing and final retrieval remain separate.
-/

namespace Transformer.GPTMini.Semantics

/-- The threshold lies halfway between positions 2P and 2P+1.
Source: the actual reciprocal-prefix BOS mass and MQAR's fixed table size. -/
noncomputable def recallTableThreshold (P : ℕ) : ℝ :=
  (1 / (2 * (P : ℝ) + 1) + 1 / (2 * (P : ℝ) + 2)) / 2

/-- Half the separation of those two real marker values.
Source: the derived reciprocal-prefix signal, not an assumed gate margin. -/
noncomputable def recallTableMargin (P : ℕ) : ℝ :=
  1 / (2 * (2 * (P : ℝ) + 1) * (2 * (P : ℝ) + 2))

/-- The fixed cutoff is positive for every nonnegative pair count.
Source: both denominators are strictly positive, including at P=0. -/
theorem recallTableThreshold_pos (P : ℕ) : 0 < recallTableThreshold P := by
  unfold recallTableThreshold
  positivity

/-- The actual boundary separation has a strictly positive margin.
Source: the explicit reciprocal difference at the two adjacent boundary positions. -/
theorem recallTableMargin_pos (P : ℕ) : 0 < recallTableMargin P := by
  unfold recallTableMargin
  positivity

/-- This margin is at most one, uniformly in the table size.
Source: its explicit positive denominator, already at least four at P=0. -/
theorem recallTableMargin_le_one (P : ℕ) : recallTableMargin P ≤ 1 := by
  unfold recallTableMargin
  rw [div_le_iff₀ (by positivity)]
  have hP : (0 : ℝ) ≤ P := Nat.cast_nonneg P
  nlinarith

/-- The last table position exceeds the cutoff by exactly the stated margin.
Source: the midpoint formula evaluated at the genuine marker 1/(2P+1). -/
theorem recallTableThreshold_left (P : ℕ) :
    1 / (2 * (P : ℝ) + 1) - recallTableThreshold P = recallTableMargin P := by
  unfold recallTableThreshold recallTableMargin
  have ha : 2 * (P : ℝ) + 1 ≠ 0 := ne_of_gt (by positivity)
  have hb : 2 * (P : ℝ) + 2 ≠ 0 := ne_of_gt (by positivity)
  field_simp
  ring

/-- The first later position falls below the cutoff by the same margin.
Source: the midpoint formula evaluated at the genuine marker 1/(2P+2). -/
theorem recallTableThreshold_right (P : ℕ) :
    recallTableThreshold P - 1 / (2 * (P : ℝ) + 2) = recallTableMargin P := by
  unfold recallTableThreshold recallTableMargin
  have ha : 2 * (P : ℝ) + 1 ≠ 0 := ne_of_gt (by positivity)
  have hb : 2 * (P : ℝ) + 2 ≠ 0 := ne_of_gt (by positivity)
  field_simp
  ring

/-- Type exclusion can use the threshold itself as a sufficient negative gate margin.
Source: the threshold is the positive next-position mass plus the derived boundary margin. -/
theorem recallTableMargin_le_threshold (P : ℕ) : recallTableMargin P ≤ recallTableThreshold P := by
  have hn : (0 : ℝ) ≤ 1 / (2 * (P : ℝ) + 2) := by positivity
  linarith [recallTableThreshold_right P]

/-- Every position inside the raw table exceeds the cutoff by at least that margin.
Source: the actual causal denominator and the table boundary i ≤ 2P. -/
theorem recallTableThreshold_inside (P i : ℕ) (hi : i ≤ 2 * P) :
    recallTableMargin P ≤ 1 / ((i + 1 : ℕ) : ℝ) - recallTableThreshold P := by
  have hden : ((i + 1 : ℕ) : ℝ) ≤ 2 * (P : ℝ) + 1 := by exact_mod_cast (by omega : i + 1 ≤ 2 * P + 1)
  have hrec := one_div_le_one_div_of_le (by positivity : (0 : ℝ) < ((i + 1 : ℕ) : ℝ)) hden
  linarith [recallTableThreshold_left P]

example : (16 : ℕ) ≤ 2 * 8 := by decide

/-- Every post-table position falls below the cutoff by at least that margin.
Source: the actual causal denominator and the table boundary 2P < i. -/
theorem recallTableThreshold_after (P i : ℕ) (hi : 2 * P < i) :
    1 / ((i + 1 : ℕ) : ℝ) - recallTableThreshold P ≤ -recallTableMargin P := by
  have hden : 2 * (P : ℝ) + 2 ≤ ((i + 1 : ℕ) : ℝ) := by exact_mod_cast (by omega : 2 * P + 2 ≤ i + 1)
  have hrec := one_div_le_one_div_of_le (by positivity : (0 : ℝ) < 2 * (P : ℝ) + 2) hden
  linarith [recallTableThreshold_right P]

example : 2 * (8 : ℕ) < 17 := by decide

/-- Two ordinary ReLU2 units realize a signed gated product.
Source: the original quadratic activation; all three arguments affect its computation. -/
noncomputable def recallGateProduct (eta g z : ℝ) : ℝ :=
  (relu2 (g + eta * z) - relu2 (g - eta * z)) / (4 * eta)

/-- On the separated positive region, the actual two activations multiply gate and coordinate exactly.
Source: the difference of two squares, with both preactivations nonnegative. -/
theorem recallGateProduct_on (eta g z : ℝ) (heta : eta ≠ 0) (hgate : |eta * z| ≤ g) :
    recallGateProduct eta g z = g * z := by
  obtain ⟨hlo, hhi⟩ := abs_le.mp hgate
  have hp : 0 ≤ g + eta * z := by linarith
  have hm : 0 ≤ g - eta * z := by linarith
  unfold recallGateProduct relu2
  rw [max_eq_right hp, max_eq_right hm]
  field_simp
  ring

example : (1 : ℝ) ≠ 0 ∧ |(1 : ℝ) * 1| ≤ 2 := by norm_num

/-- On the separated negative region, both actual units vanish exactly.
Source: the original nonpositive ReLU2 branch, independent of any desired output. -/
theorem recallGateProduct_off (eta g z : ℝ) (hgate : |eta * z| ≤ -g) :
    recallGateProduct eta g z = 0 := by
  obtain ⟨hlo, hhi⟩ := abs_le.mp hgate
  have hp : g + eta * z ≤ 0 := by linarith
  have hm : g - eta * z ≤ 0 := by linarith
  rw [recallGateProduct, relu2_of_nonpos _ hp, relu2_of_nonpos _ hm]
  ring

example : |(1 : ℝ) * 1| ≤ -(-2 : ℝ) := by norm_num

/-- Genuine positive prenorm scales this homogeneous gate computation quadratically.
Source: unchanged RMSNorm followed by the two original quadratic activation units. -/
theorem recallGateProduct_scale (eta g z s : ℝ) (hs : 0 ≤ s) :
    recallGateProduct eta (s * g) (s * z) = s ^ 2 * recallGateProduct eta g z := by
  have hp : s * g + eta * (s * z) = s * (g + eta * z) := by ring
  have hm : s * g - eta * (s * z) = s * (g - eta * z) := by ring
  rw [recallGateProduct, hp, hm, relu2_mul_nonneg s _ hs, relu2_mul_nonneg s _ hs]
  unfold recallGateProduct
  ring

example : (0 : ℝ) ≤ 2 := by norm_num

/-- The derived margin controls perturbations of all actual copied coordinates bounded by four.
Source: the original XSA head bound and the fixed choice eta = margin/8. -/
theorem recallGateProduct_margin (P : ℕ) (z : ℝ) (hz : |z| ≤ 4) :
    |(recallTableMargin P / 8) * z| ≤ recallTableMargin P / 2 := by
  have hd := (recallTableMargin_pos P).le
  rw [abs_mul, abs_of_nonneg (by positivity : 0 ≤ recallTableMargin P / 8)]
  nlinarith

example : |(3 : ℝ)| ≤ 4 := by norm_num

end Transformer.GPTMini.Semantics
