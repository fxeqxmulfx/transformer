import Transformer.Grokking.CircuitEfficiency.SectionC_NativeRecurrence

/-!
# Actual CE feedback preserves the native sign region

Sources: Varma et al., arXiv:2309.02390v1, appendix C, product
logits and train CE; PyTorch 2.14.1 AdamW and norm-two clipping at
lab commit 793b191. Derive current coordinate signs from the actual
CE partials and their current factor partners. Induction closes these
signs on the retained parameter/moment trajectory.

Deviation from appendix C: plain CE, parameter-wise decoupled decay
and native AdamW replace GD with coupled circuit-norm regularization.
This is a fixed-table exact-real model. No prescribed future gradient
stream, learned GPTMini features, or rule-selection premise is used.
Nonnegative remaining decay factor is explicit; arbitrary rates need
not preserve positive parameters.
-/

namespace Transformer.Grokking.CircuitEfficiency

open Transformer.Grokking.AdamW

/-- The other independently trainable factor in the same circuit.
Source: appendix C, Gen and Mem each have two product subweights. -/
def nativeFactorPartner : Fin 4 → Fin 4 := ![1, 0, 3, 2]

/-- The correct training logit from both actual current products.
Source: appendix C, sim-overall-logits; all four factors contribute. -/
def nativeTotalScore (state : NativeSubweightState) : ℝ :=
  (state 0).parameter * (state 1).parameter +
    (state 2).parameter * (state 3).parameter

/-- A numerical sign region for all four parameters and retained
buffers. Source: native AdamW at 793b191; no accuracy is encoded. -/
def NonnegativeNativeState (state : NativeSubweightState) : Prop :=
  ∀ i, NonnegativeScalarState (state i)

/-- Every actual CE partial is its present partner times the common
negative CE slope. Source: appendix C's product logits and train CE;
the derivative identification is checked in NativeRecurrence. -/
theorem native_raw_gradient_partner (remaining : ℕ) (state : NativeSubweightState) (i : Fin 4) :
    rawNativeSubweightGradient remaining state i = (state (nativeFactorPartner i)).parameter *
      (-((remaining : ℝ) + 1) / (Real.exp (nativeTotalScore state) + (remaining : ℝ) + 1)) := by
  fin_cases i <;> simp [rawNativeSubweightGradient, nativeSubweightParameters,
    nativeFactorPartner, nativeTotalScore, subweightGradient, circuitMarginal, add_comm]

/-- Nonnegative present partners give nonpositive actual derivatives.
Source: appendix C, CE derivative; future gradients are not assumed. -/
theorem native_raw_gradient_nonpos (remaining : ℕ) (state : NativeSubweightState) (i : Fin 4)
    (hp : 0 ≤ (state (nativeFactorPartner i)).parameter) :
    rawNativeSubweightGradient remaining state i ≤ 0 := by
  rw [native_raw_gradient_partner]
  exact mul_nonpos_of_nonneg_of_nonpos hp
    (le_of_lt (table_train_ce_slope_negative remaining (nativeTotalScore state)))

example : 0 ≤ (seededNativeSubweights ((0, 1 / 200), (0, 1)) (nativeFactorPartner 0)).parameter := by
  norm_num [seededNativeSubweights, nativeFactorPartner, seededScalarState]

/-- A positive present partner gives a strictly negative derivative,
even when the differentiated parameter is zero. Source: appendix C,
one-zero-factor initialization and the actual product chain rule. -/
theorem native_raw_gradient_negative (remaining : ℕ) (state : NativeSubweightState) (i : Fin 4)
    (hp : 0 < (state (nativeFactorPartner i)).parameter) :
    rawNativeSubweightGradient remaining state i < 0 := by
  rw [native_raw_gradient_partner]
  exact mul_neg_of_pos_of_neg hp (table_train_ce_slope_negative remaining (nativeTotalScore state))

example : 0 < (seededNativeSubweights ((0, 1 / 200), (0, 1)) (nativeFactorPartner 0)).parameter := by
  norm_num [seededNativeSubweights, nativeFactorPartner, seededScalarState]

/-- The actual shared clipping wrapper preserves the derived weak
sign. Source: native clip_grad_norm_ at 793b191, positive bound. -/
theorem native_applied_gradient_nonpos (remaining : ℕ) (bound : ℝ)
    (state : NativeSubweightState) (i : Fin 4) (hb : 0 < bound)
    (hp : 0 ≤ (state (nativeFactorPartner i)).parameter) :
    appliedNativeSubweightGradient remaining bound state i ≤ 0 := by
  exact coordinate_clipped_nonpos bound (rawNativeSubweightGradient remaining state) i hb
    (native_raw_gradient_nonpos remaining state i hp)

example : (0 : ℝ) < 1 ∧
    0 ≤ (seededNativeSubweights ((0, 1 / 200), (0, 1)) (nativeFactorPartner 0)).parameter := by
  norm_num [seededNativeSubweights, nativeFactorPartner, seededScalarState]

/-- Clipping cannot remove a strictly negative actual partial at any
finite state and positive bound. Source: clip_grad_norm_ at 793b191;
its shared coefficient is strictly positive, not hard thresholding. -/
theorem native_applied_gradient_negative (remaining : ℕ) (bound : ℝ)
    (state : NativeSubweightState) (i : Fin 4) (hb : 0 < bound)
    (hp : 0 < (state (nativeFactorPartner i)).parameter) :
    appliedNativeSubweightGradient remaining bound state i < 0 := by
  exact coordinate_clipped_negative bound (rawNativeSubweightGradient remaining state) i hb
    (native_raw_gradient_negative remaining state i hp)

example : (0 : ℝ) < 1 ∧
    0 < (seededNativeSubweights ((0, 1 / 200), (0, 1)) (nativeFactorPartner 0)).parameter := by
  norm_num [seededNativeSubweights, nativeFactorPartner, seededScalarState]

/-- Nonnegative four-factor seeds initialize the numerical region
with actual zero buffers. Source: native AdamW at 793b191 and
appendix C's permitted one-zero-factor source initialization. -/
theorem native_seeded_nonnegative (a b c d : ℝ)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d) :
    NonnegativeNativeState (seededNativeSubweights ((a, b), (c, d))) := by
  intro i
  fin_cases i
  · exact seeded_scalar_nonnegative a ha
  · exact seeded_scalar_nonnegative b hb
  · exact seeded_scalar_nonnegative c hc
  · exact seeded_scalar_nonnegative d hd

example : (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 / 200 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 := by norm_num

/-- One closed actual CE step preserves all signs, including retained
moments and physical variances. Sources: appendix C's product CE and
native AdamW/clipping at 793b191; the current gradient signs are
derived from the old state, not supplied as additional hypotheses. -/
theorem native_nonnegative_step (remaining : ℕ) (bound b1 b2 eps decay rate : ℝ)
    (state : NativeSubweightState) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 ≤ rate) (hd : 0 ≤ 1 - rate * decay)
    (hs : NonnegativeNativeState state) :
    NonnegativeNativeState (nativeSubweightStep remaining bound b1 b2 eps decay rate state) := by
  intro i
  exact scalar_nonnegative_step b1 b2 eps decay rate _ (state i) hb1 h1 hb2 (le_of_lt h2) he heta hd
    (hs i) (native_applied_gradient_nonpos remaining bound state i hclip (hs _).1)

example : (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧
    0 ≤ (1 / 1000 : ℝ) ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * (1 / 10) ∧
    NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) := by
  refine ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, native_seeded_nonnegative _ _ _ _ (by norm_num)
      (by norm_num) (by norm_num) (by norm_num)⟩

/-- The entire retained CE trajectory stays in the numerical region.
Sources: appendix C product CE and native AdamW at 793b191; induction
closes the gradient/parameter feedback without assuming its future. -/
theorem native_nonnegative_path (remaining : ℕ) (bound b1 b2 eps decay rate : ℝ)
    (initial : NativeSubweightState) (n : ℕ) (hclip : 0 < bound)
    (hb1 : 0 ≤ b1) (h1 : b1 < 1) (hb2 : 0 ≤ b2) (h2 : b2 < 1)
    (he : 0 < eps) (heta : 0 ≤ rate) (hd : 0 ≤ 1 - rate * decay)
    (hs : NonnegativeNativeState initial) :
    NonnegativeNativeState (nativeSubweightPath remaining bound b1 b2 eps decay rate initial n) := by
  induction n with
  | zero => exact hs
  | succ n ih =>
    exact native_nonnegative_step remaining bound b1 b2 eps decay rate _ hclip
      hb1 h1 hb2 h2 he heta hd ih

example : NonnegativeNativeState (seededNativeSubweights ((0, 1 / 200), (0, 1))) ∧
    (0 : ℝ) < 1 ∧ 0 ≤ (9 / 10 : ℝ) ∧ (9 / 10 : ℝ) < 1 ∧
    0 ≤ (49 / 50 : ℝ) ∧ (49 / 50 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100000000 ∧
    0 ≤ (1 / 1000 : ℝ) ∧ 0 ≤ 1 - (1 / 1000 : ℝ) * (1 / 10) := by
  refine ⟨native_seeded_nonnegative _ _ _ _ (by norm_num) (by norm_num)
    (by norm_num) (by norm_num), by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩

end Transformer.Grokking.CircuitEfficiency
