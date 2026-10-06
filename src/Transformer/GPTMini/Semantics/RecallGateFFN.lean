import Transformer.GPTMini.Semantics.RecallGateSignals

/-!
# Exact outcomes of the genuine table-gating FFN

Source: the unchanged original ReLU2_FFN at f11b6e2, the actual
sixteen-unit matrix realization and the derived raw gate margins.
On a separated positive gate, the actual FFN writes a positive
multiple of the copied-key vector to the fresh slot. On a separated
negative gate, its entire output is exactly zero, including after
the genuine prenorm. The RMS square remains in the stated formula.

The local coordinate bounds below are subsequently discharged by
the real first attention residual, rather than imposed on raw data
as a prepared encoder. Protected coordinates stay available to the
second block and tied readout. These conditional operator lemmas do
not yet constitute full raw Basis correctness or a convexity claim.
-/

namespace Transformer.GPTMini.Semantics

open scoped BigOperators

/-- The gate uses the original FFN parameter record with ordinary finite matrices.
Source: W_in/W_out at f11b6e2, with exactly sixteen assigned hidden coordinates. -/
noncomputable def recallGateParameters (P : ℕ) (beta : ℝ) : FFNParams recallConfig where
  W_in := recallGateIn P
  W_out := recallGateOut P beta

/-- The actual FFN is identically zero throughout the separated negative region, including its real prenorm.
Source: both original quadratic units vanish for every copied coordinate using the derived margin. -/
theorem recallGateFFN_off (P : ℕ) (beta eps : ℝ) (x : EucSpace 64)
    (hgate : recallGateForm P x ≤ -recallTableMargin P)
    (hcopy : ∀ c : Fin 8, |x (recallSlotIndex 18 c)| ≤ 4) :
    relu2FFN (recallGateIn P) (recallGateOut P beta) (rmsNormEps eps x) = 0 := by
  rw [recallGateFFN_rms]
  apply Finset.sum_eq_zero
  intro c _
  have hd := (recallTableMargin_pos P).le
  have hm := recallGateProduct_margin P _ (hcopy c)
  have hg : |(recallTableMargin P / 8) * x (recallSlotIndex 18 c)| ≤ -recallGateForm P x := by linarith
  rw [recallGateProduct_off _ _ _ hg, mul_zero, zero_smul]

example : recallGateForm 0 (-(recallUnit 26) + (3 : ℝ) • recallUnit 18) ≤ -recallTableMargin 0 ∧
    (∀ c : Fin 8, |(-(recallUnit 26) + (3 : ℝ) • recallUnit 18) (recallSlotIndex 18 c)| ≤ 4) := by
  constructor
  · norm_num [recallGateForm_apply, recallTableThreshold, recallTableMargin, recallUnit,
      PiLp.add_apply, PiLp.neg_apply, PiLp.smul_apply, PiLp.single_apply]
  · intro c
    fin_cases c <;> norm_num [recallSlotIndex, recallUnit, PiLp.add_apply, PiLp.neg_apply, PiLp.smul_apply, PiLp.single_apply]

/-- On the positive separated region the genuine FFN writes the scaled complete compact copy exactly.
Source: the original two-square identity, all eight output matrix rows and the true RMS multiplier. -/
theorem recallGateFFN_on (P : ℕ) (beta eps : ℝ) (x : EucSpace 64)
    (hgate : recallTableMargin P ≤ recallGateForm P x)
    (hcopy : ∀ c : Fin 8, |x (recallSlotIndex 18 c)| ≤ 4) :
    relu2FFN (recallGateIn P) (recallGateOut P beta) (rmsNormEps eps x) =
      (beta * (8 / Real.sqrt (‖x‖ ^ 2 + 64 * eps)) ^ 2 * recallGateForm P x) •
        recallSlotWrite 27 (recallSlotRead 18 x) := by
  have heta : recallTableMargin P / 8 ≠ 0 := ne_of_gt (by
    have h := recallTableMargin_pos P
    positivity)
  have hprod (c : Fin 8) :
      recallGateProduct (recallTableMargin P / 8) (recallGateForm P x) (x (recallSlotIndex 18 c)) =
        recallGateForm P x * x (recallSlotIndex 18 c) := by
    have hd := (recallTableMargin_pos P).le
    have hm := recallGateProduct_margin P _ (hcopy c)
    exact recallGateProduct_on _ _ _ heta (by linarith)
  rw [recallGateFFN_rms, recallSlotWrite_apply, Finset.smul_sum]
  simp_rw [hprod, recallSlotRead_at]
  apply Finset.sum_congr rfl
  intro c _
  rw [smul_smul]
  congr 1
  ring

example : recallTableMargin 0 ≤ recallGateForm 0 (recallUnit 26 + (3 : ℝ) • recallUnit 18) ∧
    (∀ c : Fin 8, |(recallUnit 26 + (3 : ℝ) • recallUnit 18) (recallSlotIndex 18 c)| ≤ 4) := by
  constructor
  · norm_num [recallGateForm_apply, recallTableThreshold, recallTableMargin, recallUnit,
      PiLp.add_apply, PiLp.smul_apply, PiLp.single_apply]
  · intro c
    fin_cases c <;> norm_num [recallSlotIndex, recallUnit, PiLp.add_apply, PiLp.smul_apply, PiLp.single_apply]

/-- Reading the true destination slot retains exactly the whole scaled copied-key vector.
Source: the actual positive FFN formula and proved real insertion/readback matrices. -/
theorem recallGateFFN_read_on (P : ℕ) (beta eps : ℝ) (x : EucSpace 64)
    (hgate : recallTableMargin P ≤ recallGateForm P x)
    (hcopy : ∀ c : Fin 8, |x (recallSlotIndex 18 c)| ≤ 4) :
    recallSlotRead 27 (relu2FFN (recallGateIn P) (recallGateOut P beta) (rmsNormEps eps x)) =
      (beta * (8 / Real.sqrt (‖x‖ ^ 2 + 64 * eps)) ^ 2 * recallGateForm P x) • recallSlotRead 18 x := by
  rw [recallGateFFN_on P beta eps x hgate hcopy, map_smul, recallSlotRead_write]

example : recallTableMargin 0 ≤ recallGateForm 0 (recallUnit 26) ∧
    (∀ c : Fin 8, |recallUnit 26 (recallSlotIndex 18 c)| ≤ 4) := by
  constructor
  · norm_num [recallGateForm_apply, recallTableThreshold, recallTableMargin, recallUnit, PiLp.single_apply]
  · intro c
    fin_cases c <;> norm_num [recallSlotIndex, recallUnit, PiLp.single_apply]

/-- Every true gate output is supported only on the fresh gated-key interval.
Source: its complete real output matrix, independent of the gate's activation region. -/
theorem recallGateFFN_protected (P : ℕ) (beta : ℝ) (x : EucSpace 64) (i : Fin 64)
    (hi : i.val < 27 ∨ 35 ≤ i.val) :
    relu2FFN (recallGateIn P) (recallGateOut P beta) x i = 0 := by
  rw [recallGateFFN_apply]
  simp only [WithLp.ofLp_sum, Finset.sum_apply, WithLp.ofLp_smul, Pi.smul_apply,
    recallUnit, PiLp.single_apply, smul_eq_mul]
  apply Finset.sum_eq_zero
  intro c _
  have hn : i ≠ recallSlotIndex 27 c := by
    intro he
    have hv := congrArg Fin.val he
    change i.val = 27 + c.val at hv
    have hc := c.isLt
    omega
  rw [ite_eq_right hn, mul_zero]

example : (36 : Fin 64).val < 27 ∨ 35 ≤ (36 : Fin 64).val := by decide

/-- A genuinely empty predecessor-copy slot cannot create a gated key, regardless of the gate sign.
Source: the two ordinary quadratic preactivations then coincide for all eight actual coordinate pairs. -/
theorem recallGateFFN_empty_copy (P : ℕ) (beta eps : ℝ) (x : EucSpace 64)
    (hcopy : recallSlotRead 18 x = 0) :
    relu2FFN (recallGateIn P) (recallGateOut P beta) (rmsNormEps eps x) = 0 := by
  have hz (c : Fin 8) : x (recallSlotIndex 18 c) = 0 := by
    rw [← recallSlotRead_at, hcopy, PiLp.zero_apply]
  rw [recallGateFFN_rms]
  apply Finset.sum_eq_zero
  intro c _
  rw [hz]
  unfold recallGateProduct
  simp only [mul_zero, add_zero, sub_zero, sub_self, zero_div, zero_smul]

example : recallSlotRead 18 (recallUnit 26) = 0 := by
  unfold recallUnit
  exact recallSlotRead_unit_outside 18 26 (by decide)

/-- A bounded copy error becomes the same error times the actual derived gate amplitude.
Source: the true destination-slot formula and ordinary norm homogeneity, with no omitted RMS factor. -/
theorem recallGateFFN_copy_error (P : ℕ) (beta eps eta : ℝ) (x : EucSpace 64) (key : EucSpace 8)
    (hgate : recallTableMargin P ≤ recallGateForm P x)
    (hcopy : ∀ c : Fin 8, |x (recallSlotIndex 18 c)| ≤ 4)
    (herr : ‖recallSlotRead 18 x - key‖ ≤ eta) :
    let a := beta * (8 / Real.sqrt (‖x‖ ^ 2 + 64 * eps)) ^ 2 * recallGateForm P x
    ‖recallSlotRead 27 (relu2FFN (recallGateIn P) (recallGateOut P beta) (rmsNormEps eps x)) - a • key‖ ≤ |a| * eta := by
  dsimp only
  rw [recallGateFFN_read_on P beta eps x hgate hcopy, ← smul_sub, norm_smul, Real.norm_eq_abs]
  exact mul_le_mul_of_nonneg_left herr (abs_nonneg _)

example : recallTableMargin 0 ≤ recallGateForm 0 (recallUnit 26) ∧
    (∀ c : Fin 8, |recallUnit 26 (recallSlotIndex 18 c)| ≤ 4) ∧
    ‖recallSlotRead 18 (recallUnit 26) - (0 : EucSpace 8)‖ ≤ (0 : ℝ) := by
  refine ⟨?_, ?_, ?_⟩
  · norm_num [recallGateForm_apply, recallTableThreshold, recallTableMargin, recallUnit, PiLp.single_apply]
  · intro c
    fin_cases c <;> norm_num [recallSlotIndex, recallUnit, PiLp.single_apply]
  · rw [recallUnit, recallSlotRead_unit_outside 18 26 (by decide), sub_zero, norm_zero]

end Transformer.GPTMini.Semantics
