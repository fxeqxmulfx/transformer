module

public import Transformer.SequenceModel.Attention.Backend.CapturedCUDA32
public import Transformer.SequenceModel.Attention.Regularity.Head

/-!
# Rational certificates for complete captured executions

Source: `review/flat_spectrum.md`, *Complete captured execution*, L2.
The uniform local allowance is 1e-4; primitive certificates below are tighter.
All checks concern the actual stored inputs and outputs. They do not assume
an error for a whole block or a whole model.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention.Backend.Complete

open GPTMini Stability

/-- L2: the width-four source half-split rotation, with independent stored coefficients. -/
theorem table_four (c s : Fin 2 → ℝ) (x : EucSpace 4) :
    Execution.tableRotate 4 c s x = WithLp.toLp 2
      ![c 0 * x 0 - s 0 * x 2, c 1 * x 1 - s 1 * x 3,
        s 0 * x 0 + c 0 * x 2, s 1 * x 1 + c 1 * x 3] := by
  ext i
  fin_cases i <;> rfl

/-- L2: a coordinate box gives the operation allowance at widths up to sixteen. -/
theorem vector_checked {d : ℕ} (hd : d ≤ 16) (x z : EucSpace d)
    (hz : ∀ c, |z c - x c| ≤ 1 / 1000000) : ‖z - x‖ ≤ 1 / 10000 := by
  have hs : Real.sqrt (d : ℝ) ≤ 4 := Real.sqrt_le_iff.mpr
    ⟨by norm_num, by exact_mod_cast hd⟩
  have h := norm_le_coordinates (z - x) (1 / 1000000) (by norm_num) hz
  nlinarith

/-- L2: every four-coordinate RMS call is certified by its own rational scale. -/
theorem rms_checked (x z : EucSpace 4) (a : ℝ) (ha : 0 ≤ a)
    (hr : |a ^ 2 * ((∑ c, x c ^ 2) + 4 / 8388608) / 4 - 1| ≤ 1 / 1000000)
    (hz : ∀ c, |z c - a * x c| ≤ 1 / 1000000) :
    ‖z - rmsNormEps (1 / 8388608) x‖ ≤ 1 / 10000 := by
  have hm := norm_le_coordinates (z - a • x) (1 / 1000000) (by norm_num) hz
  have h := rms_residual_error (d := 4) (by decide) (eps := 1 / 8388608) (by norm_num)
    x z a (2 / 1000000) (1 / 1000000) ha
    (by rw [EuclideanSpace.real_norm_sq_eq]; norm_num at hr ⊢; exact hr)
    (by norm_num at hm ⊢; exact hm)
  norm_num at h
  linarith

/-- L2: recorded values outside the L2 clipping region have a rational square check. -/
theorem l2_checked (x z : EucSpace 4) (a : ℝ) (ha : 0 ≤ a)
    (hn : (1 / 1000000000000 : ℝ) ^ 2 ≤ ∑ c, x c ^ 2)
    (hr : |a ^ 2 * (∑ c, x c ^ 2) - 1| ≤ 1 / 1000000)
    (hz : ∀ c, |z c - a * x c| ≤ 1 / 1000000) :
    ‖z - normL2 (1 / 1000000000000) x‖ ≤ 1 / 10000 := by
  have hs : (1 / 1000000000000 : ℝ) ≤ ‖x‖ := by
    rw [← EuclideanSpace.real_norm_sq_eq] at hn
    nlinarith [norm_nonneg x]
  have hm := norm_le_coordinates (z - a • x) (1 / 1000000) (by norm_num) hz
  have h := l2_residual_error (eps := 1 / 1000000000000) (by norm_num)
    x z a (2 / 1000000) (1 / 1000000) ha
    (by rw [max_eq_left hs, EuclideanSpace.real_norm_sq_eq]; exact hr)
    (by norm_num at hm ⊢; exact hm)
  linarith

/-- L2: a fixed-order polynomial supplies a uniform analytic exp remainder on [-1,1]. -/
theorem exp_checked (x e : ℝ) (hx : |x| ≤ 1)
    (hr : |e - expPolynomial x 12| ≤ 1 / 10000000000) :
    |e - Real.exp x| ≤ 1 / 100000000 := by
  have h := exponential_error x e 12 hx (by decide) (1 / 10000000000) hr
  have hp : |x| ^ 12 ≤ 1 := pow_le_one₀ (abs_nonneg x) hx
  norm_num at h
  linarith

/-- L2: each SiLU coordinate is certified against its recorded gate input. -/
theorem silu_checked (x z e : ℝ) (hx : |x| ≤ 1) (he : 0 ≤ e)
    (hr : |e - expPolynomial (-x) 12| ≤ 1 / 10000000000)
    (hz : |z - x / (1 + e)| ≤ 1 / 1000000) :
    |z - Landscape.Neural.silu x| ≤ 2 / 1000000 := by
  have hE := exp_checked (-x) e (by simpa using hx) hr
  have h := silu_residual_error x z e (1 / 1000000) (1 / 100000000) he hE hz
  linarith

example := vector_checked (d := 4) (by decide) (EuclideanSpace.single 0 1)
  (EuclideanSpace.single 0 1) (by intro c; simp)
example := rms_checked (0 : EucSpace 4) 0 (2896309376 / 1000000)
  (by norm_num) (by norm_num) (by intro c; simp)
example := l2_checked (EuclideanSpace.single 0 1) (EuclideanSpace.single 0 1) 1
  (by norm_num) (by norm_num) (by simp) (by intro c; simp)
example := exp_checked 0 1 (by norm_num) (by norm_num [expPolynomial, Finset.sum_range_succ])
example := silu_checked 0 0 1 (by norm_num) (by norm_num)
  (by norm_num [expPolynomial, Finset.sum_range_succ]) (by norm_num)

end Transformer.SequenceModel.Attention.Backend.Complete
