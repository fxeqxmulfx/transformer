module

public import Transformer.SequenceModel.Attention.Backend.Complete.Certificate
public import Transformer.SequenceModel.Attention.Adaptive.Products

/-!
# Sharper certificates for the frozen CUDA trace

Source: `review/flat_spectrum.md`, *Trace-dependent execution bounds*, M3.
Residual checks remain at each operation's own recorded inputs. Squared
scale residuals and polynomial remainders are independently checked in Lean.
-/

@[expose] public section
noncomputable section
namespace Transformer.SequenceModel.Attention.Backend.Refined
open GPTMini Stability Execution
open scoped BigOperators

/-- M3: separate coordinate and Euclidean allowances for each stored operation. -/
theorem vector_local {d : ℕ} (η δ : ℝ) (hη : 0 ≤ η)
    (hb : Real.sqrt (d : ℝ) * η ≤ δ) (x z : EucSpace d)
    (hz : ∀ c, |z c - x c| ≤ η) : ‖z - x‖ ≤ δ :=
  (norm_le_coordinates (z - x) η hη hz).trans hb

/-- M3: every normalizer uses its own scale and multiplication allowance. -/
theorem rms_local (η δ : ℝ) (hb : 2 * η + 2 / 1000000000000 ≤ δ)
    (x z : EucSpace 4) (a : ℝ) (ha : 0 ≤ a)
    (hr : |a ^ 2 * ((∑ c, x c ^ 2) + 4 / 8388608) / 4 - 1| ≤ 1 / 1000000000000)
    (hz : ∀ c, |z c - a * x c| ≤ η) :
    ‖z - rmsNormEps (1 / 8388608) x‖ ≤ δ := by
  have hη : 0 ≤ η := (abs_nonneg _).trans (hz 0)
  have hm := norm_le_coordinates (z - a • x) η hη hz
  have h := rms_residual_error (d := 4) (by decide) (eps := 1 / 8388608) (by norm_num)
    x z a (2 * η) (1 / 1000000000000) ha
    (by rw [EuclideanSpace.real_norm_sq_eq]; norm_num at hr ⊢; exact hr)
    (by norm_num at hm ⊢; exact hm)
  norm_num at h
  linarith

/-- M3: the recorded value norm is outside the L2 clipping region. -/
theorem l2_local (η δ : ℝ) (hb : 2 * η + 1 / 1000000000000 ≤ δ)
    (x z : EucSpace 4) (a : ℝ) (ha : 0 ≤ a)
    (hn : (1 / 1000000000000 : ℝ) ^ 2 ≤ ∑ c, x c ^ 2)
    (hr : |a ^ 2 * (∑ c, x c ^ 2) - 1| ≤ 1 / 1000000000000)
    (hz : ∀ c, |z c - a * x c| ≤ η) :
    ‖z - normL2 (1 / 1000000000000) x‖ ≤ δ := by
  have hs : (1 / 1000000000000 : ℝ) ≤ ‖x‖ := by
    rw [← EuclideanSpace.real_norm_sq_eq] at hn
    nlinarith [norm_nonneg x]
  have hη : 0 ≤ η := (abs_nonneg _).trans (hz 0)
  have hm := norm_le_coordinates (z - a • x) η hη hz
  have h := l2_residual_error (eps := 1 / 1000000000000) (by norm_num)
    x z a (2 * η) (1 / 1000000000000) ha
    (by rw [max_eq_left hs, EuclideanSpace.real_norm_sq_eq]; exact hr)
    (by norm_num at hm ⊢; exact hm)
  linarith

example := vector_local (d := 4) 0 0 (by norm_num) (by norm_num) 0 0 (by intro c; simp)
example := rms_local 0 1 (by norm_num) (0 : EucSpace 4) 0 (2896309375740098659 / 1000000000000000)
  (by norm_num) (by norm_num) (by intro c; simp)
example := l2_local 0 1 (by norm_num) (EuclideanSpace.single 0 1) (EuclideanSpace.single 0 1)
  1 (by norm_num) (by norm_num) (by simp) (by intro c; simp)

end Transformer.SequenceModel.Attention.Backend.Refined
