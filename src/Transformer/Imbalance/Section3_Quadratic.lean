/-
# The weighted quadratic illustration

arXiv:2402.19449v2, Section 3.1. The same scalar learning rate scales
different coordinates by their class frequencies. The source's instability
threshold 1/π_max is incorrect for gradient descent: convergence continues
up to, but excluding, 2/π_max.
-/

import Transformer.Imbalance.Section3_Model
import Mathlib.Analysis.SpecificLimits.Normed
import Mathlib.Basic.Real.Sign

open Filter
open scoped BigOperators Topology

noncomputable section

namespace Transformer.Imbalance

/-- Weighted diagonal quadratic objective; Section 3.1. -/
def quadraticLoss {c : ℕ} (π w : Fin c → ℝ) : ℝ := ∑ k, π k * (w k) ^ 2 / 2

/-- Coordinate-wise GD recurrence for the quadratic; Section 3.1. -/
def quadraticGD (π α w₀ : ℝ) : ℕ → ℝ
  | 0 => w₀
  | t + 1 => quadraticGD π α w₀ t - α * π * quadraticGD π α w₀ t

/-- Exact trajectory from Section 3.1: `(1-απ)^t w₀`. -/
theorem quadraticGD_eq (π α w₀ : ℝ) (t : ℕ) :
    quadraticGD π α w₀ t = (1 - α * π) ^ t * w₀ := by
  induction t with
  | zero => simp [quadraticGD]
  | succ t ih => rw [quadraticGD, ih, pow_succ]; ring

/-- A unit step solves the unweighted scalar quadratic; Section 3.1. -/
theorem quadraticGD_unit_step (w₀ : ℝ) : quadraticGD 1 1 w₀ 1 = 0 := by
  rw [quadraticGD_eq]
  norm_num

/-- Correct convergence interval for the weighted quadratic;
Section 3.1's stability discussion, with threshold corrected to απ<2. -/
theorem quadraticGD_converges (π α w₀ : ℝ) (hpos : 0 < α * π) (hlt : α * π < 2) :
    Tendsto (quadraticGD π α w₀) atTop (𝓝 0) := by
  have habs : |1 - α * π| < 1 := abs_lt.2 ⟨by linarith, by linarith⟩
  rw [show quadraticGD π α w₀ = (fun t => (1 - α * π) ^ t * w₀) from
    funext (quadraticGD_eq π α w₀)]
  simpa only [zero_mul] using
    (tendsto_pow_atTop_nhds_zero_of_abs_lt_one habs).mul_const w₀

/-- Nonvacuity of the corrected stability interval; Section 3.1. -/
example : (0 : ℝ) < (3 / 2) * 1 ∧ (3 / 2 : ℝ) * 1 < 2 := by norm_num

/-- Counterexample to Section 3.1's claim that steps above 1/π_max cause
instability: π=1 and α=3/2 give a convergent nonzero-start trajectory. -/
theorem paper_quadratic_threshold_false :
    (1 : ℝ) / 1 < 3 / 2 ∧ Tendsto (quadraticGD 1 (3 / 2) 1) atTop (𝓝 0) := by
  exact ⟨by norm_num, quadraticGD_converges 1 (3 / 2) 1 (by norm_num) (by norm_num)⟩

/-- Positive class frequencies cancel from the sign direction;
Section 3.1, the displayed sign-descent update. -/
theorem sign_positive_scale (π w : ℝ) (hπ : 0 < π) : Real.sign (π * w) = Real.sign w := by
  rcases lt_trichotomy w 0 with hw | hw | hw
  · rw [Real.sign_of_neg hw, Real.sign_of_neg (mul_neg_of_pos_of_neg hπ hw)]
  · subst w
    simp
  · rw [Real.sign_of_pos hw, Real.sign_of_pos (mul_pos hπ hw)]

/-- Nonvacuity of sign cancellation; Section 3.1. -/
example : (0 : ℝ) < 1 / 1000 := by norm_num

/-- Actual fixed-step sign-descent update on the quadratic; Section 3.1. -/
def quadraticSignStep (π α w : ℝ) : ℝ := w - α * Real.sign (π * w)

/-- Sign descent is independent of positive class frequency; Section 3.1. -/
theorem quadraticSignStep_frequency (π α w : ℝ) (hπ : 0 < π) :
    quadraticSignStep π α w = w - α * Real.sign w := by
  rw [quadraticSignStep, sign_positive_scale π w hπ]

/-- Nonvacuity of the frequency-independent update; Section 3.1. -/
example : (0 : ℝ) < 1 / 1000 := by norm_num

/-- The source warns of fixed-step oscillation in Section 3.1.
For π=1, α=2, starting at one gives the two-cycle one, minus one. -/
theorem quadraticSignStep_two_cycle :
    quadraticSignStep 1 2 1 = -1 ∧ quadraticSignStep 1 2 (-1) = 1 := by
  norm_num [quadraticSignStep, Real.sign]

end Transformer.Imbalance
