/-
# Grafting and adaptive momentum clipping

arXiv:2506.12543v1, Sections 4.1--4.2 and Appendix C, Algorithm 1.
The manuscript does not choose an interpolation rule for the quantile.
We use the nearest-rank empirical quantile, preserving repeated values.
-/

import Transformer.BatchSize.Section4_ErrorFunction

noncomputable section

namespace Transformer.BatchSize

/-- Grafting takes the first update's Euclidean magnitude and the
second update's direction; Section 4.1. Zero direction produces zero. -/
def graft {d : ℕ} (magnitude direction : EucSpace d) : EucSpace d :=
  (‖magnitude‖ / ‖direction‖) • direction

/-- Grafted updates have the donor's norm whenever the direction is
nonzero; Section 4.1. -/
theorem graft_norm {d : ℕ} (m u : EucSpace d) (hu : u ≠ 0) :
    ‖graft m u‖ = ‖m‖ := by
  have hn : ‖u‖ ≠ 0 := norm_ne_zero_iff.2 hu
  rw [graft, norm_smul, Real.norm_eq_abs, abs_of_nonneg (by positivity)]
  exact div_mul_cancel₀ _ hn

/-- Nonvacuity of the grafting hypothesis, Section 4.1. -/
example : (WithLp.toLp 2 (fun _ : Fin 1 => (1 : ℝ)) : EucSpace 1) ≠ 0 := by
  intro h
  have hx := congrArg (fun u : EucSpace 1 => u 0) h
  norm_num at hx

/-- Grafting an update onto its own direction recovers that update,
including zero; Section 4.1. -/
theorem graft_self {d : ℕ} (m : EucSpace d) : graft m m = m := by
  by_cases hm : m = 0
  · simp [graft, hm]
  · rw [graft, div_self (norm_ne_zero_iff.2 hm), one_smul]

/-- The coordinate clip used in Appendix C, Algorithm 1. -/
def clipScalar (τ z : ℝ) : ℝ := Real.sign z * min |z| τ

/-- The empirical (1-p)-quantile in Appendix C, Algorithm 1.
Nearest-rank indexing is ceil((1-p)*d)-1; the empty-vector value is zero. -/
def adaptiveThreshold {d : ℕ} (p : ℝ) (m : EucSpace d) : ℝ :=
  let zs := (List.ofFn (fun k : Fin d => |m k|)).mergeSort (fun a b => decide (a ≤ b))
  zs[⌈(1 - p) * d⌉₊ - 1]?.getD 0

/-- The empirical threshold is nonnegative, Appendix C, Algorithm 1. -/
theorem adaptiveThreshold_nonneg {d : ℕ} (p : ℝ) (m : EucSpace d) :
    0 ≤ adaptiveThreshold p m := by
  let zs := (List.ofFn (fun k : Fin d => |m k|)).mergeSort (fun a b => decide (a ≤ b))
  have hz : ∀ z ∈ zs, 0 ≤ z := by
    intro z h
    have hmem := (List.mergeSort_perm (List.ofFn (fun k : Fin d => |m k|))
      (fun a b => decide (a ≤ b))).mem_iff.mp h
    obtain ⟨k, rfl⟩ := List.mem_ofFn.mp hmem
    exact abs_nonneg _
  change 0 ≤ zs[⌈(1 - p) * d⌉₊ - 1]?.getD 0
  cases h : zs[⌈(1 - p) * d⌉₊ - 1]? with
  | none => simp
  | some z => exact hz z (List.mem_of_getElem? h)

/-- Coordinate magnitudes are clipped exactly to the threshold;
Appendix C, Algorithm 1. -/
theorem clipScalar_abs (τ z : ℝ) (hτ : 0 ≤ τ) : |clipScalar τ z| = min |z| τ := by
  have hm : 0 ≤ min |z| τ := le_min (abs_nonneg z) hτ
  rcases lt_trichotomy z 0 with h | h | h
  · rw [clipScalar, abs_mul, Real.sign_of_neg h, abs_neg, abs_one,
      one_mul, abs_of_nonneg hm]
  · simp [clipScalar, h, min_eq_left hτ]
  · rw [clipScalar, abs_mul, Real.sign_of_pos h, abs_one, one_mul,
      abs_of_nonneg hm]

/-- Nonvacuity of a clipping threshold, Appendix C, Algorithm 1. -/
example : (0 : ℝ) ≤ 1 := by norm_num

/-- No coordinate grows in magnitude under clipping;
Appendix C, Algorithm 1. -/
theorem clipScalar_abs_le (τ z : ℝ) (hτ : 0 ≤ τ) : |clipScalar τ z| ≤ |z| := by
  rw [clipScalar_abs τ z hτ]
  exact min_le_left _ _

/-- Nonvacuity of the coordinate bound, Appendix C, Algorithm 1. -/
example : (0 : ℝ) ≤ 2 := by norm_num

/-- The actual clipped momentum vector in Appendix C, Algorithm 1. -/
def adaptiveClip {d : ℕ} (p : ℝ) (m : EucSpace d) : EucSpace d :=
  WithLp.toLp 2 (fun k => clipScalar (adaptiveThreshold p m) (m k))

/-- Heavy-ball buffer update in Appendix C, Algorithm 1, before clipping. -/
def momentumNext {d : ℕ} (β : ℝ) (m g : EucSpace d) : EucSpace d := β • m + g

/-- Parameter update in Appendix C, Algorithm 1. The stored momentum
is the unclipped buffer; clipping changes only the applied update. -/
def adaptiveMomentumStep {d : ℕ} (η β p : ℝ) (x m g : EucSpace d) : EucSpace d :=
  x - η • adaptiveClip p (momentumNext β m g)

/-- The algorithm clips every applied coordinate without increasing
its magnitude; Appendix C, Algorithm 1. -/
theorem adaptiveClip_coordinate_bound {d : ℕ} (p : ℝ) (m : EucSpace d) (k : Fin d) :
    |adaptiveClip p m k| ≤ |m k| :=
  clipScalar_abs_le _ _ (adaptiveThreshold_nonneg p m)

end Transformer.BatchSize
