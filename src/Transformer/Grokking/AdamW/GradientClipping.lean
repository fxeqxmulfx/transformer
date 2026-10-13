import Transformer.Grokking.AdamW.MomentBounds
import Mathlib.Algebra.BigOperators.Fin

/-!
# Exact-real native gradient clipping and its moment consequences

Source: PyTorch 2.14.1 clip_grad_norm_ at lab commit 88aa892, norm
type two and coefficient min(1, max_norm/(total_norm+1e-6)). Flatten
all always-present coordinates, preserving the source's additive
constant. Finite precision, tensor grouping and summation errors are
not identified with the exact-real Euclidean norm here.

Derive sign preservation and norm/coordinate bounds supplying the
premises of the corrected moment estimates. The source doc says
gradients are only reduced when total_norm exceeds max_norm; its
1e-6 coefficient instead reduces them in the boundary strip too.
The exact threshold and a unit-boundary counterexample are proved.
-/

namespace Transformer.Grokking.AdamW

open scoped BigOperators

/-- Exact-real flattened norm used before native clipping. Source:
PyTorch 2.14.1 clip_grad_norm_ at 88aa892, norm_type=2. -/
noncomputable def coordinateGradientNorm {dimension : ℕ} (gradient : Fin dimension → ℝ) : ℝ :=
  Real.sqrt (∑ i, gradient i ^ 2)

/-- Source clipping coefficient, including its additive 1e-6. Source:
PyTorch 2.14.1 clip_grad_norm_ at 88aa892, clip_coef_clamped. -/
noncomputable def coordinateClipFactor {dimension : ℕ}
    (bound : ℝ) (gradient : Fin dimension → ℝ) : ℝ :=
  min 1 (bound / (coordinateGradientNorm gradient + 1 / 1000000))

/-- The actual scalar multiplication applied to every flattened
coordinate. Source: PyTorch 2.14.1 clip_grad_norm_ at 88aa892. -/
noncomputable def coordinateClippedGradient {dimension : ℕ}
    (bound : ℝ) (gradient : Fin dimension → ℝ) : Fin dimension → ℝ :=
  fun i => coordinateClipFactor bound gradient * gradient i

/-- Flattened Euclidean norm scales by absolute scalar. Source:
native norm_type=2 at 88aa892, derived from the finite squared sum. -/
theorem coordinate_norm_scaling {dimension : ℕ} (scale : ℝ) (gradient : Fin dimension → ℝ) :
    coordinateGradientNorm (fun i => scale * gradient i) = |scale| * coordinateGradientNorm gradient := by
  have hs : (∑ i, (scale * gradient i) ^ 2) = scale ^ 2 * (∑ i, gradient i ^ 2) := by
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i hi
    ring
  unfold coordinateGradientNorm
  rw [hs, Real.sqrt_mul (sq_nonneg scale), Real.sqrt_sq_eq_abs]

/-- Every coordinate magnitude is bounded by that flattened norm.
Source: native norm_type=2 at 88aa892; no task-specific input bound. -/
theorem coordinate_abs_le_norm {dimension : ℕ} (gradient : Fin dimension → ℝ) (i : Fin dimension) :
    |gradient i| ≤ coordinateGradientNorm gradient := by
  have hs : gradient i ^ 2 ≤ ∑ j, gradient j ^ 2 :=
    Finset.single_le_sum (fun j hj => sq_nonneg (gradient j)) (Finset.mem_univ i)
  have hr := Real.sqrt_le_sqrt hs
  simpa [Real.sqrt_sq_eq_abs, coordinateGradientNorm] using hr

/-- A positive native clip bound gives a strictly positive scaling,
even for a zero raw gradient. Source: clip_grad_norm_ at 88aa892. -/
theorem coordinate_clip_factor_pos {dimension : ℕ} (bound : ℝ) (gradient : Fin dimension → ℝ)
    (hb : 0 < bound) : 0 < coordinateClipFactor bound gradient := by
  have hn := Real.sqrt_nonneg (∑ i, gradient i ^ 2)
  unfold coordinateClipFactor coordinateGradientNorm
  exact lt_min (by norm_num) (div_pos hb (by linarith))

example : (0 : ℝ) < 1 := by norm_num

/-- Clipping never amplifies a coordinate's multiplier. Source:
clip_grad_norm_ at 88aa892, clamp to maximum one. -/
theorem coordinate_clip_factor_le_one {dimension : ℕ} (bound : ℝ) (gradient : Fin dimension → ℝ) :
    coordinateClipFactor bound gradient ≤ 1 := min_le_left _ _

/-- The applied multiplier times the original norm is bounded by
max_norm. Source: clip_grad_norm_ at 88aa892, including the positive
additive constant; this is not a floating-point certificate. -/
theorem coordinate_clip_scaled_norm_bound {dimension : ℕ} (bound : ℝ) (gradient : Fin dimension → ℝ)
    (hb : 0 < bound) : coordinateClipFactor bound gradient * coordinateGradientNorm gradient ≤ bound := by
  have hc := coordinate_clip_factor_pos bound gradient hb
  have hn := Real.sqrt_nonneg (∑ i, gradient i ^ 2)
  have hd : 0 < coordinateGradientNorm gradient + 1 / 1000000 := by
    unfold coordinateGradientNorm
    linarith
  have hr := min_le_right (1 : ℝ) (bound / (coordinateGradientNorm gradient + 1 / 1000000))
  have hm := (le_div_iff₀ hd).mp hr
  change coordinateClipFactor bound gradient * coordinateGradientNorm gradient ≤ bound
  unfold coordinateClipFactor at hc ⊢
  nlinarith

example : (0 : ℝ) < 1 := by norm_num

/-- The clipped vector itself has norm at most the native bound.
Source: clip_grad_norm_ at 88aa892 and the proved norm scaling. -/
theorem coordinate_clipped_norm_bound {dimension : ℕ} (bound : ℝ) (gradient : Fin dimension → ℝ)
    (hb : 0 < bound) : coordinateGradientNorm (coordinateClippedGradient bound gradient) ≤ bound := by
  unfold coordinateClippedGradient
  rw [coordinate_norm_scaling, abs_of_nonneg (le_of_lt (coordinate_clip_factor_pos bound gradient hb))]
  exact coordinate_clip_scaled_norm_bound bound gradient hb

example : (0 : ℝ) < 1 := by norm_num

/-- Native clipping supplies the coordinate bound needed by the
causal moment estimates. Source: clip_grad_norm_ at 88aa892. -/
theorem coordinate_clipped_abs_bound {dimension : ℕ} (bound : ℝ) (gradient : Fin dimension → ℝ)
    (i : Fin dimension) (hb : 0 < bound) : |coordinateClippedGradient bound gradient i| ≤ bound := by
  exact le_trans (coordinate_abs_le_norm (coordinateClippedGradient bound gradient) i)
    (coordinate_clipped_norm_bound bound gradient hb)

example : (0 : ℝ) < 1 := by norm_num

/-- A nonpositive raw CE coordinate remains nonpositive under the
actual clipping coefficient. Source: clip_grad_norm_ at 88aa892. -/
theorem coordinate_clipped_nonpos {dimension : ℕ} (bound : ℝ) (gradient : Fin dimension → ℝ)
    (i : Fin dimension) (hb : 0 < bound) (hg : gradient i ≤ 0) :
    coordinateClippedGradient bound gradient i ≤ 0 := by
  exact mul_nonpos_of_nonneg_of_nonpos (le_of_lt (coordinate_clip_factor_pos bound gradient hb)) hg

example : (0 : ℝ) < 1 ∧ (fun _ : Fin 1 => (-1 : ℝ)) 0 ≤ 0 := by norm_num

/-- Strict negative coordinates are not erased by native clipping.
Source: clip_grad_norm_ at 88aa892, positive finite bound and 1e-6. -/
theorem coordinate_clipped_negative {dimension : ℕ} (bound : ℝ) (gradient : Fin dimension → ℝ)
    (i : Fin dimension) (hb : 0 < bound) (hg : gradient i < 0) :
    coordinateClippedGradient bound gradient i < 0 := by
  exact mul_neg_of_pos_of_neg (coordinate_clip_factor_pos bound gradient hb) hg

example : (0 : ℝ) < 1 ∧ (fun _ : Fin 1 => (-1 : ℝ)) 0 < 0 := by norm_num

/-- Correct the source doc's nominal threshold: no reduction requires
norm+1e-6<=bound, rather than norm<=bound. Source: PyTorch 2.14.1
clip_grad_norm_ at 88aa892, documented coefficient and implementation. -/
theorem coordinate_clip_exact_threshold {dimension : ℕ} (bound : ℝ) (gradient : Fin dimension → ℝ) :
    coordinateClipFactor bound gradient = 1 ↔ coordinateGradientNorm gradient + 1 / 1000000 ≤ bound := by
  have hn := Real.sqrt_nonneg (∑ i, gradient i ^ 2)
  have hd : 0 < coordinateGradientNorm gradient + 1 / 1000000 := by
    unfold coordinateGradientNorm
    linarith
  unfold coordinateClipFactor
  rw [min_eq_left_iff, le_div_iff₀ hd, one_mul]

/-- Counterexample to the source doc's "only" condition: exact unit
norm at a unit max_norm is already reduced by the 1e-6 coefficient.
Source: clip_grad_norm_ at 88aa892, boundary wording corrected above. -/
theorem unit_boundary_clipping_counterexample :
    coordinateGradientNorm (fun _ : Fin 1 => (1 : ℝ)) = 1 ∧
      coordinateClipFactor 1 (fun _ : Fin 1 => (1 : ℝ)) = 1000000 / 1000001 := by
  norm_num [coordinateGradientNorm, coordinateClipFactor]

end Transformer.Grokking.AdamW
