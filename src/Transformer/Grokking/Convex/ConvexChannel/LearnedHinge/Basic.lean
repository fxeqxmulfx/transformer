module

public import Transformer.Grokking.Convex.ConvexRecall.Loss
public import Mathlib.Analysis.Normed.Module.RCLike.Real
public import Mathlib.Analysis.Real.Sqrt

/-!
# Recover learned representation and readout from a joint ReLU coordinate

Source: `review/convex/moments.md`, CM75. This construction uses actual ReLU,
not the fixed random gates in arXiv:2211.11052v1, §3.4. A joint vector `u`
recovers readout amplitude `sqrt ‖u‖` and representation `u / sqrt ‖u‖`.
Their actual product through the positively homogeneous activation equals
ReLU applied to `u`, at every input, including outside a training set.

The two factors are coupled coordinates of one neuron. Convexity in two
independently optimized factors is not claimed. The balance uses the stated
ambient norm; it does not transfer an unspecified weight-decay penalty.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedHinge

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The actual rectified activation, CM75. -/
def hinge (t : ℝ) : ℝ := max t 0

/-- Learned positive readout amplitude in the balanced coordinates, CM75. -/
noncomputable def amplitude (u : E) : ℝ := Real.sqrt ‖u‖

/-- Learned representation vector, with the zero vector recovered as zero,
CM75. The input functional is not used to choose these weights. -/
noncomputable def representation (u : E) : E := (amplitude u)⁻¹ • u

/-- Actual representation, activation, and learned readout computation,
CM75. The following identity is a theorem, not its definition. -/
noncomputable def neuron (u : E) (input : E →ₗ[ℝ] ℝ) : ℝ :=
  amplitude u * hinge (input (representation u))

omit [NormedSpace ℝ E] in
/-- The recovered readout is always nonnegative, CM75. -/
theorem amplitude_nonneg (u : E) : 0 ≤ amplitude u := Real.sqrt_nonneg _

omit [NormedSpace ℝ E] in
/-- Every nonzero joint coordinate has a positive learned readout, CM75. -/
theorem amplitude_pos {u : E} (hu : u ≠ 0) : 0 < amplitude u :=
  Real.sqrt_pos.mpr (norm_pos_iff.mpr hu)

/-- Multiplying the recovered factors returns the whole original vector,
CM75. No coordinate is omitted at zero or at a small norm. -/
theorem recover (u : E) : amplitude u • representation u = u := by
  by_cases hu : u = 0
  · subst u
    simp [amplitude, representation]
  · rw [representation, smul_smul, mul_inv_cancel₀ (amplitude_pos hu).ne', one_smul]

/-- The recovered representation norm equals its learned readout amplitude,
CM75. This fixes a scaling gauge rather than freezing either layer. -/
theorem balanced (u : E) : ‖representation u‖ = amplitude u := by
  by_cases hu : u = 0
  · subst u
    simp [amplitude, representation]
  · have ha := amplitude_pos hu
    have hs : amplitude u ^ 2 = ‖u‖ := Real.sq_sqrt (norm_nonneg u)
    rw [representation, norm_smul_of_nonneg (inv_nonneg.mpr ha.le), ← hs]
    field_simp

/-- The actual two learned stages equal the joint-coordinate ReLU at every
input, CM75. This identity does not assume any activation pattern. -/
theorem neuron_eq (u : E) (input : E →ₗ[ℝ] ℝ) :
    neuron u input = hinge (input u) := by
  unfold neuron hinge
  rw [mul_max_of_nonneg _ _ (amplitude_nonneg u), mul_zero]
  congr 1
  calc
    amplitude u * input (representation u) = input (amplitude u • representation u) :=
      (input.map_smul _ _).symm
    _ = input u := by rw [recover]

/-- Every independently specified nonnegative readout and representation
has exactly the same predictions after recovery, CM75. Only their scaling
gauge can change; equality holds on every possible input. -/
theorem neuron_of_scaled (a : ℝ) (ha : 0 ≤ a) (w : E) (input : E →ₗ[ℝ] ℝ) :
    neuron (a • w) input = a * hinge (input w) := by
  rw [neuron_eq, map_smul, smul_eq_mul]
  unfold hinge
  rw [mul_max_of_nonneg _ _ ha, mul_zero]

example := amplitude_pos (u := ((4, 1) : ℝ × ℝ)) (by norm_num)
example := neuron_of_scaled 2 (by norm_num) ((4, 1) : ℝ × ℝ)
  (LinearMap.fst ℝ ℝ ℝ)

end Transformer.Grokking.Convex.ConvexChannel.LearnedHinge
