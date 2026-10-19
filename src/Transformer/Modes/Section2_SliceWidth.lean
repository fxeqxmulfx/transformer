module

public import Transformer.Modes.Section2_JointDensity

/-!
# The actual disk-slice width near level zero

Source: arXiv:2412.09080v3, §2.1, the joint-density
continuity condition of `thm:kac-rice`. For field values
`|x|<1`, every point of the actual unit disk admits `x` in
its stretched scale interval. Thus the exact density kernel
depends only on the derivative coordinate and latent slice.

At derivative coordinate `y`, the real disk slice is the
interval with endpoints `±sqrt(1-y²)`. The square root is
zero when the slice is empty and varies continuously for
all real `y`. Its values lie between zero and one.

A continuous positive extension of the scale reciprocal
equals `1/(2*(2+u))` throughout the relevant latent interval.
The extension is used only to apply ordinary interval
integration results without a singularity outside the disk.
The exact kernel identity keeps both the original field
observation and its actual support, and gives a normalized
interval formula for the actual finite joint density.
-/

@[expose] public section

open Real Set Filter MeasureTheory
open scoped ENNReal

namespace Transformer.Modes

/-- The actual half-width of a real unit-disk slice, with empty slices represented by zero.
Source: arXiv:2412.09080v3, §2.1, the actual disk support in the joint-density construction. -/
noncomputable def riceCounterSliceWidth (y : ℝ) : ℝ := Real.sqrt (1 - y ^ 2)

/-- Every actual slice width is nonnegative.
Source: arXiv:2412.09080v3, §2.1, the ordered integration endpoints of the actual joint density. -/
theorem riceCounterSliceWidth_nonneg (y : ℝ) : 0 ≤ riceCounterSliceWidth y := Real.sqrt_nonneg _

/-- Every actual disk slice has half-width at most one.
Source: arXiv:2412.09080v3, §2.1, the bounded slice domain of the actual joint density. -/
theorem riceCounterSliceWidth_le_one (y : ℝ) : riceCounterSliceWidth y ≤ 1 :=
  Real.sqrt_le_one.mpr (by nlinarith [sq_nonneg y])

/-- The actual slice half-width is continuous at every derivative coordinate, including the boundary and empty slices.
Source: arXiv:2412.09080v3, §2.1, the global derivative-coordinate continuity required by hypothesis three. -/
theorem continuous_riceCounterSliceWidth : Continuous riceCounterSliceWidth := by
  exact Real.continuous_sqrt.comp (continuous_const.sub (continuous_id.pow 2))

/-- The actual slice has positive width exactly for derivative coordinates of magnitude below one.
Source: arXiv:2412.09080v3, §2.1, the nonempty actual density support in the derivative coordinate. -/
theorem riceCounterSliceWidth_pos_iff (y : ℝ) : 0 < riceCounterSliceWidth y ↔ |y| < 1 := by
  rw [riceCounterSliceWidth, Real.sqrt_pos, ← sq_lt_one_iff_abs_lt_one]
  constructor <;> intro h <;> linarith

/-- The original real disk condition is exactly membership in the actual slice interval, for every derivative coordinate.
Source: arXiv:2412.09080v3, §2.1, the exact integration domain in the actual joint-density formula. -/
theorem riceCounterDiskDomain_slice_iff (u y : ℝ) :
    (u, y) ∈ riceCounterDiskDomain ↔ u ∈ Ioo (-riceCounterSliceWidth y) (riceCounterSliceWidth y) := by
  change u ^ 2 + y ^ 2 < 1 ↔ -riceCounterSliceWidth y < u ∧ u < riceCounterSliceWidth y
  rw [← abs_lt]
  by_cases hy : 0 ≤ 1 - y ^ 2
  · have hw : (riceCounterSliceWidth y) ^ 2 = 1 - y ^ 2 := Real.sq_sqrt hy
    rw [← abs_of_nonneg (riceCounterSliceWidth_nonneg y), ← sq_lt_sq, hw]
    constructor <;> intro h <;> linarith
  · have hw : riceCounterSliceWidth y = 0 := Real.sqrt_eq_zero_of_nonpos (le_of_not_ge hy)
    rw [hw]
    constructor
    · intro h; nlinarith [sq_nonneg u]
    · intro h; exact (not_lt_of_ge (abs_nonneg u) h).elim

/-- A continuous positive extension of the actual scale reciprocal, equal to it on the full latent disk interval.
Source: arXiv:2412.09080v3, §2.1, the actual reciprocal factor in the joint-density slice integral. -/
noncomputable def riceCounterSliceReciprocal (u : ℝ) : ℝ :=
  1 / (2 * (2 + max (-1) u))

/-- The extended reciprocal is strictly positive everywhere, because its denominator is bounded away from zero.
Source: arXiv:2412.09080v3, §2.1, positivity of the actual density on every nonempty disk slice. -/
theorem riceCounterSliceReciprocal_pos (u : ℝ) : 0 < riceCounterSliceReciprocal u := by
  unfold riceCounterSliceReciprocal
  have h := le_max_left (-1 : ℝ) u
  exact one_div_pos.mpr (by nlinarith)

/-- The extended reciprocal is continuous on the entire real line.
Source: arXiv:2412.09080v3, §2.1, the continuous integrand used for actual joint-density continuity. -/
theorem continuous_riceCounterSliceReciprocal : Continuous riceCounterSliceReciprocal := by
  unfold riceCounterSliceReciprocal
  apply continuous_const.div (by fun_prop)
  intro u
  have h := le_max_left (-1 : ℝ) u
  nlinarith

/-- The extension agrees with the original scale reciprocal on the original latent disk interval.
Source: arXiv:2412.09080v3, §2.1, preservation of the actual reciprocal factor in the joint density. -/
theorem riceCounterSliceReciprocal_eq (u : ℝ) (hu : -1 ≤ u) :
    riceCounterSliceReciprocal u = 1 / (2 * (2 + u)) := by
  rw [riceCounterSliceReciprocal, max_eq_right hu]

/-- Near field level zero, the actual kernel is exactly the positive reciprocal on the actual disk-slice interval.
Source: arXiv:2412.09080v3, §2.1, the exact local joint-density formula for hypothesis three of `thm:kac-rice`. -/
theorem riceCounterSliceKernel_near (u x y : ℝ) (hx : |x| < 1) :
    riceCounterSliceKernel u x y = (Ioo (-riceCounterSliceWidth y) (riceCounterSliceWidth y)).indicator
      (fun u => ENNReal.ofReal (riceCounterSliceReciprocal u)) u := by
  by_cases hD : (u, y) ∈ riceCounterDiskDomain
  · have hxI := riceCounterDiskDomain_value_interval (u, y) hD x hx
    have huI := (riceCounterDiskDomain_slice_iff u y).mp hD
    have hu : -1 ≤ u := (abs_lt.mp (riceCounterDiskDomain_bounds (u, y) hD).1).1.le
    rw [riceCounterSliceKernel, ite_eq_left ⟨hD, hxI⟩, indicator_of_mem huI,
      riceCounterSliceReciprocal_eq u hu]
  · have huI : u ∉ Ioo (-riceCounterSliceWidth y) (riceCounterSliceWidth y) :=
      fun h => hD ((riceCounterDiskDomain_slice_iff u y).mpr h)
    simp only [riceCounterSliceKernel, hD, false_and, ite_false, indicator_of_notMem huI]

/-- Near field level zero the actual finite joint density is the normalized reciprocal integral over the actual disk slice.
Source: arXiv:2412.09080v3, §2.1, the exact actual density formula used to verify hypothesis three's continuity. -/
theorem riceCounterJointDensityENN_near (x y : ℝ) (hx : |x| < 1) :
    riceCounterJointDensityENN (x, y) = ENNReal.ofReal Real.pi⁻¹ *
      ∫⁻ u in Ioo (-riceCounterSliceWidth y) (riceCounterSliceWidth y),
        ENNReal.ofReal (riceCounterSliceReciprocal u) := by
  unfold riceCounterJointDensityENN
  simp_rw [riceCounterSliceKernel_near _ x y hx]
  rw [lintegral_indicator measurableSet_Ioo]

/-- The actual joint density vanishes outside the original bounded observation support.
Source: arXiv:2412.09080v3, §2.1, the finite support of the actual joint-density construction. -/
theorem riceCounterJointDensityENN_zero_of_outside (q : ℝ × ℝ) (hq : ¬(|q.1| < 3 ∧ |q.2| < 1)) :
    riceCounterJointDensityENN q = 0 := by
  have hK : (fun u => riceCounterSliceKernel u q.1 q.2) = fun _ => 0 := by
    funext u
    by_contra h
    exact hq (riceCounterSliceKernel_ne_zero_bounds u q.1 q.2 h)
  rw [riceCounterJointDensityENN, hK]
  simp

/-- The real actual density likewise vanishes outside the actual bounded observation support.
Source: arXiv:2412.09080v3, §2.1, the bounded support used to construct the actual scalar marginal density. -/
theorem riceCounterJointDensity_zero_of_outside (q : ℝ × ℝ) (hq : ¬(|q.1| < 3 ∧ |q.2| < 1)) :
    riceCounterJointDensity q = 0 := by
  rw [riceCounterJointDensity, riceCounterJointDensityENN_zero_of_outside q hq, ENNReal.toReal_zero]

/-- A genuine interior latent point and nonzero observation jointly realize the kernel identity and unchanged actual reciprocal.
Source: arXiv:2412.09080v3, §2.1, simultaneous witnesses for the actual near-zero density hypotheses. -/
example : |(1 / 2 : ℝ)| < 1 ∧ (-1 : ℝ) ≤ 1 / 2 ∧
    riceCounterSliceReciprocal (1 / 2) = 1 / 5 ∧
    riceCounterSliceKernel (1 / 2) (1 / 2) (1 / 2) =
      (Ioo (-riceCounterSliceWidth (1 / 2)) (riceCounterSliceWidth (1 / 2))).indicator
        (fun u => ENNReal.ofReal (riceCounterSliceReciprocal u)) (1 / 2) := by
  refine ⟨by norm_num, by norm_num, ?_, riceCounterSliceKernel_near _ _ _ (by norm_num)⟩
  norm_num [riceCounterSliceReciprocal]

/-- A genuine observation outside the value support realizes both support-vanishing hypotheses and conclusions.
Source: arXiv:2412.09080v3, §2.1, a nonempty-support-complement example for the actual density. -/
example : ¬(|(4 : ℝ)| < 3 ∧ |(1 / 2 : ℝ)| < 1) ∧
    riceCounterJointDensityENN (4, 1 / 2) = 0 ∧ riceCounterJointDensity (4, 1 / 2) = 0 := by
  exact ⟨by norm_num, riceCounterJointDensityENN_zero_of_outside _ (by norm_num),
    riceCounterJointDensity_zero_of_outside _ (by norm_num)⟩

end Transformer.Modes
