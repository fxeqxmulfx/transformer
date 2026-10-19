module

public import Transformer.Causal.Basic
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Arctan

/-!
# A continuous periodic cusp force for the sequential-flow regularity audit

ArXiv:2411.04990v2, Appendix A, `lemma:convergence`, assumes
only C¹ energies. Their angular derivatives are continuous;
the statement does not impose C¹ regularity on those derivatives.
The proof subsequently invokes an unstable-manifold argument.

The scalar force below is a continuous periodic function with a
square-root cusp at zero. Continuity supplies the input for a genuine C¹
primitive in the subsequent energy construction. The half-angle formula is defined at every
angle, including both zeros of sine. The positive and negative
square roots are combined using continuous positive parts.

Its exact arctangent-coordinate formula is
`g(2 arctan z) = 2 sqrt(z)/(1+z²)` for nonnegative z.
Thus the actual scalar cusp equation can be expressed without a
singular tangent at the opposite point of the circle. The fourth-
power identity gives a finite trigonometric equation containing all
zeros of `sin y - a g(y)`, needed to check the source's isolated-
critical-point hypothesis for the actual periodic energies.
These algebraic and regularity identities are established before
any conclusion about trajectories or the printed lemma is used.
-/

@[expose] public section

open Real
namespace Transformer.Causal

/-- A continuous odd square root, defined through the two positive parts.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, C¹ regularity audit. -/
noncomputable def cuspSignedRoot (x : ℝ) : ℝ := Real.sqrt (max x 0) - Real.sqrt (max (-x) 0)

/-- Both positive-part square roots are continuous, including at zero.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, regularity audit. -/
theorem continuous_cuspSignedRoot : Continuous cuspSignedRoot := by
  unfold cuspSignedRoot
  fun_prop

/-- On the positive half-line this is the actual nonnegative square root.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, cusp construction. -/
theorem cuspSignedRoot_of_nonneg {x : ℝ} (hx : 0 ≤ x) : cuspSignedRoot x = Real.sqrt x := by
  simp [cuspSignedRoot, max_eq_left hx, max_eq_right (neg_nonpos.mpr hx)]

/-- On the negative half-line the negative square root is retained.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, cusp construction. -/
theorem cuspSignedRoot_of_nonpos {x : ℝ} (hx : x ≤ 0) : cuspSignedRoot x = -Real.sqrt (-x) := by
  simp [cuspSignedRoot, max_eq_right hx, max_eq_left (neg_nonneg.mpr hx)]

/-- The signed square root is odd on the whole real line.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, cusp construction. -/
theorem cuspSignedRoot_neg (x : ℝ) : cuspSignedRoot (-x) = -cuspSignedRoot x := by
  simp only [cuspSignedRoot, neg_neg]
  ring

/-- The continuous cusp takes its actual value zero at the origin.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, cusp construction. -/
@[simp] theorem cuspSignedRoot_zero : cuspSignedRoot 0 = 0 := by simp [cuspSignedRoot]

/-- Squaring the signed root gives the original absolute value.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, cusp construction. -/
theorem cuspSignedRoot_sq (x : ℝ) : cuspSignedRoot x ^ 2 = |x| := by
  rcases le_total 0 x with hx | hx
  · rw [cuspSignedRoot_of_nonneg hx, Real.sq_sqrt hx, abs_of_nonneg hx]
  · rw [cuspSignedRoot_of_nonpos hx, neg_sq, Real.sq_sqrt (neg_nonneg.mpr hx), abs_of_nonpos hx]

/-- The continuous angular cusp force, including both half-angle zeros.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, regularity audit. -/
noncomputable def cuspForce (y : ℝ) : ℝ :=
  2 * cuspSignedRoot (Real.sin (y / 2)) *
    (Real.cos (y / 2) * Real.sqrt |Real.cos (y / 2)|)

/-- The actual force is continuous at every original circle angle.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, C¹ energy requirement. -/
theorem continuous_cuspForce : Continuous cuspForce := by
  unfold cuspForce
  have h := continuous_cuspSignedRoot
  fun_prop

/-- Oddness ensures that the primitive can be chosen periodic.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, angular setting. -/
theorem cuspForce_neg (y : ℝ) : cuspForce (-y) = -cuspForce y := by
  simp only [cuspForce, neg_div, Real.sin_neg, Real.cos_neg, cuspSignedRoot_neg]
  ring

/-- The actual force has exactly the original angular period.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, particles on the circle. -/
theorem cuspForce_periodic : Function.Periodic cuspForce (2 * π) := by
  intro y
  have h : (y + 2 * π) / 2 = y / 2 + π := by ring
  simp only [cuspForce, h, Real.sin_add_pi, Real.cos_add_pi, cuspSignedRoot_neg, abs_neg]
  ring

/-- The cusp force vanishes at the original angular origin.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, cusp construction. -/
@[simp] theorem cuspForce_zero : cuspForce 0 = 0 := by simp [cuspForce]

/-- The opposite angle also retains its zero value.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, cusp construction. -/
@[simp] theorem cuspForce_pi : cuspForce π = 0 := by simp [cuspForce, Real.cos_pi_div_two]

/-- The actual squared force retains all half-angle factors.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, critical-point audit. -/
theorem cuspForce_sq (y : ℝ) : cuspForce y ^ 2 =
    4 * |Real.sin (y / 2)| * Real.cos (y / 2) ^ 2 * |Real.cos (y / 2)| := by
  simp only [cuspForce, mul_pow, cuspSignedRoot_sq, Real.sq_sqrt (abs_nonneg _)]
  ring

/-- The fourth-power identity removes both square roots exactly.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, critical-point audit. -/
theorem cuspForce_fourth_identity (y : ℝ) :
    cuspForce y ^ 4 * (1 - Real.cos y) = Real.sin y ^ 4 * (1 + Real.cos y) := by
  have hs : Real.sin y = 2 * Real.sin (y / 2) * Real.cos (y / 2) := by
    simpa only [two_mul, add_halves] using Real.sin_two_mul (y / 2)
  have hc : Real.cos y = 2 * Real.cos (y / 2) ^ 2 - 1 := by
    simpa only [two_mul, add_halves] using Real.cos_two_mul (y / 2)
  have hsq := cuspForce_sq y
  rw [show cuspForce y ^ 4 = (cuspForce y ^ 2) ^ 2 by ring, hsq, hs, hc]
  simp only [mul_pow, sq_abs]
  linear_combination (-32 * Real.sin (y / 2) ^ 2 * Real.cos (y / 2) ^ 6) *
    Real.sin_sq_add_cos_sq (y / 2)

/-- The force has this exact nonsingular formula in arctangent coordinates.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, trajectory construction. -/
theorem cuspForce_double_arctan {z : ℝ} (hz : 0 ≤ z) :
    cuspForce (2 * Real.arctan z) = 2 * Real.sqrt z / (1 + z ^ 2) := by
  have hr : 0 < Real.sqrt (1 + z ^ 2) := Real.sqrt_pos.mpr (by positivity)
  have hS : 0 ≤ Real.sin (Real.arctan z) := by rw [Real.sin_arctan]; positivity
  have hC := Real.cos_arctan_pos z
  simp only [cuspForce, mul_div_cancel_left₀ _ (by norm_num : (2 : ℝ) ≠ 0),
    cuspSignedRoot_of_nonneg hS, abs_of_pos hC]
  rw [Real.sin_arctan, Real.cos_arctan]
  have hprod : Real.sqrt (z / Real.sqrt (1 + z ^ 2)) *
      Real.sqrt (1 / Real.sqrt (1 + z ^ 2)) = Real.sqrt z / Real.sqrt (1 + z ^ 2) := by
    rw [← Real.sqrt_mul (by positivity), div_mul_div_comm, mul_one, ← pow_two,
      Real.sq_sqrt (by positivity), Real.sqrt_div hz]
  calc
    _ = 2 * (Real.sqrt (z / Real.sqrt (1 + z ^ 2)) * Real.sqrt (1 / Real.sqrt (1 + z ^ 2))) /
        Real.sqrt (1 + z ^ 2) := by ring
    _ = _ := by rw [hprod]; field_simp; rw [Real.sq_sqrt (by positivity)]

/-- A true zero of the angular velocity gives an explicit trigonometric zero.
Source: arXiv:2411.04990v2, Appendix A, `lemma:convergence`, isolated critical points. -/
theorem cuspForce_critical_relation (a y : ℝ) (h : Real.sin y = a * cuspForce y) :
    Real.sin y ^ 4 * ((1 - Real.cos y) - a ^ 4 * (1 + Real.cos y)) = 0 := by
  have hf := cuspForce_fourth_identity y
  rw [h, mul_pow] at hf
  rw [h, mul_pow]
  linear_combination (a ^ 4) * hf

/-- Positive and negative inputs and an actual angular critical point
jointly realize all sign and zero-relation hypotheses. The nonzero
arctangent value also checks the actual force coefficient. -/
example : (0 : ℝ) ≤ 1 ∧ (-1 : ℝ) ≤ 0 ∧ cuspForce (2 * Real.arctan 1) = 1 ∧
    Real.sin 0 = 1 * cuspForce 0 := by
  refine ⟨zero_le_one, by norm_num, ?_, by simp⟩
  rw [cuspForce_double_arctan zero_le_one]
  norm_num

end Transformer.Causal
