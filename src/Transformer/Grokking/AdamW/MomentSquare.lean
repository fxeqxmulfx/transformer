import Transformer.Grokking.AdamW.ScalarDenominator

/-!
# Coupled native first/second-moment square estimates

Source: PyTorch 2.14.1 non-AMSGrad AdamW, adam.py lines 445--475,
ported at 88aa892/0033b1b. Both moments consume the same actual
input, unsquared for the first and squared for the second. Derive
the algebraic coupling rather than independently bounding each buffer.

When beta1 squared is below beta2, an explicit positive numerical
scale controls the squared first moment by retained variance. Its
one-step preservation comes from a nonnegative square remainder.
The original beta1=0.9/beta2=0.98 scale is exactly 49/17.

A positive current square budget also propagates through one actual
insertion with a smaller clock-dependent budget. This prepares the
finite-history proof needed to retain bias correction and remove
the earlier clip_bound/epsilon direction ceiling. Current estimates
do not yet assert that initialized corrected histories meet that bound.

No gradient sign, magnitude, convergence or learned circuit is a
premise. The current moment/variance relation is explicit; an
initialized-path application must derive it from the native history.
The parameter, rate, decay and epsilon do not enter buffer insertion.
Neither buffer is reset and this is not AMSGrad or an instantaneous
variance substitution. Exact-real recurrence algebra alone does not
prove task generalization, stochastic optimization convergence or
floating-point kernel agreement. Frozen checkpoints remain unchanged.
-/

namespace Transformer.Grokking.AdamW

/-- Numerical native moment-square scale. Source: coupled insertions
at 88aa892/0033b1b; every argument enters the rational coefficient. -/
noncomputable def nativeMomentSquareScale (b1 b2 : ℝ) : ℝ :=
  b2 * (1 - b1) ^ 2 / ((1 - b2) * (b2 - b1 ^ 2))

/-- The legal separated memory regime gives a positive square scale.
Source: native first/second insertions at 88aa892; the strict
beta1-squared/beta2 separation is a numerical condition, not stability. -/
theorem native_moment_square_scale_pos (b1 b2 : ℝ)
    (h1 : b1 < 1) (h2 : b2 < 1) (hgap : b1 ^ 2 < b2) :
    0 < nativeMomentSquareScale b1 b2 := by
  have hb2 : 0 < b2 := by nlinarith only [hgap, sq_nonneg b1]
  have hs : 0 < (1 - b1) ^ 2 := by
    have hp : 0 < 1 - b1 := by linarith only [h1]
    simpa only [pow_two] using mul_pos hp hp
  unfold nativeMomentSquareScale
  exact div_pos (mul_pos hb2 hs) (mul_pos (by linarith only [h2]) (by linarith only [hgap]))

example : (9 / 10 : ℝ) < 1 ∧ (49 / 50 : ℝ) < 1 ∧ (9 / 10 : ℝ) ^ 2 < 49 / 50 := by norm_num

/-- The original native betas give the exact rational square scale.
Source: same-gradient native buffer insertions at 88aa892/0033b1b;
no gradient stream or fitted optimizer constant enters this value. -/
theorem native_original_moment_square_scale :
    nativeMomentSquareScale (9 / 10) (49 / 50) = (49 / 17 : ℝ) := by
  norm_num [nativeMomentSquareScale]

/-- The numerical squared first-moment bound survives an actual
same-input insertion. Source: native AdamW at 88aa892; the explicit
nonnegative-square remainder keeps both old buffers and the new input. -/
theorem native_moment_square_insert (b1 b2 moment variance gradient : ℝ)
    (hb2 : 0 ≤ b2) (h2 : b2 < 1) (hgap : b1 ^ 2 < b2)
    (hbound : moment ^ 2 ≤ nativeMomentSquareScale b1 b2 * variance) :
    (b1 * moment + (1 - b1) * gradient) ^ 2 ≤
      nativeMomentSquareScale b1 b2 * (b2 * variance + (1 - b2) * gradient ^ 2) := by
  let gap := b2 - b1 ^ 2
  have hg : 0 < b2 - b1 ^ 2 := by linarith only [hgap]
  have he : 1 - b2 ≠ 0 := by linarith only [h2]
  have hidentity : nativeMomentSquareScale b1 b2 * (b2 * variance + (1 - b2) * gradient ^ 2) -
      (b1 * moment + (1 - b1) * gradient) ^ 2 =
        b2 * (nativeMomentSquareScale b1 b2 * variance - moment ^ 2) +
          (gap * moment - b1 * (1 - b1) * gradient) ^ 2 / gap := by
    dsimp only [nativeMomentSquareScale, gap]
    field_simp [he, ne_of_gt hg]
    ring
  have hold := mul_nonneg hb2 (sub_nonneg.mpr hbound)
  have hsquare := div_nonneg (sq_nonneg (gap * moment - b1 * (1 - b1) * gradient)) (le_of_lt hg)
  linarith only [hidentity, hold, hsquare]

example : (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧ (9 / 10 : ℝ) ^ 2 < 49 / 50 ∧
    (-1 : ℝ) ^ 2 ≤ nativeMomentSquareScale (9 / 10) (49 / 50) * 1 := by
  norm_num [nativeMomentSquareScale]

/-- A positive present square budget propagates through native
insertions with its exact next numerical budget. Source: same-input
moment recurrences at 88aa892; initialized finite-history applications
must generate this present relation rather than prescribe a future one. -/
theorem native_moment_square_budget_insert (b1 b2 budget moment variance gradient : ℝ)
    (hbudget : 0 < budget) (hb2 : 0 < b2) (h2 : b2 < 1)
    (hbound : moment ^ 2 ≤ budget * variance) :
    (b1 * moment + (1 - b1) * gradient) ^ 2 ≤
      (b1 ^ 2 * budget / b2 + (1 - b1) ^ 2 / (1 - b2)) *
        (b2 * variance + (1 - b2) * gradient ^ 2) := by
  have he : 1 - b2 ≠ 0 := by linarith only [h2]
  have hidentity : (b1 ^ 2 * budget / b2 + (1 - b1) ^ 2 / (1 - b2)) *
      (b2 * variance + (1 - b2) * gradient ^ 2) - (b1 * moment + (1 - b1) * gradient) ^ 2 =
        (b1 ^ 2 + (1 - b1) ^ 2 * b2 / (budget * (1 - b2))) * (budget * variance - moment ^ 2) +
          (b1 * budget * (1 - b2) * gradient - (1 - b1) * b2 * moment) ^ 2 / (budget * b2 * (1 - b2)) := by
    field_simp [ne_of_gt hbudget, ne_of_gt hb2, he]
    ring
  have hc : 0 ≤ b1 ^ 2 + (1 - b1) ^ 2 * b2 / (budget * (1 - b2)) := by positivity
  have hold := mul_nonneg hc (sub_nonneg.mpr hbound)
  have hsquare : 0 ≤
      (b1 * budget * (1 - b2) * gradient - (1 - b1) * b2 * moment) ^ 2 / (budget * b2 * (1 - b2)) := div_nonneg
    (sq_nonneg (b1 * budget * (1 - b2) * gradient - (1 - b1) * b2 * moment)) (by positivity)
  linarith only [hidentity, hold, hsquare]

example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧
    (-1 : ℝ) ^ 2 ≤ 1 * 1 := by norm_num

/-- The whole actual scalar native path preserves a numerical
initial square guard. Source: complete retained recurrence at 88aa892;
all later inputs may vary arbitrarily and no future buffer bound or
parameter/input convergence is a hypothesis. -/
theorem scalar_native_moment_square_path (b1 b2 eps decay rate : ℝ)
    (state : ℕ → ScalarState) (gradient : ℕ → ℝ)
    (hstep : ∀ n, state (n + 1) = scalarNativeStep b1 b2 eps decay rate (state n) (gradient n))
    (hb2 : 0 ≤ b2) (h2 : b2 < 1) (hgap : b1 ^ 2 < b2)
    (hi : (state 0).moment ^ 2 ≤ nativeMomentSquareScale b1 b2 * (state 0).variance) :
    ∀ n, (state n).moment ^ 2 ≤ nativeMomentSquareScale b1 b2 * (state n).variance := by
  intro n
  induction n with
  | zero => exact hi
  | succ n ih =>
    rw [hstep n]
    exact native_moment_square_insert b1 b2 (state n).moment (state n).variance (gradient n) hb2 h2 hgap ih

example :
    (∀ n : ℕ, zeroScalarStateAt (n + 1) = scalarNativeStep (9 / 10) (49 / 50) (1 / 100000000)
      (1 / 10) (1 / 1000) (zeroScalarStateAt n) 0) ∧
    (0 : ℝ) ≤ 49 / 50 ∧ (49 / 50 : ℝ) < 1 ∧ (9 / 10 : ℝ) ^ 2 < 49 / 50 ∧
    (zeroScalarStateAt 0).moment ^ 2 ≤ nativeMomentSquareScale (9 / 10) (49 / 50) * (zeroScalarStateAt 0).variance := by
  refine ⟨fun n => (scalar_native_zero_step _ _ _ _ _ n).symm, by norm_num, by norm_num, by norm_num, ?_⟩
  norm_num [zeroScalarStateAt, nativeMomentSquareScale]

/-- An actual original-beta first insertion satisfies the coupled
square guard, but zeroing only its second buffer violates it.
Source: native insertion/reset semantics at 88aa892/0033b1b; this
is a numerical counterfactual, not a change to any frozen optimizer. -/
theorem native_original_second_reset_breaks_square_guard :
    let state := scalarNativeStep (9 / 10) (49 / 50) (1 / 100000000) (1 / 10) (1 / 1000)
      (seededScalarState 1) (-1)
    state.moment ^ 2 ≤ nativeMomentSquareScale (9 / 10) (49 / 50) * state.variance ∧
      ¬state.moment ^ 2 ≤ nativeMomentSquareScale (9 / 10) (49 / 50) * ({ state with variance := 0 } : ScalarState).variance := by
  dsimp only
  change ((9 / 10 : ℝ) * 0 + (1 - 9 / 10) * (-1)) ^ 2 ≤
      nativeMomentSquareScale (9 / 10) (49 / 50) * ((49 / 50) * 0 + (1 - 49 / 50) * (-1) ^ 2) ∧
    ¬((9 / 10 : ℝ) * 0 + (1 - 9 / 10) * (-1)) ^ 2 ≤ nativeMomentSquareScale (9 / 10) (49 / 50) * 0
  norm_num [nativeMomentSquareScale]

end Transformer.Grokking.AdamW
