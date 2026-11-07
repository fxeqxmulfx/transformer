module

public import Transformer.Grokking.GradientEvidence.BinaryObjectives
public import Transformer.Grokking.Geometry.Basic
public import Mathlib.Analysis.Calculus.Deriv.MeanValue

/-!
# Actual cross-entropy descent can increase within-answer contrast variation

Source: `review/flat_spectrum.md`, *Loss-controlled geometry*, LG4.
Two same-label examples have logits `(theta,0)` and `(2*theta,0)`, with one
shared trainable scalar. Their actual mean binary softmax CE has negative
derivative. Every positive gradient step from a nonnegative scalar lowers
the loss and increases the centered contrast energy. This is a concrete
shared-parameter classifier, not a claim about a particular transformer step.
-/

@[expose] public section
noncomputable section

namespace Transformer.Grokking.GradientEvidence

/-- The two same-label examples' actual class contrasts, LG4. -/
def sharedContrast (θ : ℝ) (i : Bool) : ℝ := if i then 2 * θ else θ

/-- Actual mean cross-entropy of the two examples, LG4. -/
def sharedContrastLoss (θ : ℝ) : ℝ := (binaryCE true θ + binaryCE true (2 * θ)) / 2

/-- Ordinary scalar gradient descent on that objective, LG4. -/
def sharedContrastStep (η θ : ℝ) : ℝ := θ - η * deriv sharedContrastLoss θ

/-- The derivative is calculated from both CE terms, including the shared
parameter's factor two; no desired training direction is defined in, LG4. -/
theorem sharedContrastLoss_hasDerivAt (θ : ℝ) :
    HasDerivAt sharedContrastLoss
      ((Real.exp θ / (Real.exp θ + 1) - 1 +
        2 * (Real.exp (2 * θ) / (Real.exp (2 * θ) + 1) - 1)) / 2) θ := by
  have h := (binaryCE_deriv true θ).add
    ((binaryCE_deriv true (2 * θ)).comp θ ((hasDerivAt_id θ).const_mul 2))
  convert h.div_const 2 using 1
  · funext x
    rfl
  · dsimp
    ring

/-- Both examples push the shared scalar upward for every finite value, LG4. -/
theorem sharedContrastLoss_deriv_neg (θ : ℝ) : deriv sharedContrastLoss θ < 0 := by
  rw [(sharedContrastLoss_hasDerivAt θ).deriv]
  have h (x : ℝ) : Real.exp x / (Real.exp x + 1) < 1 := by
    apply (div_lt_one (by positivity : 0 < Real.exp x + 1)).mpr
    linarith
  linarith [h θ, h (2 * θ)]

/-- Increasing the scalar strictly lowers the actual CE, LG4. -/
theorem sharedContrastLoss_strictAnti : StrictAnti sharedContrastLoss :=
  strictAnti_of_deriv_neg sharedContrastLoss_deriv_neg

/-- The computed training mean leaves centered energy theta squared over two,
LG4. The two contrasts are not identical when theta is nonzero. -/
theorem sharedContrast_scatter (θ : ℝ) :
    Geometry.residualOver Finset.univ (sharedContrast θ) = θ ^ 2 / 2 := by
  simp [Geometry.residualOver, Geometry.energyOver, Geometry.meanOver, sharedContrast]
  ring

/-- **Loss descent does not imply raw contrast contraction**, LG4. This is
an actual gradient step, with two nonempty same-label examples. -/
theorem descent_increases_contrast_scatter {η θ : ℝ} (hη : 0 < η) (hθ : 0 ≤ θ) :
    sharedContrastLoss (sharedContrastStep η θ) < sharedContrastLoss θ ∧
      Geometry.residualOver Finset.univ (sharedContrast θ) <
        Geometry.residualOver Finset.univ (sharedContrast (sharedContrastStep η θ)) := by
  have hg := sharedContrastLoss_deriv_neg θ
  have hstep : θ < sharedContrastStep η θ := by
    unfold sharedContrastStep
    have hp := mul_neg_of_pos_of_neg hη hg
    linarith
  refine ⟨sharedContrastLoss_strictAnti hstep, ?_⟩
  rw [sharedContrast_scatter, sharedContrast_scatter]
  nlinarith [sq_nonneg (sharedContrastStep η θ - θ)]

example := sharedContrastLoss_hasDerivAt 1
example := sharedContrastLoss_deriv_neg 1
example := sharedContrast_scatter 1
example := descent_increases_contrast_scatter (η := 1 / 10) (θ := 1)
  (by norm_num) (by norm_num)

end Transformer.Grokking.GradientEvidence
