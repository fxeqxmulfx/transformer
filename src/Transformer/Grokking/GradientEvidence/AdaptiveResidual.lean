module

public import Transformer.Grokking.GradientEvidence.ResidualDynamics
public import Transformer.Grokking.GradientEvidence.ContrastExpansion
public import Transformer.Grokking.AdamW.FirstStep

/-!
# An actual Adam first step can create a training-nullspace residual

Source: `review/flat_spectrum.md`, *Residual learning dynamics*, RD5.
One fixed feature is `(1,2)`, with binary logits `(u+2v,0)` and the first
class as target. The actual CE gradient at zero is `(-1/2,-1)`. Ordinary
gradient descent stays in the feature span. The existing bias-corrected Adam
first-update formula, with positive epsilon one, leaves that span while
lowering CE. This is a counterexample, not a replay of a saved model update.
-/

@[expose] public section
noncomputable section

namespace Transformer.Grokking.GradientEvidence

/-- Actual binary readout CE for feature `(1,2)`, RD5. -/
def axisReadoutLoss (u v : ℝ) : ℝ := binaryCE true (u + 2 * v)

/-- Its actual two partial derivatives, RD5. -/
def axisReadoutGradient (u v : ℝ) : ℝ × ℝ :=
  (deriv (fun s => axisReadoutLoss s v) u, deriv (fun s => axisReadoutLoss u s) v)

/-- Differentiating the loss fixes both coordinates and their feature scale,
RD5. Neither an update direction nor a desired residual is assumed. -/
theorem axisReadoutGradient_initial : axisReadoutGradient 0 0 = (-1 / 2, -1) := by
  have hd : HasDerivAt (binaryCE true) (-1 / 2) (2 * id (0 : ℝ)) := by
    convert binaryCE_deriv true 0 using 1 <;> norm_num
  have h := hd.comp (0 : ℝ) ((hasDerivAt_id (0 : ℝ)).const_mul 2)
  have h2 : HasDerivAt (fun s => binaryCE true (2 * s)) (-1) 0 := by
    convert h using 1 <;> norm_num [Function.comp_def]
  unfold axisReadoutGradient axisReadoutLoss
  simp only [mul_zero, add_zero, zero_add]
  rw [(binaryCE_deriv true 0).deriv, h2.deriv]
  norm_num

/-- Squared orthogonal residual to the span of `(1,2)`, RD5. -/
def axisResidualEnergy (u v : ℝ) : ℝ := (2 * u - v) ^ 2 / 5

/-- The formula is the squared norm of the computed perpendicular projection,
RD5; it is not an arbitrary failure metric. -/
theorem axisResidualEnergy_eq (u v : ℝ) :
    axisResidualEnergy u v = ((4 * u - 2 * v) / 5) ^ 2 + ((v - 2 * u) / 5) ^ 2 := by
  unfold axisResidualEnergy
  ring

/-- Existing native Adam first-step equations applied to the actual CE
gradient, RD5. Betas are 0.9/0.999, epsilon is one and decay is zero. -/
def axisAdamStep (η : ℝ) : ℝ × ℝ :=
  (AdamW.firstUpdate (9 / 10) (999 / 1000) 1 0 η 0 (axisReadoutGradient 0 0).1,
   AdamW.firstUpdate (9 / 10) (999 / 1000) 1 0 η 0 (axisReadoutGradient 0 0).2)

/-- Bias correction and the actual unequal gradient scales give this update,
RD5. The positive epsilon is retained. -/
theorem axisAdamStep_eq (η : ℝ) : axisAdamStep η = (η / 3, η / 2) := by
  have hd (g : ℝ) := AdamW.firstDirection_eq (9 / 10) (999 / 1000) 1 g
    (by norm_num) (by norm_num)
  unfold axisAdamStep
  rw [axisReadoutGradient_initial]
  simp only [AdamW.firstUpdate, hd]
  norm_num
  constructor <;> ring

/-- Ordinary Euclidean descent creates no residual in the same example,
RD5. It uses the same actual CE gradient and starting point. -/
theorem axis_gradient_step_stays_in_span (η : ℝ) :
    axisResidualEnergy (-η * (axisReadoutGradient 0 0).1)
      (-η * (axisReadoutGradient 0 0).2) = 0 := by
  rw [axisReadoutGradient_initial]
  norm_num [axisResidualEnergy]
  ring

/-- Increasing the correct binary margin lowers actual CE, RD5. -/
theorem binary_true_strictAnti : StrictAnti (binaryCE true) := by
  apply strictAnti_of_deriv_neg
  intro x
  rw [(binaryCE_deriv true x).deriv]
  have h : Real.exp x / (Real.exp x + 1) < 1 := by
    apply (div_lt_one (by positivity : 0 < Real.exp x + 1)).mpr
    linarith
  simpa using (sub_neg.mpr h)

/-- **CE descent under actual Adam can increase an unseen-direction residual**,
RD5. A fixed-feature Euclidean contraction law cannot be transferred unchanged. -/
theorem adam_creates_readout_residual {η : ℝ} (hη : 0 < η) :
    axisReadoutLoss (axisAdamStep η).1 (axisAdamStep η).2 < axisReadoutLoss 0 0 ∧
      axisResidualEnergy (axisAdamStep η).1 (axisAdamStep η).2 = η ^ 2 / 180 ∧
      0 < axisResidualEnergy (axisAdamStep η).1 (axisAdamStep η).2 := by
  rw [axisAdamStep_eq]
  refine ⟨?_, ?_, ?_⟩
  · apply binary_true_strictAnti
    dsimp
    linarith
  · unfold axisResidualEnergy
    dsimp
    ring
  · unfold axisResidualEnergy
    dsimp
    have he : (2 * (η / 3) - η / 2) = η / 6 := by ring
    rw [he]
    positivity

example := adam_creates_readout_residual (η := 1 / 10) (by norm_num)
example := axis_gradient_step_stays_in_span (1 / 10)

end Transformer.Grokking.GradientEvidence
