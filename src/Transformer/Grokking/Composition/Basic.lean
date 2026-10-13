import Transformer.Grokking.GradientEvidence.BinaryObjectives

/-!
# Coupled components under actual binary softmax cross-entropy

Source: Nanda et al., arXiv:2301.05217v1, appendix section Further
speculations on grokking, subsection Hypothesis: Phase Transitions are
inherent to composition. The source discusses useful multi-part circuits
whose isolated parts do not improve the task, and labels this speculative.

Explicit specialization: two real components supply the bilinear logit
`x*y`, competing with a zero logit for a fixed correct class. The loss is
the already verified ordinary finite-class CE, not an assumed gradient or
a quadratic surrogate. Prove both coordinate derivatives before making
any flatness or stationarity claim. The coordinate axes have constant
loss, but neither axis is a region where every gradient vanishes.

No transformer circuit, QK-normalization, generalization on new inputs or
delayed decision change is identified with this two-component calculation.
In particular, increasing its positive score can be confidence alone.
-/

namespace Transformer.Grokking.Composition

open Transformer.Grokking.GradientEvidence

/-- Actual two-class CE of a bilinear component score. Source:
arXiv:2301.05217v1, appendix compositional-circuit hypothesis; deviation:
two scalar components and one fixed target, rather than a transformer. -/
noncomputable def coupledLoss (x y : ℝ) : ℝ := binaryCE true (x * y)

/-- Both actual coordinate derivatives of that objective. Source:
the compositional specialization of arXiv:2301.05217v1 above; the desired
gradient behavior is not supplied by this definition. -/
noncomputable def coupledGradient (x y : ℝ) : ℝ × ℝ :=
  (deriv (fun u => coupledLoss u y) x, deriv (fun v => coupledLoss x v) y)

/-- The specialization retains exactly ordinary CE and its target
subtraction. Source: arXiv:2301.05217v1, appendix compositional hypothesis,
using the finite-class supervised loss verified in BinaryObjectives. -/
theorem coupledLoss_eq_exp (x y : ℝ) :
    coupledLoss x y = Real.log (1 + Real.exp (-(x * y))) := by
  unfold coupledLoss binaryCE NaiveLoss.crossEntropy
  rw [Fin.sum_univ_two]
  norm_num

/-- Actual first coordinate derivative carries the other component.
Source: arXiv:2301.05217v1, appendix Hypothesis: Phase Transitions are
inherent to composition; derived for the explicit bilinear CE model. -/
theorem coupledLoss_deriv_first (x y : ℝ) :
    HasDerivAt (fun u => coupledLoss u y) (-y / (Real.exp (x * y) + 1)) x := by
  have h := (binaryCE_deriv true (x * y)).comp x ((hasDerivAt_id x).mul_const y)
  convert h using 1
  · funext u
    rfl
  · dsimp
    have hd : Real.exp (x * y) + 1 ≠ 0 := by have hp := Real.exp_pos (x * y); positivity
    field_simp
    ring

/-- Actual second coordinate derivative is obtained by the same chain
rule, with the first component retained. Source: arXiv:2301.05217v1,
appendix compositional hypothesis; the bilinear specialization is explicit. -/
theorem coupledLoss_deriv_second (x y : ℝ) :
    HasDerivAt (fun v => coupledLoss x v) (-x / (Real.exp (x * y) + 1)) y := by
  have h := (binaryCE_deriv true (x * y)).comp y ((hasDerivAt_id y).const_mul x)
  convert h using 1
  · funext v
    rfl
  · dsimp
    have hd : Real.exp (x * y) + 1 ≠ 0 := by have hp := Real.exp_pos (x * y); positivity
    field_simp
    ring

/-- The actual gradient has both cross-component terms. Source:
arXiv:2301.05217v1, appendix compositional hypothesis; zero partner
and nonzero partner therefore require different reasoning. -/
theorem coupledGradient_eq (x y : ℝ) :
    coupledGradient x y = (-y / (Real.exp (x * y) + 1), -x / (Real.exp (x * y) + 1)) := by
  unfold coupledGradient
  rw [(coupledLoss_deriv_first x y).deriv, (coupledLoss_deriv_second x y).deriv]

/-- Changing one component while its partner is absent leaves the
actual loss unchanged. Source: arXiv:2301.05217v1, appendix multi-part
circuit argument, proved here only for the explicit bilinear CE model. -/
theorem coupledLoss_flat_axes (x y : ℝ) :
    coupledLoss x 0 = Real.log 2 ∧ coupledLoss 0 y = Real.log 2 := by
  rw [coupledLoss_eq_exp, coupledLoss_eq_exp]
  norm_num

/-- The exact all-absent state is stationary. Source: the zero-component
specialization of arXiv:2301.05217v1's appendix circuit-formation argument;
random initialization is not assumed to be exactly this state. -/
theorem coupled_origin_stationary : coupledGradient 0 0 = (0, 0) := by
  rw [coupledGradient_eq]
  norm_num

/-- The remaining coordinate on each flat axis still creates a gradient
for its absent partner. Source: arXiv:2301.05217v1, appendix composition
argument; this distinguishes a flat restriction from a stationary region. -/
theorem coupled_axis_gradients (t : ℝ) :
    coupledGradient t 0 = (0, -t / 2) ∧ coupledGradient 0 t = (-t / 2, 0) := by
  rw [coupledGradient_eq, coupledGradient_eq]
  norm_num

/-- The origin is the only stationary state of this actual CE model.
Source comparison: arXiv:2301.05217v1's appendix no-gradient circuit
argument. Flat coordinate restrictions do not make every axis point
stationary, and absence of one component is not absence of both. -/
theorem coupledGradient_zero_iff (x y : ℝ) :
    coupledGradient x y = (0, 0) ↔ x = 0 ∧ y = 0 := by
  rw [coupledGradient_eq]
  constructor
  · intro h
    have hfirst : -y / (Real.exp (x * y) + 1) = 0 := congrArg Prod.fst h
    have hsecond : -x / (Real.exp (x * y) + 1) = 0 := congrArg Prod.snd h
    have hd : Real.exp (x * y) + 1 ≠ 0 := by have hp := Real.exp_pos (x * y); positivity
    field_simp at hfirst hsecond
    constructor <;> linarith
  · rintro ⟨rfl, rfl⟩
    norm_num

/-- A present partner gives a nonzero first-component gradient at any
finite score. Source: arXiv:2301.05217v1, appendix compositional hypothesis;
small random partial components are not the exact all-zero state. -/
theorem coupled_first_gradient_nonzero (x y : ℝ) (hy : y ≠ 0) :
    (coupledGradient x y).1 ≠ 0 := by
  rw [coupledGradient_eq]
  dsimp
  intro h
  have hd : Real.exp (x * y) + 1 ≠ 0 := by have hp := Real.exp_pos (x * y); positivity
  field_simp at h
  apply hy
  linarith

example : (1 : ℝ) ≠ 0 := by norm_num

/-- Jointly aligned components improve actual CE compared with the
flat axes. Source: arXiv:2301.05217v1, appendix compositional hypothesis;
this proves loss improvement, not new held-out classification. -/
theorem coupledLoss_below_axes (x y : ℝ) (hxy : 0 < x * y) :
    coupledLoss x y < Real.log 2 := by
  rw [coupledLoss_eq_exp]
  have hp := Real.exp_pos (-(x * y))
  have he : Real.exp (-(x * y)) < 1 := by
    have h := Real.exp_lt_exp.mpr (show -(x * y) < 0 by linarith)
    simpa using h
  apply Real.log_lt_log (by positivity)
  linarith

example : 0 < (1 : ℝ) * 1 := by norm_num

end Transformer.Grokking.Composition
