/-
# Magma's alignment damping and dense moment updates

Formalization of arXiv:2602.15322v1, Sections 2-3 and Algorithm 1.
Algorithm 1 multiplies by s*m, whereas Section 5 analyzes s*m/p.
The difference is explicit below. Zero-vector cosine is extended to zero;
the paper does not prescribe a convention at a zero denominator.
-/

import Transformer.Magma.Section2_Moments
import Mathlib.Analysis.SpecialFunctions.Sigmoid
import Mathlib.Analysis.InnerProductSpace.Basic

open scoped InnerProductSpace

noncomputable section

namespace Transformer.Magma

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Cosine similarity, with the zero-denominator extension documented above.
Source: arXiv:2602.15322v1, Section 3, eq:masking_prob. -/
def cosine (momentum gradient : E) : ℝ :=
  ⟪momentum, gradient⟫_ℝ / (‖momentum‖ * ‖gradient‖)

/-- The printed alignment EMA, retaining its previous block scalar.
Source: arXiv:2602.15322v1, Section 3 and Algorithm 1. -/
def damping (τ previous : ℝ) (momentum gradient : E) : ℝ :=
  9 / 10 * previous + 1 / 10 * Real.sigmoid (cosine momentum gradient / τ)

/-- For a previous scalar in [0,1], the updated scalar lies strictly in
(0,1). This follows from sigmoid even at zero-vector inputs.
Source: arXiv:2602.15322v1, Section 3, eq:masking_prob and EMA. -/
theorem damping_mem_Ioo (τ previous : ℝ) (momentum gradient : E)
    (hprev : previous ∈ Set.Icc (0 : ℝ) 1) :
    damping τ previous momentum gradient ∈ Set.Ioo (0 : ℝ) 1 := by
  have hpos := Real.sigmoid_pos (cosine momentum gradient / τ)
  have hlt := Real.sigmoid_lt_one (cosine momentum gradient / τ)
  unfold damping
  constructor <;> nlinarith [hprev.1, hprev.2]

/-- The damping hypotheses are nonempty, including zero momentum.
Source: arXiv:2602.15322v1, Section 3. -/
example : (1 / 2 : ℝ) ∈ Set.Icc (0 : ℝ) 1 := by norm_num

/-- The scalar EMA requires the previous scale; it cannot be replaced by
a function of just the current momentum and gradient. This qualifies the
paper's informal "no additional memory" wording. Source:
arXiv:2602.15322v1, Section 3, paragraph after eq:masking_prob. -/
theorem damping_depends_on_previous (τ a b : ℝ) (momentum gradient : E) (hab : a ≠ b) :
    damping τ a momentum gradient ≠ damping τ b momentum gradient := by
  unfold damping
  intro h
  apply hab
  linarith

/-- Distinct admissible EMA states exist. Source: arXiv:2602.15322v1,
Section 3, alignment EMA. -/
example : (1 / 4 : ℝ) ≠ 1 / 2 := by norm_num

/-- Per-coordinate displacement from the algorithm, including the learning
rate already contained in direction. Source: arXiv:2602.15322v1,
Algorithm 1 and Section 3, masked update rule. -/
def algorithmDisplacement (scale direction : ℝ) (bit : Bool) : ℝ :=
  if bit then scale * direction else 0

/-- Per-coordinate displacement from the normalized SGD analysis.
Source: arXiv:2602.15322v1, Section 5, scaled masking operator. -/
def analysisDisplacement (p scale direction : ℝ) (bit : Bool) : ℝ :=
  maskCoefficient p bit * scale * direction

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- The algorithm's average displacement is p*s*direction, so damping
introduces bias. Source: arXiv:2602.15322v1, Section 3 and Algorithm 1. -/
theorem algorithmDisplacement_mean (p scale direction : ℝ) (b : ι) :
    maskExpectation p (fun mask => algorithmDisplacement scale direction (mask b)) =
      p * scale * direction := by
  rw [maskExpectation_coordinate]
  simp [algorithmDisplacement, mul_assoc]

/-- Section 5's normalized scheme has mean s*direction. Source:
arXiv:2602.15322v1, Section 5, masking operator and Magma recurrence. -/
theorem analysisDisplacement_mean (p scale direction : ℝ) (hp : p ≠ 0) (b : ι) :
    maskExpectation p (fun mask => analysisDisplacement p scale direction (mask b)) =
      scale * direction := by
  rw [maskExpectation_coordinate]
  simp [analysisDisplacement, maskCoefficient, hp, mul_assoc]

/-- Nonzero survival is possible. Source: arXiv:2602.15322v1,
Algorithm 1, p=1/2. -/
example : (1 / 2 : ℝ) ≠ 0 := by norm_num

/-- At the printed p=1/2, the normalized analysis doubles every surviving
Magma displacement. A learning-rate change is necessary to equate the
schemes. Source: arXiv:2602.15322v1, Algorithm 1 versus Section 5. -/
theorem analysis_algorithm_factor_two (scale direction : ℝ) (bit : Bool) :
    analysisDisplacement (1 / 2) scale direction bit =
      2 * algorithmDisplacement scale direction bit := by
  cases bit <;> norm_num [analysisDisplacement, algorithmDisplacement, maskCoefficient]
  ring

/-- Actual scalar base moments plus the wrapper scale and parameter.
Source: arXiv:2602.15322v1, Section 2, moment estimates, and Section 3. -/
structure ScalarState where
  position : ℝ
  momentum : ℝ
  variance : ℝ
  scale : ℝ

/-- Dense EMA moments followed by Algorithm 1's masked parameter step.
direction is the supplied base optimizer displacement, as in Algorithm 1.
Source: arXiv:2602.15322v1, Sections 2-3 and Algorithm 1. -/
def scalarMagmaStep (β₁ β₂ τ g direction : ℝ) (bit : Bool) (state : ScalarState) :
    ScalarState :=
  let m := β₁ * state.momentum + (1 - β₁) * g
  let v := β₂ * state.variance + (1 - β₂) * g ^ 2
  let s := damping τ state.scale m g
  ⟨state.position - algorithmDisplacement s direction bit, m, v, s⟩

/-- A skipped nonzero-gradient block keeps its weight but updates both
moments. This is a substantive example of dense versus lazy states, not a
claim that the resulting training converges. Source: arXiv:2602.15322v1,
Section 2, Why dense momentum updates matter, and Algorithm 1. -/
theorem skipped_block_updates_moments (direction : ℝ) :
    let after := scalarMagmaStep (9 / 10) (999 / 1000) 1 1 direction false ⟨1, 0, 0, 1 / 2⟩
    after.position = 1 ∧ after.momentum = 1 / 10 ∧ after.variance = 1 / 1000 := by
  norm_num [scalarMagmaStep, algorithmDisplacement]

end Transformer.Magma
