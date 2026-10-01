/-
# A valid replacement for the momentum-based descent-efficiency lemma

Correction of arXiv:2602.15322v1, Section 5, sup_lem:descent_lower_bound,
and Appendix A.3. Actual block-diagonal damping is positive and bounded.
The estimate permits dependence on the stochastic sample and does not
pretend that momentum alignment is alignment with the objective gradient.
-/

import Transformer.Magma.Section5_FiniteLaw

open scoped BigOperators InnerProductSpace

noncomputable section

namespace Transformer.Magma

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Uniform lower alignment and upper norm bounds of a genuine linear
damping operator. This is a property of the operator, not of convergence
or its realized direction. Source: corrected Section 5 analysis of
arXiv:2602.15322v1; block scales between a and b satisfy these bounds. -/
def DampingBounds (a b : ℝ) (S : E →L[ℝ] E) : Prop :=
  ∀ u, a * ‖u‖ ^ 2 ≤ ⟪S u, u⟫_ℝ ∧ ‖S u‖ ≤ b * ‖u‖

/-- Valid noise-coupling estimate for every sample. The true gradient v
is distinguished from its stochastic estimator g. Source: correction
of arXiv:2602.15322v1, Section 5 and Appendix A.3. -/
theorem damping_alignment_lower (a b : ℝ) (ha : 0 < a) (S : E →L[ℝ] E)
    (hS : DampingBounds a b S) (g v : E) :
    a / 2 * ‖v‖ ^ 2 - b ^ 2 / (2 * a) * ‖g - v‖ ^ 2 ≤ ⟪S g, v⟫_ℝ := by
  have hpos := (hS v).1
  have hnorm := (hS (g - v)).2
  have hc := real_inner_le_norm (-(S (g - v))) v
  simp only [inner_neg_left, norm_neg] at hc
  have hm := mul_le_mul_of_nonneg_right hnorm (norm_nonneg v)
  have heq : ⟪S g, v⟫_ℝ = ⟪S v, v⟫_ℝ + ⟪S (g - v), v⟫_ℝ := by
    rw [map_sub, inner_sub_left]
    ring
  have hraw : a * ‖v‖ ^ 2 - b * ‖g - v‖ * ‖v‖ ≤ ⟪S g, v⟫_ℝ := by
    rw [heq]
    nlinarith
  have hdiv : a * (b ^ 2 / (2 * a)) = b ^ 2 / 2 := by
    field_simp
  have hyoung : b * ‖g - v‖ * ‖v‖ ≤
      a / 2 * ‖v‖ ^ 2 + b ^ 2 / (2 * a) * ‖g - v‖ ^ 2 := by
    by_contra! h
    have hpositive := mul_pos ha (sub_pos.mpr h)
    nlinarith [sq_nonneg (a * ‖v‖ - b * ‖g - v‖),
      congrArg (fun r => r * ‖g - v‖ ^ 2) hdiv]
  linarith

/-- A concrete nonzero damping operator obeys uniform spectral and norm
bounds. Source: arXiv:2602.15322v1, Section 5, corrected domain. -/
theorem scalar_half_damping_bounds :
    DampingBounds (1 / 4) 1 ((1 / 2 : ℝ) • ContinuousLinearMap.id ℝ ℝ) := by
  intro u
  constructor
  · simp only [FunLike.coe_smul, ContinuousLinearMap.id_apply,
      Pi.smul_apply, smul_eq_mul, RCLike.inner_apply, conj_trivial, Real.norm_eq_abs, sq_abs]
    nlinarith [sq_nonneg u]
  · simp [Real.norm_eq_abs]
    nlinarith [abs_nonneg u]

/-- Positive lower damping and operator bounds are jointly satisfiable.
Source: arXiv:2602.15322v1, Section 5, corrected domain. -/
example : (0 : ℝ) < 1 / 4 ∧
    DampingBounds (1 / 4) 1 ((1 / 2 : ℝ) • ContinuousLinearMap.id ℝ ℝ) :=
  ⟨by norm_num, scalar_half_damping_bounds⟩

variable {Z : Type*} [Fintype Z]

/-- Corrected finite-law lower bound; operator damping can depend on the
same stochastic sample as g. Centered noise is measured against the actual
gradient when this helper is used in descent. Source: corrected replacement
for arXiv:2602.15322v1, Section 5, sup_lem:descent_lower_bound. -/
theorem stochastic_alignment_lower (a b σ : ℝ) (ha : 0 < a)
    (w : Z → ℝ) (hw0 : ∀ z, 0 ≤ w z) (hw : ∑ z, w z = 1)
    (g : Z → E) (v : E) (S : Z → E →L[ℝ] E) (hS : ∀ z, DampingBounds a b (S z))
    (hvariance : finiteExpectation w (fun z => ‖g z - v‖ ^ 2) ≤ σ ^ 2) :
    a / 2 * ‖v‖ ^ 2 - b ^ 2 / (2 * a) * σ ^ 2 ≤
      finiteExpectation w (fun z => ⟪S z (g z), v⟫_ℝ) := by
  have hm := finiteExpectation_mono w hw0 _ _
    (fun z => damping_alignment_lower a b ha (S z) (hS z) (g z) v)
  simp only [finiteExpectation_sub, finiteExpectation_const w hw,
    finiteExpectation_mul] at hm
  have hnonneg : 0 ≤ b ^ 2 / (2 * a) := by positivity
  have hv := mul_le_mul_of_nonneg_left hvariance hnonneg
  linarith

/-- All finite-law lower-bound hypotheses hold on a genuine quadratic
gradient with zero noise and a nonzero damping operator.
Source: arXiv:2602.15322v1, Section 5, corrected domain. -/
example : (0 : ℝ) < 1 / 4 ∧
    (∀ _ : Unit, (0 : ℝ) ≤ 1) ∧ (∑ _ : Unit, (1 : ℝ)) = 1 ∧
    (∀ _ : Unit, DampingBounds (1 / 4) 1 ((1 / 2 : ℝ) • ContinuousLinearMap.id ℝ ℝ)) ∧
    finiteExpectation (fun _ : Unit => (1 : ℝ))
      (fun _ => ‖gradient Transformer.Optimization.quadratic (1 : ℝ) -
        gradient Transformer.Optimization.quadratic (1 : ℝ)‖ ^ 2) ≤ 0 ^ 2 := by
  exact ⟨by norm_num, by simp, by simp, fun _ => scalar_half_damping_bounds,
    by simp [finiteExpectation]⟩

end Transformer.Magma
