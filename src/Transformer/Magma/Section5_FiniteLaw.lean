/-
# Finite stochastic-gradient laws

Formalization of arXiv:2602.15322v1, Section 5 and Appendix A.3.
Finite laws cover finite minibatch sampling conditioned on the current state.
Unbiasedness is an explicit correction: the source's second-moment
inequality alone neither states nor implies it.
-/

import Transformer.Optimization.Basic

open scoped BigOperators InnerProductSpace

noncomputable section

namespace Transformer.Magma

variable {Z E : Type*} [Fintype Z] [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Expectation under an explicit finite weight law. Probability-domain
conditions are supplied to order and normalization lemmas.
Source: arXiv:2602.15322v1, Section 5, conditional gradient moments. -/
def finiteExpectation (weight : Z → ℝ) (f : Z → ℝ) : ℝ := ∑ z, weight z * f z

/-- Finite expectation preserves sums. Source: arXiv:2602.15322v1,
Appendix A.3, taking total expectation. -/
theorem finiteExpectation_add (w : Z → ℝ) (f g : Z → ℝ) :
    finiteExpectation w (fun z => f z + g z) =
      finiteExpectation w f + finiteExpectation w g := by
  simp [finiteExpectation, mul_add, Finset.sum_add_distrib]

/-- Finite expectation preserves differences. Source: arXiv:2602.15322v1,
Appendix A.3, gradient-noise decomposition. -/
theorem finiteExpectation_sub (w : Z → ℝ) (f g : Z → ℝ) :
    finiteExpectation w (fun z => f z - g z) =
      finiteExpectation w f - finiteExpectation w g := by
  simp [finiteExpectation, mul_sub, Finset.sum_sub_distrib]

/-- A constant scalar factors out of expectation. Source:
arXiv:2602.15322v1, Appendix A.3. -/
theorem finiteExpectation_mul (w : Z → ℝ) (a : ℝ) (f : Z → ℝ) :
    finiteExpectation w (fun z => a * f z) = a * finiteExpectation w f := by
  unfold finiteExpectation
  rw [Finset.mul_sum]
  apply Finset.sum_congr rfl
  intro z hz
  ring

/-- Total mass one preserves constants. Source:
arXiv:2602.15322v1, Section 5, conditional expectation. -/
theorem finiteExpectation_const (w : Z → ℝ) (hw : ∑ z, w z = 1) (a : ℝ) :
    finiteExpectation w (fun _ => a) = a := by
  simp [finiteExpectation, ← Finset.sum_mul, hw]

/-- A probability law with mass one exists. Source:
arXiv:2602.15322v1, Section 5, zero-noise exact-gradient case. -/
example : ∑ z : Unit, (fun _ : Unit => (1 : ℝ)) z = 1 := by simp

/-- Nonnegative sampling weights preserve order. Source:
arXiv:2602.15322v1, Appendix A.3, noise bound. -/
theorem finiteExpectation_mono (w : Z → ℝ) (hw : ∀ z, 0 ≤ w z)
    (f g : Z → ℝ) (hfg : ∀ z, f z ≤ g z) :
    finiteExpectation w f ≤ finiteExpectation w g := by
  apply Finset.sum_le_sum
  intro z hz
  exact mul_le_mul_of_nonneg_left (hfg z) (hw z)

/-- The order hypotheses have a positive, nonempty support. Source:
arXiv:2602.15322v1, Section 5. -/
example : (∀ z : Unit, (0 : ℝ) ≤ (fun _ : Unit => (1 : ℝ)) z) ∧
    (∀ z : Unit, (fun _ : Unit => (0 : ℝ)) z ≤ (fun _ : Unit => (1 : ℝ)) z) := by simp

/-- Unbiasedness converts the raw second moment into the centered noise
second moment. This is not implied by the raw second-moment bound alone.
Source: correction to arXiv:2602.15322v1, Section 5,
assumption:layerwise_second_moment, and Appendix A.3. -/
theorem finiteVariance_eq (w : Z → ℝ) (hw : ∑ z, w z = 1) (g : Z → E) (a : E)
    (hmean : ∑ z, w z • g z = a) :
    finiteExpectation w (fun z => ‖g z - a‖ ^ 2) =
      finiteExpectation w (fun z => ‖g z‖ ^ 2) - ‖a‖ ^ 2 := by
  have hi : ∑ z, w z * ⟪g z, a⟫_ℝ = ‖a‖ ^ 2 := by
    calc
      _ = ⟪∑ z, w z • g z, a⟫_ℝ := by simp [sum_inner, real_inner_smul_left]
      _ = ‖a‖ ^ 2 := by rw [hmean, real_inner_self_eq_norm_sq]
  simp only [finiteExpectation, norm_sub_sq_real, mul_add, mul_sub,
    Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.sum_mul, hw, one_mul]
  have hi' : ∑ z, w z * (2 * ⟪g z, a⟫_ℝ) = 2 * ‖a‖ ^ 2 := by
    rw [← hi, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro z hz
    ring
  rw [hi']
  ring

/-- The unbiasedness hypotheses hold for a genuine exact derivative on
the nonconstant quadratic at x=1. Source: arXiv:2602.15322v1, Section 5. -/
example : (∑ _ : Unit, (1 : ℝ)) = 1 ∧
    (∑ _ : Unit, (1 : ℝ) • gradient Transformer.Optimization.quadratic (1 : ℝ)) =
      gradient Transformer.Optimization.quadratic (1 : ℝ) := by simp

/-- With explicit unbiasedness, the source's raw second-moment bound
controls centered gradient noise. Source: corrected assumption in
arXiv:2602.15322v1, Section 5 and Appendix A.3. -/
theorem finiteVariance_bound (w : Z → ℝ) (hw : ∑ z, w z = 1)
    (g : Z → E) (v : E) (σ : ℝ) (hmean : ∑ z, w z • g z = v)
    (hraw : finiteExpectation w (fun z => ‖g z‖ ^ 2) ≤ ‖v‖ ^ 2 + σ ^ 2) :
    finiteExpectation w (fun z => ‖g z - v‖ ^ 2) ≤ σ ^ 2 := by
  rw [finiteVariance_eq w hw g v hmean]
  linarith

/-- The centered-noise hypotheses hold for the actual quadratic gradient
with one finite sample and zero noise. Source: arXiv:2602.15322v1,
Section 5, corrected stochastic-gradient domain. -/
example :
    let v := gradient Transformer.Optimization.quadratic (1 : ℝ)
    (∑ _ : Unit, (1 : ℝ)) = 1 ∧ (∑ _ : Unit, (1 : ℝ) • v) = v ∧
      finiteExpectation (fun _ : Unit => (1 : ℝ)) (fun _ => ‖v‖ ^ 2) ≤ ‖v‖ ^ 2 + 0 ^ 2 := by
  simp [finiteExpectation]

end Transformer.Magma
