/-
# Corrected descent with stochastic gradients

Replacement for arXiv:2602.15322v1, Section 5, prop:magma_descent and
sup_lem:descent_lower_bound. The statement uses global smoothness, explicit
unbiasedness, actual derivatives, and bounded positive damping. The masks
are independent of the sampled gradient because their law is a product.
-/

import Transformer.Magma.Section5_AlignmentBound
import Transformer.Magma.Section5_MaskDescent

open scoped BigOperators InnerProductSpace

noncomputable section

namespace Transformer.Magma

open Transformer.Optimization

variable {Z ι E : Type*} [Fintype Z] [Fintype ι] [DecidableEq ι]
  [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

omit [Fintype ι] [DecidableEq ι] [CompleteSpace E] in
/-- The norm bound on damping controls its raw second moment, even when
the operator depends on the same sample as the gradient. Source:
arXiv:2602.15322v1, Appendix A.2, corrected operator bound. -/
theorem damped_second_moment (a b σ : ℝ) (w : Z → ℝ) (hw0 : ∀ z, 0 ≤ w z)
    (g : Z → E) (v : E) (S : Z → E →L[ℝ] E) (hS : ∀ z, DampingBounds a b (S z))
    (hraw : finiteExpectation w (fun z => ‖g z‖ ^ 2) ≤ ‖v‖ ^ 2 + σ ^ 2) :
    finiteExpectation w (fun z => ‖S z (g z)‖ ^ 2) ≤ b ^ 2 * (‖v‖ ^ 2 + σ ^ 2) := by
  have hpoint (z : Z) : ‖S z (g z)‖ ^ 2 ≤ b ^ 2 * ‖g z‖ ^ 2 := by
    have hn := (hS z (g z)).2
    have hb : 0 ≤ b * ‖g z‖ := (norm_nonneg _).trans hn
    have hs := (sq_le_sq₀ (norm_nonneg _) hb).mpr hn
    simpa only [mul_pow] using hs
  have hm := finiteExpectation_mono w hw0 _ _ hpoint
  rw [finiteExpectation_mul] at hm
  exact hm.trans (mul_le_mul_of_nonneg_left hraw (sq_nonneg b))

/-- The second-moment hypotheses hold on the exact quadratic gradient
with nonzero scalar damping. Source: arXiv:2602.15322v1, Section 5. -/
example : (∀ _ : Unit, (0 : ℝ) ≤ 1) ∧
    (∀ _ : Unit, DampingBounds (1 / 4) 1 ((1 / 2 : ℝ) • ContinuousLinearMap.id ℝ ℝ)) ∧
    finiteExpectation (fun _ : Unit => (1 : ℝ))
      (fun _ => ‖gradient quadratic (1 : ℝ)‖ ^ 2) ≤
        ‖gradient quadratic (1 : ℝ)‖ ^ 2 + 0 ^ 2 := by
  exact ⟨by simp, fun _ => scalar_half_damping_bounds, by simp [finiteExpectation]⟩

/-- Corrected one-step stochastic descent. It concerns the normalized
SGD update of Section 5, not arbitrary adaptive base directions in
Algorithm 1. Global smoothness replaces coordinate smoothness, and
unbiasedness is explicit. Uniform damping bounds permit sample-dependent
momentum and EMA states. The constants are coarser than the refuted
source estimate. Source: arXiv:2602.15322v1, Section 5, both descent lemmas
and Appendix A.2--A.3. -/
theorem stochastic_descent_corrected (loss : E → ℝ) (L rate p a b σ : ℝ) (x : E)
    (hf : SmoothObjective loss L) (hL : 0 < L) (hrate : 0 < rate)
    (hp : 0 < p) (hp' : p ≤ 1) (ha : 0 < a)
    (hstep : L * rate * b ^ 2 ≤ a * p / 2)
    (w : Z → ℝ) (hw0 : ∀ z, 0 ≤ w z) (hw : ∑ z, w z = 1)
    (g : Z → E) (hmean : ∑ z, w z • g z = gradient loss x)
    (hraw : finiteExpectation w (fun z => ‖g z‖ ^ 2) ≤ ‖gradient loss x‖ ^ 2 + σ ^ 2)
    (S : Z → E →L[ℝ] E) (hS : ∀ z, DampingBounds a b (S z))
    (update : Z → ι → E) (hsum : ∀ z, ∑ j, update z j = S z (g z))
    (horth : ∀ z j k, j ≠ k → ⟪update z j, update z k⟫_ℝ = 0) :
    finiteExpectation w (fun z => maskExpectation p
      (fun mask => loss (x - rate • maskedSum p (update z) mask))) ≤
      loss x - rate * a / 4 * ‖gradient loss x‖ ^ 2 +
        (rate * b ^ 2 / (2 * a) + L * rate ^ 2 * b ^ 2 / (2 * p)) * σ ^ 2 := by
  have hpoint (z : Z) := mask_descent_corrected loss L rate p x (update z)
    hf hp hp' (horth z)
  simp only [hsum] at hpoint
  have hm := finiteExpectation_mono w hw0 _ _ hpoint
  simp only [finiteExpectation_add, finiteExpectation_sub,
    finiteExpectation_const w hw, finiteExpectation_mul] at hm
  have hv := finiteVariance_bound w hw g (gradient loss x) σ hmean hraw
  have halign := stochastic_alignment_lower a b σ ha w hw0 hw g
    (gradient loss x) S hS hv
  have henergy := damped_second_moment a b σ w hw0 g (gradient loss x) S hS hraw
  have hscaled := mul_le_mul_of_nonneg_left halign hrate.le
  have hc : 0 ≤ L * rate ^ 2 / (2 * p) := by positivity
  have henergy' := mul_le_mul_of_nonneg_left henergy hc
  have hcoef : L * rate ^ 2 * b ^ 2 / (2 * p) ≤ rate * a / 4 := by
    apply (div_le_iff₀ (by positivity : 0 < 2 * p)).mpr
    nlinarith [mul_le_mul_of_nonneg_left hstep hrate.le]
  have hcoef' := mul_le_mul_of_nonneg_right hcoef (sq_nonneg ‖gradient loss x‖)
  ring_nf at hm hscaled henergy' hcoef' ⊢
  linarith only [hm, hscaled, henergy', hcoef']

/-- All corrected stochastic-descent hypotheses are jointly satisfiable
by an actual smooth, nonconstant objective, exact gradients, a nonzero
one-block damping update, and a positive step. Source:
arXiv:2602.15322v1, Section 5, corrected domain. -/
example :
    SmoothObjective quadratic 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 100 ∧
    (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 / 4 ∧
    (1 : ℝ) * (1 / 100) * 1 ^ 2 ≤ (1 / 4) * (1 / 2) / 2 ∧
    (∀ _ : Unit, (0 : ℝ) ≤ 1) ∧ (∑ _ : Unit, (1 : ℝ)) = 1 ∧
    (∑ _ : Unit, (1 : ℝ) • gradient quadratic (1 : ℝ)) = gradient quadratic (1 : ℝ) ∧
    finiteExpectation (fun _ : Unit => (1 : ℝ)) (fun _ => ‖gradient quadratic (1 : ℝ)‖ ^ 2) ≤
      ‖gradient quadratic (1 : ℝ)‖ ^ 2 + 0 ^ 2 ∧
    (∀ _ : Unit, DampingBounds (1 / 4) 1 ((1 / 2 : ℝ) • ContinuousLinearMap.id ℝ ℝ)) ∧
    (∀ _ : Unit, (∑ _ : Unit, (1 / 2 : ℝ) * gradient quadratic (1 : ℝ)) =
      ((1 / 2 : ℝ) • ContinuousLinearMap.id ℝ ℝ) (gradient quadratic (1 : ℝ))) ∧
    (∀ (_ : Unit) (j k : Unit), j ≠ k →
      ⟪(1 / 2 : ℝ) * gradient quadratic (1 : ℝ),
        (1 / 2 : ℝ) * gradient quadratic (1 : ℝ)⟫_ℝ = 0) := by
  refine ⟨quadratic_smooth, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num, by simp, by simp, by simp,
    by simp [finiteExpectation], fun _ => scalar_half_damping_bounds, by simp, ?_⟩
  intro z j k h
  exact (h (Subsingleton.elim j k)).elim

end Transformer.Magma
