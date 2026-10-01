/-
# The limiting weighted objective

arXiv:1904.03590v4, Algorithm 1, AMSGradW training extension.
Decoupled decay corresponds at equilibrium to the diagonal quadratic
penalty formed from the actual limiting denominator. It need not equal
ordinary isotropic L2 regularization or minimize the original loss.
-/

import Transformer.AMSGradW.Convergence

open scoped BigOperators InnerProductSpace Topology
open Filter

noncomputable section

namespace Transformer.AMSGradW

variable {d : ℕ}

/-- The fixed objective selected by a frozen diagonal denominator.
Source: arXiv:1904.03590v4, Algorithm 1, AMSGradW extension. -/
def weightedLoss (f : TrainingSpace d → ℝ) (wd : ℝ) (D : Fin d → ℝ)
    (x : TrainingSpace d) : ℝ := f x + wd / 2 * ∑ i, D i * x i ^ 2

/-- The equilibrium globally minimizes the weighted objective for a
loss with a convex first-order lower model. Its objective gap controls
Euclidean distance strictly when epsilon and decay are positive.
Source: arXiv:1904.03590v4, §4, AMSGradW extension; convexity is additional. -/
theorem equilibrium_gap (ε wd : ℝ) (D : Fin d → ℝ) (f : TrainingSpace d → ℝ)
    (star x : TrainingSpace d) (hwd : 0 < wd)
    (hD : ∀ i, ε ≤ D i) (hf : Optimization.StrongLowerModel f 0)
    (hstar : ∀ i, gradient f star i + wd * D i * star i = 0) :
    wd * ε / 2 * ‖x - star‖ ^ 2 ≤ weightedLoss f wd D x - weightedLoss f wd D star := by
  have hl := hf star x
  simp only [zero_div, zero_mul, add_zero] at hl
  have hi : ⟪gradient f star, x - star⟫_ℝ =
      ∑ i, (x i - star i) * gradient f star i := by
    rw [EuclideanSpace.inner_eq_star_dotProduct]
    simp only [dotProduct, PiLp.sub_apply, Pi.star_apply, star_trivial]
  have heq : ⟪gradient f star, x - star⟫_ℝ +
      wd / 2 * ((∑ i, D i * x i ^ 2) - ∑ i, D i * star i ^ 2) =
        wd / 2 * ∑ i, D i * (x i - star i) ^ 2 := by
    rw [hi, ← Finset.sum_sub_distrib, Finset.mul_sum,
      ← Finset.sum_add_distrib, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i himem
    have hg : gradient f star i = -wd * D i * star i := by linarith [hstar i]
    rw [hg]
    ring
  have hs : ε * ‖x - star‖ ^ 2 ≤ ∑ i, D i * (x i - star i) ^ 2 := by
    rw [EuclideanSpace.norm_sq_eq]
    simp only [Real.norm_eq_abs, sq_abs, PiLp.sub_apply]
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum fun i himem =>
      mul_le_mul_of_nonneg_right (hD i) (sq_nonneg _)
  have hm := mul_le_mul_of_nonneg_left hs (show 0 ≤ wd / 2 by positivity)
  dsimp only [weightedLoss]
  nlinarith

/-- The gap assumptions hold for a nonconstant convex loss and positive
diagonal decay. Source: arXiv:1904.03590v4, §4, AMSGradW extension. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 2 ∧ (∀ i : Fin 1, (1 : ℝ) ≤ (fun _ => 1) i) ∧
    Optimization.StrongLowerModel (Optimization.energy : TrainingSpace 1 → ℝ) 0 ∧
    (∀ i : Fin 1, gradient Optimization.energy (0 : TrainingSpace 1) i +
      2 * (1 : ℝ) * (0 : TrainingSpace 1) i = 0) := by
  refine ⟨by norm_num, by norm_num, fun i => le_rfl, ?_, ?_⟩
  · intro x y
    have h := Optimization.energy_models.2 x y
    simp only [zero_div, zero_mul, add_zero]
    nlinarith [sq_nonneg ‖y - x‖]
  · intro i
    simp [Optimization.energy_gradient]

/-- Continuity of the genuine frozen weighted loss.
Source: arXiv:1904.03590v4, §4, AMSGradW training extension. -/
theorem weightedLoss_continuous (f : TrainingSpace d → ℝ) (wd : ℝ) (D : Fin d → ℝ)
    (hf : Continuous f) : Continuous (weightedLoss f wd D) := by
  change Continuous (fun x : TrainingSpace d => f x + wd / 2 * ∑ i, D i * x i ^ 2)
  fun_prop

/-- A nonconstant continuous loss witnesses the assumption,
arXiv:1904.03590v4, §4, AMSGradW extension. -/
example : Continuous (Optimization.energy : TrainingSpace 1 → ℝ) :=
  Optimization.energy_models.1.1.continuous

/-- Original AMSGradW converges to the unique global minimizer of its
history-dependent weighted objective when the loss has a convex
first-order model, under the stated contraction parameters. The metric
and minimizer both exist by proof, not by trajectory assumptions.
Source: arXiv:1904.03590v4, Algorithm 1 and §4, AMSGradW extension. -/
theorem trainingRun_minimum_convergence (η ε wd β β₂ L : ℝ) (f : TrainingSpace d → ℝ)
    (initial : TrainingSpace d) (hf : Differentiable ℝ f) (hgrad : LipschitzGradient f L)
    (hconvex : Optimization.StrongLowerModel f 0)
    (hη : 0 < η) (hε : 0 < ε) (hwd : 0 < wd) (hL : 0 ≤ L)
    (hβ : 0 ≤ β) (hβ' : β < 1) (hβ₂ : 0 ≤ β₂) (hβ₂' : β₂ ≤ 1)
    (hstep : η * wd ≤ 1) (hdom : L < wd * ε) :
    ∃ (D : Fin d → ℝ) (star : TrainingSpace d), (∀ i, ε ≤ D i) ∧
      (∀ x, weightedLoss f wd D star ≤ weightedLoss f wd D x) ∧
      (∀ x, weightedLoss f wd D x = weightedLoss f wd D star → x = star) ∧
      Tendsto (fun t => (trainingRun η ε wd β β₂ f initial t).position) atTop (𝓝 star) ∧
      Tendsto (fun t => weightedLoss f wd D (trainingRun η ε wd β β₂ f initial t).position)
        atTop (𝓝 (weightedLoss f wd D star)) := by
  obtain ⟨D, star, hD, hx, hm, hmetric, hloss, heq⟩ := trainingRun_convergence
    η ε wd β β₂ L f initial hf hgrad hη hε hwd hL hβ hβ' hβ₂ hβ₂' hstep hdom
  refine ⟨D, star, hD, fun x => ?_, fun x he => ?_, hx,
    (weightedLoss_continuous f wd D hf.continuous).tendsto star |>.comp hx⟩
  · have h := equilibrium_gap ε wd D f star x hwd hD hconvex heq
    have hn : 0 ≤ wd * ε / 2 * ‖x - star‖ ^ 2 := by positivity
    linarith
  · have h := equilibrium_gap ε wd D f star x hwd hD hconvex heq
    rw [he, sub_self] at h
    have hs : ‖x - star‖ ^ 2 ≤ 0 :=
      (mul_le_mul_iff_of_pos_left (show 0 < wd * ε / 2 by positivity)).mp
        (by simpa only [mul_zero] using h)
    have hn : ‖x - star‖ = 0 := by nlinarith [norm_nonneg (x - star)]
    exact sub_eq_zero.mp (norm_eq_zero.mp hn)

/-- All convex convergence assumptions hold simultaneously with positive
momentum and decay. Source: arXiv:1904.03590v4, §4, AMSGradW extension. -/
example : Differentiable ℝ (Optimization.energy : TrainingSpace 1 → ℝ) ∧
    LipschitzGradient (Optimization.energy : TrainingSpace 1 → ℝ) 1 ∧
    Optimization.StrongLowerModel (Optimization.energy : TrainingSpace 1 → ℝ) 0 ∧
    (0 : ℝ) < 1 / 4 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) ≤ 1 ∧
    (0 : ℝ) ≤ 9 / 10 ∧ (9 / 10 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 ∧
    (1 / 4 : ℝ) * 2 ≤ 1 ∧ (1 : ℝ) < 2 * 1 := by
  refine ⟨Optimization.energy_models.1.1, energy_lipschitz, ?_, by norm_num, by norm_num,
    by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num,
    by norm_num, by norm_num⟩
  intro x y
  have h := Optimization.energy_models.2 x y
  simp only [zero_div, zero_mul, add_zero]
  nlinarith [sq_nonneg ‖y - x‖]

end Transformer.AMSGradW
