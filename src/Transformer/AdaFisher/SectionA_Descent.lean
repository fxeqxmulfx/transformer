/-
# AdaFisher: corrected preconditioned descent

arXiv:2405.16397v3, Proposition 3.3, Appendix A.2, `eq:upper_bound1`.
The Hessian bounds of the loss and the spectral bounds of the Fisher
preconditioner are separate. Damping δ requires the step η L ≤ δ.
-/

import Transformer.AdaFisher.Section3_Algorithm

open scoped BigOperators

noncomputable section

namespace Transformer.AdaFisher

variable {d : ℕ}

/-- Squared Euclidean norm used in the convergence proofs, Appendix A.2. -/
def squaredNorm (x : Fin d → ℝ) : ℝ := ∑ i, x i ^ 2

/-- Euclidean pairing used in Appendix A.2, `eq:upper_bound1`. -/
def pairing (x y : Fin d → ℝ) : ℝ := ∑ i, x i * y i

/-- The usual L-smooth upper model, Proposition 3.3, Appendix A.2.
This is an objective/gradient predicate, not a convergence assumption. -/
def SmoothUpperModel (J : (Fin d → ℝ) → ℝ) (g : (Fin d → ℝ) → Fin d → ℝ)
    (L : ℝ) : Prop := ∀ x y,
  J y ≤ J x + pairing (g x) (y - x) + L / 2 * squaredNorm (y - x)

/-- Strong-convexity lower model, Proposition 3.3, Appendix A.2. -/
def StrongLowerModel (J : (Fin d → ℝ) → ℝ) (g : (Fin d → ℝ) → Fin d → ℝ)
    (μ : ℝ) : Prop := ∀ x y,
  J x + pairing (g x) (y - x) + μ / 2 * squaredNorm (y - x) ≤ J y

/-- Unmomented diagonal NGD step of Proposition 3.3. -/
def gradientStep (η : ℝ) (f grad x : Fin d → ℝ) : Fin d → ℝ :=
  fun i => x i - η * grad i / f i

/-- Weighted gradient energy in `eq:upper_bound1`, Appendix A.2. -/
def gradientEnergy (f grad : Fin d → ℝ) : ℝ := ∑ i, grad i ^ 2 / f i

/-- Squared Euclidean norms are nonnegative, Appendix A.2. -/
theorem squaredNorm_nonneg (x : Fin d → ℝ) : 0 ≤ squaredNorm x :=
  Finset.sum_nonneg fun i _ => sq_nonneg (x i)

/-- Lower/upper Fisher bounds compare its energy to Euclidean norms,
Proposition 3.3, Appendix A.2. The upper norm bound alone is insufficient. -/
theorem energy_comparisons (δ B : ℝ) (f grad : Fin d → ℝ)
    (hδ : 0 < δ) (hf : ∀ i, δ ≤ f i ∧ f i ≤ B) :
    δ * squaredNorm (precondition f grad) ≤ gradientEnergy f grad ∧
    squaredNorm grad ≤ B * gradientEnergy f grad := by
  have heq (i : Fin d) : f i * (grad i / f i) ^ 2 = grad i ^ 2 / f i := by
    have hfi : f i ≠ 0 := ne_of_gt (lt_of_lt_of_le hδ (hf i).1)
    field_simp [hfi]
  constructor
  · simp only [squaredNorm, gradientEnergy, precondition, Finset.mul_sum]
    apply Finset.sum_le_sum
    intro i hi
    rw [← heq i]
    exact mul_le_mul_of_nonneg_right (hf i).1 (sq_nonneg _)
  · simp only [squaredNorm, gradientEnergy, Finset.mul_sum]
    apply Finset.sum_le_sum
    intro i hi
    have hfi : 0 < f i := lt_of_lt_of_le hδ (hf i).1
    have hprod : f i * (grad i ^ 2 / f i) = grad i ^ 2 := by field_simp
    calc
      _ = f i * (grad i ^ 2 / f i) := hprod.symm
      _ ≤ _ := mul_le_mul_of_nonneg_right (hf i).2 (div_nonneg (sq_nonneg _) hfi.le)

example : (0 : ℝ) < 1 ∧ (∀ i : Fin 1, 1 ≤ (fun _ => (1 : ℝ)) i ∧
    (fun _ => (1 : ℝ)) i ≤ 2) := by norm_num

/-- The smooth model evaluated at the actual step, Proposition 3.3,
Appendix A.2. No Hessian/Fisher identification is used. -/
theorem smooth_gradientStep (J : (Fin d → ℝ) → ℝ)
    (g : (Fin d → ℝ) → Fin d → ℝ) (L η : ℝ) (f x : Fin d → ℝ)
    (hJ : SmoothUpperModel J g L) :
    J (gradientStep η f (g x) x) ≤ J x - η * gradientEnergy f (g x) +
      L / 2 * η ^ 2 * squaredNorm (precondition f (g x)) := by
  have hm := hJ x (gradientStep η f (g x) x)
  have hp : pairing (g x) (gradientStep η f (g x) x - x) =
      -η * gradientEnergy f (g x) := by
    simp only [pairing, gradientStep, Pi.sub_apply, gradientEnergy, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i hi
    ring
  have hn : squaredNorm (gradientStep η f (g x) x - x) =
      η ^ 2 * squaredNorm (precondition f (g x)) := by
    simp only [squaredNorm, gradientStep, Pi.sub_apply, precondition, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i hi
    ring
  rw [hp, hn] at hm
  convert hm using 1
  ring

example : SmoothUpperModel (fun _ : Fin 1 → ℝ => 0) (fun _ _ => 0) 1 := by
  intro x y
  simp [pairing]
  exact squaredNorm_nonneg (y - x)

/-- Corrected sufficient decrease, Proposition 3.3, `eq:upper_bound1`.
The source conflates objective Hessian bounds with Fisher bounds and states
η≤1/L. The valid condition is η L≤δ for Fisher entries in `[δ,B]`, giving
decrease at least η/(2B) times the squared gradient norm. -/
theorem corrected_descent (J : (Fin d → ℝ) → ℝ)
    (g : (Fin d → ℝ) → Fin d → ℝ) (L η δ B : ℝ) (f x : Fin d → ℝ)
    (hJ : SmoothUpperModel J g L) (hη : 0 ≤ η) (hδ : 0 < δ) (hB : 0 < B)
    (hstep : L * η ≤ δ) (hf : ∀ i, δ ≤ f i ∧ f i ≤ B) :
    J (gradientStep η f (g x) x) ≤ J x - η / (2 * B) * squaredNorm (g x) := by
  have hm := smooth_gradientStep J g L η f x hJ
  obtain ⟨hc, hg⟩ := energy_comparisons δ B f (g x) hδ hf
  have hn := squaredNorm_nonneg (precondition f (g x))
  have hq : L * η * squaredNorm (precondition f (g x)) ≤ gradientEnergy f (g x) :=
    (mul_le_mul_of_nonneg_right hstep hn).trans hc
  have hq' := mul_le_mul_of_nonneg_left hq hη
  have hg' : squaredNorm (g x) / B ≤ gradientEnergy f (g x) :=
    (div_le_iff₀ hB).mpr (by simpa [mul_comm] using hg)
  have hg'' := mul_le_mul_of_nonneg_left hg' hη
  have hdiv : η / (2 * B) * squaredNorm (g x) = η * (squaredNorm (g x) / B) / 2 := by ring
  rw [hdiv]
  nlinarith

example : SmoothUpperModel (fun _ : Fin 1 → ℝ => 0) (fun _ _ => 0) 1 ∧
    (0 : ℝ) ≤ 1 / 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    1 * (1 / 2 : ℝ) ≤ 1 ∧ (∀ i : Fin 1, 1 ≤ (fun _ => (1 : ℝ)) i ∧
      (fun _ => (1 : ℝ)) i ≤ 1) := by
  refine ⟨?_, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩
  intro x y
  simp [pairing]
  exact squaredNorm_nonneg (y - x)

end Transformer.AdaFisher
