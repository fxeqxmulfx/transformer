/-
# AdaFisher: a corrected convex convergence theorem

arXiv:2405.16397v3, Proposition 3.3 and Appendix A.2.
Strong convexity μ, smoothness L, and Fisher entries in [δ,B] give a
geometric objective bound with step η L ≤ δ and η μ ≤ B. Unlike the
printed result, the step and rate explicitly account for the preconditioner.
-/

import Transformer.AdaFisher.SectionA_Descent
import Mathlib.Analysis.SpecificLimits.Basic

open scoped BigOperators
open Filter Topology

noncomputable section

namespace Transformer.AdaFisher

variable {d : ℕ}

/-- Euclidean quadratic witnessing all corrected convex assumptions,
Proposition 3.3; its gradient is the identity vector. -/
def vectorQuadratic (x : Fin d → ℝ) : ℝ := squaredNorm x / 2

/-- Exact smooth/strong lower model of the witness, Proposition 3.3. -/
theorem vectorQuadratic_model (x y : Fin d → ℝ) :
    vectorQuadratic y = vectorQuadratic x + pairing x (y - x) +
      (1 / 2 : ℝ) * squaredNorm (y - x) := by
  simp only [vectorQuadratic, squaredNorm, pairing, Pi.sub_apply,
    Finset.sum_div, Finset.mul_sum]
  rw [← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i hi
  ring

/-- The hypotheses of the corrected convex theorem are satisfiable,
Proposition 3.3, by the usual Euclidean quadratic. -/
theorem vectorQuadratic_models :
    SmoothUpperModel (vectorQuadratic (d := d)) id 1 ∧
      StrongLowerModel (vectorQuadratic (d := d)) id 1 := by
  constructor <;> intro x y <;> dsimp only [id]
  · exact (vectorQuadratic_model x y).le
  · exact (vectorQuadratic_model x y).ge

/-- Strong convexity gives the gradient domination inequality used in
Proposition 3.3's convergence argument. This follows from the lower model,
not from identifying Fisher with the objective Hessian. -/
theorem strong_model_gradient_bound (J : (Fin d → ℝ) → ℝ)
    (g : (Fin d → ℝ) → Fin d → ℝ) (μ : ℝ) (hμ : 0 < μ)
    (hJ : StrongLowerModel J g μ) (x star : Fin d → ℝ) :
    2 * μ * (J x - J star) ≤ squaredNorm (g x) := by
  have hmodel := hJ x star
  have hsq : -squaredNorm (g x) ≤ 2 * μ * pairing (g x) (star - x) +
      μ ^ 2 * squaredNorm (star - x) := by
    simp only [squaredNorm, pairing, Pi.sub_apply, ← Finset.sum_neg_distrib,
      Finset.mul_sum, ← Finset.sum_add_distrib]
    apply Finset.sum_le_sum
    intro i hi
    nlinarith [sq_nonneg (μ * (star i - x i) + g x i)]
  have hscaled := mul_le_mul_of_nonneg_left hmodel (by linarith : 0 ≤ 2 * μ)
  nlinarith

example : (0 : ℝ) < 1 ∧ StrongLowerModel (vectorQuadratic (d := 1)) id 1 :=
  ⟨by norm_num, vectorQuadratic_models.2⟩

/-- The actual unmomented iteration in Proposition 3.3. -/
def gradientRun (η : ℝ) (f : ℕ → Fin d → ℝ) (g : (Fin d → ℝ) → Fin d → ℝ)
    (initial : Fin d → ℝ) : ℕ → Fin d → ℝ
  | 0 => initial
  | t + 1 => gradientStep η (f t) (g (gradientRun η f g initial t))
      (gradientRun η f g initial t)

/-- Algorithm 1 with β=0 and no weight decay is the exact unmomented
iteration in Proposition 3.3. The gradient stream consists of the actual
gradients at its matching iterates, §3.3–3.4 and Table 1. -/
theorem adaFisherRun_zero_gradient (η : ℝ) (f : ℕ → Fin d → ℝ)
    (g : (Fin d → ℝ) → Fin d → ℝ) (initial : Fin d → ℝ) (t : ℕ) :
    adaFisherRun (fun _ => η) 0 0 f (fun k => g (gradientRun η f g initial k)) initial t =
      gradientRun η f g initial t := by
  induction t with
  | zero => rfl
  | succ t ih =>
    rw [adaFisherRun, ih, correctedMomentum_zero, gradientRun]
    funext i
    simp [parameterStep, gradientStep, mul_div_assoc]

/-- Corrected objective rate for Proposition 3.3. The source's `α≤1/L`
and preconditioner-independent `1/k` constant are refuted in
`Section3_ConvexFalse`. These explicitly changed conditions prove
`gap_t ≤ (1-ημ/B)^t gap_0` for the same unmomented diagonal update. -/
theorem corrected_convex_rate (J : (Fin d → ℝ) → ℝ)
    (g : (Fin d → ℝ) → Fin d → ℝ) (L μ η δ B : ℝ)
    (f : ℕ → Fin d → ℝ) (initial star : Fin d → ℝ)
    (hupper : SmoothUpperModel J g L) (hlower : StrongLowerModel J g μ)
    (hμ : 0 < μ) (hη : 0 ≤ η) (hδ : 0 < δ) (hB : 0 < B)
    (hstep : L * η ≤ δ) (hrate : η * μ ≤ B)
    (hf : ∀ t i, δ ≤ f t i ∧ f t i ≤ B) (t : ℕ) :
    J (gradientRun η f g initial t) - J star ≤
      (1 - η * μ / B) ^ t * (J initial - J star) := by
  have hq : 0 ≤ 1 - η * μ / B := sub_nonneg.mpr ((div_le_one hB).mpr hrate)
  induction t with
  | zero => simp [gradientRun]
  | succ t ih =>
    let x := gradientRun η f g initial t
    have hd := corrected_descent J g L η δ B (f t) x hupper hη hδ hB hstep (hf t)
    have hp := strong_model_gradient_bound J g μ hμ hlower x star
    have hp' := mul_le_mul_of_nonneg_left hp
      (div_nonneg hη (by positivity : 0 ≤ 2 * B))
    have heq : η / (2 * B) * (2 * μ * (J x - J star)) =
        η * μ / B * (J x - J star) := by ring
    rw [heq] at hp'
    calc
      J (gradientRun η f g initial (t + 1)) - J star ≤
          (1 - η * μ / B) * (J x - J star) := by
        rw [gradientRun]
        change J (gradientStep η (f t) (g x) x) - J star ≤ _
        nlinarith
      _ ≤ (1 - η * μ / B) * ((1 - η * μ / B) ^ t * (J initial - J star)) :=
        mul_le_mul_of_nonneg_left ih hq
      _ = _ := by rw [pow_succ]; ring

example :
    SmoothUpperModel (vectorQuadratic (d := 1)) id 1 ∧
    StrongLowerModel (vectorQuadratic (d := 1)) id 1 ∧
    (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    1 * (1 / 2 : ℝ) ≤ 1 ∧ (1 / 2 : ℝ) * 1 ≤ 1 ∧
    (∀ (t : ℕ) (i : Fin 1), 1 ≤ (fun _ _ => (1 : ℝ)) t i ∧
      (fun _ _ => (1 : ℝ)) t i ≤ 1) := by
  refine ⟨vectorQuadratic_models.1, vectorQuadratic_models.2, ?_⟩
  norm_num

end Transformer.AdaFisher
