module

public import Transformer.SequenceModel.Transfer.Basic

/-!
# Finite source updates and a certified nonlinear remainder

Source: `review/flat_spectrum.md`, *Transformer learning bridges*, T3.
The actual Schedule-Free update includes averaging drift and query decay.
The cross-Jacobian term is exact at first order. A bound on derivative
variation along the actual step certifies the finite prediction error.
Only that segment needs differentiability, so XSA is not assumed globally smooth.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Transfer

open ScheduleFree Rotate90.Learning
open scoped RealInnerProductSpace

variable {I A B : Type*} [Fintype I] [Fintype A] [Fintype B]

/-- The source averaging and decay contribution, with the gradient term removed. -/
def drift (p : AdamW.Parameters) (t : ℕ) (s : AdamW.TrainingState I) : Weights I :=
  AdamW.coefficient p t s.history • (WithLp.toLp 2 s.base - evaluated p s) -
    (AdamW.coefficient p t s.history * AdamW.rate p t * p.adam.decay) • WithLp.toLp 2 s.point

/-- T3: actual source movement separates drift from its actual preconditioned gradient. -/
theorem source_displacement (p : AdamW.Parameters) (hβ : p.adam.betaOne ≠ 0)
    (t : ℕ) (s : AdamW.TrainingState I) (g : I → ℝ) :
    evaluated p (AdamW.trainingStep p t s g) - evaluated p s =
      drift p t s - (AdamW.coefficient p t s.history * AdamW.rate p t) •
        preconditioner p t s g (WithLp.toLp 2 g) := by
  rw [evaluation_displacement p hβ, preconditioner_gradient]
  unfold drift
  abel

/-- First-order prediction movement including both source drift terms. -/
def predictionStep (p : AdamW.Parameters) (t : ℕ) (s : AdamW.TrainingState I)
    (g : I → ℝ) (eval : Weights I →L[ℝ] Weights A)
    (train : Weights I →L[ℝ] Weights B) (r : Weights B) : Weights A :=
  eval (drift p t s) - (AdamW.coefficient p t s.history * AdamW.rate p t) •
    cross eval train (preconditioner p t s g) r

omit [Fintype A] in
/-- T3: the cross operator comes from applying the evaluation Jacobian to
the genuine optimizer displacement. The gradient certificate is supplied by
the chain rule for the actual training loss, not a frozen-gradient assumption. -/
theorem linearized_source_step (p : AdamW.Parameters) (hβ : p.adam.betaOne ≠ 0)
    (t : ℕ) (s : AdamW.TrainingState I) (g : I → ℝ)
    (eval : Weights I →L[ℝ] Weights A) (train : Weights I →L[ℝ] Weights B) (r : Weights B)
    (hg : WithLp.toLp 2 g = train.adjoint r) :
    eval (evaluated p (AdamW.trainingStep p t s g) - evaluated p s) =
      predictionStep p t s g eval train r := by
  rw [source_displacement p hβ, map_sub, map_smul, hg]
  rfl

/-- T3: the real nonlinear prediction after one actual source step is within
C times the step length of the derived first-order prediction. The premise
bounds derivative variation on the segment, not the desired endpoint output. -/
theorem finite_prediction_error (p : AdamW.Parameters) (hβ : p.adam.betaOne ≠ 0)
    (t : ℕ) (s : AdamW.TrainingState I) (g : I → ℝ)
    (prediction : Weights I → Weights A) (eval : Weights I →L[ℝ] Weights A)
    (train : Weights I →L[ℝ] Weights B) (r : Weights B)
    (hg : WithLp.toLp 2 g = train.adjoint r) (C : ℝ)
    (hF : ∀ w ∈ segment ℝ (evaluated p s) (evaluated p (AdamW.trainingStep p t s g)),
      DifferentiableAt ℝ prediction w)
    (hJ : ∀ w ∈ segment ℝ (evaluated p s) (evaluated p (AdamW.trainingStep p t s g)),
      ‖fderiv ℝ prediction w - eval‖ ≤ C) :
    ‖prediction (evaluated p (AdamW.trainingStep p t s g)) - prediction (evaluated p s) -
      predictionStep p t s g eval train r‖ ≤
      C * ‖evaluated p (AdamW.trainingStep p t s g) - evaluated p s‖ := by
  rw [← linearized_source_step p hβ t s g eval train r hg]
  exact (convex_segment (𝕜 := ℝ) _ _).norm_image_sub_le_of_norm_fderiv_le' hF hJ
    (left_mem_segment ℝ _ _) (right_mem_segment ℝ _ _)

/-- T3: a positive first-order advantage larger than the certified remainder
forces a true finite loss decrease. The theorem is local to the actual segment. -/
theorem loss_decreases_of_transfer {F : Weights I → ℝ} {x y : Weights I}
    {D : Weights I →L[ℝ] ℝ} {C : ℝ}
    (hF : ∀ w ∈ segment ℝ x y, DifferentiableAt ℝ F w)
    (hJ : ∀ w ∈ segment ℝ x y, ‖fderiv ℝ F w - D‖ ≤ C)
    (hprogress : D (y - x) + C * ‖y - x‖ < 0) : F y < F x := by
  have h := (convex_segment (𝕜 := ℝ) x y).norm_image_sub_le_of_norm_fderiv_le' hF hJ
    (left_mem_segment ℝ x y) (right_mem_segment ℝ x y)
  have hup := (abs_le.mp (show |F y - F x - D (y - x)| ≤ C * ‖y - x‖ from h)).2
  linarith

private def p₀ : AdamW.Parameters := ⟨⟨3/1000, 1, 0, 9/10, 98/100, 1/100000000⟩, 0, 2⟩
private def s₀ : AdamW.TrainingState (Fin 2) :=
  ⟨fun i => i.val, fun _ => 1, ⟨fun _ => 1, 1, 3/1000⟩⟩

example := source_displacement p₀ (by norm_num [p₀]) 5 s₀ (fun i => i.val + 1)
example := linearized_source_step p₀ (by norm_num [p₀]) 5 s₀ (fun i => i.val + 1)
  (ContinuousLinearMap.id ℝ (Weights (Fin 2))) (ContinuousLinearMap.id ℝ (Weights (Fin 2)))
  (WithLp.toLp 2 (fun i => i.val + 1)) (by rw [ContinuousLinearMap.adjoint_id]; rfl)
example := finite_prediction_error p₀ (by norm_num [p₀]) 5 s₀ (fun i => i.val + 1)
  (fun x => x) (ContinuousLinearMap.id ℝ (Weights (Fin 2)))
  (ContinuousLinearMap.id ℝ (Weights (Fin 2))) (WithLp.toLp 2 (fun i => i.val + 1))
  (by rw [ContinuousLinearMap.adjoint_id]; rfl) 0
  (fun _ _ => differentiableAt_id) (by intro w hw; simp)

example : (fun w : Weights (Fin 1) => w 0) (WithLp.toLp 2 ![-1]) <
    (fun w : Weights (Fin 1) => w 0) 0 := by
  let D := (PiLp.continuousLinearEquiv 2 ℝ (fun _ : Fin 1 => ℝ)).toContinuousLinearMap
  let L := (ContinuousLinearMap.proj (0 : Fin 1)).comp D
  apply loss_decreases_of_transfer (I := Fin 1) (F := fun w => w 0)
    (x := 0) (y := WithLp.toLp 2 ![-1]) (D := L) (C := 0)
  · intro w hw
    exact L.differentiableAt
  · intro w hw
    change ‖fderiv ℝ L w - L‖ ≤ 0
    simp
  · norm_num [L, D]

end Transformer.SequenceModel.Transfer
