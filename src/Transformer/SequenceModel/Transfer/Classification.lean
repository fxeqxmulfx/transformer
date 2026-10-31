module

public import Transformer.SequenceModel.Transfer.Step

/-!
# From a certified finite prediction to classification

Source: `review/flat_spectrum.md`, *Transformer learning bridges*, T3.
The margin is checked on the derived first-order prediction. The error
radius comes from the derivative-variation bound, rather than a hypothesis
that the final nonlinear network already predicts the desired labels.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Transfer

variable {A V : Type*} [Fintype A] [Fintype V]

/-- T3: a predicted logit margin greater than twice a certified Euclidean
error radius survives the actual nonlinear step, simultaneously on all rows. -/
theorem correct_of_prediction_error (actual predicted : Weights (A × V))
    (target : A → V) (ε : ℝ) (herror : ‖actual - predicted‖ ≤ ε)
    (hmargin : ∀ a v, v ≠ target a → 2 * ε < predicted (a, target a) - predicted (a, v)) :
    ∀ a v, v ≠ target a → actual (a, v) < actual (a, target a) := by
  intro a v hv
  have h₁ := (PiLp.norm_apply_le (actual - predicted) (a, v)).trans herror
  have h₂ := (PiLp.norm_apply_le (actual - predicted) (a, target a)).trans herror
  change |actual (a, v) - predicted (a, v)| ≤ ε at h₁
  change |actual (a, target a) - predicted (a, target a)| ≤ ε at h₂
  have h₁' := (abs_le.mp h₁).2
  have h₂' := (abs_le.mp h₂).1
  have hm := hmargin a v hv
  linarith

/-- T3: an actual finite source step classifies every indexed row when
its derived transfer prediction clears the certified nonlinear remainder.
The row set may contain training, held-out, or all finite task examples. -/
theorem source_step_correct {I B : Type*} [Fintype I] [Fintype B]
    (p : ScheduleFree.AdamW.Parameters) (hβ : p.adam.betaOne ≠ 0)
    (t : ℕ) (s : ScheduleFree.AdamW.TrainingState I) (g : I → ℝ)
    (F : Weights I → Weights (A × V)) (J : Weights I →L[ℝ] Weights (A × V))
    (K : Weights I →L[ℝ] Weights B) (r : Weights B)
    (hg : WithLp.toLp 2 g = K.adjoint r) (C : ℝ) (target : A → V)
    (hF : ∀ w ∈ segment ℝ (Rotate90.Learning.evaluated p s)
      (Rotate90.Learning.evaluated p (ScheduleFree.AdamW.trainingStep p t s g)),
      DifferentiableAt ℝ F w)
    (hJ : ∀ w ∈ segment ℝ (Rotate90.Learning.evaluated p s)
      (Rotate90.Learning.evaluated p (ScheduleFree.AdamW.trainingStep p t s g)),
      ‖fderiv ℝ F w - J‖ ≤ C)
    (hmargin : ∀ a v, v ≠ target a →
      2 * (C * ‖Rotate90.Learning.evaluated p (ScheduleFree.AdamW.trainingStep p t s g) -
        Rotate90.Learning.evaluated p s‖) <
      (F (Rotate90.Learning.evaluated p s) + predictionStep p t s g J K r) (a, target a) -
      (F (Rotate90.Learning.evaluated p s) + predictionStep p t s g J K r) (a, v)) :
    ∀ a v, v ≠ target a →
      F (Rotate90.Learning.evaluated p (ScheduleFree.AdamW.trainingStep p t s g)) (a, v) <
      F (Rotate90.Learning.evaluated p (ScheduleFree.AdamW.trainingStep p t s g)) (a, target a) := by
  apply correct_of_prediction_error _ _ target _ _ hmargin
  simpa only [sub_add_eq_sub_sub] using finite_prediction_error p hβ t s g F J K r hg C hF hJ

example := correct_of_prediction_error (A := Fin 2) (V := Fin 2)
  (WithLp.toLp 2 (fun av => if av.1 = av.2 then 3 else 0))
  (WithLp.toLp 2 (fun av => if av.1 = av.2 then 3 else 0)) id 1
  (by simp) (by intro a v hv; have h : a ≠ v := Ne.symm hv; norm_num [h])

example := source_step_correct (I := Fin 1) (B := Fin 1) (A := Fin 2) (V := Fin 2)
  ⟨⟨3/1000, 1, 0, 9/10, 98/100, 1/100000000⟩, 0, 2⟩ (by norm_num) 0
  ⟨0, 0, ⟨0, 0, 0⟩⟩ (fun _ => 0)
  (fun _ => WithLp.toLp 2 (fun av => if av.1 = av.2 then 3 else 0)) 0 0 0
  (by simp; rfl) 0 id (fun _ _ => differentiableAt_const _) (by intro w hw; simp)
  (by
    intro a v hv
    have h : a ≠ v := Ne.symm hv
    norm_num [predictionStep, cross, h])

end Transformer.SequenceModel.Transfer
