module

public import Transformer.SequenceModel.Attention.Transfer
public import Transformer.SequenceModel.Attention.Stationary
public import Transformer.SequenceModel.Transfer.Classification

/-!
# Certified finite prediction transfer for the actual normalized Transformer

Source: `review/flat_spectrum.md`, *Transformer learning bridges*, T3.
Query regularity derives the true training gradient. Actual activation
regularity on the evaluation step segment derives differentiability of
every held-out logit. A derivative-variation bound then certifies a finite
prediction error, usable by `Transfer.correct_of_prediction_error`.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention

open GPTMini ScheduleFree

/-- T3: the full nonlinear Transformer obeys the derived cross-Jacobian
prediction with a certified finite remainder. All trainable coordinates,
the actual Adam history and query/evaluation distinction are retained.
No network differentiability or nuclear-norm equivalence is assumed. -/
theorem actual_finite_prediction_error (cfg : Config) {eps xeps : ℝ}
    (heps : 0 < eps) (hxeps : 0 < xeps) {T : ℕ} {A : Type*} [Fintype A]
    (angle : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (inputs : A → Fin T → Fin cfg.vocab_size) (readout : A → Fin T)
    (rows : Finset A) (target : A → Fin cfg.vocab_size)
    (p : AdamW.Parameters) (hβ : p.adam.betaOne ≠ 0) (t : ℕ)
    (s : AdamW.TrainingState (Index cfg)) (C : ℝ)
    (hquery : ∀ a, Regular cfg eps xeps angle (inputs a) (WithLp.toLp 2 s.point)) :
    let F := table cfg eps xeps angle inputs readout
    let g := (gradient (loss (predictor cfg eps xeps angle inputs readout) rows target)
      (WithLp.toLp 2 s.point)).ofLp
    let x := Rotate90.Learning.evaluated p s
    let y := Rotate90.Learning.evaluated p (AdamW.trainingStep p t s g)
    let J := fderiv ℝ F x
    (∀ w ∈ segment ℝ x y, ∀ a, Regular cfg eps xeps angle (inputs a) w) →
    (∀ w ∈ segment ℝ x y, ‖fderiv ℝ F w - J‖ ≤ C) →
    ‖F y - F x - Transfer.predictionStep p t s g J (fderiv ℝ F (WithLp.toLp 2 s.point))
      (gradient (tableLoss cfg rows target) (F (WithLp.toLp 2 s.point)))‖ ≤ C * ‖y - x‖ := by
  dsimp only
  intro hsegment hvariation
  apply Transfer.finite_prediction_error p hβ
  · exact actual_gradient cfg heps hxeps angle inputs readout rows target _ hquery
  · intro w hw
    exact (analyticAt_table cfg heps hxeps angle inputs readout w (hsegment w hw)).differentiableAt
  · exact hvariation

private abbrev cfg := witnessConfig
private def phases := sourceAngles cfg (T := 3) (fun _ => true) (fun j => j.val)
private def inputs : Fin 3 → Fin 3 → Fin 3 := fun a j => if a = j then 0 else 1
private def p₀ : AdamW.Parameters := ⟨⟨3/1000, 1, 0, 9/10, 98/100, 1/100000000⟩, 0, 2⟩

private theorem zero_eval (h : AdamW.History (Index cfg)) :
    Rotate90.Learning.evaluated p₀ ⟨0, 0, h⟩ = 0 := by
  ext i
  simp [Rotate90.Learning.evaluated, AdamW.evaluation]

private theorem zero_step_eval : Rotate90.Learning.evaluated p₀
    (AdamW.trainingStep p₀ 0 (⟨0, 0, ⟨0, 0, 0⟩⟩ : AdamW.TrainingState (Index cfg)) 0) = 0 := by
  ext i
  simp [Rotate90.Learning.evaluated, AdamW.evaluation, AdamW.trainingStep, AdamW.direction]

example := actual_finite_prediction_error cfg (eps := 1 / 8388608)
  (xeps := 1 / 1000000000000) (by norm_num) (by norm_num)
  phases inputs (fun _ => 2) {0, 1} id p₀ (by norm_num [p₀]) 0 ⟨0, 0, ⟨0, 0, 0⟩⟩ 0
  (fun _ => regular_zero _ _ (by norm_num) _ _)
  (by
    dsimp only
    simp only [WithLp.toLp_zero]
    rw [loss_gradient_zero cfg (by norm_num) (by norm_num)]
    intro w hw a
    have hz : w = 0 := by
      simpa only [WithLp.ofLp_zero, zero_eval, zero_step_eval, segment_same, Set.mem_singleton_iff] using hw
    subst w
    exact regular_zero _ _ (by norm_num) _ _)
  (by
    dsimp only
    simp only [WithLp.toLp_zero]
    rw [loss_gradient_zero cfg (by norm_num) (by norm_num)]
    intro w hw
    have hz : w = 0 := by
      simpa only [WithLp.ofLp_zero, zero_eval, zero_step_eval, segment_same, Set.mem_singleton_iff] using hw
    subst w
    rw [zero_eval, sub_self, norm_zero])

end Transformer.SequenceModel.Attention
