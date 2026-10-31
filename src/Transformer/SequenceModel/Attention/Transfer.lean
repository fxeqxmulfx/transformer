module

public import Transformer.SequenceModel.Attention.Regularity
public import Transformer.SequenceModel.Attention.Training
public import Transformer.SequenceModel.Transfer.Step

/-!
# The actual Transformer's cross-Jacobian learning operator

Source: `review/flat_spectrum.md`, *Transformer learning bridges*, T2--T3.
The finite logit table includes independent example IDs and every class.
Its Jacobian differentiates all embeddings and matrices. The outer loss
is ordinary sampled softmax cross-entropy, not a squared-loss surrogate.
-/

@[expose] public section
noncomputable section

namespace Transformer.SequenceModel.Attention

open GPTMini Landscape.Neural ScheduleFree

variable {A : Type*} [Fintype A]

/-- The complete finite logit table, with no rank restriction. -/
def table (cfg : Config) (eps xeps : ℝ) {T : ℕ}
    (angle : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (inputs : A → Fin T → Fin cfg.vocab_size) (readout : A → Fin T)
    (w : Parameters cfg) : Weights (A × Fin cfg.vocab_size) :=
  WithLp.toLp 2 (fun av => score cfg eps xeps angle (readout av.1) w (inputs av.1) av.2)

/-- Ordinary CE as a function of independently variable logits. -/
def tableLoss (cfg : Config) (rows : Finset A) (target : A → Fin cfg.vocab_size)
    (z : Weights (A × Fin cfg.vocab_size)) : ℝ :=
  meanCrossEntropy rows (fun a v => z (a, v)) target

/-- T2: the finite table is jointly analytic at actual regular weights. -/
theorem analyticAt_table (cfg : Config) {eps xeps : ℝ} (heps : 0 < eps) (hxeps : 0 < xeps)
    {T : ℕ} (angle : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (inputs : A → Fin T → Fin cfg.vocab_size) (readout : A → Fin T) (w : Parameters cfg)
    (hreg : ∀ a, Regular cfg eps xeps angle (inputs a) w) :
    AnalyticAt ℝ (table cfg eps xeps angle inputs readout) w := by
  have hs : AnalyticAt ℝ (fun x : Parameters cfg => fun av : A × Fin cfg.vocab_size =>
      score cfg eps xeps angle (readout av.1) x (inputs av.1) av.2) w :=
    AnalyticAt.pi (fun av => analyticAt_score cfg heps hxeps angle (inputs av.1) w
      (hreg av.1) (readout av.1) av.2)
  exact ((PiLp.continuousLinearEquiv 2 ℝ
    (fun _ : A × Fin cfg.vocab_size => ℝ)).symm.analyticAt _).comp hs

/-- T2: CE in logit coordinates is analytic without any network hypothesis. -/
theorem analyticAt_tableLoss (cfg : Config) (rows : Finset A) (target : A → Fin cfg.vocab_size)
    (z : Weights (A × Fin cfg.vocab_size)) : AnalyticAt ℝ (tableLoss cfg rows target) z := by
  exact analyticAt_meanCrossEntropy (P := Weights (A × Fin cfg.vocab_size))
    (logits := fun z a v => z.ofLp (a, v)) (x := z) rows
    (fun a _ v => analyticAt_parameter (I := A × Fin cfg.vocab_size) (a, v) z) target

/-- T2: the actual sampled gradient is J_train^T times the actual CE logit
gradient. Regularity is proved for this full forward, not assumed as a black box. -/
theorem actual_gradient (cfg : Config) {eps xeps : ℝ} (heps : 0 < eps) (hxeps : 0 < xeps)
    {T : ℕ} (angle : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (inputs : A → Fin T → Fin cfg.vocab_size) (readout : A → Fin T)
    (rows : Finset A) (target : A → Fin cfg.vocab_size) (w : Parameters cfg)
    (hreg : ∀ a, Regular cfg eps xeps angle (inputs a) w) :
    gradient (loss (predictor cfg eps xeps angle inputs readout) rows target) w =
      (fderiv ℝ (table cfg eps xeps angle inputs readout) w).adjoint
        (gradient (tableLoss cfg rows target) (table cfg eps xeps angle inputs readout w)) := by
  exact Transfer.gradient_comp
    (analyticAt_table cfg heps hxeps angle inputs readout w hreg).differentiableAt.hasFDerivAt
    (analyticAt_tableLoss cfg rows target _).differentiableAt.hasGradientAt

/-- T3: query and evaluation Jacobians give the true source step's linear
prediction, including averaging and decay. Held-out rows need not be trained. -/
theorem actual_linearized_step (cfg : Config) {eps xeps : ℝ} (heps : 0 < eps) (hxeps : 0 < xeps)
    {T : ℕ} (angle : Fin cfg.n_layers → Fin T → Fin (cfg.head_dim / 2) → ℝ)
    (inputs : A → Fin T → Fin cfg.vocab_size) (readout : A → Fin T)
    (rows : Finset A) (target : A → Fin cfg.vocab_size)
    (p : AdamW.Parameters) (hβ : p.adam.betaOne ≠ 0) (t : ℕ)
    (s : AdamW.TrainingState (Index cfg))
    (hreg : ∀ a, Regular cfg eps xeps angle (inputs a) (WithLp.toLp 2 s.point)) :
    let F := table cfg eps xeps angle inputs readout
    let g := (gradient (loss (predictor cfg eps xeps angle inputs readout) rows target)
      (WithLp.toLp 2 s.point)).ofLp
    let J := fderiv ℝ F (Rotate90.Learning.evaluated p s)
    J (Rotate90.Learning.evaluated p (AdamW.trainingStep p t s g) -
      Rotate90.Learning.evaluated p s) = Transfer.predictionStep p t s g J
        (fderiv ℝ F (WithLp.toLp 2 s.point))
        (gradient (tableLoss cfg rows target) (F (WithLp.toLp 2 s.point))) := by
  dsimp only
  apply Transfer.linearized_source_step p hβ
  exact actual_gradient cfg heps hxeps angle inputs readout rows target _ hreg

private def phases := sourceAngles witnessConfig (T := 3) (fun _ => true) (fun i => i.val)
private def inputs : Fin 3 → Fin 3 → Fin 3 := fun a j => if a = j then 0 else 1

example := analyticAt_table witnessConfig (eps := 1 / 8388608) (xeps := 1 / 1000000000000)
  (by norm_num) (by norm_num) phases inputs (fun _ => 2) 0
  (fun _ => regular_zero _ _ (by norm_num) _ _)
example := actual_gradient witnessConfig (eps := 1 / 8388608) (xeps := 1 / 1000000000000)
  (by norm_num) (by norm_num) phases inputs (fun _ => 2) {0, 1} id 0
  (fun _ => regular_zero _ _ (by norm_num) _ _)
example := actual_linearized_step witnessConfig (eps := 1 / 8388608)
  (xeps := 1 / 1000000000000) (by norm_num) (by norm_num) phases inputs (fun _ => 2) {0, 1} id
  ⟨⟨3/1000, 1, 0, 9/10, 98/100, 1/100000000⟩, 0, 2⟩ (by norm_num) 5
  ⟨fun _ => 0, fun _ => 1, ⟨fun _ => 1, 1, 3/1000⟩⟩
  (fun _ => regular_zero _ _ (by norm_num) _ _)

end Transformer.SequenceModel.Attention
