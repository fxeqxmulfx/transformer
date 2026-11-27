module

public import Transformer.Grokking.FiniteRecovery.ArgandMomentum
public import Transformer.Grokking.FiniteRecovery.ArgandSamples

/-! What weight convergence implies for the complete Argand predictor.
Cross-entropy at a finite weight limit is positive. A strict decoder at
that limit is eventually recovered on the entire finite word domain. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Filter Transformer.Optimization
open scoped Topology

variable {A : Type*} [Fintype A] {H d layers T : ℕ}

/-- Every fixed prediction loss converges along a convergent sequence
of full weight vectors. The finite limiting loss is strictly positive. -/
theorem labelLoss_at_finite_weight_limit [Nontrivial A]
    {weights : ℕ → Weights A H d layers} {limit : Weights A H d layers}
    (hw : Tendsto weights atTop (𝓝 limit)) (repeats : ℕ) (word : Word A T) (i : Fin T) (a : A) :
    let loss w := -Real.log ((Model.fromWeights w).probability repeats word i a)
    Tendsto (fun t => loss (weights t)) atTop (𝓝 (loss limit)) ∧ 0 < loss limit := by
  let model := Model.fromWeights limit
  let B := ∑ a, ‖model.embedding a‖
  have hB (b : A) : ‖model.embedding b‖ ≤ B :=
    Finset.single_le_sum (fun k _ => norm_nonneg (model.embedding k)) (Finset.mem_univ b)
  have h := model.lossGe_of_embeddingBound (by norm_num [model, Model.fromWeights]) hB repeats word i a
  exact ⟨(continuous_labelLoss_fromWeights repeats word i a).tendsto limit |>.comp hw,
    (Real.log_pos (lt_add_of_pos_right 1 (Real.exp_pos _))).trans_le h⟩

/-- A strict limiting prediction is preserved eventually, simultaneously
for every word. This applies to held-out words as well as observed ones. -/
theorem realizes_eventually_of_weight_limit
    {weights : ℕ → Weights A H d layers} {limit : Weights A H d layers}
    (hw : Tendsto weights atTop (𝓝 limit)) (repeats : ℕ) (target : Word A (T + 1) → A)
    (h : (Model.fromWeights limit).realizes repeats target) :
    ∀ᶠ t in atTop, (Model.fromWeights (weights t)).realizes repeats target := by
  unfold Model.realizes at h ⊢
  apply eventually_all.mpr
  intro word
  apply eventually_all.mpr
  intro a
  by_cases ha : a = target word
  · exact Eventually.of_forall fun _ hne => (hne ha).elim
  · have hlog (b : A) :=
      (continuous_logits_fromWeights repeats word (Fin.last T) b).tendsto limit |>.comp hw
    exact ((hlog a).eventually_lt (hlog (target word)) (h word a ha)).mono fun _ ht _ => ht

/-- A summable continuation guarantees finite weights and per-position
losses for every native initialization and every minibatch sequence. -/
theorem nativeSgd_tendsto_of_summable_rate (realScoreInit : Bool) (repeats : ℕ)
    (batches : ℕ → List (LabeledPosition A)) (rate : ℕ → ℝ)
    (draw : ComplexParameter A H d layers → ℂ) (hr : ∀ t, 0 ≤ rate t) (hsum : Summable rate) :
    let fit := nativeSgdTrainingRun realScoreInit repeats batches rate draw
    ∃ w : Weights A H d layers, Tendsto (fun t => (fit t).weights) atTop (𝓝 w) ∧
      ∀ (word : Word A T) (i : Fin T) (a : A),
        Tendsto (fun t => -Real.log ((Model.fromWeights (fit t).weights).probability repeats word i a))
          atTop (𝓝 (-Real.log ((Model.fromWeights w).probability repeats word i a))) := by
  obtain ⟨w, ht⟩ := ClippedMomentum.weights_tendsto_of_summable rate
    (fun t => batchGradient repeats (batches t)) (sgdInitialWeights realScoreInit draw) hr hsum
  exact ⟨w, ht, fun word i a => (continuous_labelLoss_fromWeights repeats word i a).tendsto w |>.comp ht⟩

example (w : Weights (Fin 2) 4 16 1) :=
  labelLoss_at_finite_weight_limit (weights := fun _ => w) tendsto_const_nhds
    1 (fun _ : Fin 8 => 0) 0 0

example (batches : ℕ → List (LabeledPosition (Fin 64))) (draw : ComplexParameter (Fin 64) 4 16 1 → ℂ) :=
  nativeSgd_tendsto_of_summable_rate (T := 32) false 1 batches (fun t => (1 / 2 : ℝ) ^ t) draw
    (fun _ => by positivity) (summable_geometric_of_lt_one (by norm_num) (by norm_num))

end Transformer.Grokking.FiniteRecovery.Argand
