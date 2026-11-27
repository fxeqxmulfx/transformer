module

public import Transformer.AMSGradW.AlternatingMomentum
public import Transformer.Grokking.FiniteRecovery.NativeConvergence

/-! The common-gradient criterion specialized to the jointly trained
Argand model from its native initialization. Recurring minibatches
constrain any candidate weight limit. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Filter Transformer.AMSGradW
open scoped Topology

variable {A : Type*} {H d layers T : ℕ}

theorem nativeTrainingRun_weights_tendsto_iff_clippedGrad [Fintype A] (repeats budget : ℕ)
    (batches : ℕ → List (LabeledPosition A)) (draw : ComplexParameter A H d layers → ℂ)
    (hw : budget * 2 / 100 ≤ budget - 1) (hd : budget * 80 / 100 < budget - 1) :
    let fit := nativeTrainingRun repeats batches (wsdRate budget) draw
    (∃ w, Tendsto (fun t => (fit t).weights) atTop (𝓝 w)) ↔
      ∃ G, Tendsto (fun t => Kernite.clip (batchGradient repeats (batches t) (fit t).weights)) atTop (𝓝 G) :=
  Kernite.run_weights_tendsto_iff_clippedGrad _ _ _
    (fun t => ⟨(wsdRate_bounds budget t).1, (wsdRate_bounds budget t).2.trans (by norm_num)⟩)
    (by norm_num) (wsdRate_tendsto_floor budget hw hd)

/-- Every infinitely recurring batch has the same clipped gradient at a
weight limit, provided its clipped gradient is continuous there. -/
theorem nativeTrainingRun_recurringBatch_gradient [Fintype A] (repeats budget : ℕ)
    (batches : ℕ → List (LabeledPosition A)) (draw : ComplexParameter A H d layers → ℂ)
    (hw : budget * 2 / 100 ≤ budget - 1) (hd : budget * 80 / 100 < budget - 1)
    (w : Weights A H d layers)
    (h : Tendsto (fun t => (nativeTrainingRun repeats batches (wsdRate budget) draw t).weights) atTop (𝓝 w)) :
    ∃ V : Weights A H d layers,
      ∀ batch, (∃ᶠ t in atTop, batches t = batch) →
        ContinuousAt (fun w => Kernite.clip (batchGradient repeats batch w)) w →
        Kernite.clip (batchGradient repeats batch w) =
          fun i => -(1 / 10 : ℝ) * (Real.sqrt (V i) + 1 / 100000000) * w i := by
  obtain ⟨V, _, hG⟩ := nativeTrainingRun_clippedGrad_of_weightLimit repeats budget batches draw hw hd w h
  refine ⟨V, ?_⟩
  intro batch hb hc
  obtain ⟨u, hu, hbatch⟩ := extraction_of_frequently_atTop hb
  apply tendsto_nhds_unique (hc.tendsto.comp (h.comp hu.tendsto_atTop))
  apply (hG.comp hu.tendsto_atTop).congr
  intro t
  simp only [Function.comp_def, hbatch t]

private theorem loss_zeroWeights [Fintype A] (repeats : ℕ) (word : Word A T) (i : Fin T) (a : A) :
    -Real.log ((Model.fromWeights (0 : Weights A H d layers)).probability repeats word i a) =
      Real.log (Fintype.card A) := by
  have he (b : A) : (Model.fromWeights (0 : Weights A H d layers)).embedding b = 0 := by
    ext j
    exact Complex.ext rfl rfl
  simp [Model.probability, Model.logits, Model.readout, he, one_div, Real.log_inv]

/-- A zero common gradient sends all weights to zero, because native
decoupled decay is positive. The limiting softmax is uniform. -/
theorem nativeTrainingRun_zeroClippedGrad [Fintype A] (repeats budget : ℕ)
    (batches : ℕ → List (LabeledPosition A)) (draw : ComplexParameter A H d layers → ℂ)
    (hw : budget * 2 / 100 ≤ budget - 1) (hd : budget * 80 / 100 < budget - 1) :
    let fit := nativeTrainingRun repeats batches (wsdRate budget) draw
    Tendsto (fun t => Kernite.clip (batchGradient repeats (batches t) (fit t).weights)) atTop (𝓝 0) →
      Tendsto (fun t => (fit t).weights) atTop (𝓝 0) ∧
      ∀ (word : Word A T) (i : Fin T) (a : A),
        Tendsto (fun t => -Real.log ((Model.fromWeights (fit t).weights).probability repeats word i a))
          atTop (𝓝 (Real.log (Fintype.card A))) := by
  dsimp only
  intro hG
  obtain ⟨V, _, hlim⟩ := Kernite.run_weights_tendsto_of_clippedGrad (wsdRate budget)
    (fun t => batchGradient repeats (batches t)) (initialWeights draw) 0
    (fun t => ⟨(wsdRate_bounds budget t).1, (wsdRate_bounds budget t).2.trans (by norm_num)⟩)
    (by norm_num) (wsdRate_tendsto_floor budget hw hd) hG
  have hz (i : Parameter A H d layers) : (-10 : ℝ) * (0 : Weights A H d layers) i /
        (Real.sqrt (V i) + 1 / 100000000) = 0 := by simp
  have hzero : Tendsto (fun t => (nativeTrainingRun repeats batches (wsdRate budget) draw t).weights)
      atTop (𝓝 0) := by
    simpa only [nativeTrainingRun, hz, ← Pi.zero_def] using hlim
  refine ⟨hzero, fun word i a => ?_⟩
  simpa only [Function.comp_def, loss_zeroWeights] using
    ((continuous_labelLoss_fromWeights (H := H) (d := d) (layers := layers) repeats word i a).tendsto 0).comp hzero

example (batches : ℕ → List (LabeledPosition (Fin 256))) (draw : ComplexParameter (Fin 256) 4 32 1 → ℂ) :=
  nativeTrainingRun_weights_tendsto_iff_clippedGrad 1 20000 batches draw (by decide) (by decide)

example (initial : Fin 1 → ℝ) :=
  Kernite.alternatingQuadratic_no_common_gradient (wsdRate 20000) initial
    (fun t => ⟨(wsdRate_bounds _ t).1, (wsdRate_bounds _ t).2.trans (by norm_num)⟩)
    (by norm_num) (wsdRate_tendsto_floor 20000 (by decide) (by decide))

example (initial : ℝ) :=
  Kernite.alternatingMomentum_not_tendsto (wsdRate 20000) (9 / 10) 0 (1 / 10) initial
    (by norm_num) (wsdRate_tendsto_floor 20000 (by decide) (by decide)) (by norm_num)

private theorem clip_empty_batch [Fintype A] (repeats : ℕ) (w : Weights A H d layers) :
    Kernite.clip (batchGradient repeats [] w) = 0 := by
  have he : (fun x : Kernite.Space (Parameter A H d layers) =>
      batchLoss repeats [] (WithLp.ofLp x)) = fun _ => 0 := by
    funext x
    simp [batchLoss]
  unfold batchGradient
  rw [he, gradient_fun_const]
  funext i
  simp [Kernite.clip]

example (draw : ComplexParameter (Fin 2) 1 1 1 → ℂ) := by
  have hG : Tendsto (fun t => Kernite.clip (batchGradient 1 []
      (nativeTrainingRun 1 (fun _ => []) (wsdRate 20000) draw t).weights)) atTop (𝓝 0) := by
    simp only [clip_empty_batch, tendsto_const_nhds]
  have hw := (nativeTrainingRun_zeroClippedGrad (T := 1) 1 20000 (fun _ => []) draw
    (by decide) (by decide) hG).1
  exact nativeTrainingRun_recurringBatch_gradient 1 20000 (fun _ => []) draw (by decide) (by decide) 0 hw

end Transformer.Grokking.FiniteRecovery.Argand
