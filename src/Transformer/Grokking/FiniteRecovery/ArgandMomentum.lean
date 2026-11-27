module

public import Transformer.Grokking.FiniteRecovery.ArgandBatchDifferentiability
public import Transformer.Grokking.FiniteRecovery.NativeConvergence
public import Transformer.Optimization.MomentumTail

/-! The complete Argand model with its default Kernite momentum SGD.
The equations apply to exact real parameters and the infinite continuation
of the WSD schedule. The actual runner stops at a finite budget. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Filter Transformer.Optimization
open scoped Topology

variable {A : Type*} {H d layers T : ℕ}

/-- `realScoreInit` starts the score multiplier at one, as the runner does.
Otherwise it starts at 1+i, the runner default at commit 7399c3bb. Other
coordinates follow the same supplied Gaussian draws and zero settings in both runs. -/
def sgdInitialWeights (realScoreInit : Bool) (draw : ComplexParameter A H d layers → ℂ) :
    Weights A H d layers
  | .inl (.score _, true) => if realScoreInit then 0 else 1
  | p => initialWeights draw p

theorem sgdInitialWeights_score (realScoreInit : Bool) (draw : ComplexParameter A H d layers → ℂ) (l : Fin layers) :
    complexEntry (sgdInitialWeights realScoreInit draw) (.score l) = if realScoreInit then 1 else 1 + Complex.I := by
  cases realScoreInit <;> apply Complex.ext <;> norm_num [complexEntry, sgdInitialWeights, initialWeights]

noncomputable def nativeSgdTrainingRun [Fintype A] (realScoreInit : Bool) (repeats : ℕ)
    (batches : ℕ → List (LabeledPosition A)) (rate : ℕ → ℝ)
    (draw : ComplexParameter A H d layers → ℂ) : ℕ → ClippedMomentum.State (Parameter A H d layers) :=
  ClippedMomentum.run rate (fun t => batchGradient repeats (batches t)) (sgdInitialWeights realScoreInit draw)

/-- At the native positive WSD floor, convergence of every weight is
equivalent to convergence of the accumulated clipped gradients. -/
theorem nativeSgd_wsd_converges_iff_gradient_sums [Fintype A] (realScoreInit : Bool) (repeats budget : ℕ)
    (batches : ℕ → List (LabeledPosition A)) (draw : ComplexParameter A H d layers → ℂ)
    (hw : budget * 2 / 100 ≤ budget - 1) (hd : budget * 80 / 100 < budget - 1) :
    let fit := nativeSgdTrainingRun realScoreInit repeats batches (wsdRate budget) draw
    (∃ w, Tendsto (fun t => (fit t).weights) atTop (𝓝 w)) ↔
      ∀ i, ∃ G : ℝ, Tendsto (fun n => ∑ j ∈ Finset.range n,
        Transformer.AMSGradW.Kernite.clip
          (batchGradient repeats (batches (budget - 1 + j)) (fit (budget - 1 + j)).weights) i) atTop (𝓝 G) :=
  ClippedMomentum.weights_tendsto_iff_tail_gradient_sum _ _ _ _ (by norm_num)
    (fun _ h => wsdRate_eq_floor hw hd h)

theorem nativeSgd_gradient_tendsto_zero [Fintype A] (realScoreInit : Bool) (repeats : ℕ)
    (batches : ℕ → List (LabeledPosition A)) (rate : ℕ → ℝ)
    (draw : ComplexParameter A H d layers → ℂ) {η : ℝ} (hη : 0 < η)
    (hr : Tendsto rate atTop (𝓝 η)) (w : Weights A H d layers) :
    let fit := nativeSgdTrainingRun realScoreInit repeats batches rate draw
    Tendsto (fun t => (fit t).weights) atTop (𝓝 w) →
      Tendsto (fun t => batchGradient repeats (batches t) (fit t).weights) atTop (𝓝 0) := by
  dsimp only
  intro h
  exact ClippedMomentum.tendsto_zero_of_clip
    (ClippedMomentum.zero_limits_of_weights_tendsto _ _ _ w hη hr h).2

/-- A recurring minibatch must be stationary at any finite limit of the
full trainable model. No differentiability assumption on a trajectory is needed. -/
theorem nativeSgd_recurringBatch_stationary [Fintype A] (realScoreInit : Bool) (repeats : ℕ)
    (batches : ℕ → List (LabeledPosition A)) (rate : ℕ → ℝ)
    (draw : ComplexParameter A H d layers → ℂ) {η : ℝ} (hη : 0 < η) (hr : Tendsto rate atTop (𝓝 η))
    (w : Weights A H d layers)
    (h : Tendsto (fun t => (nativeSgdTrainingRun realScoreInit repeats batches rate draw t).weights) atTop (𝓝 w))
    (batch : List (LabeledPosition A)) (hb : ∃ᶠ t in atTop, batches t = batch) :
    batchGradient repeats batch w = 0 := by
  obtain ⟨u, hu, hbatch⟩ := extraction_of_frequently_atTop hb
  have hG := nativeSgd_gradient_tendsto_zero realScoreInit repeats batches rate draw hη hr w h
  apply tendsto_nhds_unique ((continuous_batchGradient repeats batch).tendsto w |>.comp
    (h.comp hu.tendsto_atTop))
  apply (hG.comp hu.tendsto_atTop).congr
  intro t
  simp only [Function.comp_def, hbatch t]

example (batches : ℕ → List (LabeledPosition (Fin 256))) (draw : ComplexParameter (Fin 256) 4 32 1 → ℂ) :=
  nativeSgd_wsd_converges_iff_gradient_sums false 1 20000 batches draw (by decide) (by decide)

example (draw : ComplexParameter (Fin 256) 4 32 1 → ℂ) := sgdInitialWeights_score false draw 0

end Transformer.Grokking.FiniteRecovery.Argand
