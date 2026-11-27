module

public import Transformer.Grokking.FiniteRecovery.ArgandFullGradient
public import Transformer.Grokking.FiniteRecovery.ArgandBatchAnalyticLift
public import Transformer.Grokking.FiniteRecovery.Examples
public import Transformer.Optimization.AnalyticLiftMomentum

/-! All real trainable coordinates of full Argand converge on a fixed
finite batch with positive coupled decay and a sufficiently small
constant tail. The exact norm-plus-offset RMS lift supplies the local
energy inequality, including at zero activations. -/

@[expose] public section

open Filter
open scoped Topology

namespace Transformer.Grokking.FiniteRecovery.Argand

open Transformer.Optimization Transformer.Grokking.FlatSpectrum

variable {A : Type*} [Fintype A] {H d layers : ℕ}

/-- The full real-coordinate momentum orbit converges from every
native parameter draw on every fixed finite list of observations.
A finite WSD prefix may be used before choosing a constant tail rate. -/
theorem fullGradientMomentumRun_converges (realScoreInit : Bool) (repeats : ℕ)
    (batch : List (LabeledPosition A)) {lam : ℝ} (hlam : 0 < lam)
    (draw : ComplexParameter A H d layers → ℂ) (prefixRate : ℕ → ℝ) (N : ℕ) :
    ∃ η₀ > 0, ∀ η, 0 < η → η ≤ η₀ →
      let f := fullGradientObjective repeats batch lam
      let run := fullGradientMomentumRun realScoreInit repeats batch lam
        (fun t => if t < N then prefixRate t else η) draw
      let x := fun t => WithLp.toLp 2 (run t).weights
      let v := fun t => WithLp.toLp 2 (run t).velocity
      ∃ star : WeightSpace A H d layers,
        Tendsto x atTop (𝓝 star) ∧ Tendsto v atTop (𝓝 0) ∧ gradient f star = 0 ∧
        Tendsto (fun t => f (x t)) atTop (𝓝 (f star)) ∧
        Tendsto (fun t => gradient f (x t)) atTop (𝓝 0) := by
  obtain ⟨η₀, hη₀, hrun⟩ :=
    fullGradientMomentumRun_limits realScoreInit repeats batch hlam draw prefixRate N
  have hLift := analyticLift_fullGradientObjective (H := H) (d := d) (layers := layers) repeats batch lam
  refine ⟨η₀, hη₀, fun η hη hη' => ?_⟩
  obtain ⟨star, _, _, _, hstat, hloss, hvel, hgrad, hlim⟩ := hrun η hη hη'
  exact ⟨star, hlim (hLift.momentumEnergyInequality hη.le star), hvel, hstat, hloss, hgrad⟩

example (batch : List (LabeledPosition (Fin 16))) (draw : ComplexParameter (Fin 16) 4 16 2 → ℂ) :=
  fullGradientMomentumRun_converges false 2 batch (show 0 < (1 / 10 : ℝ) by norm_num)
    draw (wsdRate 20000) 19999

/-- Use each observed word once, with its final position as the label site. -/
noncomputable def observedLastPositions {T : ℕ} (target : Word A (T + 1) → A)
    (inputs : Finset (Word A (T + 1))) : List (LabeledPosition A) :=
  inputs.toList.map fun w => ⟨T, w, Fin.last T, target w⟩

example (draw : ComplexParameter (ZMod 2) 4 16 1 → ℂ) :=
  let e := wordEnumeration 2 4
  fullGradientMomentumRun_converges false 1
    (observedLastPositions (T := 7) (modularSum e) (inputsAtCells (enumeratedCut e) (staircase 15)))
    (show 0 < (1 / 10 : ℝ) by norm_num) draw (wsdRate 20000) 19999

example (draw : ComplexParameter (ZMod 2) 4 16 1 → ℂ) :=
  let e := wordEnumeration 2 8
  fullGradientMomentumRun_converges false 1
    (observedLastPositions (T := 15) (modularSum e) (inputsAtCells (enumeratedCut e) (staircase 255)))
    (show 0 < (1 / 10 : ℝ) by norm_num) draw (wsdRate 20000) 19999

example (draw : ComplexParameter (ZMod 2) 4 16 1 → ℂ) :=
  let e := wordEnumeration 2 16
  fullGradientMomentumRun_converges false 1
    (observedLastPositions (T := 31) (modularSum e) (inputsAtCells (enumeratedCut e) (staircase 65535)))
    (show 0 < (1 / 10 : ℝ) by norm_num) draw (wsdRate 20000) 19999

end Transformer.Grokking.FiniteRecovery.Argand
