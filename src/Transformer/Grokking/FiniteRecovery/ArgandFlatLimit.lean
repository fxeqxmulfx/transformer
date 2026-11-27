module

public import Transformer.Grokking.FiniteRecovery.ArgandEnergyRecovery
public import Transformer.Grokking.FiniteRecovery.ArgandStrictRecovery

/-! Transfer a Flat Spectrum certificate to the limit of full-gradient
momentum. The two inputs are a finite-prefix loss measurement and a
nuclear norm bound over a weight ball of known radius. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Filter MeasureTheory FlatSpectrum
open scoped Topology

variable {A U V : Type*} [Fintype A] {T H d layers : ℕ}

/-- The native batch objective on the selected last positions is exactly
the mean cross-entropy on the observed graph, also for an empty mask. -/
theorem batchLoss_observedLastPositions (repeats : ℕ) (target : Word A (T + 1) → A)
    (inputs : Finset (Word A (T + 1))) (w : Weights A H d layers) :
    batchLoss repeats (observedLastPositions target inputs) w =
      (Model.fromWeights w).observedLoss repeats (observe inputs target) := by
  simp [batchLoss, observedLastPositions, Model.observedLoss, Model.labelLoss, observe,
    graphEmbedding, Finset.sum_map_toList, List.map_map, Function.comp_def]
  rfl

variable [DecidableEq A] [Nonempty A] [Fintype U] [Fintype V] [DecidableEq V]

/-- A bound on the phase table throughout a radius R and a strict
finite-prefix objective certificate imply correct hidden logits of the
full weight limit on an open event. Neither hypothesis uses limit weights. -/
theorem flatSpectrum_momentum_recovery_of_prefix
    (mu : Measure (ComplexParameter A H d layers → ℂ)) [mu.IsOpenPosMeasure]
    (realScoreInit : Bool) (repeats : ℕ) (cut : U × V ≃ Word A (T + 1))
    (code : A → ℂ) (hinj : Function.Injective code) (hunit : ∀ a, ‖code a‖ = 1)
    (target : Word A (T + 1) → A) (inputs : Finset (Word A (T + 1)))
    (hunique : NuclearRecovers (observedCells cut inputs) (targetKernel cut code target)) :
    ∃ ε > 0, ∀ (R lam : ℝ), 0 ≤ R → 0 < lam →
      (∀ w : WeightSpace A H d layers, ‖w‖ ≤ R →
        nuclearNorm ((Model.fromWeights (WithLp.ofLp w)).phaseKernel repeats cut code) ≤
          nuclearNorm (targetKernel cut code target) + ε) →
      ∀ (rate : ℕ → ℝ) (N : ℕ) (start : ComplexParameter A H d layers → ℂ),
        let batch := observedLastPositions target inputs
        fullGradientObjective repeats batch lam
          (WithLp.toLp 2 (fullGradientMomentumRun realScoreInit repeats batch lam rate start N).weights) <
            min ε ((lam / 2) * R ^ 2) →
        ∃ S : Set (ComplexParameter A H d layers → ℂ), IsOpen S ∧ start ∈ S ∧ 0 < mu S ∧
          ∃ η₀ > 0, ∀ η, 0 < η → η ≤ η₀ → ∀ draw ∈ S,
            ∃ star : Weights A H d layers,
              Tendsto (fun t => (fullGradientMomentumRun realScoreInit repeats batch lam
                (fun j => if j < N then rate j else η) draw t).weights) atTop (𝓝 star) ∧
              (Model.fromWeights star).realizes repeats target := by
  obtain ⟨ε, hε, hcert⟩ := exists_observedLoss_nuclear_slack_for_realizes
    (H := H) (d := d) (layers := layers) repeats cut code hinj hunit target inputs hunique
  refine ⟨ε, hε, fun R lam hR hlam hnorm rate N start hstart => ?_⟩
  let batch := observedLastPositions target inputs
  let cap := min ε ((lam / 2) * R ^ 2)
  apply fullGradientMomentumRun_posMeasure_of_sublevel mu realScoreInit repeats batch hlam
    rate N cap start hstart (fun w => (Model.fromWeights w).realizes repeats target)
  intro w hw
  have hs := (fullGradientObjective_coercive repeats batch lam w).trans (hw.trans (min_le_right _ _))
  have hsq := (mul_le_mul_iff_right₀ (half_pos hlam)).mp hs
  have hr : ‖w‖ ≤ R := by nlinarith only [hsq, hR, norm_nonneg w]
  apply hcert _ _ (hnorm w hr)
  rw [← batchLoss_observedLastPositions]
  have hce : batchLoss repeats batch (WithLp.ofLp w) ≤ fullGradientObjective repeats batch lam w :=
    le_add_of_nonneg_right (by positivity : 0 ≤ (lam / 2) * ‖w‖ ^ 2)
  exact hce.trans (hw.trans (min_le_left _ _))

/-- The staircase certificate for the eight-variable modular sum supplies
the unique-completion hypothesis for the actual 31 observed inputs. -/
example (mu : Measure (ComplexParameter (ZMod 2) 4 16 1 → ℂ)) [mu.IsOpenPosMeasure] :=
  let e := wordEnumeration 2 4
  flatSpectrum_momentum_recovery_of_prefix mu true 1 (T := 7) (enumeratedCut e)
    ZMod.stdAddChar ZMod.injective_stdAddChar (fun _ => by simp) (modularSum e)
    (inputsAtCells (enumeratedCut e) (staircase 15))
    (by simpa only [observedCells_inputsAtCells] using modularSum_nuclear_recovers e)

end Transformer.Grokking.FiniteRecovery.Argand
