module

public import Transformer.Grokking.FiniteRecovery.ArgandMomentumLimits
public import Transformer.Grokking.FiniteRecovery.UnusedEmbedding

/-! With the original positive WSD floor, a finite stationary limit
cannot realize the binary last-token task from recurring singleton
observations of its two constant words. All model parameters are free. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Filter Transformer.PhasorV2
open scoped Topology InnerProductSpace

variable {H d layers T : ℕ}

/-- One observation for each final token, presented in alternate
singleton batches. Both labels are consistent with last-token copying. -/
def lastCopySingletons (length step : ℕ) : List (LabeledPosition Bool) :=
  let token := decide (step % 2 = 1)
  [⟨length, fun _ => token, Fin.last length, token⟩]

theorem lastCopySingletons_recurs (length : ℕ) (token : Bool) :
    ∃ᶠ step in atTop, lastCopySingletons length step = [⟨length, fun _ => token, Fin.last length, token⟩] := by
  apply frequently_atTop.mpr
  intro n
  cases token
  · exact ⟨2 * n, by omega, by simp [lastCopySingletons]⟩
  · exact ⟨2 * n + 1, by omega, by simp [lastCopySingletons]⟩

/-- A finite weight limit predicts both tokens uniformly on either
of the two observed words, for every Gaussian tensor draw. -/
theorem nativeSgd_lastCopy_finite_limit_uniform (realScoreInit : Bool) (repeats budget : ℕ)
    (draw : ComplexParameter Bool H d layers → ℂ)
    (hw : budget * 2 / 100 ≤ budget - 1) (hd : budget * 80 / 100 < budget - 1)
    (w : Weights Bool H d layers)
    (h : Tendsto (fun t => (nativeSgdTrainingRun realScoreInit repeats (lastCopySingletons T)
      (wsdRate budget) draw t).weights) atTop (𝓝 w)) (token : Bool) :
    (Model.fromWeights w).hidden repeats (fun _ : Fin (T + 1) => token) (Fin.last T) = 0 ∧
      ∀ a, (Model.fromWeights w).probability repeats (fun _ : Fin (T + 1) => token) (Fin.last T) a = 1 / 2 := by
  have hg := nativeSgd_recurringBatch_stationary realScoreInit repeats (lastCopySingletons T)
    (wsdRate budget) draw (by norm_num) (wsdRate_tendsto_floor budget hw hd) w h
    _ (lastCopySingletons_recurs T token)
  have hz := stationary_singleton_hidden_eq_zero w repeats (fun _ : Fin (T + 1) => token) (Fin.last T)
    token (!token) (by cases token <;> decide) (fun _ => by cases token <;> decide) hg
  refine ⟨hz, fun a => ?_⟩
  simp [Model.probability, Model.logits, Model.readout, hz]

/-- Finite parameter convergence to a strict solution is impossible
on this consistent copying task with the original WSD continuation. -/
theorem nativeSgd_lastCopy_no_finite_correct_limit (realScoreInit : Bool) (repeats budget : ℕ)
    (draw : ComplexParameter Bool H d layers → ℂ)
    (hw : budget * 2 / 100 ≤ budget - 1) (hd : budget * 80 / 100 < budget - 1) :
    ¬ ∃ w, Tendsto (fun t => (nativeSgdTrainingRun realScoreInit repeats (lastCopySingletons T)
      (wsdRate budget) draw t).weights) atTop (𝓝 w) ∧
        (Model.fromWeights w).realizes repeats (lastTokenTask (T := T) none) := by
  rintro ⟨w, hlim, hcorrect⟩
  have hz := (nativeSgd_lastCopy_finite_limit_uniform realScoreInit repeats budget draw hw hd w hlim false).1
  have hlt := hcorrect (fun _ => false) true (by simp [lastTokenTask])
  simp [Model.logits, Model.readout, hz, lastTokenTask] at hlt

example (draw : ComplexParameter Bool 4 16 1 → ℂ) :=
  nativeSgd_lastCopy_no_finite_correct_limit (T := 31) false 1 20000 draw (by decide) (by decide)

end Transformer.Grokking.FiniteRecovery.Argand
