module

public import Transformer.Grokking.FiniteRecovery.ArgandCandidateSelection
public import Transformer.Grokking.FiniteRecovery.ArgandFullGradientConvergence

/-! A global optimizer for the full Argand token classifier. Even proposals
can come from any optimizer. Odd proposals cover rational displacements
from the native start. Selection fits the revealed labels and minimizes
the nuclear norm of the model's own decoded phase table. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Set Filter FlatSpectrum

variable {A U V : Type*} [Fintype A] [Nonempty A] [Fintype U] [Fintype V] [DecidableEq V]
  {T H d layers : ℕ}

/-- Alternate the supplied proposals with rational displacements. -/
noncomputable def interleaveWeights (start : Weights A H d layers) (propose : ℕ → Weights A H d layers)
    (t : ℕ) : Weights A H d layers :=
  if t % 2 = 0 then propose (t / 2) else rationalCandidate start (t / 2)

/-- Search the trainable weights directly. Both the score and feasibility
are computed from the candidate network, the known cut, and the given
observations. Tied predictions are ineligible. -/
noncomputable def nuclearSearch (start : Weights A H d layers) (propose : ℕ → Weights A H d layers)
    (repeats : ℕ) (cut : U × V ≃ Word A (T + 1)) (code : A → ℂ)
    (part : Finset (Word A (T + 1) × A)) : ℕ → Weights A H d layers :=
  keepBest start (interleaveWeights start propose) (fitsObserved repeats part)
    (predictedNuclearNorm repeats cut code)

/-- A representable target has an open set of strictly correct weights.
Once a rational proposal hits it, every consistent competitor with a
different predicted function has strictly larger nuclear cost on a
recovering mask. Equality of complete weights follows from retaining
the incumbent on a tie. -/
theorem nuclearSearch_settles (start : Weights A H d layers) (propose : ℕ → Weights A H d layers)
    (repeats : ℕ) (cut : U × V ≃ Word A (T + 1)) (code : A → ℂ) (hc : Function.Injective code)
    (inputs : Finset (Word A (T + 1))) (f : Word A (T + 1) → A)
    (hr : ∀ X ∈ nativePhaseTables A H d layers repeats cut code,
      X ∈ completions (observedCells cut inputs) (targetKernel cut code f) →
      nuclearNorm X ≤ nuclearNorm (targetKernel cut code f) → X = targetKernel cut code f)
    (hrep : ∃ w : Weights A H d layers, (Model.fromWeights w).realizes repeats f) :
    ∃ w : Weights A H d layers, (∀ᶠ t in atTop, nuclearSearch start propose repeats cut code
      (observe inputs f) t = w) ∧ (Model.fromWeights w).realizes repeats f := by
  obtain ⟨i,hi⟩ := (exists_rationalCandidate_iff start repeats f).2 hrep
  let c := interleaveWeights start propose
  let j := 2 * i + 1
  have hreal : (Model.fromWeights (c j)).realizes repeats f := by
    have hodd : j % 2 ≠ 0 := by dsimp [j]; omega
    have hdiv : j / 2 = i := by dsimp [j]; omega
    simpa only [c, interleaveWeights, ite_eq_right hodd, hdiv] using hi
  have hp := (Model.fromWeights (c j)).predictedToken_eq_of_realizes repeats hreal
  have hf : fitsObserved repeats (observe inputs f) (c j) := by
    change (Model.fromWeights (c j)).realizes repeats _ ∧ _
    rw [hp]
    exact ⟨hreal,fun p hm => (mem_observe.mp hm).2⟩
  let J := predictedNuclearNorm (H := H) (d := d) (layers := layers) repeats cut code
  have he : J (c j) = nuclearNorm (targetKernel cut code f) :=
    congrArg (fun g => nuclearNorm (targetKernel cut code g)) hp
  have hopt := fitsObserved_nuclear_optimum (H := H) (d := d) (layers := layers) repeats cut code hc inputs f hr
  obtain ⟨w, hw, hcost, hstop⟩ := keepBest_settles start c (fitsObserved repeats (observe inputs f))
    J j hf (fun v hv => he.trans_le (hopt v hv).1)
  exact ⟨w, (eventually_ge_atTop (j+1)).mono (fun t ht => hstop t ht),
    (hopt w hw).2.mp (hcost.trans he).le⟩

/-- Use the actual clipped full-batch momentum trajectory for the even
proposals, including all native gates, WSD rates and coupled penalty.
No hidden label is passed to the gradient or to the selection rule. -/
noncomputable def transversalSearch (realScoreInit : Bool) (repeats : ℕ)
    (cut : U × V ≃ Word A (T + 1)) (code : A → ℂ) (part : Finset (Word A (T + 1) × A))
    (lam : ℝ) (rate : ℕ → ℝ) (draw : ComplexParameter A H d layers → ℂ) : ℕ → Weights A H d layers :=
  let batch : List (LabeledPosition A) := part.toList.map fun p => ⟨T, p.1, Fin.last T, p.2⟩
  let fit := fullGradientMomentumRun realScoreInit repeats batch lam rate draw
  nuclearSearch (sgdInitialWeights realScoreInit draw) (fun t => (fit (t+1)).weights) repeats cut code part

@[simp] theorem transversalSearch_zero (realScoreInit : Bool) (repeats : ℕ)
    (cut : U × V ≃ Word A (T + 1)) (code : A → ℂ) (part : Finset (Word A (T + 1) × A))
    (lam : ℝ) (rate : ℕ → ℝ) (draw : ComplexParameter A H d layers → ℂ) :
    transversalSearch realScoreInit repeats cut code part lam rate draw 0 = sgdInitialWeights realScoreInit draw := rfl

end Transformer.Grokking.FiniteRecovery.Argand
