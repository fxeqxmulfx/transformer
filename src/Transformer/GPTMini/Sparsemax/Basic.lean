import Transformer.GPTMini.Convex.Basic
import Mathlib.Analysis.Calculus.FDeriv.Const
import Mathlib.Analysis.Calculus.FDeriv.Congr

/-!
# Saturated regions of the actual causal sparsemax row

Extension of arXiv:2211.11052v1, §3.1, using the content-score replacement
already defined in `GPTMini.Convex.Basic`. The paper's convex positional
model is different. These results concern real-valued inference and the
backpropagation path through one row, not an entire PyTorch trajectory.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open Filter
open scoped Topology

/-- A unit score gap selects one allowed position exactly. Extension of
arXiv:2211.11052v1, §3.1: the existing quadratic penalty is `sum a²/4`,
so its half-unit cost gap is a unit gap at the original score scale. -/
theorem sparseWeights_eq_basis_of_gap {T : ℕ} (scores : Fin T → ℝ)
    (i winner : Fin T) (hw : winner ≤ i)
    (hgap : ∀ j, j ≤ i → j ≠ winner → scores j + 1 ≤ scores winner) :
    sparseWeights scores i = basis winner := by
  apply routing_minimizer_unique {j : Fin T | j ≤ i}
    (fun j => -scores j / 2) winner
  · intro j hj hne
    have h := hgap j hj hne
    linarith
  · exact (sparseWeights_spec scores i).1
  · exact (sparseWeights_spec scores i).2 _
      (basis_mem_simplex _ winner hw)

/-- The unit-gap premises are realized with two allowed positions.
Source context: §3.1's simplex, with the repository's causal mask. -/
example : ∃ scores : Fin 2 → ℝ, ∃ i winner : Fin 2,
    winner ≤ i ∧ ∀ j, j ≤ i → j ≠ winner → scores j + 1 ≤ scores winner := by
  refine ⟨fun j => if j = 0 then 2 else 0, 1, 0, by decide, ?_⟩
  intro j _ hj
  simp [hj]

/-- A strict unit gap gives a neighborhood on which the complete row is
constant. Extension of arXiv:2211.11052v1, §3.1; strictness excludes the
nondifferentiable support boundary. -/
theorem sparseWeights_eventually_eq_basis {T : ℕ} (scores : Fin T → ℝ)
    (i winner : Fin T) (hw : winner ≤ i)
    (hgap : ∀ j, j ≤ i → j ≠ winner → scores j + 1 < scores winner) :
    ∀ᶠ z in 𝓝 scores, sparseWeights z i = basis winner := by
  have heach : ∀ j : Fin T, ∀ᶠ z : Fin T → ℝ in 𝓝 scores,
      j ≤ i → j ≠ winner → z j + 1 < z winner := by
    intro j
    by_cases hj : j ≤ i
    · by_cases hn : j ≠ winner
      · have hc : ContinuousAt (fun z : Fin T → ℝ => z j + 1) scores :=
          (continuous_apply j).continuousAt.add continuousAt_const
        have he := hc.eventually_lt (continuous_apply winner).continuousAt (hgap j hj hn)
        exact he.mono fun z hz _ _ => hz
      · exact Filter.Eventually.of_forall fun z _ hne => False.elim (hn hne)
    · exact Filter.Eventually.of_forall fun z hle _ => False.elim (hj hle)
  have hall := Filter.eventually_all.mpr heach
  exact hall.mono fun z hz => sparseWeights_eq_basis_of_gap z i winner hw
    (fun j hj hn => le_of_lt (hz j hj hn))

/-- A genuine strict-gap neighborhood occurs with two competing slots.
Source context: the content-score extension of §3.1. -/
example : ∃ scores : Fin 2 → ℝ, ∃ i winner : Fin 2,
    winner ≤ i ∧ ∀ j, j ≤ i → j ≠ winner → scores j + 1 < scores winner := by
  refine ⟨fun j => if j = 0 then 3 else 0, 1, 0, by decide, ?_⟩
  intro j _ hj
  simp [hj]

/-- Every score direction has zero derivative inside a saturated region.
This is a theorem about the existing `sparseWeights`, not a stipulated
Jacobian. Extension of arXiv:2211.11052v1, §3.1. -/
theorem sparseWeights_hasFDerivAt_zero {T : ℕ} (scores : Fin T → ℝ)
    (i winner : Fin T) (hw : winner ≤ i)
    (hgap : ∀ j, j ≤ i → j ≠ winner → scores j + 1 < scores winner) :
    HasFDerivAt (𝕜 := ℝ) (fun z : Fin T → ℝ => sparseWeights z i) 0 scores := by
  exact (hasFDerivAt_const (𝕜 := ℝ) (basis winner) scores).congr_of_eventuallyEq
    (sparseWeights_eventually_eq_basis scores i winner hw hgap)

/-- The zero-derivative hypotheses admit a nonconstant score perturbation.
Source context: §3.1's simplex row, content-score extension. -/
example : (0 : Fin 2) ≤ 1 ∧
    ∀ j : Fin 2, j ≤ 1 → j ≠ 0 →
      (if j = 0 then (2 : ℝ) else 0) + 1 < (2 : ℝ) := by
  constructor
  · decide
  · intro j _ hj
    simp [hj]

/-- Any outer loss receives zero score derivative through this saturated
row when its other inputs are fixed. No smoothness assumption on the loss
is needed: the composition is locally constant. Extension of §3.1; this
does not assert that other parameter paths or AdamW momentum vanish. -/
theorem rowLoss_hasFDerivAt_zero {T : ℕ} (loss : (Fin T → ℝ) → ℝ)
    (scores : Fin T → ℝ) (i winner : Fin T) (hw : winner ≤ i)
    (hgap : ∀ j, j ≤ i → j ≠ winner → scores j + 1 < scores winner) :
    HasFDerivAt (𝕜 := ℝ) (fun z : Fin T → ℝ => loss (sparseWeights z i)) 0 scores := by
  apply (hasFDerivAt_const (𝕜 := ℝ) (loss (basis winner)) scores).congr_of_eventuallyEq
  exact (sparseWeights_eventually_eq_basis scores i winner hw hgap).mono
    fun z hz => congrArg loss hz

/-- The arbitrary outer-loss theorem has satisfiable strict-gap premises.
Source context: §3.1's inference simplex, with a squared-error outer loss. -/
example : ∃ scores : Fin 2 → ℝ, ∃ i winner : Fin 2,
    winner ≤ i ∧ ∀ j, j ≤ i → j ≠ winner → scores j + 1 < scores winner := by
  refine ⟨fun j => if j = 0 then 5 else 0, 1, 0, by decide, ?_⟩
  intro j _ hj
  simp [hj]

end Transformer.GPTMini.Sparsemax
