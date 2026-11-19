module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Estimate.Basic
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Retained.Execution

/-!
# Complete indexed retained coefficients and exact refresh identities

Source: `review/convex/moments.md`, CM106. A cache stores every original class
signal for every indexed occurrence. Historical states and signals may both
differ from current values. Identities do not call a stored signal a gradient.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Retained

open Pair Optimality
open scoped BigOperators
variable {J O : Type*}

/-- Complete mean contraction of indexed states and signals, CM106. -/
noncomputable def mean (n : ℕ) (z : Fin (n + 1) → J → State)
    (s : Fin (n + 1) → O → ℝ) : J → O → State :=
  fun j c => ((n : ℝ) + 1)⁻¹ • ∑ i, s i c • z i j

/-- Replace exactly an original-index subset, without deduplicating equal
prefixes or changing the training denominator, CM106. -/
def refresh {n : ℕ} (batch : Finset (Fin (n + 1)))
    (old new : Fin (n + 1) → O → ℝ) : Fin (n + 1) → O → ℝ :=
  fun i => if i ∈ batch then new i else old i

/-- Replace exactly the visited historical state rows, CM106. -/
def refreshStates {n : ℕ} (batch : Finset (Fin (n + 1)))
    (old new : Fin (n + 1) → J → State) : Fin (n + 1) → J → State :=
  fun i => if i ∈ batch then new i else old i

/-- Signal differences preserve every state/class contribution, CM106. -/
theorem mean_sub_signal (n : ℕ) (z : Fin (n + 1) → J → State)
    (r s : Fin (n + 1) → O → ℝ) : mean n z (r - s) = mean n z r - mean n z s := by
  funext j c
  simp only [mean, Pi.sub_apply, sub_smul, Finset.sum_sub_distrib, smul_sub]

/-- State differences preserve every indexed historical signal, CM106. -/
theorem mean_sub_state (n : ℕ) (z q : Fin (n + 1) → J → State)
    (s : Fin (n + 1) → O → ℝ) : mean n (z - q) s = mean n z s - mean n q s := by
  funext j c
  simp only [mean, Pi.sub_apply, smul_sub, Finset.sum_sub_distrib]

/-- Exact minibatch refresh uses the original training mean, retaining
duplicate records at distinct indices and any short tail, CM106. -/
theorem mean_refresh (n : ℕ) (z : Fin (n + 1) → J → State)
    (old new : Fin (n + 1) → O → ℝ) (batch : Finset (Fin (n + 1))) (j : J) (c : O) :
    (mean n z (refresh batch old new) - mean n z old) j c =
      ((n : ℝ) + 1)⁻¹ • ∑ i ∈ batch, (new i c - old i c) • z i j := by
  rw [← mean_sub_signal]
  change ((n : ℝ) + 1)⁻¹ • ∑ i, ((refresh batch old new - old) i c) • z i j = _
  have he : ∀ i, ((refresh batch old new - old) i c) • z i j =
      if i ∈ batch then (new i c - old i c) • z i j else 0 := by
    intro i
    by_cases hi : i ∈ batch <;> simp [refresh, hi]
  simp only [he, Finset.sum_ite_mem_eq]

/-- Simultaneously refresh states and signals using their complete old and
new products, CM106. This supports changed content without a full-data update. -/
theorem mean_refresh_both (n : ℕ) (oldState newState : Fin (n + 1) → J → State)
    (oldSignal newSignal : Fin (n + 1) → O → ℝ) (batch : Finset (Fin (n + 1))) (j : J) (c : O) :
    (mean n (refreshStates batch oldState newState) (refresh batch oldSignal newSignal) -
      mean n oldState oldSignal) j c = ((n : ℝ) + 1)⁻¹ • ∑ i ∈ batch,
        (newSignal i c • newState i j - oldSignal i c • oldState i j) := by
  change ((n : ℝ) + 1)⁻¹ • ∑ i, (refresh batch oldSignal newSignal) i c •
    (refreshStates batch oldState newState) i j -
      ((n : ℝ) + 1)⁻¹ • ∑ i, oldSignal i c • oldState i j = _
  rw [← smul_sub, ← Finset.sum_sub_distrib]
  have he : ∀ i, (refresh batch oldSignal newSignal) i c •
      (refreshStates batch oldState newState) i j - oldSignal i c • oldState i j =
      if i ∈ batch then newSignal i c • newState i j - oldSignal i c • oldState i j else 0 := by
    intro i
    by_cases hi : i ∈ batch <;> simp [refresh, refreshStates, hi]
  simp only [he, Finset.sum_ite_mem_eq]

/-- Transport every stored signal to the actual changed states, CM106.
This correction is necessary even if no cached signal is refreshed. -/
theorem mean_transport (n : ℕ) (z q : Fin (n + 1) → J → State)
    (s : Fin (n + 1) → O → ℝ) : mean n z s = mean n q s + mean n (z - q) s := by
  rw [mean_sub_state]
  abel

/-- Signal staleness and content staleness are distinct complete errors,
CM106; neither is silently discarded. -/
theorem mean_decomposition (n : ℕ) (z q : Fin (n + 1) → J → State)
    (r s : Fin (n + 1) → O → ℝ) :
    mean n z r - mean n q s = mean n z (r - s) + mean n (z - q) s := by
  rw [mean_sub_signal, mean_sub_state]
  abel

/-- Actual current ordered states at every original indexed prefix, CM106. -/
noncomputable def states (n : ℕ) (f : J → Parameters O) (xs : Fin (n + 1) → List O) :
    Fin (n + 1) → J → State := fun i j => runFrom (f j) 0 (f j).content.initial (xs i)

/-- Historical content may differ by occurrence and component, CM106. -/
noncomputable def historical (n : ℕ) (g : Fin (n + 1) → J → Parameters O)
    (xs : Fin (n + 1) → List O) : Fin (n + 1) → J → State :=
  fun i j => runFrom (g i j) 0 (g i j).content.initial (xs i)

variable [Fintype J] [Fintype O] [DecidableEq O]

/-- The current complete coefficient is the actual full-class CE residual
mean; no correctness assumption on a stored estimate occurs, CM106. -/
theorem actual_coefficient (n : ℕ) (f : J → Parameters O)
    (xs : Fin (n + 1) → List O) (target : Fin (n + 1) → O) (u : J → O → State) :
    jointCoefficient n f xs target u =
      mean n (states n f xs) (fun i => residual (jointScore f u (xs i)) (target i)) := by
  funext j c
  rw [joint_coefficient_apply]
  rfl

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Rotary.Readout.Capacity.Retained
