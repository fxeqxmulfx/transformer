module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Minibatch.Failure

/-!
# Retaining both gradients removes the singleton obstruction

Source: `review/flat_spectrum.md`, *Repairing minibatch and full-batch failures*,
MF2. The two stored CE gradients may have been computed at different states.
Their sum has a separating oracle, and every constant-answer vertex has
strictly larger linear cost. Thus retaining the gradients changes the oracle
decision in the exact failure witness; merely retaining old decoder choices
does not. This is not a convergence theorem for all datasets or stale gradients.

The finite-memory update identity explains how a native implementation replaces
an old sample gradient without counting repeated visits as additional examples.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.Minibatch

open scoped BigOperators
open Geometry (probability_pos probability_le_one)

private lemma separating_vertex_eq (b : Bool) :
    vertex (fun _ : Unit => true) (fun _ => observedDecoder b (fun _ => id)) =
      vertex (fun _ : Unit => true) (fun _ _ => (id : Bool → Bool)) := by
  funext q k v c
  cases k <;> cases b <;> cases v <;> cases c <;>
    simp [vertex, observedDecoder, input]

/-- The separating vertex is a singleton minimizer for either input, MF2. -/
theorem separating_minimizes_sample (W : Channel Unit Bool Bool Bool) (b : Bool)
    {A : Channel Unit Bool Bool Bool} (hA : Feasible A) :
    linearCost (sampleGradient W b) (vertex (fun _ : Unit => true) (fun _ _ => id)) ≤
      linearCost (sampleGradient W b) A := by
  have h := singleton_oracle W b true (fun _ => id) hA
  rwa [separating_vertex_eq] at h

/-- Linear costs add when the stored gradient tensors add. -/
theorem linearCost_add_gradient (G H A : Channel Unit Bool Bool Bool) :
    linearCost (G + H) A = linearCost G A + linearCost H A := by
  simp [linearCost, add_mul, Finset.sum_add_distrib]

/-- **Both retained gradients admit the separating vertex as an exact oracle**,
MF2, even when their evaluation states differ. -/
theorem memory_oracle (W₀ W₁ : Channel Unit Bool Bool Bool)
    {A : Channel Unit Bool Bool Bool} (hA : Feasible A) :
    linearCost (sampleGradient W₀ false + sampleGradient W₁ true)
        (vertex (fun _ : Unit => true) (fun _ _ => id)) ≤
      linearCost (sampleGradient W₀ false + sampleGradient W₁ true) A := by
  rw [linearCost_add_gradient, linearCost_add_gradient]
  exact add_le_add (separating_minimizes_sample W₀ false hA)
    (separating_minimizes_sample W₁ true hA)

/-- **The constant-answer vertices are strictly worse after both gradients are
retained**, MF2. This strict comparison rules out the bad singleton tie. -/
theorem memory_strictly_better_than_constant (W₀ W₁ : Channel Unit Bool Bool Bool) (c : Bool) :
    linearCost (sampleGradient W₀ false + sampleGradient W₁ true)
        (vertex (fun _ : Unit => true) (fun _ _ => id)) <
      linearCost (sampleGradient W₀ false + sampleGradient W₁ true)
        (vertex (fun _ : Unit => false) (fun (_ : Unit) (_ _ : Bool) => c)) := by
  have hp₀ := probability_pos (logits W₀ () (input false)) true
  have hp₁ := probability_pos (logits W₁ () (input true)) false
  have hle₀ := probability_le_one (logits W₀ () (input false)) false
  have hle₁ := probability_le_one (logits W₁ () (input true)) true
  cases c <;>
    simp [linearCost_add_gradient, linearCost_sampleGradient, logits, vertex, input] at * <;>
    linarith

/-- **Replacing one memory entry updates its sum by new minus old**, MF2.
Repeated minibatches replace a sample's gradient; they do not duplicate its mass. -/
theorem sum_replace_gradient {I E : Type*} [Fintype I] [DecidableEq I] [AddCommGroup E]
    (stored : I → E) (i : I) (fresh : E) :
    (∑ j, Function.update stored i fresh j) = (∑ j, stored j) + fresh - stored i := by
  rw [Finset.sum_update_of_mem (Finset.mem_univ i), Finset.sdiff_singleton_eq_erase,
    ← Finset.sum_erase_add Finset.univ stored (Finset.mem_univ i)]
  abel

/-! Feasible, nonzero gradient-evaluation states and oracle competitors. -/

example := memory_oracle
  (vertex (fun _ : Unit => false) (fun _ _ => (id : Bool → Bool)))
  (vertex (fun _ : Unit => true) (fun _ _ => (id : Bool → Bool)))
  (feasible_vertex (fun _ : Unit => false) (fun _ _ => (id : Bool → Bool)))

example := memory_strictly_better_than_constant
  (vertex (fun _ : Unit => false) (fun _ _ => (id : Bool → Bool)))
  (vertex (fun _ : Unit => true) (fun _ _ => (id : Bool → Bool))) false

example := sum_replace_gradient (fun b : Bool => if b then (2 : ℝ) else 1) false 3

section Centering

variable {K V C : Type*} [Fintype K] [Fintype V] [Fintype C] [DecidableEq V]

/-- A class-independent residual at one observed dictionary, MF2. -/
def sampleBaseline (D : K → V) (offset : ℝ) : Channel Unit K V C :=
  fun _ k v _ => if v = D k then offset else 0

/-- **A common class offset adds the same cost to every feasible model**, MF2.
The sparse implementation can subtract the softmax background from each sample
gradient without changing any linear-oracle comparison. -/
theorem linearCost_sampleBaseline [Nonempty V] (D : K → V) (offset : ℝ)
    {W : Channel Unit K V C} (hW : Feasible W) :
    linearCost (sampleBaseline D offset) W = offset := by
  calc
    _ = ∑ k, ∑ c, offset * W () k (D k) c := by
      simp only [linearCost, sampleBaseline, Finset.univ_unique, Finset.sum_singleton]
      apply Finset.sum_congr rfl
      intro k _
      rw [Finset.sum_comm]
      apply Finset.sum_congr rfl
      intro c _
      simp [ite_mul]
    _ = offset * ∑ c, logits W () D c := by
      rw [Finset.sum_comm]
      simp only [logits, Finset.mul_sum]
    _ = offset := by rw [sum_logits hW () D, mul_one]

example := linearCost_sampleBaseline (id : Bool → Bool) (2 : ℝ)
  (feasible_vertex (fun _ : Unit => true) (fun _ _ => (id : Bool → Bool)))

end Centering

end Transformer.Grokking.FlatSpectrum.ConvexChannel.Minibatch
