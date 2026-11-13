module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.BatchCE

/-!
# Finite full-batch and minibatch CE traces

Source: `review/flat_spectrum.md`, CM33. Any schedule of observed-input
minibatches and nonnegative rates preserves the box, defaults and every
initial observed margin. Actual raw CE cannot increase, including on rows
not selected by the current batch. This is preservation, not a discovery
or convergence theorem. Empty batches express the zero-gradient boundary.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth

open Matrix
open Landscape.Neural (crossEntropy)

variable {O Z C : Type*} [DecidableEq O] [Fintype C] [DecidableEq C]

/-- Every observed row's margin survives an actual normalized batch CE
step, CM33. The row can be outside the current minibatch. -/
theorem observedCEStep_margin {A rate : ℝ} (hr : 0 ≤ rate)
    (defaults : Z → C → ℝ) (batch : Finset O) (labels : O → C)
    (T : Matrix (O ⊕ Z) C ℝ) (hT : T ∈ stateTableBox A) (o : O) (c : C) :
    T (Sum.inl o) (labels o) - T (Sum.inl o) c ≤
      observedCEStep A rate defaults batch labels T (Sum.inl o) (labels o) -
      observedCEStep A rate defaults batch labels T (Sum.inl o) c := by
  rw [observedCEStep_row]
  have hb : 0 ≤ if o ∈ batch then rate / batch.card else 0 := by
    split_ifs
    · exact div_nonneg hr (Nat.cast_nonneg _)
    · exact le_refl 0
  exact rowCEStep_margin hb _ _ (hT (Sum.inl o)) c

/-- No observed row's full-vocabulary CE rises after a minibatch step,
CM33. The training loss is separable because observed identities are distinct. -/
theorem observedCEStep_loss_le {A rate : ℝ} (hr : 0 ≤ rate)
    (defaults : Z → C → ℝ) (batch : Finset O) (labels : O → C)
    (T : Matrix (O ⊕ Z) C ℝ) (hT : T ∈ stateTableBox A) (o : O) :
    crossEntropy (observedCEStep A rate defaults batch labels T (Sum.inl o)) (labels o) ≤
      crossEntropy (T (Sum.inl o)) (labels o) := by
  exact rowCE_le_of_margins _ _ _ (observedCEStep_margin hr defaults batch labels T hT o)

/-- Execute the derived batch gradient and box projection repeatedly,
CM33. The schedule may use full sets, minibatches, or no selected rows. -/
noncomputable def observedCETrace (A : ℝ) (defaults : Z → C → ℝ) (labels : O → C)
    (batches : ℕ → Finset O) (rates : ℕ → ℝ) (initial : Matrix (O ⊕ Z) C ℝ) :
    ℕ → Matrix (O ⊕ Z) C ℝ
  | 0 => initial
  | n + 1 => observedCEStep A (rates n) defaults (batches n) labels
      (observedCETrace A defaults labels batches rates initial n)

/-- Feasibility is maintained by every executed update, not supplied
as a hypothesis about the trace, CM33. -/
theorem observedCETrace_feasible {A : ℝ} (hA : 0 ≤ A) (defaults : Z → C → ℝ)
    (hd : ∀ z c, |defaults z c| ≤ A) (labels : O → C)
    (batches : ℕ → Finset O) (rates : ℕ → ℝ) (initial : Matrix (O ⊕ Z) C ℝ)
    (hi : initial ∈ defaultTableDomain A defaults) (n : ℕ) :
    observedCETrace A defaults labels batches rates initial n ∈
      defaultTableDomain A defaults := by
  cases n with
  | zero => exact hi
  | succ n => exact observedCEStep_feasible hA _ defaults hd _ labels _

/-- All initial observed margins persist through an arbitrary finite
full/minibatch schedule of nonnegative CE steps, CM33. -/
theorem observedCETrace_margin {A : ℝ} (hA : 0 ≤ A) (defaults : Z → C → ℝ)
    (hd : ∀ z c, |defaults z c| ≤ A) (labels : O → C)
    (batches : ℕ → Finset O) (rates : ℕ → ℝ) (hr : ∀ n, 0 ≤ rates n)
    (initial : Matrix (O ⊕ Z) C ℝ) (hi : initial ∈ defaultTableDomain A defaults)
    (n : ℕ) (o : O) (c : C) :
    initial (Sum.inl o) (labels o) - initial (Sum.inl o) c ≤
      observedCETrace A defaults labels batches rates initial n (Sum.inl o) (labels o) -
      observedCETrace A defaults labels batches rates initial n (Sum.inl o) c := by
  induction n with
  | zero => exact le_refl _
  | succ n ih =>
    have hf := observedCETrace_feasible hA defaults hd labels batches rates initial hi n
    exact ih.trans (observedCEStep_margin (hr n) defaults (batches n) labels _ hf.1 o c)

/-- The complete observed row CE is bounded by its initial CE at every
iterate, even if a row is never selected again, CM33. -/
theorem observedCETrace_loss_le {A : ℝ} (hA : 0 ≤ A) (defaults : Z → C → ℝ)
    (hd : ∀ z c, |defaults z c| ≤ A) (labels : O → C)
    (batches : ℕ → Finset O) (rates : ℕ → ℝ) (hr : ∀ n, 0 ≤ rates n)
    (initial : Matrix (O ⊕ Z) C ℝ) (hi : initial ∈ defaultTableDomain A defaults)
    (n : ℕ) (o : O) :
    crossEntropy (observedCETrace A defaults labels batches rates initial n (Sum.inl o))
      (labels o) ≤ crossEntropy (initial (Sum.inl o)) (labels o) := by
  exact rowCE_le_of_margins _ _ _
    (observedCETrace_margin hA defaults hd labels batches rates hr initial hi n o)

/-- Every consecutive step lowers or retains each observed row's actual
CE, rather than merely bounding it by the initializer, CM33. -/
theorem observedCETrace_loss_step {A : ℝ} (hA : 0 ≤ A) (defaults : Z → C → ℝ)
    (hd : ∀ z c, |defaults z c| ≤ A) (labels : O → C)
    (batches : ℕ → Finset O) (rates : ℕ → ℝ) (initial : Matrix (O ⊕ Z) C ℝ)
    (hi : initial ∈ defaultTableDomain A defaults) (n : ℕ) (hr : 0 ≤ rates n) (o : O) :
    crossEntropy (observedCETrace A defaults labels batches rates initial (n + 1)
      (Sum.inl o)) (labels o) ≤
    crossEntropy (observedCETrace A defaults labels batches rates initial n
      (Sum.inl o)) (labels o) := by
  exact observedCEStep_loss_le hr defaults (batches n) labels _
    (observedCETrace_feasible hA defaults hd labels batches rates initial hi n).1 o

/-- Default logits are fixed at every finite step, CM33. This does not
assert that an unobserved target agrees with the declared default. -/
theorem observedCETrace_default {A : ℝ} (hA : 0 ≤ A) (defaults : Z → C → ℝ)
    (hd : ∀ z c, |defaults z c| ≤ A) (labels : O → C)
    (batches : ℕ → Finset O) (rates : ℕ → ℝ) (initial : Matrix (O ⊕ Z) C ℝ)
    (hi : initial ∈ defaultTableDomain A defaults) (n : ℕ) (z : Z) :
    observedCETrace A defaults labels batches rates initial n (Sum.inr z) = defaults z := by
  exact (observedCETrace_feasible hA defaults hd labels batches rates initial hi n).2 z

private def exampleDefaults (z c : Bool) : ℝ := if c = z then 24 else 0
private def exampleTable : Matrix (Bool ⊕ Bool) Bool ℝ :=
  fun s c => if c = s.elim id id then 24 else 0
private theorem example_default_bound : ∀ z c, |exampleDefaults z c| ≤ 32 := by
  intro z c
  cases z <;> cases c <;> norm_num [exampleDefaults]
private theorem example_table_feasible : exampleTable ∈ defaultTableDomain 32 exampleDefaults := by
  constructor
  · intro s c
    by_cases hc : c = s.elim id id <;> norm_num [exampleTable, hc]
  · intro z
    rfl

example := observedCEStep_margin (by norm_num : (0 : ℝ) ≤ 1) exampleDefaults
  ({true} : Finset Bool) id exampleTable example_table_feasible.1 false true

example := observedCEStep_loss_le (by norm_num : (0 : ℝ) ≤ 1) exampleDefaults
  ({true} : Finset Bool) id exampleTable example_table_feasible.1 true

example := observedCETrace_feasible (by norm_num : (0 : ℝ) ≤ 32) exampleDefaults
  example_default_bound id (fun n => {decide (n % 2 = 0)}) (fun _ => 1)
  exampleTable example_table_feasible 7

example := observedCETrace_margin (by norm_num : (0 : ℝ) ≤ 32) exampleDefaults
  example_default_bound id (fun n => {decide (n % 2 = 0)}) (fun _ => 1)
  (by intro n; norm_num) exampleTable example_table_feasible 7 false true

example := observedCETrace_loss_le (by norm_num : (0 : ℝ) ≤ 32) exampleDefaults
  example_default_bound id (fun _ => Finset.univ) (fun _ => 1)
  (by intro n; norm_num) exampleTable example_table_feasible 7 false

example := observedCETrace_loss_step (by norm_num : (0 : ℝ) ≤ 32) exampleDefaults
  example_default_bound id (fun n => {decide (n % 2 = 0)}) (fun _ => 1)
  exampleTable example_table_feasible 7 (by norm_num) false

example := observedCETrace_default (by norm_num : (0 : ℝ) ≤ 32) exampleDefaults
  example_default_bound id (fun _ => ∅) (fun _ => 1) exampleTable example_table_feasible 7 true

end Transformer.Grokking.FlatSpectrum.ConvexDepth
