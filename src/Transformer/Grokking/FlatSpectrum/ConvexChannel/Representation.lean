module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Basic

/-!
# The channel family strictly extends copying

Source: `review/flat_spectrum.md`, *Convex channels and capacity*, CC3-CC4.
The old matching model embeds affinely. New parameters may generate a class
absent from the input. With one categorical context slot, the family can
represent any finite two-input classification table. This is an expressivity
result, not an inference that unseen table entries are learnable.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel

open scoped BigOperators

variable {Q K V C : Type*} [Fintype K] [Fintype V] [Fintype C]
  [DecidableEq K] [DecidableEq V] [DecidableEq C]

/-- Embedding of the old learned matcher into the larger parameter family;
this is one allowed parameter subfamily, not a decoder constraint (CC3). -/
def liftMatcher (A : Matrix K K ℝ) : Channel K K V V :=
  fun q k v c => if c = v then A q k else 0

omit [DecidableEq K] in
/-- Every old stochastic matcher is feasible in the extended model (CC3). -/
theorem feasible_liftMatcher {A : Matrix K K ℝ} (hA : ConvexRecall.IsStochastic A) :
    Feasible (liftMatcher (V := V) A) := by
  refine ⟨?_, ?_, ?_⟩
  · intro q k v c
    simp only [liftMatcher]
    split_ifs
    · exact hA.1 q k
    · exact le_rfl
  · intro q k v w
    simp [liftMatcher]
  · intro q v
    simpa [liftMatcher] using hA.2 q

omit [Fintype V] [DecidableEq K] in
/-- The embedding preserves the actual forward logits on every input (CC3). -/
theorem logits_liftMatcher (A : Matrix K K ℝ) (q : K) (D : K → V) (c : V) :
    logits (liftMatcher A) q D c = ConvexRecall.logits A q D c := by
  simp [logits, liftMatcher, ConvexRecall.logits, eq_comm]

/-- A possible learned rule selects the queried key and generates `g(q,v)`
from its value. The rule `g` belongs to the parameter witness, not the input
feature map or the model's forward implementation (CC3). -/
def generator (g : K → V → C) : Channel K K V C :=
  fun q k v c => if k = q then (if c = g q v then 1 else 0) else 0

omit [Fintype V] [DecidableEq V] in
/-- Every query-dependent categorical value transformation is representable
within the same convex domain (CC3). -/
theorem feasible_generator (g : K → V → C) : Feasible (generator g) := by
  refine ⟨?_, ?_, ?_⟩
  · intro q k v c
    simp only [generator]
    split_ifs <;> norm_num
  · intro q k v w
    by_cases h : k = q <;> simp [generator, h]
  · intro q v
    simp [generator]

omit [Fintype V] [Fintype C] [DecidableEq V] in
/-- Generated answers can depend on both query and input value (CC3). -/
theorem logits_generator (g : K → V → C) (q : K) (D : K → V) (c : C) :
    logits (generator g) q D c = if c = g q (D q) then 1 else 0 := by
  simp [logits, generator]

/-- A feasible channel outputs `true` from all-false input bindings, which
no setting of the old copy-only model can do (CC3). -/
theorem generates_absent_value :
    Feasible (generator (fun _ : Bool => Bool.not)) ∧
    logits (generator (fun _ : Bool => Bool.not)) false (fun _ => false) true = 1 ∧
    ∀ A : Matrix Bool Bool ℝ, ConvexRecall.logits A false (fun _ => false) true = 0 := by
  refine ⟨feasible_generator _, ?_, ?_⟩
  · simp [logits_generator]
  · intro A
    simp [ConvexRecall.logits]

/-- Unfactorized categorical table parameters with one input slot. `f` is
used to construct parameters for a representation witness only (CC4). -/
def table (f : Q → V → C) : Channel Q Unit V C :=
  fun q k v c => match k with | () => if c = f q v then 1 else 0

omit [Fintype V] [DecidableEq V] in
/-- Arbitrary finite two-input classification is in the feasible family;
no algebraic task-specific features are needed for representation (CC4). -/
theorem feasible_table (f : Q → V → C) : Feasible (table f) := by
  refine ⟨?_, ?_, ?_⟩
  · intro q k v c
    simp only [table]
    split_ifs <;> norm_num
  · intro q k v w
    simp [table]
  · intro q v
    simp [table]

omit [Fintype V] [Fintype C] [DecidableEq V] in
/-- The table representation produces the declared class on every pair;
this says nothing about identifying unobserved pairs from training (CC4). -/
theorem logits_table (f : Q → V → C) (q : Q) (v : V) (c : C) :
    logits (table f) q (fun _ => v) c = if c = f q v then 1 else 0 := by
  simp [logits, table]

example := feasible_liftMatcher (V := Bool) (ConvexRecall.stochastic_one (κ := Bool))

example := logits_nonneg (feasible_generator (fun _ : Bool => Bool.not))
  true (id : Bool → Bool) false

example := sum_logits (feasible_generator (fun _ : Bool => Bool.not)) true (id : Bool → Bool)

example := logits_le_one (feasible_generator (fun _ : Bool => Bool.not))
  true (id : Bool → Bool) false

example := feasible_table (fun q v : Bool => xor q v)

end Transformer.Grokking.FlatSpectrum.ConvexChannel
