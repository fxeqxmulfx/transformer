module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Optimality
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Capacity

/-!
# Flat capacity does not identify missing answers

Source: `review/flat_spectrum.md`, *Convex channels and capacity*, CC7.
Two full-rank flat Boolean tables agree on the observed row and disagree
on the held-out row. Both are feasible and both globally minimize the
same ordinary answer CE. Thus the rank bound is not a completion theorem,
even when the true target is known to be flat and capacity is sufficient.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel

open scoped BigOperators

/-- Arbitrary categorical tables minimize CE when their labels agree
with the samples. Unobserved pairs impose no additional constraint (CC7). -/
theorem isMinOn_table {Q V C S : Type*} [Fintype C] [Fintype S]
    [Nonempty V] [DecidableEq C] (f : Q → V → C) (query : S → Q)
    (data : S → Unit → V) (label : S → C)
    (hlabel : ∀ s, label s = f (query s) (data s ())) :
    IsMinOn (answerLoss query data label) {W | Feasible W} (table f) := by
  intro W hW
  apply Finset.sum_le_sum
  intro s _
  have hz : logits (table f) (query s) (data s) = ConvexRecall.targetLogits (label s) := by
    funext c
    simp only [logits, Fintype.sum_unique, table, ConvexRecall.targetLogits, hlabel s]
  rw [hz]
  exact ConvexRecall.target_crossEntropy_le _ (logits_nonneg hW (query s) (data s))
    (logits_le_one hW (query s) (data s) _)

/-- A second parameter witness: its first row equals Boolean AND and
its second row is the complement. This is used only in the counterexample (CC7). -/
def twistedBoolTable (q v : Bool) : Bool := if q then !v else false

/-- Flipping the second row's signs preserves flatness and full rank (CC7). -/
theorem flat_twisted_bool_table :
    IsFlat (outputKernel boolSign (table twistedBoolTable)
      (fun v : Bool => fun _ : Unit => v)) 2 := by
  left
  ext q r
  cases q <;> cases r <;>
    norm_num [Matrix.mul_apply, Matrix.conjTranspose_apply, outputKernel,
      logits, table, boolSign, twistedBoolTable, Fintype.sum_bool, Matrix.one_apply]

/-- Both competing flat tables attain the same global CE optimum on
all observations in the first query row, but give opposite held-out
answers. Full flat rank and adequate width do not resolve the ambiguity (CC7). -/
theorem flat_capacity_does_not_identify :
    let A := table Bool.and
    let B := table twistedBoolTable
    let query := fun _ : Bool => false
    let data := fun v : Bool => fun _ : Unit => v
    let label := fun _ : Bool => false
    Feasible A ∧ Feasible B ∧
      IsFlat (outputKernel boolSign A data) 2 ∧ IsFlat (outputKernel boolSign B data) 2 ∧
      IsMinOn (answerLoss query data label) {W | Feasible W} A ∧
      IsMinOn (answerLoss query data label) {W | Feasible W} B ∧
      (∀ v, logits A false (data v) = logits B false (data v)) ∧
      logits A true (data false) false ≠ logits B true (data false) false := by
  dsimp only
  refine ⟨feasible_table _, feasible_table _, flat_bool_table, flat_twisted_bool_table,
    isMinOn_table _ _ _ _ ?_, isMinOn_table _ _ _ _ ?_, ?_, ?_⟩
  · intro s
    simp
  · intro s
    simp [twistedBoolTable]
  · intro v
    funext c
    simp [logits_table, twistedBoolTable]
  · norm_num [logits_table, twistedBoolTable]

example := isMinOn_table Bool.and (fun _ : Bool => false)
  (fun v : Bool => fun _ : Unit => v) (fun _ => false) (fun s => by simp)

end Transformer.Grokking.FlatSpectrum.ConvexChannel
