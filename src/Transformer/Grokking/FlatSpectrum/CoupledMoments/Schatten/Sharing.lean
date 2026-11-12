module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.Optimality
public import Transformer.Grokking.FlatSpectrum.ConvexChannel.Observation.Basic

/-!
# Fixed sharing constraints preserve convexity and transport observed answers

Source: `review/flat_spectrum.md`, CM10. A fixed representative map imposes linear
equalities between probability rows. Ordinary CE still has its observed lower bound.
An observed representative determines its shared rows at that bound. The map is a
hypothesis of this model, not a conclusion inferred from coincident observed labels.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten

variable {m n C S : Type*}

/-- Fixed column representatives tie all class coordinates linearly (CM10). -/
def ColumnSharing (representative : n → n) (P : Table m n C) : Prop :=
  ∀ i j, P i j = P i (representative j)

/-- Parameter sharing is an affine constraint, hence convex (CM10). -/
theorem columnSharing_convex (representative : n → n) :
    Convex ℝ {P : Table m n C | ColumnSharing representative P} := by
  intro P hP Q hQ a b _ _ _ i j
  funext c
  change a * P i j c + b * Q i j c =
    a * P i (representative j) c + b * Q i (representative j) c
  rw [congrFun (hP i j) c, congrFun (hQ i j) c]

variable [Fintype C] [Fintype S] [DecidableEq C] [Nontrivial C]

/-- Actual ordinary-CE optimality identifies every coordinate shared with an observed
representative (CM10). This does not certify that the chosen sharing rule is correct
for unknown ground truth. -/
theorem shared_target_of_loss_le (representative : n → n) (cell : S → m × n)
    (labels : S → C) {P : Table m n C} (hP : Rows P)
    (hshare : ColumnSharing representative P) (hL : loss cell labels P ≤ idealLoss labels)
    {s : S} {i : m} {j : n} (hcell : cell s = (i, representative j)) :
    P i j = ConvexRecall.targetLogits (labels s) := by
  have ht := target_mass_of_loss_le cell labels hP hL s
  rw [hcell] at ht
  exact (hshare i j).trans
    (ConvexChannel.Observation.simplex_eq_target _ (hP.1 i (representative j))
      (hP.2 i (representative j)) ht)

example :
    (pointMass (fun (i : Bool) (_j : Bool) => i)) true true =
      ConvexRecall.targetLogits true := by
  have hL : loss (fun s : Bool => (s, false)) id
      (pointMass (fun (i : Bool) (_j : Bool) => i)) ≤ idealLoss (id : Bool → Bool) := by
    apply (loss_eq_ideal_of_fit _ _ ?_).le
    intro s
    funext c
    simp [pointMass, ConvexRecall.targetLogits]
  exact shared_target_of_loss_le (fun (_j : Bool) => false)
    (fun s : Bool => (s, false)) id
    (rows_pointMass (fun (i : Bool) (_j : Bool) => i))
    (by intro i j; rfl) hL (s := true) rfl

end Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten
