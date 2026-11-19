module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.WeightedObservations

/-!
# Select identifying anchors without deleting noisy CE observations

Source: `review/convex/moments.md`, CM61. Exact anchors are additional model
constraints, distinct from the original empirical CE records. Conflicting
labels at identical complete features force program mass to zero. Keeping
only explicitly selected anchors can restore a nonempty convex domain; the
ordinary CE still uses every original observed label and its multiplicity.
No clean-label recovery or unseen-identification theorem is asserted here.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.SelectedAnchors

open scoped BigOperators
open CoordinatePrograms

variable {P A : Type*} [Fintype P]

/-- Only the selected original indices impose exact center equations; all
copy, perspective, total-mass and complete-budget constraints remain (CM61). -/
def selectedDomain (rows : P → A → Fin 5 → ℝ) (values : P → A → ℝ)
    (keep : Set A) (B : P → ℝ) (R δ : ℝ) : Set (P × Fin 6 → ℝ) :=
  domain (fun p (a : {a // a ∈ keep}) => rows p a.val)
    (fun p (a : {a // a ∈ keep}) => values p a.val) B R δ

/-- An arbitrary fixed selection retains a convex joint domain (CM61).
Selecting anchors is outside this convexity statement. -/
theorem selected_convex (rows : P → A → Fin 5 → ℝ) (values : P → A → ℝ)
    (keep : Set A) (B : P → ℝ) (R δ : ℝ) :
    Convex ℝ (selectedDomain rows values keep B R δ) :=
  domain_convex _ _ B R δ

/-- Removing exact equations enlarges the domain, without relaxing any of
its budgets or the positive total-mass requirement (CM61). -/
theorem selection_mono (rows : P → A → Fin 5 → ℝ) (values : P → A → ℝ)
    (small large : Set A) (hsub : small ⊆ large) (B : P → ℝ) (R δ : ℝ) :
    selectedDomain rows values large B R δ ⊆ selectedDomain rows values small B R δ := by
  intro θ hθ
  refine ⟨hθ.1, hθ.2.1, hθ.2.2.1, hθ.2.2.2.1, hθ.2.2.2.2.1, ?_⟩
  intro p a
  exact hθ.2.2.2.2.2 p ⟨a.val, hsub a.property⟩

/-- Every fully anchored feasible point remains feasible after selection;
the converse need not hold, as the explicit noisy witness shows (CM61). -/
theorem full_subset (rows : P → A → Fin 5 → ℝ) (values : P → A → ℝ)
    (keep : Set A) (B : P → ℝ) (R δ : ℝ) :
    domain rows values B R δ ⊆ selectedDomain rows values keep B R δ := by
  intro θ hθ
  refine ⟨hθ.1, hθ.2.1, hθ.2.2.1, hθ.2.2.2.1, hθ.2.2.2.2.1, ?_⟩
  exact fun p a => hθ.2.2.2.2.2 p a.val

omit [Fintype P] in
/-- Two different observed coordinates at the same complete feature vector
force this program's mass to zero when both are exact anchors (CM61). -/
theorem conflicting_anchors (θ : Fin 6 → ℝ) (h : Fin 5 → ℝ) (a b : ℝ)
    (hne : a ≠ b) (ha : center θ h = θ 5 * a) (hb : center θ h = θ 5 * b) :
    θ 5 = 0 := by
  have hab : θ 5 * a = θ 5 * b := ha.symm.trans hb
  have hz : θ 5 * (a - b) = 0 := by rw [mul_sub, hab, sub_self]
  rcases mul_eq_zero.mp hz with hg | heq
  · exact hg
  · exact False.elim (hne (sub_eq_zero.mp heq))

/-- If every program faces contradictory identical-feature anchors, a
strictly positive total mass makes the fully anchored domain empty (CM61). -/
theorem all_conflicted_infeasible (rows : P → A → Fin 5 → ℝ) (values : P → A → ℝ)
    (B : P → ℝ) (R δ : ℝ) (hδ : 0 < δ) (a b : A)
    (hrows : ∀ p, rows p a = rows p b) (hvalues : ∀ p, values p a ≠ values p b)
    (θ : P × Fin 6 → ℝ) : θ ∉ domain rows values B R δ := by
  intro hθ
  have hm : ∀ p, θ (p, 5) = 0 := by
    intro p
    apply conflicting_anchors (fun j => θ (p, j)) (rows p a)
      (values p a) (values p b) (hvalues p) (hθ.2.2.2.2.2 p a)
    rw [hrows p]
    exact hθ.2.2.2.2.2 p b
  have hs : (∑ p, θ (p, 5)) = 0 := Finset.sum_eq_zero (fun p _ => hm p)
  have hmass := hθ.2.2.1
  rw [hs] at hmass
  linarith

/-- Two identical content rows have different observed labels; a third
row is the separately declared identity-copy direction (CM61). -/
def toyRows (a : Fin 3) : Fin 5 → ℝ :=
  if a = 2 then ![0, 0, 0, 0, 1] else ![1, 0, 0, 0, 0]
def toyLabels (a : Fin 3) : Bool := decide (a ≠ 0)
def toyCoordinates (c : Bool) : ℝ := if c then 1 else 0
def toyValues (a : Fin 3) : ℝ := toyCoordinates (toyLabels a)
def toyKeep : Set (Fin 3) := {2}
noncomputable def toyParameters (u : Fin 1 × Fin 6) : ℝ := ![1 / 2, 0, 0, 0, 1, 1] u.2

/-- The selected domain contains a positive-mass, nonconstant complete head.
Its content center is one half, and the retained copy equation is exact (CM61). -/
theorem toy_feasible : toyParameters ∈ selectedDomain (fun _ : Fin 1 => toyRows)
    (fun _ => toyValues) toyKeep (fun _ => 16) 16 1 := by
  norm_num [selectedDomain, domain, ParameterSharing.weightedDomain,
    Fintype.sum_prod_type, Fin.sum_univ_succ, costs, charge, center,
    toyParameters, toyRows, toyValues, toyCoordinates, toyLabels, toyKeep,
    Fin.forall_fin_succ]

/-- Retaining both noisy content equations makes the same positive-mass
domain infeasible; the anchor choice has a substantive effect (CM61). -/
theorem toy_full_infeasible (θ : Fin 1 × Fin 6 → ℝ) :
    θ ∉ domain (fun _ : Fin 1 => toyRows) (fun _ => toyValues) (fun _ => 16) 16 1 := by
  apply all_conflicted_infeasible _ _ _ _ _ (by norm_num) (0 : Fin 3) 1
  · intro p
    norm_num [toyRows]
  · intro p
    norm_num [toyValues, toyCoordinates, toyLabels]

variable {C X S : Type*} [Fintype C] [DecidableEq P]

/-- Ordinary full/minibatch CE still uses the entire supplied observed
batch, independently of which indices impose exact anchors (CM61). -/
theorem observed_batch_convex (e : C → ℝ) (he : ∀ c, |e c| ≤ 1)
    (h : P → X → Fin 5 → ℝ) (inputs : S → X) (labels : S → C) (batch : Finset S)
    (rows : P → A → Fin 5 → ℝ) (values : P → A → ℝ)
    (keep : Set A) (B : P → ℝ) (R δ : ℝ) :
    ConvexOn ℝ (selectedDomain rows values keep B R δ)
      (fun θ => Fallback.batchLoss (features h) inputs labels batch (head e θ)) :=
  domain_batchLoss_convex e he h inputs labels batch _ _ B R δ

/-- The two raw coordinates of the noisy witness satisfy the supplied class
geometry bound and are distinct; this is an explicit prior (CM61). -/
theorem toy_coordinates_bound (c : Bool) : |toyCoordinates c| ≤ 1 := by
  cases c <;> norm_num [toyCoordinates]

example : selectedDomain (fun _ : Fin 1 => toyRows) (fun _ => toyValues)
    toyKeep (fun _ => 16) 16 1 ⊆ selectedDomain (fun _ : Fin 1 => toyRows)
      (fun _ => toyValues) ∅ (fun _ => 16) 16 1 :=
  selection_mono _ _ _ _ (by intro a ha; cases ha) _ _ _

example : (fun j : Fin 6 => if j = 2 then (1 : ℝ) else 0) 5 = 0 :=
  conflicting_anchors (fun j => if j = 2 then 1 else 0) (toyRows 0) 0 1
    (by norm_num) (by norm_num [center, toyRows, Fin.sum_univ_succ])
    (by norm_num [center, toyRows, Fin.sum_univ_succ])

example (θ : Fin 1 × Fin 6 → ℝ) :
    θ ∉ domain (fun _ : Fin 1 => toyRows) (fun _ => toyValues) (fun _ => 16) 16 1 :=
  toy_full_infeasible θ

example : ConvexOn ℝ (selectedDomain (fun _ : Fin 1 => toyRows) (fun _ => toyValues)
    toyKeep (fun _ => 16) 16 1)
    (fun θ => Fallback.batchLoss (features (fun _ : Fin 1 => toyRows)) id toyLabels
      (Finset.univ : Finset (Fin 3)) (head toyCoordinates θ)) :=
  observed_batch_convex toyCoordinates toy_coordinates_bound (fun _ => toyRows)
    id toyLabels Finset.univ (fun _ => toyRows) (fun _ => toyValues) toyKeep (fun _ => 16) 16 1

example := toy_feasible

end Transformer.Grokking.FlatSpectrum.ConvexDepth.SelectedAnchors
