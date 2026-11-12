module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery.CrossEntropy

/-!
# The witness of the discovery, and a training set that has to be refused

Source: `review/flat_spectrum.md`, CM17, *Witness* and *A training set that must be refused*.

The witness is the `5 × 5` table `fiveCode⁻¹ (i + j)` of `Basic`, whose coding exchanges the
class names `0` and `1`, with twelve observed cells.

* Thirteen cells are held out, and the table shows all five classes on them
  (`five_heldout_card`, `five_table`, `five_heldout`).
* The order of the class names is not a coding of the training rectangles
  (`five_names_not_code`): the labels are not those of `i + j`, and the coding has to be found.
* From the twelve cells and labels the learner returns the whole table, and the convex
  cross-entropy program of a coding that it can find, twice the hidden one and shifted, has
  one minimizer, the one-hot table of the witness (`five_coding_discovery`).

The cross is the row `0` and the column `0` of the table of `i + j`: nine observed cells whose
graph is connected. The identity coding and the coding of the witness both have phases that
fit them, and their tables differ at the cell `(2, 2)` (`cross_two_tables`). No learner can
return both; this one answers `ambiguous` (`cross_ambiguous`), as it does whenever two tables
of this kind have one training set with a connected graph
(`discover_eq_ambiguous_of_two_tables`). So the test of the elimination cannot be dropped:
connected coverage does not fix the coding.

The ten cells are the witness without its last two: the test of the elimination still passes,
but no cell of the row `4` is observed, and the learner answers `disconnected`
(`ten_disconnected`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery

open Schatten

attribute [local instance] fact_prime_five

/-- Thirteen of the twenty-five cells of the witness are held out. -/
theorem five_heldout_card :
    (Finset.univ.filter fun ij : Fin 5 × Fin 5 => ∀ t, fiveCells t ≠ ij).card = 13 := by
  decide

/-- The table of the witness, row by row. -/
theorem five_table (i j : Fin 5) : phaseLabels fiveCode fivePhase fivePhase i j =
    (![![1, 0, 2, 3, 4], ![0, 2, 3, 4, 1], ![2, 3, 4, 1, 0], ![3, 4, 1, 0, 2],
      ![4, 1, 0, 2, 3]] : Fin 5 → Fin 5 → ZMod 5) i j := by
  revert i j
  decide

/-- Five held-out cells of the witness on which the table shows the five classes. -/
theorem five_heldout : (∀ t, fiveCells t ≠ (4, 2)) ∧ (∀ t, fiveCells t ≠ (1, 4)) ∧
    (∀ t, fiveCells t ≠ (3, 4)) ∧ (∀ t, fiveCells t ≠ (2, 1)) ∧ (∀ t, fiveCells t ≠ (0, 4)) ∧
    phaseLabels fiveCode fivePhase fivePhase 4 2 = 0 ∧
    phaseLabels fiveCode fivePhase fivePhase 1 4 = 1 ∧
    phaseLabels fiveCode fivePhase fivePhase 3 4 = 2 ∧
    phaseLabels fiveCode fivePhase fivePhase 2 1 = 3 ∧
    phaseLabels fiveCode fivePhase fivePhase 0 4 = 4 := by
  decide

/-- **The order of the class names is not a coding of the witness**: read as numbers, the
labels `1`, `0` of the row `0` and `0`, `2` of the row `1` violate their rectangle. -/
theorem five_names_not_code : (fun c : ZMod 5 => c) ∉ rectangleCodes 5 fiveCells fiveLabels :=
  fun h => absurd (h 0 1 4 5 (by decide)) (by decide)

/-- **The witness, end to end** (CM17): from the twelve cells and labels the learner returns
the whole table, and the convex cross-entropy program of the gauge `fiveGauge`, a coding that
the elimination can return, has one minimizer, the one-hot table of the witness. -/
theorem five_coding_discovery :
    discover 5 fiveCells fiveLabels 0 = .accept (phaseLabels fiveCode fivePhase fivePhase) ∧
      {P | BudgetFeasible (phaseCode fiveGauge) (fun _ => Fintype.card (Fin 5))
          (fun _ => Fintype.card (Fin 5)) P ∧
        IsMinOn (loss fiveCells fiveLabels) {Q | BudgetFeasible (phaseCode fiveGauge)
          (fun _ => Fintype.card (Fin 5)) (fun _ => Fintype.card (Fin 5)) Q} P} =
        {pointMass (phaseLabels fiveCode fivePhase fivePhase)} :=
  ⟨five_discover, discovered_argmin fiveCode fivePhase fivePhase five_real 0 (five_rank 0)
    five_connected five_matching fiveGauge fiveGaugeRow fiveGaugeCol five_gauge_fit⟩

/-- The gauge of the witness is a pinned coding: what the elimination can return. -/
theorem five_gauge_mem : (fiveGauge : ZMod 5 → ZMod 5) ∈ pinnedCodes 5 fiveCells fiveLabels 0 :=
  ⟨mem_rectangleCodes_of_fit five_gauge_fit, (by decide : fiveGauge 0 = 0)⟩

section Refusal

variable {p : ℕ} [Fact p.Prime] {C n S : Type*} {cell : S → n × n} {labels : S → C}

/-- **Two tables with one training set are refused** (CM17): if two tables with codings and
phases show the training labels and differ, and the graph of the observed cells is connected,
the learner answers `ambiguous`. -/
theorem discover_eq_ambiguous_of_two_tables (e : C ≃ ZMod p) (r s : n → ZMod p)
    (hreal : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2)
    (e' : C ≃ ZMod p) (r' s' : n → ZMod p)
    (hreal' : ∀ t, e' (labels t) = r' (cell t).1 + s' (cell t).2)
    (hne : phaseLabels e r s ≠ phaseLabels e' r' s')
    (hgraph : (patternGraph (Set.range cell)).Preconnected) (c₀ : C) :
    discover (n := n) p cell labels c₀ = .ambiguous :=
  discover_eq_ambiguous e r s hreal c₀ fun hrank => hne
    ((completes_iff e r s hreal c₀ hrank hgraph _).1 (completes_target e' r' s' hreal')).symm

end Refusal

/-- The cross: the row `0` and the column `0` of a `5 × 5` table. -/
def crossCells : Fin 5 ⊕ Fin 5 → Fin 5 × Fin 5 := Sum.elim (fun j => (0, j)) fun i => (i, 0)

/-- The labels of the cross in the table of `i + j`: the number of the column or of the row. -/
def crossLabels : Fin 5 ⊕ Fin 5 → ZMod 5 := Sum.elim fivePhase fivePhase

/-- The labels of the cross are those of the table of `i + j`. -/
lemma cross_real (t : Fin 5 ⊕ Fin 5) : (Equiv.refl (ZMod 5)) (crossLabels t) =
    fivePhase (crossCells t).1 + fivePhase (crossCells t).2 := by
  revert t
  decide

/-- The labels of the cross are also those of a table whose coding exchanges two classes. -/
lemma cross_real' (t : Fin 5 ⊕ Fin 5) : fiveCode (crossLabels t) =
    (fiveCode (fivePhase (crossCells t).1) - 1) + fiveCode (fivePhase (crossCells t).2) := by
  revert t
  decide

/-- The graph of the cross is connected. -/
lemma cross_connected : (patternGraph (Set.range crossCells)).Connected := by
  refine (rigid_iff_connected _).mp fun r s h i j => ?_
  have h1 : r i + s 0 = 0 := h (i, 0) ⟨.inr i, rfl⟩
  have h2 : r 0 + s j = 0 := h (0, j) ⟨.inl j, rfl⟩
  have h3 : r 0 + s 0 = 0 := h (0, 0) ⟨.inl 0, rfl⟩
  linear_combination h1 + h2 - h3

/-- **Two tables that show the labels of the cross differ at the cell `(2, 2)`**, which is not
observed. -/
theorem cross_two_tables : phaseLabels (Equiv.refl (ZMod 5)) fivePhase fivePhase 2 2 = 4 ∧
    phaseLabels fiveCode (fun i => fiveCode (fivePhase i) - 1) (fun j => fiveCode (fivePhase j))
      2 2 = 3 ∧ ∀ t, crossCells t ≠ (2, 2) := by
  decide

/-- **The learner refuses the cross** (CM17): its answer is `ambiguous`. -/
theorem cross_ambiguous : discover (n := Fin 5) 5 crossCells crossLabels 0 = .ambiguous :=
  discover_eq_ambiguous_of_two_tables (Equiv.refl _) fivePhase fivePhase cross_real fiveCode
    (fun i => fiveCode (fivePhase i) - 1) (fun j => fiveCode (fivePhase j)) cross_real'
    (fun h => absurd (congrFun (congrFun h 2) 2) (by decide)) cross_connected.preconnected 0

/-- The pinned codings of the cross do not have dimension one. -/
lemma cross_rank : Module.finrank (ZMod 5) (pinnedCodes 5 crossCells crossLabels 0) ≠ 1 :=
  fun hrank => absurd (congrFun (congrFun ((completes_iff (Equiv.refl _) fivePhase fivePhase
    cross_real 0 hrank cross_connected.preconnected _).1 (completes_target fiveCode
    (fun i => fiveCode (fivePhase i) - 1) (fun j => fiveCode (fivePhase j)) cross_real')) 2) 2)
    (by decide)

example : discover (n := Fin 5) 5 crossCells crossLabels 0 = .ambiguous :=
  discover_eq_ambiguous (Equiv.refl _) fivePhase fivePhase cross_real 0 cross_rank

/-- The witness without its last two cells: the row `4` and the column `4` are not observed. -/
def tenCells : Fin 10 → Fin 5 × Fin 5 := fiveCells ∘ Fin.castLE (by norm_num)

/-- The labels of the first ten cells of the witness. -/
def tenLabels : Fin 10 → ZMod 5 := fiveLabels ∘ Fin.castLE (by norm_num)

lemma ten_real (t : Fin 10) :
    fiveCode (tenLabels t) = fivePhase (tenCells t).1 + fivePhase (tenCells t).2 :=
  five_real (Fin.castLE (by norm_num) t)

/-- The ten cells still pass the test of the elimination: the rows `0` and `1` are there. -/
lemma ten_rank : Module.finrank (ZMod 5) (pinnedCodes 5 tenCells tenLabels 0) = 1 :=
  finrank_pinnedCodes_eq_one_of_two_rows fiveCode fivePhase fivePhase ten_real 0
    (fun k : Fin 4 => Fin.castLE (by norm_num) k) (fun k : Fin 4 => Fin.natAdd 4 (k.castLE
      (by norm_num) : Fin 6)) (by decide) (by decide) (by decide) (by decide) (by decide)
    (by decide)

/-- The graph of the ten cells is not connected: no cell of the row `4` is observed. -/
lemma ten_not_connected : ¬ (patternGraph (Set.range tenCells)).Connected := fun h =>
  absurd (rigid_of_preconnected (G := ℤ) h.preconnected (fun i => if i = 4 then 1 else 0)
    (fun _ => 0) (by rintro _ ⟨t, rfl⟩; revert t; decide) 4 0) (by decide)

/-- **The learner refuses the ten cells** (CM17): its answer is `disconnected`. -/
theorem ten_disconnected : discover 5 tenCells tenLabels 0 = .disconnected :=
  discover_eq_disconnected fiveCode fivePhase fivePhase ten_real 0 ten_rank ten_not_connected

end Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery
