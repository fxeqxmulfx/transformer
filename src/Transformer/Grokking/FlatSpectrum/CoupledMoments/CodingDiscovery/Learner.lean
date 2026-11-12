module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery.Completion

/-!
# The learner: from the training cells and labels to the whole table, or to a refusal

Source: `review/flat_spectrum.md`, CM17, *The learner*. Native reference:
`experiments/convex_stress_relabel/src/coding.rs` at 414431c7, `discover`, with its rejections
`no_nonconstant_phase_code`, `ambiguous_phase_code`, `nonbijective_phase_code` and
`inconsistent_observed_cycle`, and `experiments/convex_stress_relabel/phase_start`, with
`Disconnected`.

`discover p cell labels c₀` is a function of the training cells, the training labels and the
pinned class, and of nothing else: the coding of the table, its phases and the labels of the
other cells are not among its arguments. It answers (`Verdict`):

* `noCode` if the pinned codings have dimension zero, `ambiguous` if above one;
* `nonbijective` if no pinned coding is a bijection of the classes;
* `inconsistent` if no bijective coding has phases that fit the training samples;
* `disconnected` if the graph of the observed cells is not connected;
* else `accept y` for a training-compatible phase completion `y`.

For the labels of a table `e⁻¹ (r i + s j)` with `e`, `r`, `s` unknown to the learner:

* if the pinned codings have dimension one and the graph is connected, the learner returns
  the table (`discover_eq_accept`), and it accepts in no other case
  (`discover_eq_accept_iff`);
* it is never wrong: it returns the table, or answers `ambiguous` or `disconnected`
  (`discover_realizable`); the three other refusals show that the labels are those of no such
  table.

Whatever the labels, an accepted table is a training-compatible phase completion
(`completes_of_discover_eq_accept`), and the answer depends only on the set of observed cells
with their labels (`discover_congr`), so not on the order, the repetitions or the batches of
the samples. The elimination and the search of the graph that compute these conditions are not
formalized: the conditions are stated by what they compute.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery

open Schatten

variable {p : ℕ} [Fact p.Prime] {C n S : Type*} {cell : S → n × n} {labels : S → C}

/-- The answers of the learner: five refusals, or a table. -/
inductive Verdict (n C : Type*) : Type _
  | noCode
  | ambiguous
  | nonbijective
  | inconsistent
  | disconnected
  | accept (table : n → n → C)

open Classical in
/-- The answer for a space `K` of pinned codings, a family `Q` of completions and a statement
of connectivity, in the order of the native checks. -/
noncomputable def Verdict.of (K : Submodule (ZMod p) (C → ZMod p)) (Q : (n → n → C) → Prop)
    (connected : Prop) : Verdict n C :=
  if Module.finrank (ZMod p) K = 0 then .noCode
  else if Module.finrank (ZMod p) K ≠ 1 then .ambiguous
  else if ¬ ∃ w : C ≃ ZMod p, ⇑w ∈ K then .nonbijective
  else if h : ∃ y, Q y then (if connected then .accept h.choose else .disconnected)
  else .inconsistent

/-- **The training-only learner** (CM17): its arguments are the modulus, the training cells,
the training labels and the pinned class. -/
noncomputable def discover (p : ℕ) [Fact p.Prime] (cell : S → n × n) (labels : S → C)
    (c₀ : C) : Verdict n C :=
  Verdict.of (pinnedCodes p cell labels c₀) (Completes p cell labels)
    (patternGraph (Set.range cell)).Connected

/-- Whatever the labels, an accepted table is a training-compatible phase completion. -/
theorem completes_of_discover_eq_accept {c₀ : C} {y : n → n → C}
    (h : discover p cell labels c₀ = .accept y) : Completes p cell labels y := by
  rw [discover, Verdict.of] at h
  split_ifs at h with h1 h2 h3 h4 h5
  exact Verdict.accept.inj h ▸ h4.choose_spec

/-- A bijective pinned coding of a table with a coding and phases. -/
lemma exists_equiv_mem_pinnedCodes (e : C ≃ ZMod p) (r s : n → ZMod p)
    (hreal : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2) (c₀ : C) :
    ∃ w : C ≃ ZMod p, ⇑w ∈ pinnedCodes p cell labels c₀ :=
  ⟨e.trans (Equiv.subRight (e c₀)), code_sub_mem_pinnedCodes e r s hreal c₀⟩

/-- **The learner returns the table** (CM17), at every cell, when the pinned codings have
dimension one and the graph of the observed cells is connected. -/
theorem discover_eq_accept (e : C ≃ ZMod p) (r s : n → ZMod p)
    (hreal : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2) (c₀ : C)
    (hrank : Module.finrank (ZMod p) (pinnedCodes p cell labels c₀) = 1)
    (hgraph : (patternGraph (Set.range cell)).Connected) :
    discover p cell labels c₀ = .accept (phaseLabels e r s) := by
  have hex : ∃ y, Completes p cell labels y := ⟨_, completes_target e r s hreal⟩
  rw [discover, Verdict.of, ite_eq_right (by rw [hrank]; exact one_ne_zero),
    ite_eq_right (not_not.2 hrank),
    ite_eq_right (not_not.2 (exists_equiv_mem_pinnedCodes e r s hreal c₀)), dite_eq_left hex,
    ite_eq_left hgraph]
  exact congrArg Verdict.accept
    ((completes_iff e r s hreal c₀ hrank hgraph.preconnected _).1 hex.choose_spec)

/-- The learner answers `ambiguous` when the pinned codings of such a table have dimension
other than one. -/
theorem discover_eq_ambiguous (e : C ≃ ZMod p) (r s : n → ZMod p)
    (hreal : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2) (c₀ : C)
    (hrank : Module.finrank (ZMod p) (pinnedCodes p cell labels c₀) ≠ 1) :
    discover (n := n) p cell labels c₀ = .ambiguous := by
  rw [discover, Verdict.of, ite_eq_right (finrank_pinnedCodes_ne_zero e r s hreal c₀),
    ite_eq_left hrank]

/-- The learner answers `disconnected` when the pinned codings of such a table have dimension
one and the graph of the observed cells is not connected. -/
theorem discover_eq_disconnected (e : C ≃ ZMod p) (r s : n → ZMod p)
    (hreal : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2) (c₀ : C)
    (hrank : Module.finrank (ZMod p) (pinnedCodes p cell labels c₀) = 1)
    (hgraph : ¬ (patternGraph (Set.range cell)).Connected) :
    discover p cell labels c₀ = .disconnected := by
  have hex : ∃ y, Completes p cell labels y := ⟨_, completes_target e r s hreal⟩
  rw [discover, Verdict.of, ite_eq_right (by rw [hrank]; exact one_ne_zero),
    ite_eq_right (not_not.2 hrank),
    ite_eq_right (not_not.2 (exists_equiv_mem_pinnedCodes e r s hreal c₀)), dite_eq_left hex,
    ite_eq_right hgraph]

/-- **On such a table the learner is never wrong** (CM17): it returns the table, or answers
that the coding is ambiguous, or that the observed cells are not connected. -/
theorem discover_realizable (e : C ≃ ZMod p) (r s : n → ZMod p)
    (hreal : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2) (c₀ : C) :
    discover p cell labels c₀ = .accept (phaseLabels e r s) ∨
      discover (n := n) p cell labels c₀ = .ambiguous ∨
      discover (n := n) p cell labels c₀ = .disconnected := by
  by_cases hrank : Module.finrank (ZMod p) (pinnedCodes p cell labels c₀) = 1
  · by_cases hgraph : (patternGraph (Set.range cell)).Connected
    · exact Or.inl (discover_eq_accept e r s hreal c₀ hrank hgraph)
    · exact Or.inr (Or.inr (discover_eq_disconnected e r s hreal c₀ hrank hgraph))
  · exact Or.inr (Or.inl (discover_eq_ambiguous e r s hreal c₀ hrank))

/-- **The learner accepts such a table exactly under the two observable conditions** (CM17),
and then with the table itself. -/
theorem discover_eq_accept_iff (e : C ≃ ZMod p) (r s : n → ZMod p)
    (hreal : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2) (c₀ : C) (y : n → n → C) :
    discover p cell labels c₀ = .accept y ↔
      Module.finrank (ZMod p) (pinnedCodes p cell labels c₀) = 1 ∧
        (patternGraph (Set.range cell)).Connected ∧ y = phaseLabels e r s := by
  constructor
  · intro h
    by_cases hrank : Module.finrank (ZMod p) (pinnedCodes p cell labels c₀) = 1
    · by_cases hgraph : (patternGraph (Set.range cell)).Connected
      · rw [discover_eq_accept e r s hreal c₀ hrank hgraph] at h
        exact ⟨hrank, hgraph, (Verdict.accept.inj h).symm⟩
      · rw [discover_eq_disconnected e r s hreal c₀ hrank hgraph] at h
        cases h
    · rw [discover_eq_ambiguous e r s hreal c₀ hrank] at h
      cases h
  · rintro ⟨hrank, hgraph, rfl⟩
    exact discover_eq_accept e r s hreal c₀ hrank hgraph

/-- **The answer depends only on the set of observed cells with their labels** (CM17): not on
the order of the samples, their repetitions or their grouping into batches. -/
theorem discover_congr {S' : Type*} {cell' : S' → n × n} {labels' : S' → C} (c₀ : C)
    (h : Set.range (fun t => (cell t, labels t)) = Set.range fun t => (cell' t, labels' t)) :
    discover p cell labels c₀ = discover p cell' labels' c₀ := by
  have h1 : Set.range cell = Prod.fst '' Set.range (fun t => (cell t, labels t)) := by
    rw [← Set.range_comp]
    rfl
  have h2 : Set.range cell' = Prod.fst '' Set.range (fun t => (cell' t, labels' t)) := by
    rw [← Set.range_comp]
    rfl
  rw [discover, discover, pinnedCodes_congr c₀ h, completes_congr h, h1, h2, h]

/-! The witness: from its twelve cells and labels the learner returns the whole table. -/

attribute [local instance] fact_prime_five

/-- **The learner returns the table of the witness.** -/
theorem five_discover :
    discover 5 fiveCells fiveLabels 0 = .accept (phaseLabels fiveCode fivePhase fivePhase) :=
  discover_eq_accept fiveCode fivePhase fivePhase five_real 0 (five_rank 0) five_connected

example : Completes 5 fiveCells fiveLabels (phaseLabels fiveCode fivePhase fivePhase) :=
  completes_of_discover_eq_accept five_discover

example := discover_realizable fiveCode fivePhase fivePhase five_real 3

example := exists_equiv_mem_pinnedCodes fiveCode fivePhase fivePhase five_real 0

example (y : Fin 5 → Fin 5 → ZMod 5) :=
  discover_eq_accept_iff fiveCode fivePhase fivePhase five_real 0 y

/-- The samples in the reverse order give the same answer. -/
example : discover 5 (fiveCells ∘ Fin.rev) (fiveLabels ∘ Fin.rev) 0 =
    discover 5 fiveCells fiveLabels 0 :=
  discover_congr 0 (Fin.rev_surjective.range_comp fun t => (fiveCells t, fiveLabels t))

end Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery
