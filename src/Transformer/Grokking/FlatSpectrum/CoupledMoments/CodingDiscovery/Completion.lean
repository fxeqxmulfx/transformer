module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery.Gauge

/-!
# Every training-compatible phase completion is the table

Source: `review/flat_spectrum.md`, CM17, *The completion is the table*. CM13 and CM14 stop
before it: "Observed phase consistency alone does not prove that unknown ground truth satisfies
it" and "It does not prove that unobserved true labels follow the learned phase equations".
Native reference at 414431c7: `propose` of `experiments/convex_stress_relabel/phase_start`,
which with `analyze` of `graph_audit` builds the table `w⁻¹ (ρ i + σ j)` from a coding `w` and
phase potentials `ρ`, `σ` found on the observed cells.

A table is a training-compatible phase completion (`Completes`) if a bijection of the classes
onto `ZMod p` and phases fit every training sample and produce it. This is a property of the
training set and the table: the cells that are not observed do not enter.

Let the labels be those of a table `e⁻¹ (r i + s j)`, let the pinned codings have dimension
one, and let the graph of the observed cells be connected. Then every training-compatible
phase completion is that table, on every cell (`completion_eq_target`, `completes_iff`): the
coding of the completion is an affine gauge of `e` (`Gauge`), so its phases and the gauged
phases of the table agree as sums on the observed cells, and on a connected graph everywhere
(`rigid_of_preconnected`, for potentials in `ZMod p`).

What is assumed about the cells that are not observed is only that the table has such a
coding and phases, which the learner does not know.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery

open Schatten

variable {p : ℕ} [Fact p.Prime] {C n S : Type*} {cell : S → n × n} {labels : S → C}

/-- **Every bijective coding with phases that fit the training samples gives the table**
(CM17), on every cell, when the pinned codings have dimension one and the graph of the
observed cells is connected. -/
theorem completion_eq_target (e : C ≃ ZMod p) (r s : n → ZMod p)
    (hreal : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2) (c₀ : C)
    (hrank : Module.finrank (ZMod p) (pinnedCodes p cell labels c₀) = 1)
    (hgraph : (patternGraph (Set.range cell)).Preconnected)
    (w : C ≃ ZMod p) (ρ σ : n → ZMod p)
    (hfit : ∀ t, w (labels t) = ρ (cell t).1 + σ (cell t).2) :
    phaseLabels w ρ σ = phaseLabels e r s := by
  obtain ⟨α, -, hα⟩ := exists_gauge e r s hreal c₀ hrank
    (code_sub_mem_pinnedCodes w ρ σ hfit c₀) (code_sub_ne_zero w c₀)
  have hrig := rigid_of_preconnected hgraph (fun i => ρ i - (α * (r i - e c₀) + w c₀))
    (fun j => σ j - α * s j) (by
      rintro _ ⟨t, rfl⟩
      have h2 : w (labels t) - w c₀ = α * (e (labels t) - e c₀) := hα (labels t)
      change ρ (cell t).1 - (α * (r (cell t).1 - e c₀) + w c₀) +
        (σ (cell t).2 - α * s (cell t).2) = 0
      linear_combination -hfit t + h2 + α * hreal t)
  funext i j
  change w.symm (ρ i + σ j) = e.symm (r i + s j)
  rw [Equiv.symm_apply_eq]
  have h1 : ρ i - (α * (r i - e c₀) + w c₀) + (σ j - α * s j) = 0 := hrig i j
  have h2 : w (e.symm (r i + s j)) - w c₀ = α * (e (e.symm (r i + s j)) - e c₀) := hα _
  rw [e.apply_symm_apply] at h2
  linear_combination h1 - h2

/-- A table is a **training-compatible phase completion** (CM14): a bijection of the classes
onto `ZMod p` and phases fit every training sample and produce the table. -/
def Completes (p : ℕ) [NeZero p] (cell : S → n × n) (labels : S → C) (y : n → n → C) : Prop :=
  ∃ (w : C ≃ ZMod p) (ρ σ : n → ZMod p),
    (∀ t, w (labels t) = ρ (cell t).1 + σ (cell t).2) ∧ y = phaseLabels w ρ σ

/-- A completion shows the training labels at the training cells. -/
lemma Completes.fit {y : n → n → C} (h : Completes p cell labels y) (t : S) :
    y (cell t).1 (cell t).2 = labels t := by
  obtain ⟨w, ρ, σ, hfit, rfl⟩ := h
  exact (Equiv.symm_apply_eq w).2 (hfit t).symm

/-- A table with a coding and phases is a completion of each of its training sets. -/
lemma completes_target (e : C ≃ ZMod p) (r s : n → ZMod p)
    (hreal : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2) :
    Completes p cell labels (phaseLabels e r s) :=
  ⟨e, r, s, hreal, rfl⟩

/-- **The table is the only training-compatible phase completion** (CM17), when the pinned
codings have dimension one and the graph of the observed cells is connected. -/
theorem completes_iff (e : C ≃ ZMod p) (r s : n → ZMod p)
    (hreal : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2) (c₀ : C)
    (hrank : Module.finrank (ZMod p) (pinnedCodes p cell labels c₀) = 1)
    (hgraph : (patternGraph (Set.range cell)).Preconnected) (y : n → n → C) :
    Completes p cell labels y ↔ y = phaseLabels e r s :=
  ⟨fun ⟨w, ρ, σ, hfit, hy⟩ =>
    hy.trans (completion_eq_target e r s hreal c₀ hrank hgraph w ρ σ hfit),
    fun h => h ▸ completes_target e r s hreal⟩

variable {S' : Type*} {cell' : S' → n × n} {labels' : S' → C}

/-- A completion of a training set is a completion of every part of it. -/
lemma Completes.anti {y : n → n → C} (h : Completes p cell labels y)
    (hsub : Set.range (fun t => (cell' t, labels' t)) ⊆ Set.range fun t => (cell t, labels t)) :
    Completes p cell' labels' y := by
  obtain ⟨w, ρ, σ, hfit, hy⟩ := h
  refine ⟨w, ρ, σ, fun t' => ?_, hy⟩
  obtain ⟨t, ht⟩ := hsub ⟨t', rfl⟩
  simp only [Prod.mk.injEq] at ht
  rw [← ht.1, ← ht.2]
  exact hfit t

/-- The completions depend only on the set of observed cells with their labels. -/
theorem completes_congr
    (h : Set.range (fun t => (cell t, labels t)) = Set.range fun t => (cell' t, labels' t)) :
    Completes p cell labels = Completes p cell' labels' :=
  funext fun _ => propext ⟨fun hc => hc.anti h.ge, fun hc => hc.anti h.le⟩

/-! The witness: the graph of its twelve observed cells is connected, and every completion of
them is the table of its coding, at the thirteen cells held out as well. -/

attribute [local instance] fact_prime_five

/-- The observed cells of the witness connect all rows and columns. -/
lemma five_connected : (patternGraph (Set.range fiveCells)).Connected := by
  refine (rigid_iff_connected _).mp fun r s h i j => ?_
  have h00 : r 0 + s 0 = 0 := h (0, 0) ⟨0, rfl⟩
  have h01 : r 0 + s 1 = 0 := h (0, 1) ⟨1, rfl⟩
  have h02 : r 0 + s 2 = 0 := h (0, 2) ⟨2, rfl⟩
  have h03 : r 0 + s 3 = 0 := h (0, 3) ⟨3, rfl⟩
  have h10 : r 1 + s 0 = 0 := h (1, 0) ⟨4, rfl⟩
  have h22 : r 2 + s 2 = 0 := h (2, 2) ⟨8, rfl⟩
  have h33 : r 3 + s 3 = 0 := h (3, 3) ⟨9, rfl⟩
  have h44 : r 4 + s 4 = 0 := h (4, 4) ⟨10, rfl⟩
  have h40 : r 4 + s 0 = 0 := h (4, 0) ⟨11, rfl⟩
  have hr : ∀ i, r i = r 0 := by
    intro i
    fin_cases i
    · rfl
    · change r 1 = r 0
      linear_combination h10 - h00
    · change r 2 = r 0
      linear_combination h22 - h02
    · change r 3 = r 0
      linear_combination h33 - h03
    · change r 4 = r 0
      linear_combination h40 - h00
  have hs : ∀ j, s j = -r 0 := by
    intro j
    fin_cases j
    · change s 0 = -r 0
      linear_combination h00
    · change s 1 = -r 0
      linear_combination h01
    · change s 2 = -r 0
      linear_combination h02
    · change s 3 = -r 0
      linear_combination h03
    · change s 4 = -r 0
      linear_combination h44 - h40 + h00
  rw [hr i, hs j, add_neg_cancel]

/-- A gauge of the coding of the witness: twice the coding, shifted to vanish at the class `0`. -/
def fiveGauge : ZMod 5 ≃ ZMod 5 :=
  fiveCode.trans ((Equiv.subRight 1).trans (Equiv.mulLeft₀ 2 (by decide)))

/-- The row phases of the gauge. -/
def fiveGaugeRow (i : Fin 5) : ZMod 5 := 2 * (fivePhase i - 1)

/-- The column phases of the gauge. -/
def fiveGaugeCol (j : Fin 5) : ZMod 5 := 2 * fivePhase j

/-- The gauge and its phases fit the twelve samples. -/
lemma five_gauge_fit (t : Fin 12) :
    fiveGauge (fiveLabels t) = fiveGaugeRow (fiveCells t).1 + fiveGaugeCol (fiveCells t).2 := by
  revert t
  decide

/-- The gauge and its phases give the table of the witness. -/
example : phaseLabels fiveGauge fiveGaugeRow fiveGaugeCol =
    phaseLabels fiveCode fivePhase fivePhase :=
  completion_eq_target fiveCode fivePhase fivePhase five_real 0 (five_rank 0)
    five_connected.preconnected _ _ _ five_gauge_fit

example (y : Fin 5 → Fin 5 → ZMod 5) := completes_iff fiveCode fivePhase fivePhase five_real 0
  (five_rank 0) five_connected.preconnected y

example (t : Fin 12) :
    phaseLabels fiveCode fivePhase fivePhase (fiveCells t).1 (fiveCells t).2 = fiveLabels t :=
  (completes_target fiveCode fivePhase fivePhase five_real).fit t

example : Completes 5 (fiveCells ∘ Fin.rev) (fiveLabels ∘ Fin.rev) =
    Completes 5 fiveCells fiveLabels :=
  completes_congr (Fin.rev_surjective.range_comp fun t => (fiveCells t, fiveLabels t))

/-- The table of the witness completes its first eight samples as well. -/
example : Completes 5 (fiveCells ∘ Fin.castLE (by norm_num : 8 ≤ 12))
    (fiveLabels ∘ Fin.castLE (by norm_num : 8 ≤ 12)) (phaseLabels fiveCode fivePhase fivePhase) :=
  (completes_target fiveCode fivePhase fivePhase five_real).anti
    (Set.range_comp_subset_range _ fun t => (fiveCells t, fiveLabels t))

end Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery
