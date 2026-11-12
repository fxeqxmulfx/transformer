module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery.Learner

/-!
# The convex cross-entropy program of the discovered coding has one minimizer: the table

Source: `review/flat_spectrum.md`, CM17, *The bridge to CE*, on CM13 and CM14:
"CM13 further identifies this candidate as the only ideal probability table under the nuclear
budget. This conclusion concerns the compatible model completion. It does not prove that
unobserved true labels follow the learned phase equations."

The program is that of CM5 and CM14 in the coding `w` that the learner found: probability rows
`P`, the operator norm and the nuclear norm of every moment `M_f (P)`, `f : ZMod p`, of the
code bank `exp (2π i f w(c) / p)` at most the size `N` of the table, and the ordinary
cross-entropy of the training samples, with all classes in its log-sum-exp (`loss`).

Let the labels be those of a table `e⁻¹ (r i + s j)` with `e`, `r`, `s` unknown to the learner,
and let the pinned codings have dimension one. For a bijective coding `w` with phases that fit
the training samples, as every nonzero pinned coding is (`Gauge`):

* with a connected graph of observed cells, the one-hot table that the learner builds from
  `w` and its phases is the one-hot table of the target, and it is feasible and minimizes the
  cross-entropy (`discovered_minimum`): the minimizer is constructed, not searched;
* with a perfect matching among the observed cells as well, it is the only minimizer
  (`discovered_argmin`, `pinned_argmin`), so a minimizer gives probability one to the label of
  the table at every cell, observed or not (`discovered_prediction`).

`coding_discovery` joins this with the learner. The statements are about exact minimizers over
the reals: a table whose loss is within a tolerance of the ideal loss is not covered, and no
run of a solver is claimed to reach the minimizer.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery

open Schatten

variable {p : ℕ} [Fact p.Prime] {C : Type*} [Fintype C] [DecidableEq C] {n : Type*} [Fintype n]
  [DecidableEq n] {S : Type*} [Fintype S] {cell : S → n × n} {labels : S → C}

/-- **The table built from the training set is the one-hot table of the target, and a feasible
minimizer of the cross-entropy** (CM17), in the program of any bijective coding `w` with phases
`ρ`, `σ` that fit the training samples. -/
theorem discovered_minimum (e : C ≃ ZMod p) (r s : n → ZMod p)
    (hreal : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2) (c₀ : C)
    (hrank : Module.finrank (ZMod p) (pinnedCodes p cell labels c₀) = 1)
    (hgraph : (patternGraph (Set.range cell)).Preconnected)
    (w : C ≃ ZMod p) (ρ σ : n → ZMod p)
    (hfit : ∀ t, w (labels t) = ρ (cell t).1 + σ (cell t).2) :
    pointMass (phaseLabels w ρ σ) = pointMass (phaseLabels e r s) ∧
      BudgetFeasible (phaseCode w) (fun _ => Fintype.card n) (fun _ => Fintype.card n)
        (pointMass (phaseLabels e r s)) ∧
      IsMinOn (loss cell labels) {P | BudgetFeasible (phaseCode w) (fun _ => Fintype.card n)
        (fun _ => Fintype.card n) P} (pointMass (phaseLabels e r s)) := by
  have h := completion_eq_target e r s hreal c₀ hrank hgraph w ρ σ hfit
  have hm := phaseCandidate_minimum w ρ σ cell labels hfit
  rw [h] at hm
  exact ⟨by rw [h], hm⟩

/-- **The program of the discovered coding has one minimizer, the one-hot table of the target**
(CM17): with a connected graph of observed cells that contains a perfect matching, the feasible
minimizers of the cross-entropy are that table alone. -/
theorem discovered_argmin (e : C ≃ ZMod p) (r s : n → ZMod p)
    (hreal : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2) (c₀ : C)
    (hrank : Module.finrank (ZMod p) (pinnedCodes p cell labels c₀) = 1)
    (hgraph : (patternGraph (Set.range cell)).Connected)
    (hmatch : ∃ τ : Equiv.Perm n, ∀ i, (i, τ i) ∈ Set.range cell)
    (w : C ≃ ZMod p) (ρ σ : n → ZMod p)
    (hfit : ∀ t, w (labels t) = ρ (cell t).1 + σ (cell t).2) :
    {P | BudgetFeasible (phaseCode w) (fun _ => Fintype.card n) (fun _ => Fintype.card n) P ∧
      IsMinOn (loss cell labels) {Q | BudgetFeasible (phaseCode w) (fun _ => Fintype.card n)
        (fun _ => Fintype.card n) Q} P} = {pointMass (phaseLabels e r s)} := by
  obtain ⟨heq, hfeas, hmin⟩ :=
    discovered_minimum e r s hreal c₀ hrank hgraph.preconnected w ρ σ hfit
  ext P
  refine ⟨fun ⟨hP, hPmin⟩ => ?_, fun hP => ?_⟩
  · have hideal : loss cell labels (pointMass (phaseLabels w ρ σ)) = idealLoss labels := by
      refine loss_eq_ideal_of_fit cell labels fun t => ?_
      have ht := Completes.fit (p := p) ⟨w, ρ, σ, hfit, rfl⟩ t
      funext c
      simp [pointMass, ht, ConvexRecall.targetLogits]
    have hL : loss cell labels P ≤ idealLoss labels := by
      rw [← hideal, heq]
      exact hPmin hfeas
    have : Nontrivial C := e.nontrivial
    rcases isEmpty_or_nonempty n with hn | hn
    · exact funext fun i => isEmptyElim i
    · exact (phaseCandidate_unique w ρ σ cell labels hfit hgraph hmatch hP.1.1
        (fun f _ => hP.2 f) hL).trans heq
  · rw [Set.mem_singleton_iff.1 hP]
    exact ⟨hfeas, hmin⟩

/-- **For a nonzero pinned coding the program has one minimizer, the one-hot table of the
target** (CM17): the coding is the one that the elimination returns, with the bijection that
`Gauge` proves for it. -/
theorem pinned_argmin (e : C ≃ ZMod p) (r s : n → ZMod p)
    (hreal : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2) (c₀ : C)
    (hrank : Module.finrank (ZMod p) (pinnedCodes p cell labels c₀) = 1)
    (hgraph : (patternGraph (Set.range cell)).Connected)
    (hmatch : ∃ τ : Equiv.Perm n, ∀ i, (i, τ i) ∈ Set.range cell)
    {z : C → ZMod p} (hz : z ∈ pinnedCodes p cell labels c₀) (hz0 : z ≠ 0) :
    {P | BudgetFeasible (phaseCode (Equiv.ofBijective z
        (bijective_of_mem_pinnedCodes e r s hreal c₀ hrank hz hz0)))
        (fun _ => Fintype.card n) (fun _ => Fintype.card n) P ∧
      IsMinOn (loss cell labels) {Q | BudgetFeasible (phaseCode (Equiv.ofBijective z
        (bijective_of_mem_pinnedCodes e r s hreal c₀ hrank hz hz0)))
        (fun _ => Fintype.card n) (fun _ => Fintype.card n) Q} P} =
      {pointMass (phaseLabels e r s)} := by
  obtain ⟨ρ, σ, -, hfit⟩ := exists_phases_of_mem_pinnedCodes e r s hreal c₀ hrank hz hz0
  exact discovered_argmin e r s hreal c₀ hrank hgraph hmatch _ ρ σ hfit

/-- **A minimizer of the program predicts the table at every cell** (CM17), observed or not:
probability one for the label of the table, zero for every other class. -/
theorem discovered_prediction (e : C ≃ ZMod p) (r s : n → ZMod p)
    (hreal : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2) (c₀ : C)
    (hrank : Module.finrank (ZMod p) (pinnedCodes p cell labels c₀) = 1)
    (hgraph : (patternGraph (Set.range cell)).Connected)
    (hmatch : ∃ τ : Equiv.Perm n, ∀ i, (i, τ i) ∈ Set.range cell)
    (w : C ≃ ZMod p) (ρ σ : n → ZMod p)
    (hfit : ∀ t, w (labels t) = ρ (cell t).1 + σ (cell t).2) {P : Table n n C}
    (hP : BudgetFeasible (phaseCode w) (fun _ => Fintype.card n) (fun _ => Fintype.card n) P)
    (hmin : IsMinOn (loss cell labels) {Q | BudgetFeasible (phaseCode w)
      (fun _ => Fintype.card n) (fun _ => Fintype.card n) Q} P) (i j : n) :
    P i j (phaseLabels e r s i j) = 1 ∧ ∀ c, c ≠ phaseLabels e r s i j → P i j c = 0 := by
  have hmem : P ∈ ({pointMass (phaseLabels e r s)} : Set (Table n n C)) :=
    discovered_argmin e r s hreal c₀ hrank hgraph hmatch w ρ σ hfit ▸
      (show P ∈ {P | _ ∧ _} from ⟨hP, hmin⟩)
  rw [Set.mem_singleton_iff.1 hmem]
  exact ⟨by simp [pointMass], fun c hc => by simp [pointMass, hc]⟩

/-- **Discovery of the coding, end to end** (CM17). The labels of the training cells are those
of a table `e⁻¹ (r i + s j)`, with `e`, `r`, `s` unknown. If the pinned codings of the training
rectangles have dimension one and the observed cells are connected and contain a perfect
matching, the learner returns the table, and for every nonzero pinned coding the convex
cross-entropy program has one minimizer: the one-hot table of that table. -/
theorem coding_discovery (e : C ≃ ZMod p) (r s : n → ZMod p)
    (hreal : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2) (c₀ : C)
    (hrank : Module.finrank (ZMod p) (pinnedCodes p cell labels c₀) = 1)
    (hgraph : (patternGraph (Set.range cell)).Connected)
    (hmatch : ∃ τ : Equiv.Perm n, ∀ i, (i, τ i) ∈ Set.range cell) :
    discover p cell labels c₀ = .accept (phaseLabels e r s) ∧
      ∀ (z : C → ZMod p) (hz : z ∈ pinnedCodes p cell labels c₀) (hz0 : z ≠ 0),
        {P | BudgetFeasible (phaseCode (Equiv.ofBijective z
            (bijective_of_mem_pinnedCodes e r s hreal c₀ hrank hz hz0)))
            (fun _ => Fintype.card n) (fun _ => Fintype.card n) P ∧
          IsMinOn (loss cell labels) {Q | BudgetFeasible (phaseCode (Equiv.ofBijective z
            (bijective_of_mem_pinnedCodes e r s hreal c₀ hrank hz hz0)))
            (fun _ => Fintype.card n) (fun _ => Fintype.card n) Q} P} =
          {pointMass (phaseLabels e r s)} :=
  ⟨discover_eq_accept e r s hreal c₀ hrank hgraph, fun _ hz hz0 =>
    pinned_argmin e r s hreal c₀ hrank hgraph hmatch hz hz0⟩

/-! The witness: its diagonal is observed, so the observed cells contain a perfect matching. -/

attribute [local instance] fact_prime_five

/-- The observed cells of the witness contain the diagonal. -/
lemma five_matching : ∃ τ : Equiv.Perm (Fin 5), ∀ i, (i, τ i) ∈ Set.range fiveCells :=
  ⟨Equiv.refl _, fun i => by
    fin_cases i
    exacts [⟨0, rfl⟩, ⟨5, rfl⟩, ⟨8, rfl⟩, ⟨9, rfl⟩, ⟨10, rfl⟩]⟩

example := discovered_argmin fiveCode fivePhase fivePhase five_real 0 (five_rank 0)
  five_connected five_matching fiveGauge fiveGaugeRow fiveGaugeCol five_gauge_fit

example := pinned_argmin fiveCode fivePhase fivePhase five_real 0 (five_rank 0) five_connected
  five_matching five_mem (code_sub_ne_zero fiveCode 0)

example := coding_discovery fiveCode fivePhase fivePhase five_real 0 (five_rank 0)
  five_connected five_matching

/-- A minimizer in the program of the gauge predicts the table at the held-out cell `(4, 2)`. -/
example :=
  have h := discovered_minimum fiveCode fivePhase fivePhase five_real 0 (five_rank 0)
    five_connected.preconnected fiveGauge fiveGaugeRow fiveGaugeCol five_gauge_fit
  discovered_prediction fiveCode fivePhase fivePhase five_real 0 (five_rank 0) five_connected
    five_matching fiveGauge fiveGaugeRow fiveGaugeCol five_gauge_fit h.2.1 h.2.2 4 2

end Transformer.Grokking.FlatSpectrum.CoupledMoments.CodingDiscovery
