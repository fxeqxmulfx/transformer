module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedPropagation.Witness

/-!
# Feasible endpoints from surviving domains

Source: `review/flat_spectrum.md`, CM19, endpoint construction. A native solver
needs a feasible point that keeps the bank-certified rows. Any choice of one
phase from each domain of any surviving branch gives such a point, by the
same Cartesian-product output union used in identification. This choice need
not fit the observations; no CE optimality of this fallback point is claimed.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedPropagation

open Finset Schatten

variable {q : ℕ} [NeZero q] {n S C K : Type*} [Fintype n] [DecidableEq n]
  [Fintype S] [Fintype C] [DecidableEq C] [Fintype K]
  {decoder : K → ZMod q → C} {cell : S → n × n} {labels : S → C}

omit [NeZero q] [Fintype n] [DecidableEq n] in
/-- Selecting one element of each nonempty finite domain gives a retained
assignment. No observed-constraint consistency follows from nonemptiness. -/
theorem live_has_selection {D : Domains n q} (hD : Live D) :
    ∃ r s, Contains D r s := by
  choose r hr using hD.1
  choose s hs using hD.2
  exact ⟨r, s, hr, hs⟩

omit [Fintype C] in
/-- Every Cartesian choice in any final branch contributes to the complete
bank union, whether or not that choice fits all training labels. -/
theorem branch_mem_answers (k : K) {z : ZMod q} (anchor : S) (steps : ℕ)
    (hz : z ∈ anchors (decoder k) (labels anchor)) {r s : n → ZMod q}
    (hD : Contains (run (decoder k) cell labels steps
      (initial (cell anchor).1 (cell anchor).2 z)) r s) (i j : n) :
    decoder k (r i + s j) ∈ bankAnswers decoder cell labels anchor steps i j := by
  apply mem_biUnion.mpr
  refine ⟨k, mem_univ _, ?_⟩
  exact mem_biUnion.mpr ⟨z, hz, mem_outputs hD i j⟩

omit [Fintype C] in
/-- A phase table is in its fixed-decoder spectral domain at budgets `N`,
without an observed-fit hypothesis. -/
theorem phase_endpoint (d : ZMod q → C) (r s : n → ZMod q) :
    DecodedFeasible d (phaseCode (Equiv.refl (ZMod q)))
      (fun _ => Fintype.card n) (fun _ => Fintype.card n)
      (pointMass (fun i j => d (r i + s j))) := by
  refine ⟨pointMass (phaseLabels (Equiv.refl (ZMod q)) r s),
    phaseCandidate_budget (Equiv.refl (ZMod q)) r s, ?_⟩
  simp only [decode_pointMass, phaseLabels, Equiv.refl_symm, Equiv.refl_apply]

omit [Fintype C] in
/-- **A feasible fallback preserving every certified label** (CM19). The phases
are arbitrary selections from a final branch. Only membership in its domains,
not correctness of its unobserved labels or an observed fit, is assumed. -/
theorem branch_endpoint (k : K) {z : ZMod q} (anchor : S) (steps : ℕ)
    (hz : z ∈ anchors (decoder k) (labels anchor)) {r s : n → ZMod q}
    (hD : Contains (run (decoder k) cell labels steps
      (initial (cell anchor).1 (cell anchor).2 z)) r s) :
    DecodedFeasible (decoder k) (phaseCode (Equiv.refl (ZMod q)))
        (fun _ => Fintype.card n) (fun _ => Fintype.card n)
        (pointMass (fun i j => decoder k (r i + s j))) ∧
      CertifiedFeasible decoder cell labels anchor steps
        (pointMass (fun i j => decoder k (r i + s j))) := by
  have hp := phase_endpoint (decoder k) r s
  refine ⟨hp, decoderHull_member _ _ _ _ hp, fun i j label hanswer => ?_⟩
  have hm := branch_mem_answers k anchor steps hz hD i j
  rw [hanswer, mem_singleton] at hm
  funext c
  simp [pointMass, ConvexRecall.targetLogits, hm]

omit [Fintype C] in
/-- Every live branch supplies a feasible point in the certified restriction.
This does not assert consistency or existence of an ideal training fit. -/
theorem live_endpoint (k : K) {z : ZMod q} (anchor : S) (steps : ℕ)
    (hz : z ∈ anchors (decoder k) (labels anchor))
    (hD : Live (run (decoder k) cell labels steps
      (initial (cell anchor).1 (cell anchor).2 z))) :
    ∃ r s : n → ZMod q, CertifiedFeasible decoder cell labels anchor steps
      (pointMass (fun i j => decoder k (r i + s j))) := by
  obtain ⟨r, s, hretained⟩ := live_has_selection hD
  exact ⟨r, s, (branch_endpoint k anchor steps hz hretained).2⟩

omit [Fintype C] in
/-- Mixing two tables with the same certified rows keeps those rows even
before a spectral feasibility check. This is an affine identity for any `t`. -/
theorem certified_rows_blend (anchor : S) (steps : ℕ) {P Q : Table n n C}
    (hP : ∀ i j label, bankAnswers decoder cell labels anchor steps i j = {label} →
      P i j = ConvexRecall.targetLogits label)
    (hQ : ∀ i j label, bankAnswers decoder cell labels anchor steps i j = {label} →
      Q i j = ConvexRecall.targetLogits label) (t : ℝ) :
    ∀ i j label, bankAnswers decoder cell labels anchor steps i j = {label} →
      ((1 - t) • P + t • Q) i j = ConvexRecall.targetLogits label := by
  intro i j label hanswer
  change (1 - t) • P i j + t • Q i j = _
  rw [hP i j label hanswer, hQ i j label hanswer, ← add_smul, sub_add_cancel, one_smul]

/-! Partial coverage: a valid fallback that gets an observed label wrong. -/

/-- A concrete choice from the unpruned witness domains. -/
def wideZeroPhases : Fin 2 → ZMod 4 := ![0, 0]

private theorem zero_anchor : (0 : ZMod 4) ∈ anchors (wideBank 0) (wideLabels 2) := by decide

private theorem zero_retained : Contains
    (run (wideBank 0) threeCells wideLabels 0 (initial 1 0 0)) wideZeroPhases wideZeroPhases := by
  unfold Contains
  decide

private theorem fitting_retained : Contains
    (run (wideBank 0) threeCells wideLabels 0 (initial 1 0 0)) ![3, 0] wideZeroPhases := by
  unfold Contains
  decide

/-- At zero sweeps the certificate fixes the false anchor, while the selected
all-zero phase table predicts false at the observed true cell `(0,0)`. -/
theorem wide_partial_endpoint :
    CertifiedFeasible wideBank threeCells wideLabels 2 0
        (pointMass (fun i j => wideBank 0 (wideZeroPhases i + wideZeroPhases j))) ∧
      ¬ DecodedRecovery.Fits (wideBank 0) threeCells wideLabels wideZeroPhases wideZeroPhases := by
  refine ⟨(branch_endpoint 0 2 0 zero_anchor zero_retained).2, ?_⟩
  intro hfit
  exact (by decide : wideBank 0 (wideZeroPhases 0 + wideZeroPhases 0) ≠ wideLabels 0) (hfit 0)

example := live_has_selection (live_of_contains zero_retained)
example := branch_mem_answers 0 2 0 zero_anchor zero_retained 1 0
example := branch_endpoint 0 2 0 zero_anchor zero_retained
example := live_endpoint 0 2 0 zero_anchor (live_of_contains zero_retained)
example := certified_rows_blend 2 0 wide_partial_endpoint.1.2
  (branch_endpoint 0 2 0 zero_anchor fitting_retained).2.2 (1 / 2)

/-- The two tables in the mixture example differ at an observed, uncertified cell. -/
example : pointMass (fun i j => wideBank 0 (wideZeroPhases i + wideZeroPhases j)) ≠
    pointMass (fun i j => wideBank 0 (![3, 0] i + wideZeroPhases j)) := by
  intro h
  have he := congrArg (fun P => P 0 0 true) h
  norm_num [pointMass, wideBank, wideDecoder, wideZeroPhases] at he
  exact (by decide : (0 : ZMod 4) ≠ 3) he

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedPropagation
