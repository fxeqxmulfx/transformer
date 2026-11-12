module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery.PhaseHull

/-!
# The certificate inside the spectral cross-entropy program

Source: `review/flat_spectrum.md`, CM18. The declared convex program is ordinary
original-vocabulary cross-entropy over the tables `decode d P` with `P` in the spectral domain
of CM15 at the budgets `N`; CM16 adds the linear equalities of an even latent table.

* `certified_minimum`: from any phases that fit the training samples one obtains a table of
  the symmetric domain that minimizes the training cross-entropy over the whole CM15 domain,
  and at every covered cell its row is the one-hot row of the unknown target's label.
* That program has other minimizers, and one of them can be wrong at a covered cell
  (`Obstruction.lean`). `CertifiedFeasible` adds to the program the linear equalities that fix
  the covered rows to the certified labels, which are computed from training samples alone.
  The restricted domain is convex, contains the table above, and every minimizer over it is a
  minimizer of the unrestricted program with the target's rows at the covered cells
  (`certified_restriction`). The equalities are imposed, not derived from the minimization:
  the content is that they agree with the target and cost nothing in training cross-entropy.

Exact real arithmetic throughout; no iteration, tolerance or convergence is involved.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery

open Schatten

variable {q : ℕ} [NeZero q] {C n S : Type*} [Fintype C] [DecidableEq C] [Fintype n]
  [DecidableEq n] [Fintype S] {d : ZMod q → C} {cell : S → n × n} {labels : S → C}

/-- **A minimizer built from training with the target's answers** (CM18). The phases `r`, `s`
are any that fit the training samples, for instance those of a certificate that fits them
all; `r'`, `s'` are the phases of the unknown target. -/
theorem certified_minimum (hq : Odd q) (hd : PairedDecoder d) {c : Certificate n q}
    (hc : Accepts d cell labels c) {r s r' s' : n → ZMod q} (hfit : Fits d cell labels r s)
    (hfit' : Fits d cell labels r' s') :
    PairedFeasible d (Fintype.card n) (Fintype.card n) (pointMass (fun i j => d (r i + s j))) ∧
      IsMinOn (loss cell labels)
        {Q | DecodedFeasible d (phaseCode (Equiv.refl (ZMod q)))
          (fun _ => Fintype.card n) (fun _ => Fintype.card n) Q}
        (pointMass (fun i j => d (r i + s j))) ∧
      ∀ i j, Pinned cell c (.inl i) → Pinned cell c (.inr j) →
        pointMass (fun i j => d (r i + s j)) i j =
          ConvexRecall.targetLogits (d (r' i + s' j)) := by
  refine ⟨pairedPhase_feasible d hd.even r s, (decodedPhase_minimum d r s cell labels hfit).2,
    fun i j hi hj => ?_⟩
  funext y
  simp [pointMass, ConvexRecall.targetLogits, fits_agree hq hd hc hfit hfit' hi hj]

/-- **The convex restriction** (CM18): the CM15 domain at the budgets `N` with every covered
row fixed to the label that the certificate proposes. -/
def CertifiedFeasible (d : ZMod q → C) (cell : S → n × n) (c : Certificate n q)
    (Q : Table n n C) : Prop :=
  DecodedFeasible d (phaseCode (Equiv.refl (ZMod q)))
      (fun _ => Fintype.card n) (fun _ => Fintype.card n) Q ∧
    ∀ i j, Pinned cell c (.inl i) → Pinned cell c (.inr j) →
      Q i j = ConvexRecall.targetLogits (c.decoded d i j)

omit [Fintype C] [Fintype S] in
/-- The added equalities are linear: the restricted domain is convex. -/
theorem certifiedFeasible_convex (d : ZMod q → C) (cell : S → n × n) (c : Certificate n q) :
    Convex ℝ {Q : Table n n C | CertifiedFeasible d cell c Q} := by
  intro P hP Q hQ a b ha hb hab
  refine ⟨decodedFeasible_convex d _ _ _ hP.1 hQ.1 ha hb hab, fun i j hi hj => ?_⟩
  change a • P i j + b • Q i j = _
  rw [hP.2 i j hi hj, hQ.2 i j hi hj, ← add_smul, hab, one_smul]

/-- **The restriction keeps the optimum and fixes the covered answers** (CM18). The table is
realizable with the unknown phases `r`, `s`. Its decoded table lies in the restricted domain,
and every minimizer of the training cross-entropy over the restricted domain minimizes it over
the whole CM15 domain and has the target's rows at the covered cells. -/
theorem certified_restriction (hq : Odd q) (hd : PairedDecoder d) {c : Certificate n q}
    (hc : Accepts d cell labels c) {r s : n → ZMod q} (hfit : Fits d cell labels r s) :
    CertifiedFeasible d cell c (pointMass (fun i j => d (r i + s j))) ∧
      ∀ Q, CertifiedFeasible d cell c Q →
        IsMinOn (loss cell labels) {Q | CertifiedFeasible d cell c Q} Q →
        IsMinOn (loss cell labels)
          {Q | DecodedFeasible d (phaseCode (Equiv.refl (ZMod q)))
            (fun _ => Fintype.card n) (fun _ => Fintype.card n) Q} Q ∧
        ∀ i j, Pinned cell c (.inl i) → Pinned cell c (.inr j) →
          Q i j = ConvexRecall.targetLogits (d (r i + s j)) := by
  have hV : CertifiedFeasible d cell c (pointMass (fun i j => d (r i + s j))) := by
    refine ⟨(decodedPhase_minimum d r s cell labels hfit).1, fun i j hi hj => ?_⟩
    funext y
    simp [pointMass, ConvexRecall.targetLogits, accepted_decoding hq hd hc hfit hi hj]
  refine ⟨hV, fun Q hQ hmin => ⟨?_, fun i j hi hj => ?_⟩⟩
  · rintro T ⟨P, hP, rfl⟩
    exact ((isMinOn_iff.1 hmin _ hV).trans (hull_target_minimum hfit).2.2.le).trans
      (idealLoss_le cell labels (rows_decode d hP.1.1))
  · rw [hQ.2 i j hi hj, accepted_decoding hq hd hc hfit hi hj]

/-! The folded table with the withheld cell `(2, 1)`. -/

example := certified_minimum (by decide) (pairedDecoder_fold 2) heldCert_accepts held_fits
  held_fits_reflected
example := certifiedFeasible_convex (foldDecoder 2) heldCells heldCert
example := certified_restriction (by decide) (pairedDecoder_fold 2) heldCert_accepts held_fits

/-- A minimizer over the restricted domain exists: the decoded table of the fitting phases. -/
example : IsMinOn (loss heldCells heldLabels)
    {Q | CertifiedFeasible (foldDecoder 2) heldCells heldCert Q}
    (pointMass (fun i j => foldDecoder 2 (foldedRow i + foldedColumn j))) := fun Q hQ =>
  (hull_target_minimum held_fits).2.2.le.trans (idealLoss_le heldCells heldLabels
    (by obtain ⟨P, hP, rfl⟩ := hQ.1; exact rows_decode _ hP.1.1))

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery
