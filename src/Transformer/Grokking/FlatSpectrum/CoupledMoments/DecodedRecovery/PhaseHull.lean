module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery.Rigidity

/-!
# Cross-entropy minimizers over the hull of decoded phase tables

Source: `review/flat_spectrum.md`, CM18. The spectral relaxation of CM15 and CM16 contains
every decoded phase table and is convex, so it contains their convex hull `phaseHull`. The
hull is a smaller convex family that depends on the decoder and the side alone. On it the
certificate bounds the error at a covered cell by the misfit at the training cells
(`hull_defect_le`): both sides are affine, a vertex that fits the training samples has the
certified label by `accepted_decoding`, and a vertex that does not fit pays at least one.

Hence every table of the hull that attains the ideal training cross-entropy, and so every
minimizer of ordinary cross-entropy over the hull when the table is realizable, has the
one-hot row of the target at every covered cell (`hull_minimizer_decoding`). Latent
probabilities are not identified and are not mentioned. The same statement fails on the
spectral relaxation itself (`Obstruction.lean`). No projection onto the hull and no way to
optimize over it is given here.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery

open Schatten

variable {q : ℕ} {C n S : Type*} [Fintype C] [DecidableEq C] {d : ZMod q → C}

/-- **The changed convex family** (CM18): the convex hull of the decoded tables of all row and
column phases. -/
def phaseHull (d : ZMod q → C) (n : Type*) : Set (Table n n C) :=
  convexHull ℝ {Q | ∃ r s : n → ZMod q, Q = pointMass (fun i j => d (r i + s j))}

/-- The tables of the hull are tables of probability rows. -/
theorem phaseHull_rows {Q : Table n n C} (hQ : Q ∈ phaseHull d n) : Rows Q := by
  refine convexHull_min (t := {Q : Table n n C | Rows Q}) ?_ rows_convex hQ
  rintro Q ⟨r, s, rfl⟩
  exact rows_pointMass _

variable [Fintype n] [DecidableEq n]

omit [Fintype C] in
/-- The hull lies in the spectral relaxation of CM15 with the budgets `N`. -/
theorem phaseHull_subset_decoded [NeZero q] (d : ZMod q → C) :
    phaseHull d n ⊆ {Q | DecodedFeasible d (phaseCode (Equiv.refl (ZMod q)))
      (fun _ => Fintype.card n) (fun _ => Fintype.card n) Q} := by
  refine convexHull_min ?_ (decodedFeasible_convex d _ _ _)
  rintro Q ⟨r, s, rfl⟩
  exact ⟨pointMass (phaseLabels (Equiv.refl (ZMod q)) r s), phaseCandidate_budget _ r s,
    decode_pointMass d _⟩

omit [Fintype C] in
/-- For an even decoder the hull lies in the symmetric relaxation of CM16. -/
theorem phaseHull_subset_paired [NeZero q] (hd : PairedDecoder d) :
    phaseHull d n ⊆ {Q | PairedFeasible d (Fintype.card n) (Fintype.card n) Q} := by
  refine convexHull_min ?_ (pairedFeasible_convex d _ _)
  rintro Q ⟨r, s, rfl⟩
  exact pairedPhase_feasible d hd.even r s

variable [Fintype S] {cell : S → n × n} {labels : S → C}

omit [Fintype C] [Fintype n] [DecidableEq n] in
/-- **The defect at a covered cell is at most the misfit at the training cells** (CM18), for
every table of the hull. -/
theorem hull_defect_le (hq : Odd q) (hd : PairedDecoder d) {c : Certificate n q}
    (hc : Accepts d cell labels c) {i j : n} (hi : Pinned cell c (.inl i))
    (hj : Pinned cell c (.inr j)) {Q : Table n n C} (hQ : Q ∈ phaseHull d n) :
    1 - Q i j (c.decoded d i j) ≤ ∑ t, (1 - Q (cell t).1 (cell t).2 (labels t)) := by
  refine convexHull_min (t := {Q : Table n n C |
    1 - Q i j (c.decoded d i j) ≤ ∑ t, (1 - Q (cell t).1 (cell t).2 (labels t))}) ?_ ?_ hQ
  · rintro Q ⟨r, s, rfl⟩
    have hnn (t : S) : 0 ≤ 1 - pointMass (fun i j => d (r i + s j)) (cell t).1 (cell t).2
        (labels t) := by
      simp only [pointMass]
      split_ifs <;> norm_num
    by_cases hfit : Fits d cell labels r s
    · have h0 : pointMass (fun i j => d (r i + s j)) i j (c.decoded d i j) = 1 := by
        simp [pointMass, accepted_decoding hq hd hc hfit hi hj]
      change 1 - pointMass (fun i j => d (r i + s j)) i j (c.decoded d i j) ≤ _
      rw [h0, sub_self]
      exact Finset.sum_nonneg fun t _ => hnn t
    · obtain ⟨t, ht⟩ := not_forall.1
        (show ¬ ∀ t, d (r (cell t).1 + s (cell t).2) = labels t from hfit)
      have h1 : 1 - pointMass (fun i j => d (r i + s j)) (cell t).1 (cell t).2 (labels t) = 1 := by
        simp [pointMass, Ne.symm ht]
      calc 1 - pointMass (fun i j => d (r i + s j)) i j (c.decoded d i j) ≤ 1 := by
            simp only [pointMass]
            split_ifs <;> norm_num
        _ = 1 - pointMass (fun i j => d (r i + s j)) (cell t).1 (cell t).2 (labels t) := h1.symm
        _ ≤ _ := Finset.single_le_sum (f := fun t => 1 - pointMass (fun i j => d (r i + s j))
            (cell t).1 (cell t).2 (labels t)) (fun t _ => hnn t) (Finset.mem_univ t)
  · intro P hP Q hQ a b ha hb hab
    change 1 - (a * P i j (c.decoded d i j) + b * Q i j (c.decoded d i j)) ≤
      ∑ t, (1 - (a * P (cell t).1 (cell t).2 (labels t) + b * Q (cell t).1 (cell t).2 (labels t)))
    have e : ∑ t, (1 - (a * P (cell t).1 (cell t).2 (labels t) +
        b * Q (cell t).1 (cell t).2 (labels t))) =
        a * ∑ t, (1 - P (cell t).1 (cell t).2 (labels t)) +
          b * ∑ t, (1 - Q (cell t).1 (cell t).2 (labels t)) := by
      rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
      exact Finset.sum_congr rfl fun t _ => by linear_combination (-1 : ℝ) * hab
    have hP' : 1 - P i j (c.decoded d i j) ≤ ∑ t, (1 - P (cell t).1 (cell t).2 (labels t)) := hP
    have hQ' : 1 - Q i j (c.decoded d i j) ≤ ∑ t, (1 - Q (cell t).1 (cell t).2 (labels t)) := hQ
    rw [e]
    linarith [mul_le_mul_of_nonneg_left hP' ha, mul_le_mul_of_nonneg_left hQ' hb]

omit [Fintype n] [DecidableEq n] in
/-- **A table of the hull with the ideal training cross-entropy has the certified rows**
(CM18): at a covered cell its row is the one-hot row of the proposed label. -/
theorem hull_fit_decoding [Nontrivial C] (hq : Odd q) (hd : PairedDecoder d)
    {c : Certificate n q} (hc : Accepts d cell labels c) {i j : n}
    (hi : Pinned cell c (.inl i)) (hj : Pinned cell c (.inr j)) {Q : Table n n C}
    (hQ : Q ∈ phaseHull d n) (hL : loss cell labels Q ≤ idealLoss labels) :
    Q i j = ConvexRecall.targetLogits (c.decoded d i j) := by
  have hR := phaseHull_rows hQ
  have hm (t : S) := target_mass_of_loss_le cell labels hR hL t
  have he := hull_defect_le hq hd hc hi hj hQ
  simp only [hm, sub_self, Finset.sum_const_zero] at he
  exact ConvexChannel.Observation.simplex_eq_target _ (hR.1 i j) (hR.2 i j)
    (le_antisymm (coordinate_le_one hR i j _) (by linarith))

omit [Fintype n] [DecidableEq n] in
/-- The decoded table of fitting phases lies in the hull and minimizes the training
cross-entropy over it. -/
theorem hull_target_minimum {r s : n → ZMod q} (hfit : Fits d cell labels r s) :
    pointMass (fun i j => d (r i + s j)) ∈ phaseHull d n ∧
      IsMinOn (loss cell labels) (phaseHull d n) (pointMass (fun i j => d (r i + s j))) ∧
      loss cell labels (pointMass (fun i j => d (r i + s j))) = idealLoss labels := by
  have hL : loss cell labels (pointMass (fun i j => d (r i + s j))) = idealLoss labels := by
    apply loss_eq_ideal_of_fit
    intro t
    funext y
    simp [pointMass, hfit t, ConvexRecall.targetLogits]
  exact ⟨subset_convexHull ℝ _ ⟨r, s, rfl⟩,
    fun Q hQ => hL.le.trans (idealLoss_le cell labels (phaseHull_rows hQ)), hL⟩

omit [Fintype n] [DecidableEq n] in
/-- **Minimizers of cross-entropy over the hull predict the target at covered cells** (CM18).
The table is realizable with the unknown phases `r`, `s`; `Q` is any minimizer of the ordinary
training cross-entropy over the hull. -/
theorem hull_minimizer_decoding [Nontrivial C] (hq : Odd q) (hd : PairedDecoder d)
    {c : Certificate n q} (hc : Accepts d cell labels c) {r s : n → ZMod q}
    (hfit : Fits d cell labels r s) {i j : n} (hi : Pinned cell c (.inl i))
    (hj : Pinned cell c (.inr j)) {Q : Table n n C} (hQ : Q ∈ phaseHull d n)
    (hmin : IsMinOn (loss cell labels) (phaseHull d n) Q) :
    Q i j = ConvexRecall.targetLogits (d (r i + s j)) := by
  obtain ⟨hV, -, hL⟩ := hull_target_minimum hfit
  rw [accepted_decoding hq hd hc hfit hi hj]
  exact hull_fit_decoding hq hd hc hi hj hQ ((isMinOn_iff.1 hmin _ hV).trans hL.le)

/-! The folded table with the withheld cell `(2, 1)`. -/

example := phaseHull_rows (hull_target_minimum held_fits).1
example := phaseHull_subset_decoded (n := Fin 3) (foldDecoder 2)
example := phaseHull_subset_paired (n := Fin 3) (pairedDecoder_fold 2)
example := hull_defect_le (by decide) (pairedDecoder_fold 2) heldCert_accepts held_row held_col
  (hull_target_minimum held_fits).1
example := hull_fit_decoding (by decide) (pairedDecoder_fold 2) heldCert_accepts held_row
  held_col (hull_target_minimum held_fits).1 (hull_target_minimum held_fits).2.2.le
example := hull_minimizer_decoding (by decide) (pairedDecoder_fold 2) heldCert_accepts
  held_fits_reflected held_row held_col (hull_target_minimum held_fits).1
  (hull_target_minimum held_fits).2.1

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery
