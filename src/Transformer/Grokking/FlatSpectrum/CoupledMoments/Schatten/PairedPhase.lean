module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.Paired
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.DecodedHull

/-!
# Symmetric latent phase witnesses and ordinary cross-entropy

Source: `review/flat_spectrum.md`, CM16. A phase point and its negative have
the same output under a fixed even decoder. Their average has equal paired
probabilities and satisfies the original radius-N, nuclear-budget-N bounds.
Observed fitting certifies ordinary CE optimality, without a recovery claim.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten

variable {q : ℕ} [NeZero q] {n C S : Type*}

/-- Half of the probability at a phase and half at its negative (CM16).
When the two phases coincide, their masses add to one. -/
noncomputable def pairedPhase (r s : n → ZMod q) : Table n n (ZMod q) :=
  pairLatent (pointMass (fun i j => r i + s j))

omit [NeZero q] in
/-- The constructed phase distribution satisfies the additional equalities (CM16). -/
theorem pairedPhase_even (r s : n → ZMod q) : EvenLatent (pairedPhase r s) :=
  pairLatent_even _

variable [Fintype n] [DecidableEq n]

/-- No rank increase in a factorization is required as a domain constraint:
convexity gives both spectral upper bounds for the averaged witness (CM16). -/
theorem pairedPhase_budget (r s : n → ZMod q) :
    BudgetFeasible (phaseCode (Equiv.refl (ZMod q)))
      (fun _ => Fintype.card n) (fun _ => Fintype.card n) (pairedPhase r s) := by
  apply budget_pairLatent
  convert phaseCandidate_budget (Equiv.refl (ZMod q)) r s using 1
  rfl

variable [Fintype C] [DecidableEq C]

omit [Fintype n] [DecidableEq n] [Fintype C] in
/-- A fixed even decoder produces deterministic original labels from this
generally nondeterministic latent distribution (CM16). -/
theorem decode_pairedPhase (d : ZMod q → C) (hd : ∀ z, d (-z) = d z)
    (r s : n → ZMod q) :
    decode d (pairedPhase r s) = pointMass (fun i j => d (r i + s j)) := by
  rw [pairedPhase, decode_pairLatent d hd, decode_pointMass]

omit [Fintype C] in
/-- Every paired phase witness is a feasible output prediction (CM16). -/
theorem pairedPhase_feasible (d : ZMod q → C) (hd : ∀ z, d (-z) = d z)
    (r s : n → ZMod q) :
    PairedFeasible d (Fintype.card n) (Fintype.card n)
      (pointMass (fun i j => d (r i + s j))) := by
  exact ⟨pairedPhase r s, pairedPhase_budget r s, pairedPhase_even r s,
    decode_pairedPhase d hd r s⟩

variable [Fintype S]

/-- Original-vocabulary ordinary CE is convex on the paired prediction domain (CM16). -/
theorem paired_loss_convex (d : ZMod q → C) (R B : ℝ)
    (cell : S → n × n) (labels : S → C) :
    ConvexOn ℝ {Q | PairedFeasible d R B Q} (loss cell labels) := by
  exact (loss_convex cell labels).subset (Set.subset_univ _) (pairedFeasible_convex d R B)

/-- A feasible symmetric latent exact fit is a global ordinary-CE minimum (CM16).
There is no requirement to identify one latent state for each observed label. -/
theorem paired_fit_minimum (d : ZMod q → C) (R B : ℝ)
    (cell : S → n × n) (labels : S → C) {P : Table n n (ZMod q)}
    (hP : BudgetFeasible (phaseCode (Equiv.refl (ZMod q))) (fun _ => R) (fun _ => B) P)
    (heven : EvenLatent P)
    (hfit : ∀ t, decode d P (cell t).1 (cell t).2 = ConvexRecall.targetLogits (labels t)) :
    PairedFeasible d R B (decode d P) ∧
      IsMinOn (loss cell labels) {Q | PairedFeasible d R B Q} (decode d P) := by
  refine ⟨⟨P, hP, heven, rfl⟩, ?_⟩
  rintro Q ⟨T, hT, _, rfl⟩
  rw [loss_eq_ideal_of_fit cell labels hfit]
  exact idealLoss_le cell labels (rows_decode d hT.1.1)

/-- Decoded training compatibility constructs a symmetric feasible CE minimizer (CM16). -/
theorem pairedPhase_minimum (d : ZMod q → C) (hd : ∀ z, d (-z) = d z)
    (r s : n → ZMod q) (cell : S → n × n) (labels : S → C)
    (hfit : ∀ t, d (r (cell t).1 + s (cell t).2) = labels t) :
    PairedFeasible d (Fintype.card n) (Fintype.card n)
        (pointMass (fun i j => d (r i + s j))) ∧
      IsMinOn (loss cell labels)
        {Q | PairedFeasible d (Fintype.card n) (Fintype.card n) Q}
        (pointMass (fun i j => d (r i + s j))) := by
  have hf : ∀ t, decode d (pairedPhase r s) (cell t).1 (cell t).2 =
      ConvexRecall.targetLogits (labels t) := by
    intro t
    rw [decode_pairedPhase d hd]
    funext c
    simp [pointMass, hfit t, ConvexRecall.targetLogits]
  simpa only [decode_pairedPhase d hd] using paired_fit_minimum d _ _ cell labels
    (pairedPhase_budget r s) (pairedPhase_even r s) hf

/-- The same symmetric witness attains the CE lower bound even against the
larger hull of all fixed decoders (CM16). This is not a discovery algorithm. -/
theorem pairedPhase_hull_minimum (d : ZMod q → C) (hd : ∀ z, d (-z) = d z)
    (r s : n → ZMod q) (cell : S → n × n) (labels : S → C)
    (hfit : ∀ t, d (r (cell t).1 + s (cell t).2) = labels t) :
    PairedFeasible d (Fintype.card n) (Fintype.card n)
        (pointMass (fun i j => d (r i + s j))) ∧
      IsMinOn (loss cell labels)
        (DecoderHull (phaseCode (Equiv.refl (ZMod q)))
          (fun _ => Fintype.card n) (fun _ => Fintype.card n))
        (pointMass (fun i j => d (r i + s j))) := by
  exact ⟨pairedPhase_feasible d hd r s,
    (decodedPhase_hull_minimum d r s cell labels hfit).2⟩

/-! Inhabited certificates use an even decoder of three states into two classes. -/

private def witnessDecoder (z : ZMod 3) : Bool := decide (z ≠ 0)
private theorem witness_even (z : ZMod 3) : witnessDecoder (-z) = witnessDecoder z := by
  simp [witnessDecoder]
private def witnessLabels (t : Fin 3) : Bool := decide (t ≠ 0)
private theorem witness_fit (t : Fin 3) :
    witnessDecoder (cornerPhase (threeCells t).1 + cornerPhase (threeCells t).2) =
      witnessLabels t := by
  fin_cases t <;> norm_num [witnessDecoder, witnessLabels, cornerPhase, threeCells]

example := pairedPhase_even cornerPhase cornerPhase
example := pairedPhase_budget cornerPhase cornerPhase
example := decode_pairedPhase witnessDecoder witness_even cornerPhase cornerPhase
example := pairedPhase_feasible witnessDecoder witness_even cornerPhase cornerPhase
example := paired_loss_convex witnessDecoder 2 2 threeCells witnessLabels
example := paired_fit_minimum witnessDecoder 2 2 threeCells witnessLabels
  (by simpa using pairedPhase_budget cornerPhase cornerPhase)
  (pairedPhase_even cornerPhase cornerPhase)
  (by
    intro t
    rw [decode_pairedPhase witnessDecoder witness_even]
    funext c
    simp [pointMass, witness_fit t, ConvexRecall.targetLogits])
example := pairedPhase_minimum witnessDecoder witness_even cornerPhase cornerPhase
  threeCells witnessLabels witness_fit
example := pairedPhase_hull_minimum witnessDecoder witness_even cornerPhase cornerPhase
  threeCells witnessLabels witness_fit
example := witness_even 1
example := witness_fit 1

end Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten
