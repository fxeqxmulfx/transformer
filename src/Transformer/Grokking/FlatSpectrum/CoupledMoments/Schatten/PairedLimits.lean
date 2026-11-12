module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.PairedCoding

/-!
# A constructor obstruction and the remaining identification limit

Source: `review/flat_spectrum.md`, CM16. A complete cyclic column permutation
forces the same class masses in every row, for every decoder. Convex mixtures
preserve that restriction. The larger symmetric latent family admits a table
outside it, but still has distinct observed CE minima with different unseen
answers. Neither capacity nor observed fitting alone proves recovery.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten

variable {m n C : Type*} [Fintype n]

/-- Every input row has the same total mass of each output class (CM16). -/
def EqualRowMass (P : Table m n C) : Prop := ∀ i k c, ∑ j, P i j c = ∑ j, P k j c

/-- The constructor's equal-mass restriction survives all convex mixtures (CM16). -/
theorem equalRowMass_convex : Convex ℝ {P : Table m n C | EqualRowMass P} := by
  intro P hP Q hQ a b _ _ _ i k c
  simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul, Finset.sum_add_distrib,
    ← Finset.mul_sum, hP i k c, hQ i k c]

variable {q : ℕ} [NeZero q] [DecidableEq C]

/-- A column permutation traverses each latent phase once in every row (CM16).
Row phases are arbitrary; the decoder may merge classes in any way. -/
theorem permutationPhase_row_mass (d : ZMod q → C) (r : m → ZMod q)
    (s : n ≃ ZMod q) (i : m) (c : C) :
    (∑ j, pointMass (fun i j => d (r i + s j)) i j c) =
      ∑ z : ZMod q, if c = d z then (1 : ℝ) else 0 := by
  apply Fintype.sum_equiv (s.trans (Equiv.addLeft (r i)))
  intro j
  rfl

/-- The old permutation-phase constructor necessarily has equal row histograms (CM16). -/
theorem permutationPhase_equalRowMass (d : ZMod q → C) (r : m → ZMod q)
    (s : n ≃ ZMod q) : EqualRowMass (pointMass (fun i j => d (r i + s j))) := by
  intro i k c
  rw [permutationPhase_row_mass d r s i c, permutationPhase_row_mass d r s k c]

/-- The three-class witness violates that invariant (CM16). -/
theorem folded_unequalRowMass : ¬ EqualRowMass (pointMass foldedLabels) := by
  intro h
  have hc := h 0 1 2
  simp only [Fin.sum_univ_succ] at hc
  norm_num [pointMass, foldedLabels] at hc

/-- No decoder or choice of row phases repairs the three-state permutation model (CM16). -/
theorem folded_no_permutationPhase (d : ZMod 3 → Fin 3) (r : Fin 3 → ZMod 3)
    (s : Fin 3 ≃ ZMod 3) :
    pointMass (fun i j => d (r i + s j)) ≠ pointMass foldedLabels := by
  intro he
  have h := permutationPhase_equalRowMass d r s
  rw [he] at h
  exact folded_unequalRowMass h

/-- Mixing any number of these restricted constructors still cannot fit the witness (CM16).
This excludes their hull, not the full latent probability domain of CM15. -/
theorem folded_not_permutationHull :
    pointMass foldedLabels ∉ convexHull ℝ
      {P : Table (Fin 3) (Fin 3) (Fin 3) |
        ∃ (d : ZMod 3 → Fin 3) (r : Fin 3 → ZMod 3) (s : Fin 3 ≃ ZMod 3),
          P = pointMass (fun i j => d (r i + s j))} := by
  intro h
  apply folded_unequalRowMass
  apply convexHull_min (t := {P | EqualRowMass P}) ?_ equalRowMass_convex h
  rintro P ⟨d, r, s, rfl⟩
  exact permutationPhase_equalRowMass d r s

/-- A symmetric latent probability table admits the previously excluded point (CM16). -/
theorem folded_new_capacity :
    PairedFeasible (foldDecoder 2) 3 3 (pointMass foldedLabels) ∧
      pointMass foldedLabels ∉ convexHull ℝ
        {P : Table (Fin 3) (Fin 3) (Fin 3) |
          ∃ (d : ZMod 3 → Fin 3) (r : Fin 3 → ZMod 3) (s : Fin 3 ≃ ZMod 3),
            P = pointMass (fun i j => d (r i + s j))} := by
  exact ⟨folded_paired_admitted, folded_not_permutationHull⟩

/-- The two-class paired decoder merges +1 and -1 and keeps zero separate (CM16). -/
def pairedBoolDecoder (z : ZMod 3) : Bool := decide (z ≠ 0)

/-- The ambiguity witness uses the same even-decoder hypothesis (CM16). -/
theorem pairedBoolDecoder_even (z : ZMod 3) : pairedBoolDecoder (-z) = pairedBoolDecoder z := by
  simp [pairedBoolDecoder]

/-- The shared observed labels of the two completions (CM16). -/
def pairedThreeLabels (t : Fin 3) : Bool := decide (t ≠ 0)

private theorem positive_fit (t : Fin 3) :
    pairedBoolDecoder (cornerPhase (threeCells t).1 + cornerPhase (threeCells t).2) =
      pairedThreeLabels t := by
  fin_cases t <;> norm_num [pairedBoolDecoder, pairedThreeLabels, cornerPhase, threeCells]

private theorem negative_fit (t : Fin 3) :
    pairedBoolDecoder (cornerPhase (threeCells t).1 + -cornerPhase (threeCells t).2) =
      pairedThreeLabels t := by
  fin_cases t <;> norm_num [pairedBoolDecoder, pairedThreeLabels, cornerPhase, threeCells]

/-- Same decoder, symmetry, budgets, connected graph and perfect matching; two
global ordinary-CE minima predict opposite unseen labels (CM16). -/
theorem paired_unseen_ambiguity :
    let domain := {Q : Table (Fin 2) (Fin 2) Bool | PairedFeasible pairedBoolDecoder 2 2 Q}
    let P := pointMass (fun i j => pairedBoolDecoder (cornerPhase i + cornerPhase j))
    let Q := pointMass (fun i j => pairedBoolDecoder (cornerPhase i + -cornerPhase j))
    P ∈ domain ∧ Q ∈ domain ∧
      IsMinOn (loss threeCells pairedThreeLabels) domain P ∧
      IsMinOn (loss threeCells pairedThreeLabels) domain Q ∧
      P 1 1 true = 1 ∧ Q 1 1 true = 0 ∧
      (patternGraph (Set.range threeCells)).Connected ∧
      ∃ σ : Equiv.Perm (Fin 2), ∀ i, (i, σ i) ∈ Set.range threeCells := by
  have hp := pairedPhase_minimum pairedBoolDecoder pairedBoolDecoder_even
    cornerPhase cornerPhase threeCells pairedThreeLabels positive_fit
  have hn := pairedPhase_minimum pairedBoolDecoder pairedBoolDecoder_even
    cornerPhase (fun j => -cornerPhase j) threeCells pairedThreeLabels negative_fit
  simp only [Fintype.card_fin, Nat.cast_ofNat] at hp hn
  refine ⟨hp.1, hn.1, hp.2, hn.2, ?_, ?_, corner_decoded_graph_coverage⟩
  · norm_num [pointMass, cornerPhase, pairedBoolDecoder]
    decide
  · norm_num [pointMass, cornerPhase, pairedBoolDecoder]

example := permutationPhase_row_mass (id : ZMod 3 → ZMod 3)
  (fun i : Fin 2 => (i.val : ZMod 3)) (Equiv.refl (ZMod 3)) 0 1
example := permutationPhase_equalRowMass (id : ZMod 3 → ZMod 3)
  (fun i : Fin 2 => (i.val : ZMod 3)) (Equiv.refl (ZMod 3))
example := pairedBoolDecoder_even 1
example := positive_fit 1
example := negative_fit 1

end Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten
