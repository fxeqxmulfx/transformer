module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.PairedPhase

/-!
# A vocabulary-only paired decoder

Source: `review/flat_spectrum.md`, CM16. For C = k+1 original classes, use
q = 2k+1 latent states and merge the pair z,-z. Every original class has a
preimage. These are categorical output coordinates, not input arithmetic
features. The three-class example separates this construction from the
permutation-phase constructor of `experiments/convex_stress_decode`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten

/-- Fold opposite latent states into k+1 output coordinates (CM16). -/
def foldDecoder (k : ℕ) (z : ZMod (2 * k + 1)) : Fin (k + 1) :=
  ⟨min z.val (2 * k + 1 - z.val), by
    have h := ZMod.val_lt z
    have h₁ := min_le_left z.val (2 * k + 1 - z.val)
    have h₂ := min_le_right z.val (2 * k + 1 - z.val)
    omega⟩

/-- The proposed decoder meets exactly the symmetry hypothesis of CM16. -/
theorem foldDecoder_even (k : ℕ) (z : ZMod (2 * k + 1)) :
    foldDecoder k (-z) = foldDecoder k z := by
  apply Fin.ext
  change min (-z).val (2 * k + 1 - (-z).val) = min z.val (2 * k + 1 - z.val)
  by_cases hz : z = 0
  · subst z
    simp
  · rw [ZMod.neg_val, ite_eq_right hz]
    have h := ZMod.val_lt z
    have he : 2 * k + 1 - (2 * k + 1 - z.val) = z.val := by omega
    rw [he, min_comm]

/-- The nonnegative half contains one representative of each original class (CM16). -/
theorem foldDecoder_representative (k : ℕ) (c : Fin (k + 1)) :
    foldDecoder k (c.val : ZMod (2 * k + 1)) = c := by
  apply Fin.ext
  change min (c.val : ZMod (2 * k + 1)).val
    (2 * k + 1 - (c.val : ZMod (2 * k + 1)).val) = c.val
  rw [ZMod.val_natCast_of_lt (by omega : c.val < 2 * k + 1)]
  exact min_eq_left (by omega)

/-- No original output class is deleted from the decoder or CE vocabulary (CM16). -/
theorem foldDecoder_surjective (k : ℕ) : Function.Surjective (foldDecoder k) := by
  intro c
  exact ⟨(c.val : ZMod (2 * k + 1)), foldDecoder_representative k c⟩

/-- Each decoder fiber is precisely one opposite pair, with a singleton at zero (CM16).
This identifies what a forward-only decoded constraint must sum over. -/
theorem foldDecoder_fiber (k : ℕ) (z : ZMod (2 * k + 1)) (c : Fin (k + 1)) :
    foldDecoder k z = c ↔
      z = (c.val : ZMod (2 * k + 1)) ∨ z = -(c.val : ZMod (2 * k + 1)) := by
  have hc : (c.val : ZMod (2 * k + 1)).val = c.val :=
    ZMod.val_natCast_of_lt (by omega)
  constructor
  · intro he
    have hv : min z.val (2 * k + 1 - z.val) = c.val := congrArg Fin.val he
    by_cases hs : z.val ≤ 2 * k + 1 - z.val
    · left
      apply ZMod.val_injective
      rw [hc]
      rw [min_eq_left hs] at hv
      exact hv
    · right
      have hz : z ≠ 0 := by
        intro hz
        subst z
        simp at hs
      apply neg_eq_iff_eq_neg.mp
      apply ZMod.val_injective
      rw [ZMod.neg_val, ite_eq_right hz, hc]
      rw [min_eq_right (by omega)] at hv
      exact hv
  · rintro (rfl | rfl)
    · exact foldDecoder_representative k c
    · rw [foldDecoder_even]
      exact foldDecoder_representative k c

/-- A fixed categorical renaming retains the paired property (CM16). -/
theorem renamed_foldDecoder_even {C : Type*} (k : ℕ) (e : Fin (k + 1) ≃ C)
    (z : ZMod (2 * k + 1)) : e (foldDecoder k (-z)) = e (foldDecoder k z) := by
  rw [foldDecoder_even]

/-- Every output class is retained after a fixed categorical renaming (CM16). -/
theorem renamed_foldDecoder_surjective {C : Type*} (k : ℕ) (e : Fin (k + 1) ≃ C) :
    Function.Surjective (fun z => e (foldDecoder k z)) := by
  intro c
  obtain ⟨z, hz⟩ := foldDecoder_surjective k (e.symm c)
  exact ⟨z, (congrArg e hz).trans (e.apply_symm_apply c)⟩

/-- Three categorical outputs with unequal counts across rows (CM16). -/
def foldedLabels : Fin 3 → Fin 3 → Fin 3 := ![![0, 1, 2], ![1, 0, 1], ![2, 1, 0]]

/-- The two input axes need only three of the five latent phases (CM16). -/
def foldedRow (i : Fin 3) : ZMod 5 := i.val

/-- Opposite categorical coordinates for the second axis of the witness (CM16). -/
def foldedColumn (j : Fin 3) : ZMod 5 := -(j.val : ZMod 5)

/-- Five latent states realize the table without a bijection from either input axis (CM16). -/
theorem folded_decoding :
    (fun i j => foldDecoder 2 (foldedRow i + foldedColumn j)) = foldedLabels := by
  funext i j
  fin_cases i <;> fin_cases j <;> decide

/-- Its symmetric latent table meets the same radius-N and nuclear-budget-N bounds (CM16).
Here input side N=3, latent states q=5 and output vocabulary C=3 are distinct quantities. -/
theorem folded_paired_admitted :
    PairedFeasible (foldDecoder 2) 3 3 (pointMass foldedLabels) := by
  simpa only [folded_decoding, Fintype.card_fin, Nat.cast_ofNat] using
    pairedPhase_feasible (foldDecoder 2) (foldDecoder_even 2) foldedRow foldedColumn

/-- The admitted model has the usual original-vocabulary CE certificate (CM16).
This full-observation example is capacity evidence, not a held-out recovery theorem. -/
theorem folded_paired_minimum :
    IsMinOn (loss (id : Fin 3 × Fin 3 → Fin 3 × Fin 3) (fun ij => foldedLabels ij.1 ij.2))
      {Q | PairedFeasible (foldDecoder 2) 3 3 Q} (pointMass foldedLabels) := by
  have h := pairedPhase_minimum (foldDecoder 2) (foldDecoder_even 2) foldedRow foldedColumn
    (id : Fin 3 × Fin 3 → Fin 3 × Fin 3) (fun ij => foldedLabels ij.1 ij.2)
    (fun ij => congrArg (fun f => f ij.1 ij.2) folded_decoding)
  simpa only [folded_decoding, Fintype.card_fin, Nat.cast_ofNat] using h.2

example := foldDecoder_even 2 1
example := foldDecoder_representative 2 2
example := foldDecoder_surjective 2
example := foldDecoder_fiber 2 3 2
example := renamed_foldDecoder_even 2 (Equiv.refl (Fin 3)) 1
example := renamed_foldDecoder_surjective 2 (Equiv.refl (Fin 3))

end Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten
