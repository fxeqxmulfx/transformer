module

public import Transformer.Grokking.FiniteRecovery.DivisionCertificate
public import Mathlib.Algebra.Field.ZMod

/-! A kernel-checkable witness for the mask used by the modular-division
experiment. Row zero is the constant numerator. Row i+1 and column j
are the nonzero residues i+1 and j+1. The native data loader exports
each row as a bit vector of observed denominator indices. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand

open Set FlatSpectrum

private theorem unit_val_pos (u : DivisionUnit) : 0 < u.val.val := by
  let : Fact (Nat.Prime 97) := ⟨by decide⟩
  apply Nat.pos_of_ne_zero
  intro h
  apply u.ne_zero
  exact ZMod.val_injective 97 h

def divisionUnitIndex : DivisionUnit ≃ Fin 96 where
  toFun u := ⟨u.val.val-1, by have := u.val.val_lt; have := unit_val_pos u; omega⟩
  invFun i := by
    letI : Fact (Nat.Prime 97) := ⟨by decide⟩
    exact Units.mk0 (G₀ := ZMod 97) (⟨i.val+1,by omega⟩ : Fin 97) (by
      intro h
      have := congrArg Fin.val h
      change i.val+1 = 0 at this
      omega)
  left_inv u := by
    apply Units.ext
    apply ZMod.val_injective 97
    change u.val.val - 1 + 1 = u.val.val
    have := unit_val_pos u
    omega
  right_inv i := Fin.ext (by rfl)

/-- Bit i corresponds to denominator i+1. -/
def bitDivisionMask (rows : Fin 97 → ℕ) : Set DivisionInput :=
  {x | (rows (x.1.elim 0 (fun u => (divisionUnitIndex u).succ))).testBit (divisionUnitIndex x.2).val}

/-- The three integer witnesses are enough to check graph connectivity
without traversing the graph in the theorem prover. The conclusion
quantifies over all unknown factors and all 239-token score tables. -/
theorem bitMask_certifies_division (rows : Fin 97 → ℕ) (bridge cover : Fin 96 → Fin 96) (z : Fin 96)
    (hrows : ∀ i, (rows i.succ).testBit (bridge i).val = true ∧ (rows 1).testBit (bridge i).val = true)
    (hcols : ∀ j, (rows (cover j).succ).testBit j.val = true)
    (hz : (rows 0).testBit z.val = true) (score : DivisionInput → Fin 239 → ℝ) :
    classCertificate (bitDivisionMask rows) (factorClass unitToken) (fun x => divisionAnswer x) score ↔
      StrictOn score divisionAnswer := by
  let e := divisionUnitIndex
  apply division_certificate_iff
  · apply preconnected_of_row_bridges _ (e.symm 0)
    · intro i
      obtain ⟨h,h'⟩ := hrows (e i)
      refine ⟨e.symm (bridge (e i)),?_,?_⟩
      · change (rows (e i).succ).testBit (e (e.symm (bridge (e i)))).val = true
        simpa using h
      · change (rows (e (e.symm 0)).succ).testBit (e (e.symm (bridge (e i)))).val = true
        simpa using h'
    · intro j
      refine ⟨e.symm (cover (e j)),?_⟩
      change (rows (e (e.symm (cover (e j)))).succ).testBit (e j).val = true
      simpa using hcols (e j)
  · refine ⟨e.symm z,?_⟩
    change (rows 0).testBit (e (e.symm z)).val = true
    simpa using hz

private def crossRows (i : Fin 97) : ℕ :=
  if i = 1 then 2^96 - 1 else 1

/-- A sparse example has an observed anchor row, one cell in each
other unit row and a single observation of the zero numerator. -/
example (score : DivisionInput → Fin 239 → ℝ) :
    classCertificate (bitDivisionMask crossRows) (factorClass unitToken) (fun x => divisionAnswer x) score ↔
      StrictOn score divisionAnswer :=
  bitMask_certifies_division crossRows (fun _ => 0) (fun _ => 0) 0
    (by decide) (by decide) (by decide) score

end Transformer.Grokking.FiniteRecovery.Argand
