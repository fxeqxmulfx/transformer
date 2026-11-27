module

public import Transformer.Grokking.FiniteRecovery.DivisionMask
public import Transformer.Grokking.FlatSpectrum.ProductCells

/-! The exact observation threshold for group-valued factors with an
additional unknown constant row. The unit graph must be connected and the
constant must be observed. For a square unit block the same optimal mask
supports unique Flat Spectrum nuclear-norm completion of its character
table (review/flat_spectrum.md, Theorem E). -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

open FlatSpectrum Set Matrix

variable {I J G Y : Type*} [AddCommGroup G] [Nontrivial G]

/-- The two graph conditions characterize identifiability for the entire
class of unknown row factors, column factors and constant-row tokens. -/
theorem factorClass_identified_iff [Nonempty I] [Nonempty J] (code : G ↪ Y) (S : Set (Option I × J)) :
    InjOn (fun f : Option I × J → Y => fun x : S => f x) (factorClass code) ↔
      (patternGraph (unitMask S)).Connected ∧ ∃ j, (none,j) ∈ S := by
  refine ⟨fun hid => ?_,fun ⟨hc,hz⟩ => factorClass_identified code S hc.preconnected hz⟩
  constructor
  · apply connected_of_rigid (G := G)
    intro r c h
    let z := code 0
    have he : factorTable code r c z = factorTable code 0 0 z := by
      apply hid ⟨r,c,z,rfl⟩ ⟨0,0,z,rfl⟩
      funext x
      obtain ⟨⟨i,j⟩,hx⟩ := x
      cases i with
      | none => rfl
      | some i => simpa [factorTable] using congrArg code (h (i,j) hx)
    intro i j
    exact code.injective (by simpa [factorTable] using congrFun he (some i,j))
  · by_contra! h
    obtain ⟨g,hg⟩ := exists_ne (0 : G)
    have he : factorTable (I := I) (J := J) code 0 0 (code g) = factorTable code 0 0 (code 0) := by
      apply hid ⟨0,0,code g,rfl⟩ ⟨0,0,code 0,rfl⟩
      funext x
      obtain ⟨⟨i,j⟩,hx⟩ := x
      cases i with
      | none => exact (h j hx).elim
      | some i => rfl
    exact hg (code.injective (congrFun he (none,Classical.choice ‹Nonempty J›)))

def liftUnitCell (x : I × J) : Option I × J := (some x.1,x.2)

theorem liftUnitCell_injective : Function.Injective (liftUnitCell (I := I) (J := J)) := by
  rintro ⟨i,j⟩ ⟨k,l⟩ he
  simpa [liftUnitCell] using he

private theorem none_not_image (Ω : Set (I × J)) (j : J) : (none,j) ∉ liftUnitCell '' Ω := by
  rintro ⟨⟨i,k⟩,_,he⟩
  have h : some i = (none : Option I) := congrArg Prod.fst he
  cases h

/-- A spanning unit graph costs at least |I|+|J|−1 labels. The unknown
constant row requires one further label, even for an unrestricted learner. -/
theorem factor_observation_lower [Finite I] [Finite J] [Nonempty I] [Nonempty J]
    (code : G ↪ Y) (S : Set (Option I × J))
    (hid : InjOn (fun f : Option I × J → Y => fun x : S => f x) (factorClass code)) :
    Nat.card I + Nat.card J ≤ S.ncard := by
  obtain ⟨hc,j,hj⟩ := (factorClass_identified_iff code S).mp hid
  have h := card_add_card_le_ncard_add_one_of_connected hc
  have hsub : insert (none,j) (liftUnitCell '' unitMask S) ⊆ S := by
    apply insert_subset hj
    rintro _ ⟨ij,hij,rfl⟩
    exact hij
  have he := ncard_le_ncard hsub
  rw [ncard_insert_of_notMem (none_not_image _ j),ncard_image_of_injective _ liftUnitCell_injective] at he
  exact h.trans he

/-- The diagonal, one full unit row and one observation of the constant. -/
def factorMask (anchor : I) : Set (Option I × I) :=
  insert (none,anchor) (liftUnitCell '' diagRow anchor)

theorem unitMask_factorMask (anchor : I) : unitMask (factorMask anchor) = diagRow anchor := by
  ext ⟨i,j⟩
  change (some i,j) ∈ insert (none,anchor) (liftUnitCell '' diagRow anchor) ↔ _
  simp only [mem_insert_iff,Prod.mk.injEq,reduceCtorEq,false_and,false_or,mem_image]
  constructor
  · rintro ⟨x,hx,h⟩
    exact liftUnitCell_injective (show liftUnitCell x = liftUnitCell (i,j) from h) ▸ hx
  · exact fun h => ⟨(i,j),h,rfl⟩

theorem ncard_factorMask [Fintype I] (anchor : I) :
    (factorMask anchor).ncard = 2 * Fintype.card I := by
  rw [factorMask,ncard_insert_of_notMem (none_not_image _ _),ncard_image_of_injective _ liftUnitCell_injective,
    ncard_diagRow]
  have hp := Fintype.card_pos_iff.mpr ⟨anchor⟩
  omega

/-- The same 2N labels identify every unknown factored token table; any
fixed mask with fewer labels leaves at least two indistinguishable tables. -/
theorem isLeast_factor_observations [Fintype I] (anchor : I) (code : G ↪ Y) :
    IsLeast {k : ℕ | ∃ S : Set (Option I × I), S.ncard = k ∧
      InjOn (fun f : Option I × I → Y => fun x : S => f x) (factorClass code)} (2 * Fintype.card I) := by
  let : Nonempty I := ⟨anchor⟩
  refine ⟨⟨factorMask anchor,ncard_factorMask anchor,(factorClass_identified_iff _ _).2 ?_⟩,?_⟩
  · rw [unitMask_factorMask]
    exact ⟨diagRow_connected _,anchor,mem_insert _ _⟩
  · rintro k ⟨S,rfl,hs⟩
    simpa [two_mul,Nat.card_eq_fintype_card] using factor_observation_lower code S hs

/-- The unit portion includes a perfect matching. For any unit-modulus
rank-one readout its unique nuclear-norm minimizer is the original table. -/
theorem factorMask_nuclear_completion [Fintype I] [DecidableEq I] (anchor : I) {u v : I → ℂ}
    (h : ∀ i j, ‖vecMulVec u v i j‖ = 1) :
    let Ω := unitMask (factorMask anchor)
    ∀ X ∈ {X' : Matrix I I ℂ | ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec u v ij.1 ij.2},
      (IsMinOn nuclearNorm {X' | ∀ ij ∈ Ω, X' ij.1 ij.2 = vecMulVec u v ij.1 ij.2} X ↔ X = vecMulVec u v) := by
  let : Nonempty I := ⟨anchor⟩
  dsimp only
  rw [unitMask_factorMask]
  exact (forall_isMinOn_vecMulVec_iff_connected_of_entries h).2
    ⟨diagRow_connected _,Equiv.refl _,fun i => Or.inl ⟨i,rfl⟩⟩

namespace Argand

/-- For the 96 nonzero residues of the original 97-modulus experiment, 191
unit cells and one constant-row cell achieve the exact class minimum. -/
theorem division_isLeast_192 :
    IsLeast {k : ℕ | ∃ S : Set DivisionInput, S.ncard = k ∧
      InjOn (fun f : DivisionInput → Fin 239 => fun x : S => f x) (factorClass unitToken)} 192 := by
  let : Nontrivial (Additive DivisionUnit) := divisionUnitIndex.symm.injective.nontrivial
  have hcard : Fintype.card DivisionUnit = 96 := (Fintype.card_congr divisionUnitIndex).trans (Fintype.card_fin 96)
  simpa [hcard] using isLeast_factor_observations (1 : DivisionUnit) unitToken

example := factorMask_nuclear_completion (1 : DivisionUnit) (u := fun _ => 1) (v := fun _ => 1)
  (by simp [vecMulVec_apply])

end Argand
end Transformer.Grokking.FiniteRecovery
