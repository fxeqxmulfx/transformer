module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.KernelCodes.Witness
public import Mathlib.Data.Fintype.Card
public import Mathlib.Data.Set.Finite.Range
public import Mathlib.Data.Fintype.Sum

/-!
# Finite states from the actual range of full vector codes

Source: `review/flat_spectrum.md`, CM34. Only codes reached by active input
pairs become states. The number of states is bounded by the number of such
pairs even when the field or basis is large. A fixed observed-class bank
provides representatives; its explicit fallback has no correctness promise.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.KernelCodes

variable {R I J C K A D S : Type*} [CommRing R]
variable {M : Submodule R (Coordinates I J C → R)}

/-- The actual range of the vector-valued query map, CM34. The whole
ambient vector space is not materialized as a categorical state space. -/
abbrev CodeStates (b : Module.Basis K R M) :=
  ↥(Set.range (fun x : I × J => queryCode b x.1 x.2))

/-- Every active pair maps to its full vector code in the actual range, CM34. -/
noncomputable def codeState (b : Module.Basis K R M) (x : I × J) : CodeStates b :=
  ⟨queryCode b x.1 x.2, ⟨x, rfl⟩⟩

noncomputable instance codeStatesFintype [Finite I] [Finite J] (b : Module.Basis K R M) :
    Fintype (CodeStates b) := Fintype.ofFinite _

noncomputable instance codeStatesDecidableEq (b : Module.Basis K R M) :
    DecidableEq (CodeStates b) := Classical.decEq _

/-- The number of full-vector states depends on the active input product,
CM34. In particular the number of states need not be the field size to the nullity. -/
theorem codeState_card_bound [Fintype I] [Fintype J] (b : Module.Basis K R M) :
    Fintype.card (CodeStates b) ≤ Fintype.card I * Fintype.card J := by
  have h := Fintype.card_range_le (fun x : I × J => queryCode b x.1 x.2)
  simpa only [Fintype.card_prod] using h

/-- Ordered observed-class search supplies a representative, with an
explicit fallback on refusal, CM34. No target or evaluation score enters it. -/
noncomputable def codeRepresentative (b : Module.Basis K R M) (bank : List C)
    (fallback : C) (z : CodeStates b) : C :=
  (chooseClass b z.val bank).getD fallback

/-- An accepted match determines the representative before any CE
training, CM34. The fallback matters only when the search refuses. -/
theorem representative_of_match (b : Module.Basis K R M) (bank : List C) (fallback : C)
    (i : I) (j : J) {c : C} (h : chooseClass b (queryCode b i j) bank = some c) :
    codeRepresentative b bank fallback (codeState b (i, j)) = c := by
  unfold codeRepresentative codeState
  rw [h, Option.getD_some]

/-- A refused state keeps the stated fallback and makes no identifying
claim, CM34. This branch is explicit in the total complex-head default. -/
theorem representative_of_refusal (b : Module.Basis K R M) (bank : List C) (fallback : C)
    (i : I) (j : J) (h : chooseClass b (queryCode b i j) bank = none) :
    codeRepresentative b bank fallback (codeState b (i, j)) = fallback := by
  unfold codeRepresentative codeState
  rw [h]
  rfl

/-- A complete-basis change cannot change the resulting representative
when the bank order and fallback stay fixed, CM34. -/
theorem representative_basis_eq {L : Type*} (b : Module.Basis K R M)
    (d : Module.Basis L R M) (bank : List C) (fallback : C) (i : I) (j : J) :
    codeRepresentative b bank fallback (codeState b (i, j)) =
      codeRepresentative d bank fallback (codeState d (i, j)) := by
  unfold codeRepresentative codeState
  rw [chooseClass_basis_eq b d]

/-- A successful full-vector search supplies a correct declared decoded
default under the actual unknown-code realizability premises, CM34. These
premises concern the task, and are not inferred from observed CE or flatness. -/
theorem representative_readout_sound
    {cell : S → I × J} {labels : S → C} {i0 : I} {c0 : C}
    (b : Module.Basis K R (observedKernel cell labels i0 c0))
    (name : C → A) (e : A → R) (q : A → D)
    (hq : ∀ a a', e a = e a' → q a = q a')
    (r : I → R) (s : J → R) (y : I → J → A)
    (hreal : ∀ i j, e (y i j) = r i + s j)
    (hobs : ∀ t, y (cell t).1 (cell t).2 = name (labels t))
    (bank : List C) (fallback : C) (i : I) (j : J) {c : C}
    (hfound : chooseClass b (queryCode b i j) bank = some c) :
    q (name (codeRepresentative b bank fallback (codeState b (i, j)))) = q (y i j) := by
  rw [representative_of_match b bank fallback i j hfound]
  exact (chooseClass_readout_sound b name e q hq r s y hreal hobs i j bank hfound).symm

example : Fintype.card (CodeStates diagonalBasis) ≤ 4 := by
  simpa using codeState_card_bound diagonalBasis

example : codeRepresentative diagonalBasis [true, false] false
    (codeState diagonalBasis (true, true)) = true := by
  apply representative_of_match
  have hm : queryCode diagonalBasis true true = classCode diagonalBasis true :=
    observed_code_match (fun b : Bool => (b, b)) id false false diagonalBasis true
  rw [hm]
  exact chooseClass_first _ _ _

example : codeRepresentative diagonalBasis [false, true] false
    (codeState diagonalBasis (true, false)) = false :=
  representative_of_refusal _ _ _ _ _ diagonal_refuses

example := representative_basis_eq diagonalBasis shearedBasis [false, true] false true false

open DecodedConsequences (aliasCells aliasLabels aliasPhases aliasTarget alias_observed)

example : (codeRepresentative aliasBasis [(0, false)] (1, true)
    (codeState aliasBasis ((2 : Fin 3), 2))).1 = (aliasTarget true 2 2).1 :=
  representative_readout_sound aliasBasis id Prod.fst Prod.fst (fun _ _ h => h)
    aliasPhases aliasPhases (aliasTarget true) (alias_observed.2.2.2 true)
    (alias_observed.2.2.1 true) [(0, false)] (1, true) 2 2 (c := (0, false))
    (by rw [alias_match]; exact chooseClass_first _ _ [])

/-- Even the alias witness uses at most nine actual latent states,
CM34. A raw vocabulary alias does not force enumeration of an ambient vector space. -/
theorem alias_state_count_bound : Fintype.card (CodeStates aliasBasis) ≤ 9 := by
  simpa using codeState_card_bound aliasBasis

/-- Observed-input refinement adds precisely the supplied observation
count to the finite latent range, CM34. This charges defaults and observations. -/
theorem codeState_refined_card_bound [Fintype I] [Fintype J]
    (b : Module.Basis K R M) (observed : Finset (I × J)) :
    Fintype.card (↥observed ⊕ CodeStates b) ≤
      observed.card + Fintype.card I * Fintype.card J := by
  simpa only [Fintype.card_sum, Fintype.card_coe] using
    Nat.add_le_add_left (codeState_card_bound b) observed.card

/-- The exact-real64N coefficient allowance has the same product bound,
CM34. This is a coefficient budget, not a bound on basis or Fourier storage. -/
theorem codeState_budget_bound [Fintype I] [Fintype J]
    (b : Module.Basis K R M) (observed : Finset (I × J)) :
    64 * ((observed.card : ℝ) + (Fintype.card (CodeStates b) : ℝ)) ≤
      64 * ((observed.card : ℝ) + Fintype.card I * Fintype.card J) := by
  have h : (Fintype.card (CodeStates b) : ℝ) ≤
      (Fintype.card I : ℝ) * Fintype.card J := by
    exact_mod_cast codeState_card_bound b
  gcongr

example := codeState_refined_card_bound diagonalBasis ({(false, false)} : Finset (Bool × Bool))

example := codeState_budget_bound diagonalBasis ({(false, false)} : Finset (Bool × Bool))

end Transformer.Grokking.FlatSpectrum.CoupledMoments.KernelCodes

