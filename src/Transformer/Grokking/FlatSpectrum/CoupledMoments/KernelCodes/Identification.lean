module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.KernelCodes.Basic
public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedConsequences.Quotient
public import Mathlib.LinearAlgebra.Basis.VectorSpace

/-!
# Conditional unseen answers from a full-kernel match

Source: `review/flat_spectrum.md`, CM34. Normalize the unknown compatible
target code into the actual observed kernel, then apply the basis comparison
theorem. Observed class names can embed into a larger raw vocabulary. Unknown
raw aliases need not have been observed, and matching does not identify their
raw names without a compatible declared readout. The positive example has a
genuinely unobserved query and two possible raw aliases.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.KernelCodes

variable {R I J C A D K S : Type*} [CommRing R]
variable {cell : S → I × J} {labels : S → C} {i0 : I} {c0 : C}

/-- A checked full-kernel match identifies the target code under the
explicit additive target and observed-fit hypotheses, CM34. The target code
is used only in this proof, not supplied to `queryCode` or the class search. -/
theorem matched_code_sound (b : Module.Basis K R (observedKernel cell labels i0 c0))
    (name : C → A) (e : A → R) (r : I → R) (s : J → R) (y : I → J → A)
    (hreal : ∀ i j, e (y i j) = r i + s j)
    (hobs : ∀ t, y (cell t).1 (cell t).2 = name (labels t))
    (i : I) (j : J) (c : C) (hmatch : queryCode b i j = classCode b c) :
    e (y i j) = e (name c) := by
  let w := pack r s (e ∘ name)
  have hw : ∀ t, queryAction (cell t).1 (cell t).2 (labels t) w = 0 := by
    intro t
    change r (cell t).1 + s (cell t).2 - e (name (labels t)) = 0
    rw [← hobs t, hreal]
    ring
  have hn := normalize_mem cell labels i0 c0 w hw
  have hz := (code_match_iff b i j c).mp hmatch ⟨normalize i0 c0 w, hn⟩
  rw [normalize_query] at hz
  change r i + s j - e (name c) = 0 at hz
  rw [hreal]
  exact sub_eq_zero.mp hz

/-- Compatible decoding identifies the unchanged task answer, CM34.
Compatibility concerns the unknown target code on the whole raw vocabulary;
it is not inferred from the observed vector-code fibers or low training CE. -/
theorem matched_readout_sound (b : Module.Basis K R (observedKernel cell labels i0 c0))
    (name : C → A) (e : A → R) (q : A → D)
    (hq : ∀ a a', e a = e a' → q a = q a')
    (r : I → R) (s : J → R) (y : I → J → A)
    (hreal : ∀ i j, e (y i j) = r i + s j)
    (hobs : ∀ t, y (cell t).1 (cell t).2 = name (labels t))
    (i : I) (j : J) (c : C) (hmatch : queryCode b i j = classCode b c) :
    q (y i j) = q (name c) := by
  apply hq
  exact matched_code_sound b name e r s y hreal hobs i j c hmatch

/-- A successful observed-class search supplies the checked equality
needed for code identification, CM34. Its bank never receives unseen labels. -/
theorem chooseClass_code_sound (b : Module.Basis K R (observedKernel cell labels i0 c0))
    (name : C → A) (e : A → R) (r : I → R) (s : J → R) (y : I → J → A)
    (hreal : ∀ i j, e (y i j) = r i + s j)
    (hobs : ∀ t, y (cell t).1 (cell t).2 = name (labels t))
    (i : I) (j : J) (bank : List C) {c : C}
    (hfound : chooseClass b (queryCode b i j) bank = some c) :
    e (y i j) = e (name c) := by
  apply matched_code_sound b name e r s y hreal hobs i j c
  exact chooseClass_match b _ bank hfound

/-- Successful search preserves a compatible declared readout, CM34,
without claiming raw-label equality or universal search coverage. -/
theorem chooseClass_readout_sound (b : Module.Basis K R (observedKernel cell labels i0 c0))
    (name : C → A) (e : A → R) (q : A → D)
    (hq : ∀ a a', e a = e a' → q a = q a')
    (r : I → R) (s : J → R) (y : I → J → A)
    (hreal : ∀ i j, e (y i j) = r i + s j)
    (hobs : ∀ t, y (cell t).1 (cell t).2 = name (labels t))
    (i : I) (j : J) (bank : List C) {c : C}
    (hfound : chooseClass b (queryCode b i j) bank = some c) :
    q (y i j) = q (name c) := by
  apply hq
  exact chooseClass_code_sound b name e r s y hreal hobs i j bank hfound

open DecodedConsequences (aliasCells aliasLabels aliasPhases aliasTarget alias_observed)

/-- A complete basis of CM30's actual finite alias witness, CM34. Its
existence is standard vector-space linear algebra, not an answer hypothesis. -/
noncomputable def aliasBasis := Module.Free.chooseBasis (ZMod 2)
  (observedKernel (R := ZMod 2) aliasCells aliasLabels (0 : Fin 3) (0, false))

/-- The three actual observed equations force a match at the unobserved
(2,2) cell in every direction of the complete basis, CM34. -/
theorem alias_match : queryCode aliasBasis (2 : Fin 3) 2 = classCode aliasBasis (0, false) := by
  apply (code_match_iff _ _ _ _).mpr
  intro w
  have h02 := w.property.1 (6 : Fin 8)
  have h20 := w.property.1 (2 : Fin 8)
  have h00 := w.property.1 (0 : Fin 8)
  change w.val (.inl 0) + w.val (.inr (.inl 2)) - w.val (.inr (.inr (0, false))) = 0 at h02
  change w.val (.inl 2) + w.val (.inr (.inl 0)) - w.val (.inr (.inr (0, false))) = 0 at h20
  change w.val (.inl 0) + w.val (.inr (.inl 0)) - w.val (.inr (.inr (0, false))) = 0 at h00
  change w.val (.inl 2) + w.val (.inr (.inl 2)) - w.val (.inr (.inr (0, false))) = 0
  linear_combination h02 + h20 - h00

example : (aliasTarget true 2 2).1 = (0 : ZMod 2) :=
  matched_code_sound aliasBasis id Prod.fst aliasPhases aliasPhases (aliasTarget true)
    (alias_observed.2.2.2 true) (alias_observed.2.2.1 true) 2 2 (0, false) alias_match

example : (aliasTarget true 2 2).1 = (0 : ZMod 2) :=
  matched_readout_sound aliasBasis id Prod.fst Prod.fst (fun _ _ h => h)
    aliasPhases aliasPhases (aliasTarget true) (alias_observed.2.2.2 true)
    (alias_observed.2.2.1 true) 2 2 (0, false) alias_match

example : (aliasTarget true 2 2).1 = (0 : ZMod 2) :=
  chooseClass_code_sound aliasBasis id Prod.fst aliasPhases aliasPhases (aliasTarget true)
    (alias_observed.2.2.2 true) (alias_observed.2.2.1 true) 2 2 [(0, false)] (c := (0, false))
    (by rw [alias_match]; exact chooseClass_first _ _ [])

example : (aliasTarget true 2 2).1 = (0 : ZMod 2) :=
  chooseClass_readout_sound aliasBasis id Prod.fst Prod.fst (fun _ _ h => h)
    aliasPhases aliasPhases (aliasTarget true) (alias_observed.2.2.2 true)
    (alias_observed.2.2.1 true) 2 2 [(0, false)] (c := (0, false))
    (by rw [alias_match]; exact chooseClass_first _ _ [])

example : aliasTarget false 2 2 ≠ aliasTarget true 2 2 :=
  DecodedConsequences.alias_raw_obstruction.1

-- Neither raw alias is selected by the unknown target in the search definition.
/-- The full-kernel comparison certifies a genuinely unobserved cell,
CM34. It is not merely interpolation of an original observed row. -/
theorem alias_unobserved_match :
    (∀ t, aliasCells t ≠ (2, 2)) ∧
    queryCode aliasBasis (2 : Fin 3) 2 = classCode aliasBasis (0, false) := by
  exact ⟨alias_observed.2.1, alias_match⟩

/-- Raw identification still fails for the full-kernel search, CM34:
its accepted representative can differ from a compatible unseen raw target.
The positive decoded examples above retain the original quotient answer. -/
theorem alias_search_raw_obstruction :
    chooseClass aliasBasis (queryCode aliasBasis (2 : Fin 3) 2) [(0, false)] =
      some (aliasTarget false 2 2) ∧
    chooseClass aliasBasis (queryCode aliasBasis (2 : Fin 3) 2) [(0, false)] ≠
      some (aliasTarget true 2 2) := by
  rw [alias_match, chooseClass_first]
  decide

end Transformer.Grokking.FlatSpectrum.CoupledMoments.KernelCodes
