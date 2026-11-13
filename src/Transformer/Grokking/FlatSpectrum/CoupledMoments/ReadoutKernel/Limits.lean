module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.ReadoutKernel.Witness

/-!
# Necessary hypotheses for derived readout-fiber constraints

Source: `review/flat_spectrum.md`, CM35. Compatibility alone does not collapse
readout aliases when the code alphabet is larger. Counting unused declared
answers also cannot replace readout surjectivity. An unknown code incompatible
with the readout cannot inherit the factorization conclusion.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.ReadoutKernel

/-- A surjective two-answer readout from three distinct latent codes, CM35. -/
def wideReadout (a : ZMod 3) : Bool := decide (a = 1)

/-- Equal readout values need not imply equal compatible codes when the
code alphabet is larger, CM35. Thus dropping the cardinality condition would
add false class equations, even for an injective unknown raw code. -/
theorem larger_code_counterexample :
    Function.Surjective wideReadout ∧
    (∀ a b : ZMod 3, id a = id b → wideReadout a = wideReadout b) ∧
    Fintype.card Bool < Fintype.card (ZMod 3) ∧
    wideReadout 0 = wideReadout 2 ∧ (0 : ZMod 3) ≠ 2 ∧
    ¬ ∃ φ : Bool ≃ ZMod 3, ∀ a, a = φ (wideReadout a) := by
  refine ⟨?_, ?_, by decide, by decide, by decide, ?_⟩
  · intro b
    cases b
    · exact ⟨0, by decide⟩
    · exact ⟨1, by decide⟩
  · intro a b h
    exact congrArg wideReadout h
  · rintro ⟨φ, hφ⟩
    have hz : (0 : ZMod 3) = 2 := by
      calc
        0 = φ (wideReadout 0) := hφ 0
        _ = φ (wideReadout 2) := congrArg φ (by decide)
        _ = 2 := (hφ 2).symm
    exact (by decide : (0 : ZMod 3) ≠ 2) hz

/-- Counting an unused answer does not justify alias equations, CM35.
Here the nominal code/answer cardinalities agree and the identity code is
compatible, but the readout is not surjective and the codes of its aliases differ. -/
theorem unused_answer_counterexample :
    ¬ Function.Surjective (fun _ : Bool => false) ∧
    (∀ a b : Bool, id a = id b → (fun _ : Bool => false) a = (fun _ : Bool => false) b) ∧
    Fintype.card Bool = Fintype.card Bool ∧
    ¬ ∃ φ : Bool ≃ Bool, ∀ a, a = φ false := by
  refine ⟨?_, ?_, rfl, ?_⟩
  · intro h
    obtain ⟨a, ha⟩ := h true
    exact Bool.false_ne_true ha
  · intro a b h
    cases h
    rfl
  · rintro ⟨φ, hφ⟩
    exact Bool.false_ne_true ((hφ false).trans (hφ true).symm)

/-- Equal cardinalities and a surjective readout do not establish unknown
code compatibility, CM35. A constant code cannot factor through a bijection
of the two distinct answers. -/
theorem incompatible_code_counterexample :
    Function.Surjective (id : Bool → Bool) ∧
    Fintype.card Bool = Fintype.card Bool ∧
    ¬ (∀ a b : Bool, (fun _ : Bool => false) a = (fun _ : Bool => false) b → a = b) ∧
    ¬ ∃ φ : Bool ≃ Bool, ∀ a, false = φ a := by
  refine ⟨fun b => ⟨b, rfl⟩, rfl, ?_, ?_⟩
  · intro h
    exact Bool.false_ne_true (h false true rfl)
  · rintro ⟨φ, hφ⟩
    exact Bool.false_ne_true (φ.injective ((hφ false).symm.trans (hφ true)))

end Transformer.Grokking.FlatSpectrum.CoupledMoments.ReadoutKernel
