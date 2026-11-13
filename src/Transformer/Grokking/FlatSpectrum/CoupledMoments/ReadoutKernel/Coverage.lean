module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.ReadoutKernel.ComplexHead

/-!
# Declared grouping preserves full-kernel match coverage

Source: `review/flat_spectrum.md`, CM36. Pull grouped coefficient vectors back
through the recorded-name readout. The actual observed equations and both
corresponding gauges are preserved. Completeness then transfers every raw
query match, without an unseen-target or unknown-code premise. Native resource
refusals and conditional correctness remain separate.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.ReadoutKernel

open KernelCodes

variable {R I J C D K L S : Type*} [CommRing R]

/-- Copy axis coefficients and compose only the class coefficients with the
explicit declared readout, CM36. This is a linear map of actual coordinates. -/
def readoutPullback (q : C → D) :
    (Coordinates I J D → R) →ₗ[R] (Coordinates I J C → R) where
  toFun w := pack (fun i => w (.inl i)) (fun j => w (.inr (.inl j)))
    (fun c => w (.inr (.inr (q c))))
  map_add' w v := by
    funext x
    rcases x with i | (j | c) <;> rfl
  map_smul' a w := by
    funext x
    rcases x with i | (j | c) <;> rfl

/-- The copied signed matrix row is the corresponding grouped row, CM36. -/
theorem pullback_query (q : C → D) (w : Coordinates I J D → R) (i : I) (j : J) (c : C) :
    queryAction i j c (readoutPullback q w) = queryAction i j (q c) w := rfl

/-- Every grouped fitting vector pulls back to the original fitting space,
with both gauges checked rather than replaced by stronger constraints, CM36. -/
theorem pullback_mem (q : C → D) (cell : S → I × J) (labels : S → C) (i0 : I) (c0 : C)
    (w : Coordinates I J D → R) (hw : w ∈ observedKernel cell (q ∘ labels) i0 (q c0)) :
    readoutPullback q w ∈ observedKernel cell labels i0 c0 := by
  refine ⟨?_, ?_, ?_⟩
  · intro t
    rw [pullback_query]
    exact hw.1 t
  · exact hw.2.1
  · exact hw.2.2

/-- Grouping cannot discard a full-kernel match to a recorded raw class,
CM36. The conclusion concerns algebraic coverage, not an unseen target. -/
theorem raw_match_grouped (q : C → D) {cell : S → I × J} {labels : S → C}
    {i0 : I} {c0 : C}
    (raw : Module.Basis K R (observedKernel cell labels i0 c0))
    (grouped : Module.Basis L R (observedKernel cell (q ∘ labels) i0 (q c0)))
    (i : I) (j : J) (c : C) (hm : queryCode raw i j = classCode raw c) :
    queryCode grouped i j = classCode grouped (q c) := by
  apply (code_match_iff grouped i j (q c)).mpr
  intro w
  have hw := pullback_mem q cell labels i0 c0 w.val w.property
  have h := (code_match_iff raw i j c).mp hm ⟨readoutPullback q w.val, hw⟩
  change queryAction i j c (readoutPullback q w.val) = 0 at h
  rw [pullback_query] at h
  exact h

variable {M : Submodule R (Coordinates I J C → R)}

/-- A successful ordered search returns a member of its recorded bank,
CM36. The bank supplies no unobserved class names. -/
theorem search_mem (b : Module.Basis K R M) (z : K → R) (bank : List C) {c : C}
    (h : chooseClass b z bank = some c) : c ∈ bank := by
  induction bank with
  | nil => cases h
  | cons a rest ih =>
    classical
    by_cases ha : z = classCode b a
    · simp only [chooseClass, ha, ite_true] at h
      cases h
      exact List.mem_cons_self
    · simp only [chooseClass, ha, ite_false] at h
      exact List.mem_cons_of_mem a (ih h)

/-- An actual matching member makes the ordered search succeed, even if
an earlier member has the same vector, CM36. No injectivity is assumed. -/
theorem search_exists (b : Module.Basis K R M) (z : K → R) (bank : List C) {c : C}
    (hc : c ∈ bank) (hm : z = classCode b c) : ∃ d, chooseClass b z bank = some d := by
  induction bank with
  | nil => cases hc
  | cons a rest ih =>
    classical
    by_cases ha : z = classCode b a
    · exact ⟨a, by simp only [chooseClass, ha, ite_true]⟩
    · have hr : c ∈ rest := by
        rcases List.mem_cons.mp hc with h | h
        · subst c
          exact False.elim (ha hm)
        · exact h
      obtain ⟨d, hd⟩ := ih hr
      exact ⟨d, by simpa only [chooseClass, ha, ite_false] using hd⟩

/-- Mapping the original bank through the declared readout cannot turn
an accepted full-kernel query into refusal, CM36. The selected group can
require the separate CM35 target premises to be called a correct answer. -/
theorem raw_search_grouped (q : C → D) {cell : S → I × J} {labels : S → C}
    {i0 : I} {c0 : C}
    (raw : Module.Basis K R (observedKernel cell labels i0 c0))
    (grouped : Module.Basis L R (observedKernel cell (q ∘ labels) i0 (q c0)))
    (i : I) (j : J) (bank : List C) {c : C}
    (h : chooseClass raw (queryCode raw i j) bank = some c) :
    ∃ d, chooseClass grouped (queryCode grouped i j) (bank.map q) = some d := by
  have hc := search_mem raw _ bank h
  have hm := raw_match_grouped q raw grouped i j c (chooseClass_match raw _ bank h)
  exact search_exists grouped _ (bank.map q) (List.mem_map_of_mem hc) hm

private def groupedExample : Coordinates Bool Bool (ZMod 2) → ZMod 2 :=
  pack cornerPhase cornerPhase id

private theorem groupedExample_mem : groupedExample ∈ cornerGroupedKernel := by
  refine ⟨?_, rfl, rfl⟩
  intro t
  fin_cases t <;> norm_num [groupedExample, pack, cornerPhase, cornerCells,
    cornerLabels, Function.comp_apply, queryAction]

example := pullback_query (Prod.fst ∘ cornerLabels) groupedExample true false (1 : Fin 3)

example : readoutPullback (Prod.fst ∘ cornerLabels) groupedExample ∈ cornerRawKernel :=
  pullback_mem _ cornerCells id false 0 groupedExample groupedExample_mem

example : queryCode cornerGroupedBasis true false = classCode cornerGroupedBasis 1 :=
  raw_match_grouped (Prod.fst ∘ cornerLabels) cornerRawBasis cornerGroupedBasis true false 1
    (observed_code_match cornerCells id false 0 cornerRawBasis 1)

example : (0 : Fin 3) ∈ [0, 1, 2] :=
  search_mem cornerRawBasis (classCode cornerRawBasis 0) [0, 1, 2]
    (chooseClass_first cornerRawBasis 0 [1, 2])

example : ∃ c, chooseClass cornerRawBasis (classCode cornerRawBasis 1) [0, 1, 2] = some c :=
  search_exists cornerRawBasis _ [0, 1, 2] (c := 1) (by decide) rfl

example : ∃ d, chooseClass cornerGroupedBasis (queryCode cornerGroupedBasis false false)
    ([0, 1, 2].map (Prod.fst ∘ cornerLabels)) = some d := by
  apply raw_search_grouped (Prod.fst ∘ cornerLabels) cornerRawBasis cornerGroupedBasis
    false false [0, 1, 2] (c := 0)
  have h := observed_code_match cornerCells id false 0 cornerRawBasis 0
  change queryCode cornerRawBasis false false = classCode cornerRawBasis 0 at h
  rw [h]
  exact chooseClass_first _ _ _

/-- Coverage inclusion can be strict on the actual three-corner systems,
CM36. The missing query is still absent from all original observations. -/
theorem corner_coverage_strict :
    (∀ t, cornerCells t ≠ (true, true)) ∧
    chooseClass cornerRawBasis (queryCode cornerRawBasis true true) [0, 1, 2] = none ∧
    chooseClass cornerGroupedBasis (queryCode cornerGroupedBasis true true) [0, 1] = some 0 := by
  refine ⟨corner_unobserved, corner_raw_refuses, ?_⟩
  rw [corner_grouped_match]
  exact chooseClass_first _ _ _

end Transformer.Grokking.FlatSpectrum.CoupledMoments.ReadoutKernel
