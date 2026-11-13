module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.ReadoutKernel.Coverage

/-!
# Grouped coverage with independently chosen gauge anchors

Source: `review/flat_spectrum.md`, CM37. Raw and grouped complete kernels
need not choose corresponding gauge anchors. Normalize the actual coefficient
pullback with CM34's two harmless shifts. Every query action is unchanged, so
coverage inclusion survives unrelated row and class anchors. This does not
certify an unknown target or justify an arbitrary earlier-model replacement.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.ReadoutKernel

open KernelCodes

variable {R I J C D K L S : Type*} [CommRing R]

/-- Normalizing the pulled-back coefficients satisfies the actual raw
equations and gauges even with different grouped anchors, CM37. -/
theorem normalized_pullback_mem (q : C → D) (cell : S → I × J) (labels : S → C)
    (i0 i1 : I) (c0 : C) (d0 : D) (w : Coordinates I J D → R)
    (hw : w ∈ observedKernel cell (q ∘ labels) i1 d0) :
    normalize i0 c0 (readoutPullback q w) ∈ observedKernel cell labels i0 c0 := by
  apply normalize_mem
  intro t
  rw [pullback_query]
  exact hw.1 t

/-- Full raw matches transfer to the grouped system with any independently
chosen anchors, CM37. No unknown-target or equal-cardinality premise is used. -/
theorem raw_match_grouped_gauges (q : C → D) {cell : S → I × J} {labels : S → C}
    {i0 i1 : I} {c0 : C} {d0 : D}
    (raw : Module.Basis K R (observedKernel cell labels i0 c0))
    (grouped : Module.Basis L R (observedKernel cell (q ∘ labels) i1 d0))
    (i : I) (j : J) (c : C) (hm : queryCode raw i j = classCode raw c) :
    queryCode grouped i j = classCode grouped (q c) := by
  apply (code_match_iff grouped i j (q c)).mpr
  intro w
  have hw := normalized_pullback_mem q cell labels i0 i1 c0 d0 w.val w.property
  have h := (code_match_iff raw i j c).mp hm ⟨_, hw⟩
  change queryAction i j c (normalize i0 c0 (readoutPullback q w.val)) = 0 at h
  rw [normalize_query, pullback_query] at h
  exact h

/-- Ordered mapped-bank search retains every raw acceptance regardless
of either system's gauge anchors, CM37. Correctness needs CM35 separately. -/
theorem raw_search_grouped_gauges (q : C → D) {cell : S → I × J} {labels : S → C}
    {i0 i1 : I} {c0 : C} {d0 : D}
    (raw : Module.Basis K R (observedKernel cell labels i0 c0))
    (grouped : Module.Basis L R (observedKernel cell (q ∘ labels) i1 d0))
    (i : I) (j : J) (bank : List C) {c : C}
    (h : chooseClass raw (queryCode raw i j) bank = some c) :
    ∃ d, chooseClass grouped (queryCode grouped i j) (bank.map q) = some d := by
  have hc := search_mem raw _ bank h
  have hm := raw_match_grouped_gauges q raw grouped i j c (chooseClass_match raw _ bank h)
  exact search_exists grouped _ (bank.map q) (List.mem_map_of_mem hc) hm

/-- For an unchanged label system, query matches are independent of both
gauge anchors and the complete basis, CM37. This is an equivalence. -/
theorem gauge_match_iff {cell : S → I × J} {labels : S → C} {i0 i1 : I} {c0 c1 : C}
    (first : Module.Basis K R (observedKernel cell labels i0 c0))
    (second : Module.Basis L R (observedKernel cell labels i1 c1))
    (i : I) (j : J) (c : C) :
    queryCode first i j = classCode first c ↔ queryCode second i j = classCode second c := by
  constructor
  · exact raw_match_grouped_gauges id first second i j c
  · exact raw_match_grouped_gauges id second first i j c

/-- Holding the class bank and its order fixed, the actual search returns
the same class or refusal after either gauge anchor changes, CM37. -/
theorem gauge_search_eq {cell : S → I × J} {labels : S → C} {i0 i1 : I} {c0 c1 : C}
    (first : Module.Basis K R (observedKernel cell labels i0 c0))
    (second : Module.Basis L R (observedKernel cell labels i1 c1))
    (i : I) (j : J) (bank : List C) :
    chooseClass first (queryCode first i j) bank =
      chooseClass second (queryCode second i j) bank := by
  classical
  induction bank with
  | nil => rfl
  | cons c rest ih =>
    have hc := gauge_match_iff first second i j c
    by_cases h : queryCode first i j = classCode first c
    · simp only [chooseClass, h, hc.mp h, ite_true]
    · have h' : queryCode second i j ≠ classCode second c := fun h' => h (hc.mpr h')
      simp only [chooseClass, h, h', ite_false, ih]

private def shiftedGroupedFit : Coordinates Bool Bool (ZMod 2) → ZMod 2 :=
  normalize true 1 (pack cornerPhase cornerPhase id)

private theorem shiftedGroupedFit_mem : shiftedGroupedFit ∈
    observedKernel cornerCells (Prod.fst ∘ cornerLabels) true 1 := by
  apply normalize_mem
  intro t
  fin_cases t <;> norm_num [pack, cornerPhase, cornerCells, cornerLabels,
    Function.comp_apply, queryAction]

private noncomputable def shiftedGroupedBasis := Module.Free.chooseBasis (ZMod 2)
  (observedKernel (R := ZMod 2) cornerCells (Prod.fst ∘ cornerLabels) true 1)

private noncomputable def shiftedRawBasis := Module.Free.chooseBasis (ZMod 2)
  (observedKernel (R := ZMod 2) cornerCells id true (2 : Fin 3))

example : normalize false (0 : Fin 3)
    (readoutPullback (Prod.fst ∘ cornerLabels) shiftedGroupedFit) ∈ cornerRawKernel :=
  normalized_pullback_mem _ cornerCells id false true 0 1 shiftedGroupedFit shiftedGroupedFit_mem

example : queryCode shiftedGroupedBasis true false = classCode shiftedGroupedBasis 1 :=
  raw_match_grouped_gauges (Prod.fst ∘ cornerLabels) cornerRawBasis shiftedGroupedBasis
    true false 1 (observed_code_match cornerCells id false 0 cornerRawBasis 1)

example : ∃ d, chooseClass shiftedGroupedBasis (queryCode shiftedGroupedBasis false false)
    ([0, 1, 2].map (Prod.fst ∘ cornerLabels)) = some d := by
  apply raw_search_grouped_gauges (Prod.fst ∘ cornerLabels) cornerRawBasis shiftedGroupedBasis
    false false [0, 1, 2] (c := 0)
  have h := observed_code_match cornerCells id false 0 cornerRawBasis 0
  change queryCode cornerRawBasis false false = classCode cornerRawBasis 0 at h
  rw [h]
  exact chooseClass_first _ _ _

example := gauge_match_iff cornerRawBasis shiftedRawBasis true true 0

example : chooseClass shiftedRawBasis (queryCode shiftedRawBasis true true) [0, 1, 2] = none := by
  rw [← gauge_search_eq cornerRawBasis shiftedRawBasis true true [0, 1, 2]]
  exact corner_raw_refuses

end Transformer.Grokking.FlatSpectrum.CoupledMoments.ReadoutKernel
