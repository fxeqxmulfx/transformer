module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.ReadoutKernel.Identification
public import Mathlib.Algebra.CharP.Two
public import Mathlib.Tactic.FinCases
public import Mathlib.Tactic.LinearCombination

/-!
# Strictly more unseen-query coverage after derived readout grouping

Source: `review/flat_spectrum.md`, CM35. Three observed corners have three
different raw labels but two declared answers. The ungrouped complete kernel
refuses the fourth corner. Grouping the known answer fibers forces a match,
and the original unknown-code premises prove its declared answer. Two raw
targets still differ at that genuinely unobserved query.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.ReadoutKernel

open KernelCodes

/-- Three original observed cells, with the fourth corner absent, CM35. -/
def cornerCells : Fin 3 → Bool × Bool :=
  ![(false, false), (true, false), (false, true)]

/-- Distinct raw labels whose last two entries share an answer, CM35. -/
def cornerLabels : Fin 3 → ZMod 2 × Bool := ![(0, false), (1, false), (1, true)]

/-- Nonconstant categorical axis phases for the concrete witness, CM35. -/
def cornerPhase (b : Bool) : ZMod 2 := if b then 1 else 0

/-- Both raw completions fit the same three observations, CM35. -/
def cornerTarget (which i j : Bool) : ZMod 2 × Bool :=
  if i then (if j then (0, which) else (1, false))
  else (if j then (1, true) else (0, false))

/-- Actual ungrouped matrix, with exactly CM34's two shift gauges, CM35. -/
abbrev cornerRawKernel := observedKernel (R := ZMod 2) cornerCells id false (0 : Fin 3)

/-- Both free directions satisfy all three original equations, CM35. -/
def cornerRawVector (x y : ZMod 2) : Coordinates Bool Bool (Fin 3) → ZMod 2 :=
  pack (fun i => if i then x else 0) (fun j => if j then y else 0) ![0, x, y]

/-- The displayed directions belong to the actual observed kernel, CM35. -/
theorem cornerRawVector_mem (x y : ZMod 2) : cornerRawVector x y ∈ cornerRawKernel := by
  refine ⟨?_, rfl, rfl⟩
  intro t
  fin_cases t <;> simp [cornerCells, cornerRawVector, queryAction, pack]

/-- A complete basis, without selecting either displayed direction, CM35. -/
noncomputable def cornerRawBasis := Module.Free.chooseBasis (ZMod 2) cornerRawKernel

/-- Each candidate disagrees with a genuine original-kernel direction,
so the full vector query matches no original raw class, CM35. -/
theorem corner_raw_no_match (c : Fin 3) :
    queryCode cornerRawBasis true true ≠ classCode cornerRawBasis c := by
  intro h
  have hall := (code_match_iff cornerRawBasis true true c).mp h
  fin_cases c
  · have hz := hall ⟨cornerRawVector 1 0, cornerRawVector_mem 1 0⟩
    norm_num [queryAction, cornerRawVector, pack] at hz
  · have hz := hall ⟨cornerRawVector 0 1, cornerRawVector_mem 0 1⟩
    norm_num [queryAction, cornerRawVector, pack] at hz
  · have hz := hall ⟨cornerRawVector 1 0, cornerRawVector_mem 1 0⟩
    norm_num [queryAction, cornerRawVector, pack] at hz

/-- Complete ungrouped search really refuses the unseen query, CM35. -/
theorem corner_raw_refuses :
    chooseClass cornerRawBasis (queryCode cornerRawBasis true true) [0, 1, 2] = none := by
  classical
  simp only [chooseClass, corner_raw_no_match, ite_false]

/-- The new kernel changes only observed labels by the declared readout,
and keeps the same observations and legitimate gauges, CM35. -/
abbrev cornerGroupedKernel := observedKernel (R := ZMod 2) cornerCells
  (Prod.fst ∘ cornerLabels) false 0

/-- A complete basis of the grouped original equations, CM35. -/
noncomputable def cornerGroupedBasis := Module.Free.chooseBasis (ZMod 2) cornerGroupedKernel

/-- The three grouped equations force the unseen answer-code match in
every compatible direction, CM35. The extra constraint is derived from the
finite-code premises rather than asserted about the unknown raw target. -/
theorem corner_grouped_match :
    queryCode cornerGroupedBasis true true = classCode cornerGroupedBasis 0 := by
  apply (code_match_iff _ _ _ _).mpr
  intro w
  have h00 := w.property.1 (0 : Fin 3)
  have h10 := w.property.1 (1 : Fin 3)
  have h01 := w.property.1 (2 : Fin 3)
  change w.val (.inl false) + w.val (.inr (.inl false)) -
    w.val (.inr (.inr 0)) = 0 at h00
  change w.val (.inl true) + w.val (.inr (.inl false)) -
    w.val (.inr (.inr 1)) = 0 at h10
  change w.val (.inl false) + w.val (.inr (.inl true)) -
    w.val (.inr (.inr 1)) = 0 at h01
  change w.val (.inl true) + w.val (.inr (.inl true)) -
    w.val (.inr (.inr 0)) = 0
  have h0 := CharTwo.add_self_eq_zero (w.val (.inr (.inr 0)))
  have h1 := CharTwo.add_self_eq_zero (w.val (.inr (.inr 1)))
  linear_combination h10 + h01 - h00 + h1 - h0

/-- The accepted query is absent from the actual observation list, CM35. -/
theorem corner_unobserved : ∀ t, cornerCells t ≠ (true, true) := by
  intro t
  fin_cases t <;> decide

/-- Both raw completions satisfy the actual observed labels, CM35. -/
theorem corner_target_observed (which : Bool) :
    ∀ t, cornerTarget which (cornerCells t).1 (cornerCells t).2 = cornerLabels t := by
  intro t
  fin_cases t <;> simp [cornerCells, cornerLabels, cornerTarget]

/-- The unknown raw code is additive for both completions, CM35. -/
theorem corner_target_realized (which : Bool) :
    ∀ i j, (cornerTarget which i j).1 = cornerPhase i + cornerPhase j := by
  intro i j
  cases which <;> cases i <;> cases j <;> decide

/-- Correct grouping still leaves a genuine unknown raw alias, CM35. -/
theorem corner_raw_ambiguity :
    cornerTarget false true true ≠ cornerTarget true true true ∧
    (cornerTarget false true true).1 = (cornerTarget true true true).1 := by
  decide

example : cornerRawVector 1 1 ∈ cornerRawKernel := cornerRawVector_mem 1 1
example : queryCode cornerRawBasis true true ≠ classCode cornerRawBasis 0 :=
  corner_raw_no_match 0
example : ∀ t, cornerTarget true (cornerCells t).1 (cornerCells t).2 = cornerLabels t :=
  corner_target_observed true
example : ∀ i j, (cornerTarget true i j).1 = cornerPhase i + cornerPhase j :=
  corner_target_realized true

example : (cornerTarget true true true).1 = (0 : ZMod 2) :=
  named_readout_sound (labels := cornerLabels) Prod.fst cornerGroupedBasis id
    (fun _ => rfl) Prod.fst (fun d => ⟨(d, false), rfl⟩) (fun _ _ h => h) rfl
    cornerPhase cornerPhase (cornerTarget true) (corner_target_realized true)
    (corner_target_observed true) true true 0 corner_grouped_match

example : (cornerTarget true true true).1 = (0 : ZMod 2) :=
  matched_readout_sound Prod.fst cornerGroupedBasis Prod.fst
    (fun d => ⟨(d, false), rfl⟩) (fun _ _ h => h) rfl
    cornerPhase cornerPhase (cornerTarget true) (corner_target_realized true)
    (corner_target_observed true) true true 0 corner_grouped_match

example : (cornerTarget true true true).1 = (0 : ZMod 2) :=
  choose_readout_sound Prod.fst cornerGroupedBasis Prod.fst
    (fun d => ⟨(d, false), rfl⟩) (fun _ _ h => h) rfl
    cornerPhase cornerPhase (cornerTarget true) (corner_target_realized true)
    (corner_target_observed true) true true [0, 1]
    (by rw [corner_grouped_match]; exact chooseClass_first _ _ _)

end Transformer.Grokking.FlatSpectrum.CoupledMoments.ReadoutKernel
