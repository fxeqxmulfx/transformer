/-
# Phase model — a spurious local minimizer of a generating partial table

Below `n²/8` every local minimizer of `V_full` is the phases of a character
up to periods (`exists_sub_characterPhase_mem_periods_of_isLocalMin`).  This
fails for partial tables `S` whose rows generate those of the full table
over `ℤ`, although the zeros of `V_S` are still the characters up to periods
(`divisionEnergy_eq_zero_iff_of_mem_span`).  For `n = 5` the fifteen pairs of
`spuriousTable`, the nine of the forms `(x, 0)` and `(x, x)` and six more,
generate the full table (`mem_span_divisionRow_spuriousTable`).  At the
integer phases `j = (0, 1, 1, 1, 3)` the six have the defects `1` for
`(4, 1)`, `(4, 3)`, `(3, 4)` and `-1` for `(1, 3)`, `(2, 1)`, `(3, 1)`
modulo `6`, and both triples of rows sum to `-2e₁ - e₃`.  So `(π/3) j` is a
twisted state: a local minimizer of energy `3 < 5²/8`
(`isLocalMin_divisionEnergy_spuriousTable`, `divisionEnergy_spuriousTable`)
that is not a character up to periods
(`exists_mem_span_isLocalMin_divisionEnergy_eq_three`,
`not_forall_exists_sub_characterPhase_mem_periods_of_isLocalMin`).
-/

import Transformer.PhaseModel.Twisted

open Real
open scoped Finset

namespace Transformer.PhaseModel

/-- **The fifteen pairs of the counterexample** in `ℤ/5`: the nine of the
forms `(x, 0)` and `(x, x)`, of row `-e₀`, and `(4, 1)`, `(4, 3)`, `(3, 4)`,
`(1, 3)`, `(2, 1)`, `(3, 1)`. -/
def spuriousTable : Finset (Fin 5 × Fin 5) :=
  {(0, 0), (1, 0), (1, 1), (2, 0), (2, 2), (3, 0), (3, 3), (4, 0), (4, 4), (4, 1), (4, 3),
    (3, 4), (1, 3), (2, 1), (3, 1)}

/-- **The fifteen pairs generate the full table over `ℤ`**: with `v` the row
of `p` and the integer `k = (v₁ + 2v₂ + 3v₃ + 4v₄)/5`,
`v = -v₀ r₀₀ + (v₂ + v₃ + v₄ - 2k) r₂₁ + (v₃ + v₄ - 2k) r₃₁ + v₄ r₄₁ - k r₁₃`
for the rows `r_{xy}` of `(x, y)`. -/
theorem mem_span_divisionRow_spuriousTable (p : Fin 5 × Fin 5) :
    divisionRow p ∈ Submodule.span ℤ (divisionRow '' (spuriousTable : Set (Fin 5 × Fin 5))) := by
  have hmem : ∀ q ∈ spuriousTable, ∀ c : ℤ, c • divisionRow q ∈
      Submodule.span ℤ (divisionRow '' (spuriousTable : Set (Fin 5 × Fin 5))) :=
    fun q hq c => Submodule.smul_mem _ c (Submodule.subset_span ⟨q, hq, rfl⟩)
  have key : ∀ p : Fin 5 × Fin 5, let v := divisionRow p
      let k := (v 1 + 2 * v 2 + 3 * v 3 + 4 * v 4) / 5
      v = (-v 0) • divisionRow (0, 0) + (v 2 + v 3 + v 4 - 2 * k) • divisionRow (2, 1) +
        (v 3 + v 4 - 2 * k) • divisionRow (3, 1) + v 4 • divisionRow (4, 1) +
        (-k) • divisionRow (1, 3) := by
    decide
  have h := key p
  dsimp only at h
  rw [h]
  exact add_mem (add_mem (add_mem (add_mem (hmem _ (by decide) _) (hmem _ (by decide) _))
    (hmem _ (by decide) _)) (hmem _ (by decide) _)) (hmem _ (by decide) _)

/-- **The twisted state of the counterexample is a local minimizer**: the
phases `(π/3)(0, 1, 1, 1, 3)` of `V_S` for the fifteen pairs. -/
theorem isLocalMin_divisionEnergy_spuriousTable :
    IsLocalMin (divisionEnergy spuriousTable) ((π / 3) • intVec ![0, 1, 1, 1, 3]) :=
  isLocalMin_divisionEnergy_twisted mem_span_divisionRow_spuriousTable (by decide) (by decide)

/-- **The energy of the twisted state is `3`**: six pairs have the cosine
`1/2`, the nine others `1`. -/
theorem divisionEnergy_spuriousTable :
    divisionEnergy spuriousTable ((π / 3) • intVec ![0, 1, 1, 1, 3]) = 3 := by
  rw [divisionEnergy_twisted (by decide),
    show #{p ∈ spuriousTable | defect ![0, 1, 1, 1, 3] p % 6 ≠ 0} = 6 by decide]
  norm_num

/-- **A generating partial table with a spurious local minimizer**: for
`n = 5` some `S` generates the full table over `ℤ`, so that the zeros of
`V_S` are the characters up to periods, and yet `V_S` has a local minimizer
`ψ` of energy `3`, below `5²/8`, that is not a character up to periods. -/
theorem exists_mem_span_isLocalMin_divisionEnergy_eq_three :
    ∃ S : Finset (Fin 5 × Fin 5),
      (∀ p, divisionRow p ∈ Submodule.span ℤ (divisionRow '' (S : Set (Fin 5 × Fin 5)))) ∧
      ∃ ψ : EucSpace 5, IsLocalMin (divisionEnergy S) ψ ∧ divisionEnergy S ψ = 3 ∧
        ∀ j : ℤ, ψ - characterPhase 5 j ∉ periods 5 := by
  refine ⟨spuriousTable, mem_span_divisionRow_spuriousTable, _,
    isLocalMin_divisionEnergy_spuriousTable, divisionEnergy_spuriousTable, fun j hj => ?_⟩
  have h := (divisionEnergy_eq_zero_iff_of_mem_span mem_span_divisionRow_spuriousTable _).2
    ⟨j, hj⟩
  rw [divisionEnergy_spuriousTable] at h
  norm_num at h

/-- **The landscape of `V_full` below `n²/8` does not pass to generating
partial tables**: `exists_sub_characterPhase_mem_periods_of_isLocalMin`, with
the full table replaced by a set `S` of pairs whose rows generate it over
`ℤ`, is false already for `n = 5`
(`exists_mem_span_isLocalMin_divisionEnergy_eq_three`). -/
theorem not_forall_exists_sub_characterPhase_mem_periods_of_isLocalMin :
    ¬∀ (n : ℕ) (S : Finset (Fin n × Fin n)), 4 ≤ n →
      (∀ p, divisionRow p ∈ Submodule.span ℤ (divisionRow '' (S : Set (Fin n × Fin n)))) →
      ∀ ψ, IsLocalMin (divisionEnergy S) ψ → divisionEnergy S ψ < n ^ 2 / 8 →
        ∃ j : ℤ, ψ - characterPhase n j ∈ periods n := by
  intro h
  obtain ⟨S, hS, ψ, hmin, hV, hψ⟩ := exists_mem_span_isLocalMin_divisionEnergy_eq_three
  obtain ⟨j, hj⟩ := h 5 S (by norm_num) hS ψ hmin (by rw [hV]; norm_num)
  exact hψ j hj

end Transformer.PhaseModel
