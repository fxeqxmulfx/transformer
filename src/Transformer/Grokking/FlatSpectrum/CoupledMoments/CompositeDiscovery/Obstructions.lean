module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.CompositeDiscovery.Witness

/-!
# Missing axes cannot be silently pinned for identification

Source: `review/flat_spectrum.md`, CM28. CM27's native constructor explicitly
chooses zero for missing axes. That is permitted as a proposal, but it cannot
be used in the observed identification certificate. A missing nonanchor row
supports an independent kernel vector, so the real joint module is not cyclic
when a bijective-code target exists. The learner must refuse this condition.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.CompositeDiscovery

variable {q : ℕ} [Nontrivial (ZMod q)] {n C S : Type*}
variable {cell : S → n × n} {labels : S → C}

/-- A genuinely missing row leaves a free direction beyond the code gauge,
CM28. Pinning it to zero would incorrectly erase this observable obstruction. -/
theorem missing_row_not_cyclic (e : C ≃ ZMod q) (r s : n → ZMod q)
    (hreal : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2)
    (i₀ i : n) (c₀ : C) (hi : i ≠ i₀) (hmiss : ∀ t, (cell t).1 ≠ i) :
    ¬ Cyclic (jointKernel (ZMod q) cell labels i₀ c₀) := by
  classical
  let x : n ⊕ (n ⊕ C) → ZMod q := fun v => if v = .inl i then 1 else 0
  have hx : x ∈ jointKernel (ZMod q) cell labels i₀ c₀ := by
    refine ⟨fun t => ?_, ?_, ?_⟩
    · simp [x, hmiss t]
    · simp [x, Ne.symm hi]
    · simp [x]
  intro hK
  obtain ⟨a, ha⟩ := scalar_of_cyclic hK (gaugeVector_mem e r s i₀ c₀ hreal) hx
    (gaugeVector_one e r s i₀ c₀)
  have hc := congrFun ha (.inr (.inr (e.symm (1 + e c₀))))
  change 0 = a * (e (e.symm (1 + e c₀)) - e c₀) at hc
  rw [e.apply_symm_apply] at hc
  have ha0 : a = 0 := by linear_combination -hc
  have hi1 : (1 : ZMod q) = a * (r i - r i₀) := by
    simpa [x, gaugeVector, pack] using congrFun ha (.inl i)
  rw [ha0, zero_mul] at hi1
  exact one_ne_zero hi1

/-- The observed-only learner refuses rather than selecting an arbitrary
answer for a missing row, under the stated model, CM28. -/
theorem missing_row_refused (e : C ≃ ZMod q) (r s : n → ZMod q)
    (hreal : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2)
    (i₀ i : n) (c₀ : C) (hi : i ≠ i₀) (hmiss : ∀ t, (cell t).1 ≠ i) :
    discover q cell labels i₀ c₀ = none := by
  rw [discover, ite_eq_right (missing_row_not_cyclic e r s hreal i₀ i c₀ hi hmiss)]

/-- A single complete observed row still shows all six categorical classes. -/
def onlyRowCells (j : Fin 7) : Fin 7 × Fin 7 := (0, j)

/-- Actual labels of that row, with zero and one exchanged as above. -/
def onlyRowLabels : Fin 7 → ZMod 6 := ![1, 0, 2, 3, 4, 5, 1]

/-- This missing-axis counterexample is structurally realizable, CM28. -/
theorem onlyRow_real (j : Fin 7) : sixCode (onlyRowLabels j) =
    sixPhase (onlyRowCells j).1 + sixPhase (onlyRowCells j).2 := by
  revert j
  decide

local instance : Fact (1 < 6) := ⟨by decide⟩

example : ¬ Cyclic (jointKernel (ZMod 6) onlyRowCells onlyRowLabels 0 1) :=
  missing_row_not_cyclic sixCode sixPhase sixPhase onlyRow_real 0 1 1
    (by decide) (by decide)

example : discover 6 onlyRowCells onlyRowLabels 0 1 = none :=
  missing_row_refused sixCode sixPhase sixPhase onlyRow_real 0 1 1
    (by decide) (by decide)

end Transformer.Grokking.FlatSpectrum.CoupledMoments.CompositeDiscovery
