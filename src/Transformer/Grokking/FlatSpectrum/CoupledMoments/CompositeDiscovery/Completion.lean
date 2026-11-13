module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.CompositeDiscovery.Basic

/-!
# Identification from the observed joint module

Source: `review/flat_spectrum.md`, CM28. The latent cyclic domain may be
composite and input phases may repeat. The unknown target has a bijective class
code; neither it nor any unseen label is an argument of the learner.

Cyclicity is imposed on the full joint observed module, not just on class
rectangles with arbitrary pins for missing axes. It therefore also certifies
the phase completion. This identifies a table inside the categorical cyclic
model. It does not identify every completion in the larger convex spectral
hull, and it does not make the code-discovery step jointly convex.

The specification uses classical choice only after the observable algebraic
check. Bounded native elimination may refuse to produce its certificate;
this file asserts neither a runtime bound nor completeness of a capped search.
`Witness` instantiates every conditional result on a composite held-out table.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.CompositeDiscovery

open Schatten

variable {q : ℕ} {n C S : Type*} {cell : S → n × n} {labels : S → C}

/-- Every bijective phase completion of the observations equals the unknown
realizable table when the actual joint module is cyclic, CM28. -/
theorem completion_eq_target (e : C ≃ ZMod q) (r s : n → ZMod q)
    (hreal : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2) (i₀ : n) (c₀ : C)
    (hK : Cyclic (jointKernel (ZMod q) cell labels i₀ c₀))
    (w : C ≃ ZMod q) (ρ σ : n → ZMod q)
    (hfit : ∀ t, w (labels t) = ρ (cell t).1 + σ (cell t).2) :
    phaseLabels w ρ σ = phaseLabels e r s := by
  obtain ⟨a, ha⟩ := scalar_of_cyclic hK
    (gaugeVector_mem e r s i₀ c₀ hreal) (gaugeVector_mem w ρ σ i₀ c₀ hfit)
    (gaugeVector_one e r s i₀ c₀)
  funext i j
  change w.symm (ρ i + σ j) = e.symm (r i + s j)
  rw [Equiv.symm_apply_eq]
  have hr := congrFun ha (.inl i)
  have hs := congrFun ha (.inr (.inl j))
  have hc := congrFun ha (.inr (.inr (e.symm (r i + s j))))
  change ρ i - ρ i₀ = a * (r i - r i₀) at hr
  change σ j + ρ i₀ - w c₀ = a * (s j + r i₀ - e c₀) at hs
  change w (e.symm (r i + s j)) - w c₀ =
    a * (e (e.symm (r i + s j)) - e c₀) at hc
  rw [e.apply_symm_apply] at hc
  linear_combination hr + hs - hc

/-- A table arising from some bijective code and phases that fit only the
observed cells, CM28. No true or held-out answers occur in this predicate. -/
def Completes (q : ℕ) (cell : S → n × n) (labels : S → C)
    (y : n → n → C) : Prop :=
  ∃ (w : C ≃ ZMod q) (ρ σ : n → ZMod q),
    (∀ t, w (labels t) = ρ (cell t).1 + σ (cell t).2) ∧ y = phaseLabels w ρ σ

/-- The entire set of model-compatible completions is a singleton under
unknown structural realizability and the observable joint condition, CM28. -/
theorem completes_iff (e : C ≃ ZMod q) (r s : n → ZMod q)
    (hreal : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2) (i₀ : n) (c₀ : C)
    (hK : Cyclic (jointKernel (ZMod q) cell labels i₀ c₀)) (y : n → n → C) :
    Completes q cell labels y ↔ y = phaseLabels e r s := by
  constructor
  · rintro ⟨w, ρ, σ, hfit, hy⟩
    exact hy.trans (completion_eq_target e r s hreal i₀ c₀ hK w ρ σ hfit)
  · rintro rfl
    exact ⟨e, r, s, hreal, rfl⟩

open Classical in
/-- The observed-only learner specification, CM28. It checks the joint
module, then returns a compatible completion if one exists. There is no
argument for a discovered code, an optimizer or any unobserved answer. -/
noncomputable def discover (q : ℕ) (cell : S → n × n) (labels : S → C)
    (i₀ : n) (c₀ : C) : Option (n → n → C) :=
  if Cyclic (jointKernel (ZMod q) cell labels i₀ c₀) then
    if h : ∃ y, Completes q cell labels y then some h.choose else none
  else none

/-- The specified learner returns the full unknown table from its observed
samples under structural realizability and the observable module check, CM28. -/
theorem discover_eq_some (e : C ≃ ZMod q) (r s : n → ZMod q)
    (hreal : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2) (i₀ : n) (c₀ : C)
    (hK : Cyclic (jointKernel (ZMod q) cell labels i₀ c₀)) :
    discover q cell labels i₀ c₀ = some (phaseLabels e r s) := by
  have hex : ∃ y, Completes q cell labels y := ⟨_, e, r, s, hreal, rfl⟩
  rw [discover, ite_eq_left hK, dite_eq_left hex]
  exact congrArg some ((completes_iff e r s hreal i₀ c₀ hK _).mp hex.choose_spec)

/-- Acceptance exposes the actual two training-only conditions, CM28. This
does not require structural realizability or an optimizer hypothesis. -/
theorem conditions_of_discover {i₀ : n} {c₀ : C} {y : n → n → C}
    (h : discover q cell labels i₀ c₀ = some y) :
    Cyclic (jointKernel (ZMod q) cell labels i₀ c₀) ∧ Completes q cell labels y := by
  unfold discover at h
  split_ifs at h with hK hex
  have hy : hex.choose = y := by injection h
  exact ⟨hK, hy ▸ hex.choose_spec⟩

/-- Under the unknown cyclic model, acceptance cannot select a wrong unseen
answer, CM28. Refusal is retained if the observable condition fails. -/
theorem discover_sound (e : C ≃ ZMod q) (r s : n → ZMod q)
    (hreal : ∀ t, e (labels t) = r (cell t).1 + s (cell t).2)
    {i₀ : n} {c₀ : C} {y : n → n → C}
    (h : discover q cell labels i₀ c₀ = some y) : y = phaseLabels e r s := by
  obtain ⟨hK, hy⟩ := conditions_of_discover h
  exact (completes_iff e r s hreal i₀ c₀ hK y).mp hy

end Transformer.Grokking.FlatSpectrum.CoupledMoments.CompositeDiscovery
