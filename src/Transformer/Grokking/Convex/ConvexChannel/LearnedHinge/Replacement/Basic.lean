module

public import Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.PatternReduction.Complex

/-!
# Replace one learned component and retain the complete complex forward

Source: `review/convex/moments.md`, CM83. A query replaces one complete real
or imaginary vector in one slot, class and sign. The other component is
retained. The entire changed complex neuron is recovered again, including
its representation and readout. The equality below keeps the original
per-slot sum; it does not assume that separately recovered factors stay
unchanged when their joint vector changes.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Replacement

open scoped BigOperators
open PatternReduction

variable {D J C : Type*}

/-- Replace one full component vector and retain the other, CM83. -/
def component (part : Bool) (value : D → ℝ) (old : D → ℂ) : D → ℂ :=
  join (if part then value else project true old)
    (if part then project false old else value)

/-- Both component projections are exact, CM83, at every coordinate. -/
theorem component_project (part selected : Bool) (value : D → ℝ) (old : D → ℂ) :
    project selected (component part value old) =
      if selected = part then value else project selected old := by
  cases part <;> cases selected <;> simp [component, project_join]

variable [DecidableEq J] [DecidableEq C]

/-- Change exactly one slot/class/sign vector, CM83. Object identities and
all other learned vectors are retained. -/
def replace (p : Parameters J C (D → ℂ)) (slot : J) (object : C)
    (sign part : Bool) (value : D → ℝ) : Parameters J C (D → ℂ) :=
  fun j c s => if j = slot ∧ c = object ∧ s = sign then
    component part value (p slot object sign) else p j c s

/-- A replacement has exactly one changed full real-coordinate cell, CM83.
This includes the unchanged opposite component of the selected neuron. -/
theorem replace_project (p : Parameters J C (D → ℂ)) (slot : J) (object : C)
    (sign part : Bool) (value : D → ℝ) (j : J) (c : C) (s k : Bool) :
    project k (replace p slot object sign part value j c s) =
      if j = slot ∧ c = object ∧ s = sign ∧ k = part then value
      else project k (p j c s) := by
  by_cases h : j = slot ∧ c = object ∧ s = sign
  · rcases h with ⟨rfl, rfl, rfl⟩
    simp [replace, component_project]
  · have hn : ¬ (j = slot ∧ c = object ∧ s = sign ∧ k = part) := by
      intro hh
      exact h ⟨hh.1, hh.2.1, hh.2.2.1⟩
    simp [replace, h, hn]

/-- Unselected whole complex vectors are unchanged, CM83; their recovered
representations and readouts can therefore be reused. -/
theorem replace_other (p : Parameters J C (D → ℂ)) (slot : J) (object : C)
    (sign part : Bool) (value : D → ℝ) (j : J) (c : C) (s : Bool)
    (h : ¬ (j = slot ∧ c = object ∧ s = sign)) :
    replace p slot object sign part value j c s = p j c s := by
  unfold replace
  exact ite_eq_right h

variable [Fintype D]

/-- The actual full complex neuron at every address is either recovered
from the replacement vector or is exactly the old neuron, CM83. -/
theorem replace_neuron (p : Parameters J C (D → ℂ)) (slot : J) (object : C)
    (sign part : Bool) (value : D → ℝ) (input : (D → ℂ) →ₗ[ℝ] ℂ)
    (j : J) (c : C) (s : Bool) :
    complexNeuron (replace p slot object sign part value j c s) input =
      if j = slot ∧ c = object ∧ s = sign then
        complexNeuron (component part value (p slot object sign)) input
      else complexNeuron (p j c s) input := by
  by_cases h : j = slot ∧ c = object ∧ s = sign
  · simp only [replace, ite_eq_left h]
  · rw [replace_other p slot object sign part value j c s h, ite_eq_right h]

variable [Fintype J]

/-- Full complex inference equals replacement of exactly one whole neuron
inside the original per-slot sum, CM83. No activation or output is omitted. -/
theorem replace_forward (p : Parameters J C (D → ℂ)) (slot : J) (object : C)
    (sign part : Bool) (value : D → ℝ) (input : (D → ℂ) →ₗ[ℝ] ℂ) (c : C) :
    complexForward (replace p slot object sign part value) input c =
      ∑ j, ((if j = slot ∧ c = object ∧ true = sign then
        complexNeuron (component part value (p slot object sign)) input
        else complexNeuron (p j c true) input) -
      (if j = slot ∧ c = object ∧ false = sign then
        complexNeuron (component part value (p slot object sign)) input
        else complexNeuron (p j c false) input)) := by
  unfold complexForward
  apply Finset.sum_congr rfl
  intro j hj
  rw [replace_neuron, replace_neuron]

/-- Only the selected object's complex score can change, CM83, by the
signed difference of the complete new and old neuron outputs. This is an
exact complex identity, not an assertion about floating reassociation. -/
theorem replace_forward_change (p : Parameters J C (D → ℂ)) (slot : J) (object : C)
    (sign part : Bool) (value : D → ℝ) (input : (D → ℂ) →ₗ[ℝ] ℂ) (c : C) :
    complexForward (replace p slot object sign part value) input c =
      complexForward p input c + if c = object then
        (if sign then (1 : ℂ) else -1) *
          (complexNeuron (component part value (p slot object sign)) input -
            complexNeuron (p slot object sign) input) else 0 := by
  rw [replace_forward]
  by_cases hc : c = object
  · subst c
    rw [ite_eq_left rfl]
    let delta := (if sign then (1 : ℂ) else -1) *
      (complexNeuron (component part value (p slot object sign)) input -
        complexNeuron (p slot object sign) input)
    have hsum : (∑ j, ((if j = slot ∧ object = object ∧ true = sign then
        complexNeuron (component part value (p slot object sign)) input
        else complexNeuron (p j object true) input) -
      (if j = slot ∧ object = object ∧ false = sign then
        complexNeuron (component part value (p slot object sign)) input
        else complexNeuron (p j object false) input))) =
        ∑ j, (complexNeuron (p j object true) input -
          complexNeuron (p j object false) input + if j = slot then delta else 0) := by
      apply Finset.sum_congr rfl
      intro j hj
      by_cases hj : j = slot
      · subst j
        cases sign <;> simp [delta]
      · simp [hj]
    rw [hsum, Finset.sum_add_distrib]
    simp [complexForward, delta]
  · simp [hc, complexForward]

example := replace_other (D := Bool) (J := Bool) (C := Bool) 0 true true true false
  (fun _ => 1) false true true (by simp)

end Transformer.Grokking.Convex.ConvexChannel.LearnedHinge.Replacement
