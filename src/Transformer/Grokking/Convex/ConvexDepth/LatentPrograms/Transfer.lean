module

public import Transformer.Grokking.Convex.ConvexDepth.LatentPrograms.Initialization

/-!
# Accepted latent-program predictions identify actual complex logits

Source: `review/convex/moments.md`, CM53. Observation constraints force every
active program's known decoder row. Consensus therefore identifies the full
complex prediction for every feasible joint tensor, not only the initializer.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexDepth.LatentPrograms

open scoped BigOperators
open Landscape.Neural (crossEntropy)

variable {J X S C : Type*} [Fintype J] [Fintype C] [DecidableEq S] [DecidableEq C]

/-- At an accepted query, every joint row is fixed by original observations
or its zero program mass, CM53. Unknown defaults cannot change this row. -/
theorem accepted_row (program : J → X → S) (order : List J) (samples : List (X × C))
    {q : Parameters J S C} (hq : q ∈ domain program order samples) (x : X) (y : C)
    (hp : predict program order samples x = some y) (j : J) (c : C) :
    q.2 j (program j x) c = if c = y then q.1 j else 0 := by
  by_cases hz : q.1 j = 0
  · simpa [hz] using coefficient_zero program order samples hq j hz (program j x) c
  · have hj := nonzero_mem_retained program order samples hq j hz
    have found := ((predict_some_iff program order samples x y).mp hp).2 j hj
    obtain ⟨t, ht, hs, hy⟩ := lookup_some_mem (program j) samples (program j x) y found
    have h := hq.2.2.2.2 j t ht c
    simpa [hs, hy] using h

variable [Fintype S]

omit [Fintype C] in
/-- Shared unit total mass and forced raw rows give the exact full complex
margin, CM53. All blocks and both coefficient coordinates remain in the head. -/
theorem forced_logits (U : Matrix S S ℂ) (hU : U ∈ Matrix.unitaryGroup S ℂ)
    (program : J → X → S) (margin : ℝ) (q : Parameters J S C)
    (hs : ∑ j, q.1 j = 1) (x : X) (y : C)
    (hf : ∀ j c, q.2 j (program j x) c = if c = y then q.1 j else 0) :
    headLogits (features U program) (head U margin q) x =
      fun c => if c = y then margin else 0 := by
  funext c
  rw [logits U hU]
  simp only [hf]
  by_cases hc : c = y
  · simp [hc, hs]
  · simp [hc]

/-- Every feasible joint tensor exactly fits the original finite-margin
logits, CM53. Ordinary observed CE is constant on this compatible domain. -/
theorem observed_logits (U : Matrix S S ℂ) (hU : U ∈ Matrix.unitaryGroup S ℂ)
    (program : J → X → S) (order : List J) (samples : List (X × C)) (margin : ℝ)
    {q : Parameters J S C} (hq : q ∈ domain program order samples)
    (t : X × C) (ht : t ∈ samples) :
    headLogits (features U program) (head U margin q) t.1 =
      fun c => if c = t.2 then margin else 0 := by
  apply forced_logits U hU program margin q hq.1.2 t.1 t.2
  exact fun j c => hq.2.2.2.2 j t ht c

/-- Executable agreement determines the actual raw logits for every
feasible joint tensor, CM53, even if some other latent states are unobserved. -/
theorem accepted_logits (U : Matrix S S ℂ) (hU : U ∈ Matrix.unitaryGroup S ℂ)
    (program : J → X → S) (order : List J) (samples : List (X × C)) (margin : ℝ)
    {q : Parameters J S C} (hq : q ∈ domain program order samples)
    (x : X) (y : C) (hp : predict program order samples x = some y) :
    headLogits (features U program) (head U margin q) x =
      fun c => if c = y then margin else 0 := by
  apply forced_logits U hU program margin q hq.1.2 x y
  exact accepted_row program order samples hq x y hp

/-- Structural realizability identifies the accepted raw class with the
unknown target, CM53, in the actual complete complex model. -/
theorem target_logits (U : Matrix S S ℂ) (hU : U ∈ Matrix.unitaryGroup S ℂ)
    (program : J → X → S) (order : List J) (samples : List (X × C)) (margin : ℝ)
    {q : Parameters J S C} (hq : q ∈ domain program order samples) (target : X → C)
    (hreal : ∃ j ∈ order, ∃ decoder : S → C, ∀ x, decoder (program j x) = target x)
    (hobs : ∀ t ∈ samples, t.2 = target t.1) (x : X) (y : C)
    (hp : predict program order samples x = some y) :
    headLogits (features U program) (head U margin q) x =
      fun c => if c = target x then margin else 0 := by
  have hy := predict_sound program order samples target hreal hobs x y hp
  rw [accepted_logits U hU program order samples margin hq x y hp, hy]

/-- The identified target has ordinary full-vocabulary finite-margin CE,
CM53. This is the actual complex readout loss, not a surrogate decoder loss. -/
theorem target_ce (U : Matrix S S ℂ) (hU : U ∈ Matrix.unitaryGroup S ℂ)
    (program : J → X → S) (order : List J) (samples : List (X × C)) (margin : ℝ)
    {q : Parameters J S C} (hq : q ∈ domain program order samples) (target : X → C)
    (hreal : ∃ j ∈ order, ∃ decoder : S → C, ∀ x, decoder (program j x) = target x)
    (hobs : ∀ t ∈ samples, t.2 = target t.1) (x : X) (y : C)
    (hp : predict program order samples x = some y) :
    crossEntropy (headLogits (features U program) (head U margin q) x) (target x) =
      Real.log (1 + ((Fintype.card C : ℝ) - 1) * Real.exp (-margin)) := by
  rw [target_logits U hU program order samples margin hq target hreal hobs x y hp]
  exact fixed_margin_ce _ _

/-- A positive margin makes the accepted raw class strictly preferred to
every competing class in every feasible complete complex head, CM53. -/
theorem accepted_strict (U : Matrix S S ℂ) (hU : U ∈ Matrix.unitaryGroup S ℂ)
    (program : J → X → S) (order : List J) (samples : List (X × C)) {margin : ℝ}
    (hm : 0 < margin) {q : Parameters J S C} (hq : q ∈ domain program order samples)
    (x : X) (y : C) (hp : predict program order samples x = some y) (c : C) (hc : c ≠ y) :
    headLogits (features U program) (head U margin q) x c <
      headLogits (features U program) (head U margin q) x y := by
  rw [accepted_logits U hU program order samples margin hq x y hp]
  simpa [hc] using hm

/-- End-to-end training-only construction, capacity and accepted-target
transfer, CM53. Existence of a feasible initialized tensor is a conclusion;
neither that tensor, its decoder nor an optimizer outcome is supplied. -/
theorem learned_target [DecidableEq J]
    (U : Matrix S S ℂ) (hU : U ∈ Matrix.unitaryGroup S ℂ)
    (program : J → X → S) (order : List J) (samples : List (X × C))
    (default : C) (margin : ℝ) (target : X → C)
    (hreal : ∃ j ∈ order, ∃ decoder : S → C, ∀ x, decoder (program j x) = target x)
    (hobs : ∀ t ∈ samples, t.2 = target t.1) (x : X) (y : C)
    (hp : predict program order samples x = some y) :
    ∃ q, initializeTensor program order samples default = some q ∧
      q ∈ domain program order samples ∧
      head U margin q ∈ headBudget (2 * |margin| * Fintype.card S) ∧
      headLogits (features U program) (head U margin q) x =
        (fun c => if c = target x then margin else 0) ∧
      crossEntropy (headLogits (features U program) (head U margin q) x) (target x) =
        Real.log (1 + ((Fintype.card C : ℝ) - 1) * Real.exp (-margin)) := by
  obtain ⟨q, hi, hq⟩ := realizable_initialize program order samples default target hreal hobs
  exact ⟨q, hi, hq, head_budget U hU program order samples margin hq,
    target_logits U hU program order samples margin hq target hreal hobs x y hp,
    target_ce U hU program order samples margin hq target hreal hobs x y hp⟩

example := accepted_row exampleProgram [false, true] exampleSamples examplePoint_mem
  (true, true) true (by decide) true true

example := forced_logits (1 : Matrix Bool Bool ℂ) (Matrix.unitaryGroup Bool ℂ).one_mem
  exampleProgram 24 examplePoint (by norm_num [examplePoint, ofDecoders, Fintype.sum_bool])
  (true, true) true
  (by intro j c; cases j <;> cases c <;> norm_num [examplePoint, ofDecoders, exampleProgram])

example := observed_logits (1 : Matrix Bool Bool ℂ) (Matrix.unitaryGroup Bool ℂ).one_mem
  exampleProgram [false, true] exampleSamples 24 examplePoint_mem
  ((false, true), true) (by simp [exampleSamples])

example := accepted_logits (1 : Matrix Bool Bool ℂ) (Matrix.unitaryGroup Bool ℂ).one_mem
  exampleProgram [false, true] exampleSamples 24 examplePoint_mem (true, true) true (by decide)

example := target_logits (1 : Matrix Bool Bool ℂ) (Matrix.unitaryGroup Bool ℂ).one_mem
  exampleProgram [false, true] exampleSamples 24 examplePoint_mem Prod.snd
  ⟨true, by simp, id, by simp [exampleProgram]⟩ (by simp [exampleSamples])
  (true, true) true (by decide)

example := target_ce (1 : Matrix Bool Bool ℂ) (Matrix.unitaryGroup Bool ℂ).one_mem
  exampleProgram [false, true] exampleSamples 24 examplePoint_mem Prod.snd
  ⟨true, by simp, id, by simp [exampleProgram]⟩ (by simp [exampleSamples])
  (true, true) true (by decide)

example := accepted_strict (1 : Matrix Bool Bool ℂ) (Matrix.unitaryGroup Bool ℂ).one_mem
  exampleProgram [false, true] exampleSamples (by norm_num : (0 : ℝ) < 24)
  examplePoint_mem (true, true) true (by decide) false (by decide)

example := learned_target (1 : Matrix Bool Bool ℂ) (Matrix.unitaryGroup Bool ℂ).one_mem
  exampleProgram [false, true] exampleSamples false 24 Prod.snd
  ⟨true, by simp, id, by simp [exampleProgram]⟩ (by simp [exampleSamples])
  (true, true) true (by decide)

end Transformer.Grokking.Convex.ConvexDepth.LatentPrograms
