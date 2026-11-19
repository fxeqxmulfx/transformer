module

public import Transformer.Grokking.Convex.ConvexDepth.BoundedLatent.Complex

/-!
# Accepted latent states retain their actual full complex predictions

Source: `review/convex/moments.md`, CM64. Learned confidence changes the margin,
while observed latent coverage and agreement identify the raw class only under
CM53's separately stated unknown-program realizability premise.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexDepth.BoundedLatent

open scoped BigOperators
open LatentPrograms (Parameters features head)

variable {J X S C : Type*} [Fintype J] [Fintype C] [DecidableEq S] [DecidableEq C]

omit [Fintype J] in
/-- An accepted query forces every active raw row; all other program rows
are zero by their shared mass, CM64. Confidence does not fill an unknown state. -/
theorem accepted_row (program : J → X → S) (order : List J) (samples : List (X × C))
    {q : Parameters J S C} (hq : q ∈ cone program order samples) (x : X) (y : C)
    (hp : LatentPrograms.predict program order samples x = some y) (j : J) (c : C) :
    q.2 j (program j x) c = if c = y then q.1 j else 0 := by
  by_cases hz : q.1 j = 0
  · simpa [hz] using coefficient_zero program order samples hq j hz (program j x) c
  · have hj := nonzero_retained program order samples hq j hz
    have found := ((LatentPrograms.predict_some_iff program order samples x y).mp hp).2 j hj
    obtain ⟨t, ht, hs, hy⟩ :=
      LatentPrograms.lookup_some_mem (program j) samples (program j x) y found
    have h := hq.2.2.2.2 j t ht c
    simpa [hs, hy] using h

variable [Fintype S]

omit [Fintype C] in
/-- The actual complete complex prediction at forced rows has the learned
total mass as its margin, CM64, with no product of learned factors. -/
theorem forced_logits (U : Matrix S S ℂ) (hU : U ∈ Matrix.unitaryGroup S ℂ)
    (program : J → X → S) (q : Parameters J S C) (x : X) (y : C)
    (hf : ∀ j c, q.2 j (program j x) c = if c = y then q.1 j else 0) :
    headLogits (features U program) (head U 1 q) x =
      fun c => if c = y then mass q else 0 := by
  funext c
  rw [LatentPrograms.logits U hU]
  simp only [hf, one_mul]
  by_cases hc : c = y
  · simp [hc, mass]
  · simp [hc]

/-- The finite raw logits on every observed record are forced by its joint
anchor equations, CM64. Their margin is learned total mass, not a fixed constant. -/
theorem observed_logits (U : Matrix S S ℂ) (hU : U ∈ Matrix.unitaryGroup S ℂ)
    (program : J → X → S) (order : List J) (samples : List (X × C)) {δ R : ℝ}
    {q : Parameters J S C} (hq : q ∈ domain program order samples δ R)
    (t : X × C) (ht : t ∈ samples) :
    headLogits (features U program) (head U 1 q) t.1 =
      fun c => if c = t.2 then mass q else 0 := by
  apply forced_logits U hU program q t.1 t.2
  exact fun j c => hq.1.2.2.2.2 j t ht c

/-- Observable state coverage and agreement identify the actual raw logit
vector for every bounded feasible tensor, CM64. Unknown-state defaults cannot
change an accepted vector, even when the optimizer changes program masses. -/
theorem accepted_logits (U : Matrix S S ℂ) (hU : U ∈ Matrix.unitaryGroup S ℂ)
    (program : J → X → S) (order : List J) (samples : List (X × C)) {δ R : ℝ}
    {q : Parameters J S C} (hq : q ∈ domain program order samples δ R)
    (x : X) (y : C) (hp : LatentPrograms.predict program order samples x = some y) :
    headLogits (features U program) (head U 1 q) x =
      fun c => if c = y then mass q else 0 := by
  apply forced_logits U hU program q x y
  exact accepted_row program order samples hq.1 x y hp

/-- Positive lower total mass makes the accepted raw class a strict winner
through the complete complex product, CM64. Unknown queries are not covered. -/
theorem accepted_strict (U : Matrix S S ℂ) (hU : U ∈ Matrix.unitaryGroup S ℂ)
    (program : J → X → S) (order : List J) (samples : List (X × C)) {δ R : ℝ}
    (hδ : 0 < δ) {q : Parameters J S C} (hq : q ∈ domain program order samples δ R)
    (x : X) (y : C) (hp : LatentPrograms.predict program order samples x = some y)
    (c : C) (hc : c ≠ y) :
    headLogits (features U program) (head U 1 q) x c <
      headLogits (features U program) (head U 1 q) x y := by
  rw [accepted_logits U hU program order samples hq x y hp]
  simpa [hc] using lt_of_lt_of_le hδ hq.2.1

/-- The unknown target and its many-to-one decoder remain soundness
hypotheses, CM64. Neither is supplied to the actual observed filter or head. -/
theorem target_logits (U : Matrix S S ℂ) (hU : U ∈ Matrix.unitaryGroup S ℂ)
    (program : J → X → S) (order : List J) (samples : List (X × C)) {δ R : ℝ}
    {q : Parameters J S C} (hq : q ∈ domain program order samples δ R) (target : X → C)
    (hreal : ∃ j ∈ order, ∃ decoder : S → C, ∀ x, decoder (program j x) = target x)
    (hobs : ∀ t ∈ samples, t.2 = target t.1) (x : X) (y : C)
    (hp : LatentPrograms.predict program order samples x = some y) :
    headLogits (features U program) (head U 1 q) x =
      fun c => if c = target x then mass q else 0 := by
  have hy := LatentPrograms.predict_sound program order samples target hreal hobs x y hp
  rw [accepted_logits U hU program order samples hq x y hp, hy]

example := accepted_row LatentPrograms.exampleProgram [false, true]
  LatentPrograms.exampleSamples examplePoint_mem.1 (true, true) true (by decide) true true

example := forced_logits (1 : Matrix Bool Bool ℂ) (Matrix.unitaryGroup Bool ℂ).one_mem
  LatentPrograms.exampleProgram examplePoint (true, true) true
  (accepted_row _ _ _ examplePoint_mem.1 (true, true) true (by decide))

example := observed_logits (1 : Matrix Bool Bool ℂ) (Matrix.unitaryGroup Bool ℂ).one_mem
  LatentPrograms.exampleProgram [false, true] LatentPrograms.exampleSamples examplePoint_mem
  ((false, true), true) (by decide)

example := accepted_logits (1 : Matrix Bool Bool ℂ) (Matrix.unitaryGroup Bool ℂ).one_mem
  LatentPrograms.exampleProgram [false, true] LatentPrograms.exampleSamples examplePoint_mem
  (true, true) true (by decide)

example := accepted_strict (1 : Matrix Bool Bool ℂ) (Matrix.unitaryGroup Bool ℂ).one_mem
  LatentPrograms.exampleProgram [false, true] LatentPrograms.exampleSamples
  (by norm_num : (0 : ℝ) < 1) examplePoint_mem (true, true) true (by decide) false (by decide)

example := target_logits (1 : Matrix Bool Bool ℂ) (Matrix.unitaryGroup Bool ℂ).one_mem
  LatentPrograms.exampleProgram [false, true] LatentPrograms.exampleSamples examplePoint_mem
  Prod.snd ⟨true, by simp, id, by simp [LatentPrograms.exampleProgram]⟩
  (by simp [LatentPrograms.exampleSamples]) (true, true) true (by decide)

end Transformer.Grokking.Convex.ConvexDepth.BoundedLatent
