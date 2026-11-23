module

public import Transformer.Phasor.ModularChainScore

/-!
# Dividend and divisor copy in the original two-register GPT task

Source: DiffGPT `mqar_div97.copy_queries`, `Task.tensors` and
`current_grokking.adapters.associative` at 5a37c515. Each independently
sampled memory appears with all four ordered query pairs. The first
operand is the requested register for dividend copy and the second for
divisor copy. These are statements about the external dataset and score.
-/

@[expose] public section

namespace Transformer.Phasor.AssociativeCopyScore

variable {A : Type*}

/-- A frame retains the order of both writes and both query positions.
All entries are already encoded, including the shared empty marker. -/
def frame (k v : Bool → A) (marker : A) (a b : Bool) : List A :=
  [k false,marker,v false,k true,marker,v true,k a,k b]

/-- Read the first queried key from the two displayed writes. -/
def dividend [DecidableEq A] : List A → Option A
  | [k₀,_,v₀,k₁,_,v₁,q,_] =>
    if q = k₁ then some v₁ else if q = k₀ then some v₀ else none
  | _ => none

/-- Read the second queried key in the canonical divisor control. -/
def divisor [DecidableEq A] : List A → Option A
  | [k₀,_,v₀,k₁,_,v₁,_,q] =>
    if q = k₁ then some v₁ else if q = k₀ then some v₀ else none
  | _ => none

/-- The source draws distinct register keys. Both reads retrieve the
corresponding loaded value, including identical queries or equal values. -/
theorem reads [DecidableEq A] (k v : Bool → A) (marker : A) (a b : Bool)
    (distinct : k false ≠ k true) :
    dividend (frame k v marker a b) = some (v a) ∧
      divisor (frame k v marker a b) = some (v b) := by
  cases a <;> cases b <;> simp [frame,dividend,divisor,distinct]

/-- Within one complete source quartet the original right-query target
on the swapped query pair is exactly the new left-query target. -/
theorem swap_target [DecidableEq A] (k v : Bool → A) (marker : A) (a b : Bool)
    (distinct : k false ≠ k true) :
    dividend (frame k v marker a b) = divisor (frame k v marker b a) := by
  rw [(reads k v marker a b distinct).1,(reads k v marker b a distinct).2]

/-- With different stored values, keeping the divisor target would
change the label of each mixed query pair and preserve both repeated pairs. -/
theorem roles_agree_iff [DecidableEq A] (k v : Bool → A) (marker : A) (a b : Bool)
    (distinct : k false ≠ k true) (unequal : v false ≠ v true) :
    dividend (frame k v marker a b) = divisor (frame k v marker a b) ↔ a = b := by
  rw [(reads k v marker a b distinct).1,(reads k v marker a b distinct).2]
  cases a <;> cases b <;> simp [unequal, Ne.symm unequal]

/-- The mask selects only position seven, after the entire eight-token
prompt. For any dense causal predictor the source exact score is therefore
equivalent to predicting the value stored under the first query. -/
theorem score_iff (k v : Bool → A) (marker : A) (a b : Bool) (predict : List A → A) :
    (∀ i y, (ModularChainScore.targets 7 (v a))[i]? = some (some y) →
      predict ((frame k v marker a b).take (i+1)) = y) ↔
      predict (frame k v marker a b) = v a := by
  have full : (frame k v marker a b).take 8 = frame k v marker a b := by simp [frame]
  constructor
  · intro h
    simpa only [full] using h 7 _ ((ModularChainScore.supervised_iff ..).2 ⟨rfl,rfl⟩)
  · intro h i y hy
    obtain ⟨rfl,rfl⟩ := (ModularChainScore.supervised_iff ..).1 hy
    simpa only [full] using h

private def keys : Bool → ℕ | false => 37 | true => 91
private def values : Bool → ℕ | false => 210 | true => 176

example : dividend (frame keys values 0 true false) = some 176 ∧
    divisor (frame keys values 0 true false) = some 210 := reads keys values 0 true false (by decide)

example : dividend (frame keys values 0 false true) = divisor (frame keys values 0 true false) :=
  swap_target keys values 0 false true (by decide)

example : dividend (frame keys values 0 false true) ≠ divisor (frame keys values 0 false true) := by
  rw [ne_eq, roles_agree_iff keys values 0 false true (by decide) (by decide)]
  decide

end Transformer.Phasor.AssociativeCopyScore
