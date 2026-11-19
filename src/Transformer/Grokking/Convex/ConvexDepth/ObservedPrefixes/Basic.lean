module

public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Data.Finset.Union
public import Mathlib.Data.Finset.Range
public import Mathlib.Tactic

/-!
# Finite observed pools closed under nonempty prefixes

Source: `review/convex/moments.md`, CM58. A truth signature used inside an
inclusive count must include every nonempty prefix of each observed word.
Closing a finite input pool needs at most the sum of its word lengths. Equal
signatures on that closure preserve counts there, without identifying values
on any new word. BOS is input framing and is absent from these symbol words.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexDepth.ObservedPrefixes

open scoped BigOperators

variable {A : Type*} [DecidableEq A]

/-- Include every nonempty prefix using only the finite input words, CM58. -/
def prefixClosure (words : Finset (List A)) : Finset (List A) :=
  words.biUnion fun w => (Finset.range w.length).image fun i => w.take (i + 1)

/-- Every prefix read by an inclusive count is still in the pool, CM58. -/
def PrefixClosed (pool : Finset (List A)) : Prop :=
  ∀ w ∈ pool, ∀ i < w.length, w.take (i + 1) ∈ pool

/-- Observed truth signatures agree; no claim about unobserved words, CM58. -/
def ObservedEq (pool : Finset (List A)) (p q : List A → Bool) : Prop :=
  ∀ w ∈ pool, p w = q w

/-- Inclusive count of the predicate's values at all nonempty prefixes, CM58. -/
def prefixCount (p : List A → Bool) (w : List A) : ℕ :=
  ∑ i ∈ Finset.range w.length, if p (w.take (i + 1)) then 1 else 0

/-- Prefix closure contains no empty word (CM58). -/
theorem closure_positive (words : Finset (List A)) {w : List A}
    (hw : w ∈ prefixClosure words) : 0 < w.length := by
  obtain ⟨v, hv, hvw⟩ := Finset.mem_biUnion.mp hw
  obtain ⟨i, hi, rfl⟩ := Finset.mem_image.mp hvw
  have hil : i < v.length := Finset.mem_range.mp hi
  rw [List.length_take]
  omega

/-- A nonempty observed word is itself retained (CM58). -/
theorem observed_mem_closure (words : Finset (List A)) {w : List A}
    (hw : w ∈ words) (hpos : 0 < w.length) : w ∈ prefixClosure words := by
  apply Finset.mem_biUnion.mpr
  refine ⟨w, hw, Finset.mem_image.mpr ?_⟩
  refine ⟨w.length - 1, Finset.mem_range.mpr (by omega), ?_⟩
  have hlen : w.length - 1 + 1 = w.length := by omega
  rw [hlen, List.take_length]

/-- Taking a prefix of a stored prefix stays in the finite closure (CM58). -/
theorem closure_closed (words : Finset (List A)) : PrefixClosed (prefixClosure words) := by
  intro w hw i hi
  obtain ⟨v, hv, hvw⟩ := Finset.mem_biUnion.mp hw
  obtain ⟨k, hk, rfl⟩ := Finset.mem_image.mp hvw
  rw [List.length_take] at hi
  apply Finset.mem_biUnion.mpr
  refine ⟨v, hv, Finset.mem_image.mpr ?_⟩
  refine ⟨i, Finset.mem_range.mpr (by omega), ?_⟩
  rw [List.take_take]
  congr 1
  omega

/-- Every closed pool containing the observed inputs contains their closure,
so no unseen test input is needed to construct this pool (CM58). -/
theorem closure_minimal (words pool : Finset (List A))
    (hclosed : PrefixClosed pool) (hwords : words ⊆ pool) : prefixClosure words ⊆ pool := by
  intro w hw
  obtain ⟨v, hv, hvw⟩ := Finset.mem_biUnion.mp hw
  obtain ⟨i, hi, rfl⟩ := Finset.mem_image.mp hvw
  exact hclosed v (hwords hv) i (Finset.mem_range.mp hi)

/-- The closure needs at most one entry per input position; duplicate prefixes
can only reduce its cardinality. This is a finite storage bound (CM58). -/
theorem closure_card_le (words : Finset (List A)) :
    (prefixClosure words).card ≤ ∑ w ∈ words, w.length := by
  unfold prefixClosure
  apply le_trans Finset.card_biUnion_le
  apply Finset.sum_le_sum
  intro w hw
  exact le_trans Finset.card_image_le (le_of_eq (Finset.card_range w.length))

/-- Repeated closure adds no new observed inputs (CM58). -/
theorem closure_idempotent (words : Finset (List A)) :
    prefixClosure (prefixClosure words) = prefixClosure words := by
  apply Finset.Subset.antisymm
  · exact closure_minimal _ _ (closure_closed words) (fun w hw => hw)
  · intro w hw
    exact observed_mem_closure _ hw (closure_positive words hw)

omit [DecidableEq A] in
/-- An inclusive Boolean count is bounded independently of predicate depth,
which bounds the exact integer values needed by the native search (CM58). -/
theorem prefixCount_le_length (p : List A → Bool) (w : List A) :
    prefixCount p w ≤ w.length := by
  have h : prefixCount p w ≤ ∑ i ∈ Finset.range w.length, (1 : ℕ) := by
    apply Finset.sum_le_sum
    intro i hi
    split <;> omega
  simpa using h

omit [DecidableEq A] in
/-- Equal observed signatures imply equal inclusive counts only when every
prefix inspected by that count is covered by the same pool (CM58). -/
theorem prefixCount_congr (pool : Finset (List A)) (p q : List A → Bool)
    (hclosed : PrefixClosed pool) (hpq : ObservedEq pool p q) {w : List A}
    (hw : w ∈ pool) : prefixCount p w = prefixCount q w := by
  apply Finset.sum_congr rfl
  intro i hi
  have hp := hpq (w.take (i + 1)) (hclosed w hw i (Finset.mem_range.mp hi))
  rw [hp]

/-- A two-position input provides a nonempty closed witness (CM58). -/
def toyPool : Finset (List Bool) := prefixClosure {[true, false]}

/-- One input-only predicate reads the current symbol (CM58). -/
def lastTrue (w : List Bool) : Bool := decide (w.getLast? = some true)

/-- Another input-only predicate is true only at the first position (CM58). -/
def onlyFirst (w : List Bool) : Bool := decide (w.length = 1)

/-- The two nonconstant signatures agree on both observed prefixes (CM58). -/
theorem toy_observed : ObservedEq toyPool lastTrue onlyFirst := by
  intro w hw
  have hpool : toyPool = {[true], [true, false]} := by decide
  rw [hpool] at hw
  have hcases : w = [true] ∨ w = [true, false] := by
    simpa only [Finset.mem_insert, Finset.mem_singleton] using hw
  rcases hcases with rfl | rfl <;> norm_num [lastTrue, onlyFirst]

example : 0 < ([true] : List Bool).length :=
  closure_positive {[true, false]} (by decide)

example : [true, false] ∈ prefixClosure ({[true, false]} : Finset (List Bool)) :=
  observed_mem_closure _ (by simp) (by norm_num)

example : PrefixClosed toyPool := closure_closed _

example : prefixClosure ({[true, false]} : Finset (List Bool)) ⊆ toyPool :=
  closure_minimal _ _ (closure_closed _) (by
    intro w hw
    have hw' : w = [true, false] := by simpa using hw
    subst w
    exact observed_mem_closure _ (by simp) (by norm_num))

example : prefixCount lastTrue [true, false] = prefixCount onlyFirst [true, false] :=
  prefixCount_congr toyPool lastTrue onlyFirst (closure_closed _) toy_observed
    (by decide)

end Transformer.Grokking.Convex.ConvexDepth.ObservedPrefixes
