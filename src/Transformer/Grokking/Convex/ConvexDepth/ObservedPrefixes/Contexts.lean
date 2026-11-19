module

public import Transformer.Grokking.Convex.ConvexDepth.ObservedPrefixes.Basic

/-!
# Substitute observed-equivalent subprograms inside nested counts

Source: `review/convex/moments.md`, CM58. The context grammar includes Boolean
constants, current symbols, negation, conjunction and count comparisons with
integer offsets and an optional third count. Simultaneous replacement of
holes preserves every observed value on a prefix-closed pool. It preserves a
decoded training fit, not answers on arbitrary new words. Explicit examples
separate incomplete-prefix and unseen-input failures.
-/

@[expose] public section

namespace Transformer.Grokking.Convex.ConvexDepth.ObservedPrefixes

/-- Finite expression contexts, with replaceable input-only subprogram holes
and arbitrary finite nesting of inclusive counts (CM58). -/
inductive Context (A I : Type*) where
  | hole : I → Context A I
  | constant : Bool → Context A I
  | symbol : A → Context A I
  | negate : Context A I → Context A I
  | both : Context A I → Context A I → Context A I
  | less : Context A I → Context A I → ℤ → Context A I
  | lessAdd : Context A I → Context A I → Context A I → ℤ → Context A I

variable {A I C : Type*} [DecidableEq A]

/-- Actual prefix semantics. Counts evaluate the child at every earlier
nonempty prefix, including the current position (CM58). -/
def evaluate (env : I → List A → Bool) : Context A I → List A → Bool
  | .hole i, w => env i w
  | .constant b, _ => b
  | .symbol a, w => decide (w.getLast? = some a)
  | .negate c, w => !(evaluate env c w)
  | .both a b, w => evaluate env a w && evaluate env b w
  | .less a b k, w => decide
      ((prefixCount (evaluate env a) w : ℤ) < (prefixCount (evaluate env b) w : ℤ) + k)
  | .lessAdd a b c k, w => decide
      ((prefixCount (evaluate env a) w : ℤ) < (prefixCount (evaluate env b) w : ℤ) +
        (prefixCount (evaluate env c) w : ℤ) + k)

/-- Simultaneous replacement of observed-equivalent holes preserves the
entire nested context on a prefix-closed pool (CM58). -/
theorem context_congr (pool : Finset (List A)) (env alt : I → List A → Bool)
    (hclosed : PrefixClosed pool) (heq : ∀ i, ObservedEq pool (env i) (alt i))
    (c : Context A I) : ObservedEq pool (evaluate env c) (evaluate alt c) := by
  induction c with
  | hole i => exact heq i
  | constant b => intro w hw; rfl
  | symbol a => intro w hw; rfl
  | negate c ih =>
    intro w hw
    change (!evaluate env c w) = (!evaluate alt c w)
    rw [ih w hw]
  | both a b iha ihb =>
    intro w hw
    change (evaluate env a w && evaluate env b w) =
      (evaluate alt a w && evaluate alt b w)
    rw [iha w hw, ihb w hw]
  | less a b k iha ihb =>
    intro w hw
    simp only [evaluate]
    rw [prefixCount_congr pool _ _ hclosed iha hw,
      prefixCount_congr pool _ _ hclosed ihb hw]
  | lessAdd a b c k iha ihb ihc =>
    intro w hw
    simp only [evaluate]
    rw [prefixCount_congr pool _ _ hclosed iha hw,
      prefixCount_congr pool _ _ hclosed ihb hw,
      prefixCount_congr pool _ _ hclosed ihc hw]

/-- A signature representative preserves every raw observed label through
the same decoder. The decoder can be many-to-one; no unseen label is assumed
or concluded. Search may still truncate distinct signature classes (CM58). -/
theorem representative_preserves_fit (pool : Finset (List A))
    (env alt : I → List A → Bool) (decode : Bool → C) (labels : List A → C)
    (hclosed : PrefixClosed pool) (heq : ∀ i, ObservedEq pool (env i) (alt i))
    (c : Context A I) (hfit : ∀ w ∈ pool, decode (evaluate env c w) = labels w) :
    ∀ w ∈ pool, decode (evaluate alt c w) = labels w := by
  intro w hw
  have hvalue := context_congr pool env alt hclosed heq c w hw
  rw [← hvalue]
  exact hfit w hw

/-- This nonconstant observed context compares an elapsed-position count
with a replaceable subprogram count (CM58). -/
def toyContext : Context Bool Unit := .less (.constant true) (.hole ()) 1

/-- Both observed prefixes retain their context values under replacement,
even though the input-only subprograms are globally different (CM58). -/
theorem toy_context_observed : ObservedEq toyPool
    (evaluate (fun () => lastTrue) toyContext)
    (evaluate (fun () => onlyFirst) toyContext) := by
  apply context_congr toyPool _ _ (closure_closed _)
  intro i
  cases i
  exact toy_observed

/-- Both observed truth values occur; the two contexts disagree at the new
word [true,true]. This is an explicit limit of training signatures (CM58). -/
theorem toy_context_values :
    evaluate (fun () => lastTrue) toyContext [true] = true ∧
    evaluate (fun () => lastTrue) toyContext [true, false] = false ∧
    evaluate (fun () => lastTrue) toyContext [true, true] = true ∧
    evaluate (fun () => onlyFirst) toyContext [true, true] = false := by
  decide

/-- The unseen count difference is substantive, not merely syntax (CM58). -/
theorem toy_unseen_counts : prefixCount lastTrue [true, true] = 2 ∧
    prefixCount onlyFirst [true, true] = 1 := by
  decide

/-- Final-word agreement alone misses an earlier prefix: both predicates are
false at [true,false], but their inclusive counts are one and zero (CM58). -/
theorem incomplete_pool_counterexample :
    ObservedEq ({[true, false]} : Finset (List Bool)) lastTrue
      (fun w => decide (w = [false])) ∧
    prefixCount lastTrue [true, false] = 1 ∧
    prefixCount (fun w => decide (w = [false])) [true, false] = 0 := by
  constructor
  · intro w hw
    have hword : w = [true, false] := by simpa using hw
    subst w
    decide
  · decide

/-- The missing first prefix is exactly what defeats count congruence in
the previous counterexample (CM58). -/
theorem final_word_pool_not_closed :
    ¬ PrefixClosed ({[true, false]} : Finset (List Bool)) := by
  intro hclosed
  have hprefix := hclosed [true, false] (by simp) 0 (by norm_num)
  norm_num at hprefix

example : ObservedEq toyPool
    (evaluate (fun () => lastTrue) toyContext)
    (evaluate (fun () => onlyFirst) toyContext) :=
  context_congr _ _ _ (closure_closed _) (by intro i; cases i; exact toy_observed) _

example :=
  representative_preserves_fit (I := Unit) (C := ℕ) toyPool
    (fun () => lastTrue) (fun () => onlyFirst) (fun b => if b then 10 else 7)
    (fun w => if evaluate (fun () => lastTrue) toyContext w then 10 else 7)
    (closure_closed _) (by intro i; cases i; exact toy_observed) toyContext
    (by intro w hw; rfl)

end Transformer.Grokking.Convex.ConvexDepth.ObservedPrefixes
