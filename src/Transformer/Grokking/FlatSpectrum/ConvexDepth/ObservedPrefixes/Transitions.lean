module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ObservedPrefixes.Contexts

/-!
# Necessary latent bits at observed threshold crossings

Source: `review/convex/moments.md`, CM59. One new symbol changes a difference
of inclusive Boolean counts by a-b-e. A crossing from below an integer
threshold to at least that threshold forces (a,b,e)=(true,false,false).
A downward crossing with e=false forces (a,b)=(false,true). With e=true,
the weaker disjunction a=false or b=true is necessary. These are necessary
constraints conditional on a candidate root expression, not new observed
labels and not a sufficiency or unseen-identification theorem.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth.ObservedPrefixes

open scoped BigOperators

/-- A single actual update of two counted predicates and one extra count,
with supplied Boolean values at the new prefix (CM59). -/
def unitStep (x : ℤ) (a b e : Bool) : ℤ :=
  x + (if a then 1 else 0) - (if b then 1 else 0) - (if e then 1 else 0)

/-- Each step lies between minus two and plus one, independent of the
internal depth of the counted predicates (CM59). -/
theorem unitStep_bounds (x : ℤ) (a b e : Bool) :
    x - 2 ≤ unitStep x a b e ∧ unitStep x a b e ≤ x + 1 := by
  cases a <;> cases b <;> cases e <;> simp [unitStep] <;> omega

/-- An upward integer-threshold crossing identifies all three current bits
and both boundary values (CM59). -/
theorem upward_crossing {x k : ℤ} {a b e : Bool}
    (hbefore : x < k) (hafter : k ≤ unitStep x a b e) :
    a = true ∧ b = false ∧ e = false ∧ x = k - 1 ∧ unitStep x a b e = k := by
  cases a <;> cases b <;> cases e <;> simp_all [unitStep] <;> omega

/-- A downward crossing without an extra-count increment identifies both
current child bits and the integer boundary (CM59). -/
theorem downward_crossing {x k : ℤ} {a b : Bool}
    (hbefore : k ≤ x) (hafter : unitStep x a b false < k) :
    a = false ∧ b = true ∧ x = k ∧ unitStep x a b false = k - 1 := by
  cases a <;> cases b <;> simp_all [unitStep] <;> omega

/-- With an extra increment, a downward crossing gives a disjunction; it
cannot identify each child bit separately (CM59). -/
theorem downward_with_extra {x k : ℤ} {a b : Bool}
    (hbefore : k ≤ x) (hafter : unitStep x a b true < k) :
    (a = false ∨ b = true) ∧ x ≤ k + 1 ∧ k - 2 ≤ unitStep x a b true := by
  cases a <;> cases b <;> simp_all [unitStep] <;> omega

/-- An extra-count increment excludes an upward threshold crossing (CM59). -/
theorem extra_excludes_upward {x k : ℤ} (a b : Bool) (hbefore : x < k) :
    unitStep x a b true < k := by
  cases a <;> cases b <;> simp_all [unitStep] <;> omega

variable {A : Type*}

/-- Appending one symbol contributes precisely the predicate at that new
prefix to its inclusive count (CM59). No causality hypothesis is hidden. -/
theorem prefixCount_append (p : List A → Bool) (w : List A) (a : A) :
    prefixCount p (w ++ [a]) = prefixCount p w + (if p (w ++ [a]) then 1 else 0) := by
  unfold prefixCount
  rw [List.length_append, List.length_singleton, Finset.sum_range_succ]
  have htake : (w ++ [a]).take (w.length + 1) = w ++ [a] := by
    have hlen : (w ++ [a]).length = w.length + 1 := by simp
    rw [← hlen, List.take_length]
  rw [htake]
  congr 1
  apply Finset.sum_congr rfl
  intro i hi
  rw [List.take_append_of_le_length (by have := Finset.mem_range.mp hi; omega)]

/-- The actual root comparison's integer count difference, with an optional
extra predicate represented by a Boolean function (CM59). -/
def countDifference (p q r : List A → Bool) (w : List A) : ℤ :=
  (prefixCount p w : ℤ) - (prefixCount q w : ℤ) - (prefixCount r w : ℤ)

/-- The abstract integer step is exactly the change of the counted root
expression after one symbol (CM59). -/
theorem countDifference_append (p q r : List A → Bool) (w : List A) (a : A) :
    countDifference p q r (w ++ [a]) =
      unitStep (countDifference p q r w) (p (w ++ [a])) (q (w ++ [a]))
        (r (w ++ [a])) := by
  simp only [countDifference, prefixCount_append, unitStep]
  split_ifs <;> omega

/-- Observed upward crossings of an actual nested-count difference force
current child values, conditional on this candidate comparison (CM59). -/
theorem countDifference_upward (p q r : List A → Bool) (w : List A) (a : A) (k : ℤ)
    (hbefore : countDifference p q r w < k)
    (hafter : k ≤ countDifference p q r (w ++ [a])) :
    p (w ++ [a]) = true ∧ q (w ++ [a]) = false ∧ r (w ++ [a]) = false := by
  rw [countDifference_append] at hafter
  obtain ⟨hp, hq, hr, _, _⟩ := upward_crossing hbefore hafter
  exact ⟨hp, hq, hr⟩

/-- The downward condition is equally about the actual count expression;
the zero extra increment is an explicit hypothesis (CM59). -/
theorem countDifference_downward (p q r : List A → Bool) (w : List A) (a : A) (k : ℤ)
    (hbefore : k ≤ countDifference p q r w)
    (hafter : countDifference p q r (w ++ [a]) < k) (hextra : r (w ++ [a]) = false) :
    p (w ++ [a]) = false ∧ q (w ++ [a]) = true := by
  rw [countDifference_append, hextra] at hafter
  obtain ⟨hp, hq, _, _⟩ := downward_crossing hbefore hafter
  exact ⟨hp, hq⟩

/-- Three different current child pairs all permit a downward crossing when
the extra count increments. The separate child bits are not identified (CM59). -/
theorem extra_downward_ambiguity :
    unitStep 0 false false true < 0 ∧ unitStep 0 true true true < 0 ∧
      unitStep 0 false true true < 0 := by
  norm_num [unitStep]

/-- Index the admissible right count for a true integer comparison exactly,
including negative thresholds; no label or program assumption enters (CM59). -/
theorem comparison_right_lower (a b e k : ℤ) :
    a < b + e + k ↔ a - e - k + 1 ≤ b := by
  omega

/-- The false comparison is the complementary upper bound on the right
count, so an index can intersect both truth constraints exactly (CM59). -/
theorem comparison_right_upper (a b e k : ℤ) :
    ¬ a < b + e + k ↔ b ≤ a - e - k := by
  omega

/-- A lower bound on a retained left count gives a necessary right bound
for any matching pair; no unseen-realizability claim is made (CM59). -/
theorem retained_left_lower {a lo b e k : ℤ} (ha : lo ≤ a) (ht : a < b + e + k) :
    lo - e - k + 1 ≤ b := by omega

/-- The complementary necessary right bound uses a retained left maximum (CM59). -/
theorem retained_left_upper {a hi b e k : ℤ} (ha : a ≤ hi) (ht : ¬ a < b + e + k) :
    b ≤ hi - e - k := by omega

/-- A retained right maximum bounds every matching left count (CM59). -/
theorem retained_right_upper {a b hi e k : ℤ} (hb : b ≤ hi) (ht : a < b + e + k) :
    a ≤ hi + e + k - 1 := by omega

/-- A retained right minimum bounds a false comparison's left count (CM59). -/
theorem retained_right_lower {a b lo e k : ℤ} (hb : lo ≤ b) (ht : ¬ a < b + e + k) :
    lo + e + k ≤ a := by omega

example : (0 : ℤ) - 0 - 0 + 1 ≤ 3 :=
  retained_left_lower (a := 2) (by norm_num) (by norm_num)

example : (1 : ℤ) ≤ 4 - 0 - 0 :=
  retained_left_upper (a := 2) (by norm_num) (by norm_num)

example : (2 : ℤ) ≤ 4 + 0 + 0 - 1 :=
  retained_right_upper (b := 3) (by norm_num) (by norm_num)

example : (0 : ℤ) + 0 + 0 ≤ 2 :=
  retained_right_lower (b := 1) (by norm_num) (by norm_num)

example : true = true ∧ false = false ∧ false = false ∧
    (0 : ℤ) = 1 - 1 ∧ unitStep 0 true false false = 1 :=
  upward_crossing (by norm_num) (by norm_num [unitStep])

example : false = false ∧ true = true ∧ (0 : ℤ) = 0 ∧
    unitStep 0 false true false = 0 - 1 :=
  downward_crossing (by norm_num) (by norm_num [unitStep])

example : (false = false ∨ false = true) ∧ (0 : ℤ) ≤ 0 + 1 ∧
    (0 : ℤ) - 2 ≤ unitStep 0 false false true :=
  downward_with_extra (by norm_num) (by norm_num [unitStep])

example : unitStep 0 true false true < 1 :=
  extra_excludes_upward true false (by norm_num)

example : lastTrue [true] = true ∧ (fun w => !lastTrue w) [true] = false ∧
    (fun w => decide (w = [false])) [true] = false :=
  countDifference_upward lastTrue (fun w => !lastTrue w) (fun w => decide (w = [false]))
    [] true 1 (by decide) (by decide)

example : (fun w => decide (w = [false])) [true] = false ∧ lastTrue [true] = true :=
  countDifference_downward (fun w => decide (w = [false])) lastTrue
    (fun w => decide (w = [false, false])) [] true 0 (by decide) (by decide) (by decide)

example : prefixCount lastTrue ([true, false] ++ [true]) =
    prefixCount lastTrue [true, false] + 1 :=
  prefixCount_append lastTrue [true, false] true

example : countDifference lastTrue onlyFirst lastTrue ([false] ++ [true]) =
    unitStep (countDifference lastTrue onlyFirst lastTrue [false]) true false true :=
  countDifference_append lastTrue onlyFirst lastTrue [false] true

end Transformer.Grokking.FlatSpectrum.ConvexDepth.ObservedPrefixes
