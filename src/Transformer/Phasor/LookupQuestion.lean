module

public import Mathlib.Data.List.Basic

/-!
# The authored lookup-question study

Source: the “N-Hop Lookup” synthetic example in `stress_tests_for_gpt.md`.
The outer scorer applies the printed tables in order and then selects a
displayed letter. The model sees the question text and supplies the letter.
-/

@[expose] public section

namespace Transformer.Phasor.LookupQuestion

/-- Follow each displayed table in order. A missing key yields no answer. -/
def follow {A : Type*} [BEq A] : List (List (A × A)) → A → Option A
  | [], x => some x
  | table :: rest, x => (table.lookup x).bind (follow rest)

theorem append {A : Type*} [BEq A] (left right : List (List (A × A))) (x : A) :
    follow (left ++ right) x = (follow left x).bind (follow right) := by
  induction left generalizing x with
  | nil => rfl
  | cons t ts ih =>
    have eq : follow (ts ++ right) = fun y => (follow ts y).bind (follow right) := funext ih
    simp [follow, Option.bind_assoc, eq]

/-- Uniformly renaming all fictional entries commutes with following the tables
whenever each displayed lookup commutes with that renaming. -/
theorem rename {A B : Type*} [BEq A] [BEq B] (f : A → B)
    (left : List (List (A × A))) (right : List (List (B × B))) (x : A)
    (maps : List.Forall₂ (fun a b => ∀ k, (a.lookup k).map f = b.lookup (f k)) left right) :
    (follow left x).map f = follow right (f x) := by
  induction maps generalizing x with
  | nil => rfl
  | cons eq _ ih =>
    simp only [follow, ← eq]
    cases h : List.lookup x _ with
    | none => rfl
    | some y => exact ih y

/-- Shuffling rows preserves every lookup of a finite restriction of a function,
including a query whose key does not occur in the table. -/
theorem shuffle {A B : Type*} [DecidableEq A] (f : A → B) (left right : List (A × B))
    (perm : left.Perm right) (fits : ∀ p ∈ left, p.2 = f p.1) (k : A) :
    left.lookup k = right.lookup k := by
  have form (rows : List (A × B)) (ok : ∀ p ∈ rows, p.2 = f p.1) :
      rows.lookup k = if k ∈ rows.map Prod.fst then some (f k) else none := by
    induction rows with
    | nil => simp
    | cons p ps ih =>
      obtain ⟨x, y⟩ := p
      have value : y = f x := ok _ List.mem_cons_self
      have tail := ih (fun p hp => ok p (List.mem_cons_of_mem _ hp))
      by_cases eq : k = x
      · subst x
        simp [value]
      · have test : (k == x) = false := beq_eq_false_iff_ne.mpr eq
        simp only [List.lookup_cons, test, List.map_cons, List.mem_cons, eq, false_or, tail]
  rw [form left fits, form right (fun p hp => fits p (perm.mem_iff.mpr hp))]
  simp only [(perm.map Prod.fst).mem_iff]

/-- First displayed table of the Mira question. -/
def miraFirst : List (String × String) :=
  [("Nalo", "vek"), ("Mira", "dax"), ("Sori", "lum"), ("Tavi", "pim"), ("Yuna", "ros")]

/-- Second displayed table of the Mira question. -/
def miraSecond : List (String × String) :=
  [("lum", "jade"), ("ros", "opal"), ("dax", "amber"), ("vek", "quartz"), ("pim", "onyx")]

/-- Answer choices in the printed order, indexed by their displayed value. -/
def miraOptions : List (String × String) :=
  [("jade", "A"), ("quartz", "B"), ("amber", "C"), ("onyx", "D"), ("opal", "E")]

/-- The two printed tables and the choice order give the reference answer C. -/
theorem mira : follow [miraFirst] "Mira" = some "dax" ∧
    follow [miraFirst, miraSecond] "Mira" = some "amber" ∧
    ((follow [miraFirst, miraSecond] "Mira").bind fun v => miraOptions.lookup v) = some "C" := by
  decide

example : follow [miraFirst, miraSecond] "Mira" =
    (follow [miraFirst] "Mira").bind (follow [miraSecond]) := append [miraFirst] [miraSecond] _

example : (follow [[(1, 2)]] 1).map Nat.succ = follow [[(2, 3)]] 2 := by
  apply rename
  constructor
  · intro k
    by_cases h : k = 1 <;> simp [List.lookup_singleton, h]
  · constructor

example (k : ℕ) : [(0, 1), (1, 2)].lookup k = [(1, 2), (0, 1)].lookup k := by
  apply shuffle Nat.succ _ _ (List.Perm.swap _ _ [])
  intro p hp
  rw [List.mem_cons, List.mem_singleton] at hp
  rcases hp with rfl | rfl <;> decide


/-- Each displayed value has at most one key that retrieves it. -/
def UniquePreimage {A : Type*} [BEq A] (table : List (A × A)) : Prop :=
  ∀ a b v, table.lookup a = some v → table.lookup b = some v → a = b

/-- In the synthetic lookup study, every printed table is a bijection.
Following any suffix of tables whose values have unique preimages makes
the starting key recoverable from a successful result. -/
theorem recover_start {A : Type*} [BEq A] (tables : List (List (A × A)))
    (unique : ∀ t ∈ tables, UniquePreimage t) (a b result : A)
    (left : follow tables a = some result) (right : follow tables b = some result) : a = b := by
  induction tables generalizing a b with
  | nil => simpa only [follow,Option.some.injEq] using left.trans right.symm
  | cons t ts ih =>
    cases ha : t.lookup a with
    | none => simp [follow,ha] at left
    | some x =>
      cases hb : t.lookup b with
      | none => simp [follow,hb] at right
      | some y =>
        have eq : x=y := ih (fun t ht => unique t (List.mem_cons_of_mem _ ht)) x y
          (by simpa [follow,ha] using left) (by simpa [follow,hb] using right)
        exact unique t List.mem_cons_self a b x ha (by simpa only [eq] using hb)

/-- The study cycles the first table's output column. When this supplies
a different intermediate value, both successful chains have different
reference answers through any suffix with unique preimages. -/
theorem rewire_changes {A : Type*} [BEq A] (suffix : List (List (A × A)))
    (unique : ∀ t ∈ suffix, UniquePreimage t) (x y a b : A)
    (rewired : x ≠ y) (left : follow suffix x=some a) (right : follow suffix y=some b) : a≠b := by
  rintro rfl
  exact rewired (recover_start suffix unique x y a left right)

private theorem one (k v : ℕ) : UniquePreimage [(k,v)] := by
  intro x y z hx hy
  by_cases hx' : x=k <;> by_cases hy' : y=k <;> simp_all [List.lookup_singleton]

example (x y : ℕ) (a : follow [[(11,27)]] x=some 27) (b : follow [[(11,27)]] y=some 27) : x=y :=
  recover_start _ (by intro t ht; rw [List.mem_singleton] at ht; exact ht ▸ one _ _) _ _ _ a b

private theorem fork : UniquePreimage [(1,11),(2,12)] := by
  intro a b v ha hb
  by_cases h₁ : a=1 <;> by_cases h₂ : a=2 <;> by_cases h₃ : b=1 <;> by_cases h₄ : b=2 <;>
    simp_all [List.lookup_cons,beq_eq_false_iff_ne.mpr] <;> omega

example : 11≠12 := rewire_changes [[(1,11),(2,12)]]
  (by intro t h; rw [List.mem_singleton] at h; exact h ▸ fork) 1 2 11 12 (by decide)
    (by decide) (by decide)

/-- Synthetic lookup intervention at any displayed hop, including the last.
Keep every preceding and following table fixed. Replacing the reached edge
with a different value changes the final answer whenever the remaining
tables have unique preimages and both routes finish. The alternatives may
each edit other, unreached rows to preserve the printed token counts. -/
theorem interior_rewire {A : Type*} [BEq A] (pre post : List (List (A × A)))
    (before after : List (A × A)) (start key x y : A)
    (prefix_ok : follow pre start = some key)
    (old_edge : before.lookup key = some x) (new_edge : after.lookup key = some y)
    (different : x ≠ y) (unique : ∀ t ∈ post, UniquePreimage t)
    (a b : A) (old_end : follow post x = some a) (new_end : follow post y = some b) :
    follow (pre ++ before :: post) start = some a ∧
    follow (pre ++ after :: post) start = some b ∧ a ≠ b := by
  refine ⟨?_, ?_, rewire_changes post unique x y a b different old_end new_end⟩
  · simpa only [append, prefix_ok, Option.bind_some, follow, old_edge] using old_end
  · simpa only [append, prefix_ok, Option.bind_some, follow, new_edge] using new_end

example : follow ([[(0,4)]] ++ [(4,1),(5,2)] :: [[(1,11),(2,12)]]) 0 = some 11 ∧
    follow ([[(0,4)]] ++ [(4,2),(5,1)] :: [[(1,11),(2,12)]]) 0 = some 12 ∧ 11 ≠ 12 := by
  apply interior_rewire _ _ _ _ 0 4 1 2 (by decide) (by decide) (by decide) (by decide)
  · intro t ht
    rw [List.mem_singleton] at ht
    exact ht ▸ fork
  · decide
  · decide

end Transformer.Phasor.LookupQuestion
