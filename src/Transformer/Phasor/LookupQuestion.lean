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

end Transformer.Phasor.LookupQuestion
