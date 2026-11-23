module

public import Transformer.Phasor.LookupQuestion

/-!
# Deep lookup through a fixed set of printed tables

The long synthetic depth study asks all five starting-key questions about
the very same tables and answer choices. Every reachable layer is a bijection,
so the five questions have different correct letters even when names are reused
across later layers. The model receives each printed question independently.
-/

@[expose] public section
namespace Transformer.Phasor.LookupQuestion.LongChains

/-- Every layer may use a different table, including different successors
for the same key. Only the entry reached at the current depth is needed to
verify the answer of an arbitrarily long printed chain. -/
theorem follows {A : Type*} [BEq A] (tables : List (List (A × A))) (p : ℕ → A)
    (edge : ∀ i, (hi : i < tables.length) → tables[i].lookup (p i) = some (p (i+1))) :
    follow tables (p 0) = some (p tables.length) := by
  induction tables generalizing p with
  | nil => rfl
  | cons table rest ih =>
    have first := edge 0 (by simp)
    simp only [List.getElem_cons_zero, Nat.zero_add] at first
    rw [follow,first,Option.bind_some]
    apply ih (fun i => p (i+1))
    intro i hi
    exact edge (i+1) (by simpa using hi)

/-- A prediction which ignores the question's starting key cannot be right
on two distinct successful starting keys with the same displayed tables
and choices when each table has unique preimages. -/
theorem different_questions {A : Type*} [BEq A] (tables : List (List (A × A)))
    (unique : ∀ t ∈ tables, UniquePreimage t) (x y a : A) (different : x ≠ y)
    (correct : follow tables x = some a) : follow tables y ≠ some a := by
  intro other
  exact different (recover_start tables unique x y a correct other)

example : follow [[(1,4)],[(4,2)],[(2,1)],[(1,9)]] (1:ℕ) = some 9 := by
  let p : ℕ → ℕ := fun i => [1,4,2,1,9][i]?.getD 0
  apply follows _ p
  intro i hi
  change i < 4 at hi
  have choices : i=0 ∨ i=1 ∨ i=2 ∨ i=3 := by omega
  rcases choices with rfl | rfl | rfl | rfl <;> rfl

example : follow [[(0,3)]] (1:ℕ) ≠ some 3 := by
  apply different_questions _ _ 0 1 3 (by decide) (by decide)
  intro t ht x y z hx hy
  rw [List.mem_singleton] at ht
  subst t
  by_cases l : x=0 <;> by_cases r : y=0 <;> simp_all [List.lookup_singleton]

end Transformer.Phasor.LookupQuestion.LongChains
