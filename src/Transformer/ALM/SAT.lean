/-
# From satisfiability to orthogonal vectors

`Transformer.ALM.Hardness` carries the hardness of Orthogonal Vectors as a
hypothesis.  This file removes the *combinatorial* part of that hypothesis by
proving the reduction that produces it — Williams' split-and-enumerate
construction:

* R. Williams, *A new algorithm for optimal 2-constraint satisfaction and its
  implications*, Theoret. Comput. Sci. 348 (2005), §4.

Split the variables into two halves.  For an assignment `α` of the left half
build the vector recording which clauses `α` leaves unsatisfied, and likewise
`β` on the right.  Then `α` and `β` together satisfy the formula exactly when
the two vectors are orthogonal, because a coordinate is `1` in both precisely
when that clause is satisfied by neither half.

Everything here is a theorem: the construction is explicit and the
equivalence `satisfiable_iff_orth` is proved, with no model of computation
involved.  What remains hypothetical after this file is the *time* side — the
Strong Exponential Time Hypothesis itself, and the Sparsification Lemma of
Impagliazzo–Paturi–Zane needed to keep the number of clauses linear.
-/

import Transformer.ALM.OrthVectors

namespace Transformer
namespace ALM

variable {n m : ℕ}

/-! ### Formulas over two halves -/

/-- A literal over two blocks of `n` variables: which variable, and the sign
that satisfies it. -/
abbrev Lit (n : ℕ) : Type := (Fin n ⊕ Fin n) × Bool

/-- A clause is a list of literals. -/
abbrev Clause (n : ℕ) : Type := List (Lit n)

/-- A CNF formula with `m` clauses over two blocks of `n` variables. -/
abbrev CNF (n m : ℕ) : Type := Fin m → Clause n

/-- The literal is satisfied by the left half of the assignment. -/
def lsat (α : Fin n → Bool) (l : Lit n) : Bool :=
  match l.1 with
  | .inl v => α v == l.2
  | .inr _ => false

/-- The literal is satisfied by the right half of the assignment. -/
def rsat (β : Fin n → Bool) (l : Lit n) : Bool :=
  match l.1 with
  | .inl _ => false
  | .inr v => β v == l.2

/-- The pair `(α, β)` satisfies every clause. -/
def Sat (φ : CNF n m) (α β : Fin n → Bool) : Prop :=
  ∀ c, ((φ c).any fun l => lsat α l || rsat β l) = true

/-- The formula has a satisfying assignment. -/
def Satisfiable (φ : CNF n m) : Prop := ∃ α β, Sat φ α β

/-! ### The construction -/

/-- Coordinate `c` of the left vector records that clause `c` is **not**
satisfied by the left half alone. -/
def aVec (φ : CNF n m) (α : Fin n → Bool) : BVec m :=
  fun c => !((φ c).any (lsat α))

/-- And the right vector, symmetrically. -/
def bVec (φ : CNF n m) (β : Fin n → Bool) : BVec m :=
  fun c => !((φ c).any (rsat β))

/-- A clause is satisfied by the pair exactly when it is satisfied by one of
the halves — the only fact about `List.any` the reduction uses. -/
private lemma any_or {γ : Type*} (l : List γ) (f g : γ → Bool) :
    (l.any f || l.any g) = l.any fun x => f x || g x := by
  induction l with
  | nil => rfl
  | cons a t ih =>
      simp only [List.any_cons, ← ih]
      cases f a <;> cases g a <;> simp

/-- **The reduction.**  The two vectors are orthogonal exactly when the two
halves together satisfy the formula: a coordinate is `1` in both precisely
when that clause is satisfied by neither half. -/
theorem orth_aVec_bVec_iff (φ : CNF n m) (α β : Fin n → Bool) :
    Orth (aVec φ α) (bVec φ β) ↔ Sat φ α β := by
  rw [orth_iff, Sat]
  constructor
  · intro h c
    have hc := h c
    simp only [aVec, bVec, Bool.and_eq_false_iff, Bool.not_eq_false'] at hc
    rw [← any_or]
    rcases hc with hc | hc <;> simp [hc]
  · intro h c
    have hc := h c
    rw [← any_or] at hc
    simp only [aVec, bVec, Bool.and_eq_false_iff, Bool.not_eq_false']
    rcases Bool.or_eq_true_iff.mp hc with hc | hc
    · exact Or.inl hc
    · exact Or.inr hc

/-- Satisfiability *is* an Orthogonal Vectors question over the assignments
of the two halves. -/
theorem satisfiable_iff_orth (φ : CNF n m) :
    Satisfiable φ ↔ ∃ α β, Orth (aVec φ α) (bVec φ β) := by
  unfold Satisfiable
  exact ⟨fun ⟨α, β, h⟩ => ⟨α, β, (orth_aVec_bVec_iff φ α β).mpr h⟩,
    fun ⟨α, β, h⟩ => ⟨α, β, (orth_aVec_bVec_iff φ α β).mp h⟩⟩

/-! ### As an instance of `2^n` vectors

`CostModel.Solves` asks about instances indexed by `Fin N`.  Enumerating the
assignments of one half turns the equivalence above into exactly that shape,
with `N = 2^n` vectors in dimension `m`: this is where the quadratic lower
bound on `N` becomes an exponential lower bound on `n`.
-/

/-- The assignments of one half, enumerated by `Fin (2^n)`. -/
noncomputable def assign (n : ℕ) : Fin (2 ^ n) → (Fin n → Bool) :=
  (Fintype.equivFinOfCardEq (α := Fin n → Bool) (by simp)).symm

lemma assign_surjective (n : ℕ) : Function.Surjective (assign n) :=
  (Fintype.equivFinOfCardEq (α := Fin n → Bool) (by simp)).symm.surjective

/-- **The instance.**  A formula on `2n` variables and `m` clauses becomes an
Orthogonal Vectors instance on `2^n` vectors of dimension `m`, with the same
answer.  This is the shape `CostModel.Solves` asks about. -/
theorem satisfiable_iff_ov (φ : CNF n m) :
    Satisfiable φ ↔
      ∃ i j, Orth (aVec φ (assign n i)) (bVec φ (assign n j)) := by
  rw [satisfiable_iff_orth]
  constructor
  · rintro ⟨α, β, h⟩
    obtain ⟨i, rfl⟩ := assign_surjective n α
    obtain ⟨j, rfl⟩ := assign_surjective n β
    exact ⟨i, j, h⟩
  · rintro ⟨i, j, h⟩
    exact ⟨_, _, h⟩

/-! ### Both answers are reachable -/

/-- A satisfiable formula: one clause, one literal, satisfied on the left. -/
example : Satisfiable (fun _ : Fin 1 => [((Sum.inl 0 : Fin 1 ⊕ Fin 1), true)]) := by
  refine ⟨fun _ => true, fun _ => true, fun c => ?_⟩
  simp [lsat]

/-- An unsatisfiable formula: a variable and its negation, both on the left,
so no pair of halves satisfies both clauses. -/
example : ¬ Satisfiable
    (fun c : Fin 2 =>
      [((Sum.inl 0 : Fin 1 ⊕ Fin 1), if c = 0 then true else false)]) := by
  rintro ⟨α, β, h⟩
  have h0 := h 0
  have h1 := h 1
  simp [lsat, rsat] at h0 h1
  rw [h0] at h1
  exact absurd h1 (by simp)

end ALM
end Transformer
