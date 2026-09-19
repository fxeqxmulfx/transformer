/-
# The RASP-Generalization Conjecture

Zhou et al. — arXiv:2310.16028v1, "What Algorithms can Transformers Learn?  A
Study in Length Generalization" (ICLR 2024), §3 ("Main Conjecture").

> A decoder-only autoregressive Transformer is likely to length-generalize
> when trained to completion on an algorithmic task if the following
> conditions hold.  **Realizability.** The true next-token function for the
> task can be represented by a single causal Transformer which works on all
> input lengths.  **Simplicity.** This representation is "simple", meaning it
> can be written in RASP-L.  **Diversity.** The training data is sufficiently
> diverse, such that there does not exist any shorter RASP-L program which
> agrees with the task in-distribution but not out-of-distribution.

This is a conjecture and, as the paper stresses, a *phenomenological* one:
"we do not claim Transformers will actually learn weights which are close to
the compiled weights of the RASP-L program".  Its conclusion — "likely to
length-generalize" — is not a mathematical predicate, so nothing here is
stated as a theorem.  What is written out is the three conditions, as
predicates on a task, and the conjecture itself as an implication
parameterized by whatever `LengthGeneralizes` is taken to mean:
`RASPGeneralizationConjecture` is a `Prop` to be assumed, never proved.

Two pieces of the statement stay parameters because the paper leaves them so.
`Rep` is the set of length-polymorphic maps a single causal Transformer
computes at every length (the architecture is fixed but unmodelled here, and
the paper's footnote allows weights in `ℝ ∪ {±∞}` and any uniformly
generated positional encoding).  `L` is the RASP-L-writable subset, whose
index restrictions Appendix C calls an open problem.  `size` is the program
length that "shorter" in the Diversity condition refers to.

The paper's footnote that a next-token function needing `Ω(n³)` time is
representable by no Transformer — by the Time Hierarchy Theorem, since
Transformers are simulable in `O(n²)` time — is a statement about Turing
machines, which this development does not model; it is recorded in `todo.md`.
-/

import Transformer.RASPL.Defs

namespace Transformer
namespace RASPL

open RASP

variable {V : Type*}

/-- An algorithmic task: the true next-token function, defined at every
prompt length (§3).  Prompts are nonempty, so the length is written `n + 1`. -/
def Task (V : Type*) : Type _ := (n : ℕ) → Seq (n + 1) V → V

/-- A length-polymorphic sequence-to-sequence map: one object that acts at
*every* input length, which is what "a single causal Transformer which works
on all input lengths" asks for. -/
def Prog (V : Type*) : Type _ := (n : ℕ) → Seq n V → Seq n V

/-- The next token a program predicts: autoregressive decoding reads the
output at the last position. -/
def nextTok (P : Prog V) {n : ℕ} (x : Seq (n + 1) V) : V := P (n + 1) x (Fin.last n)

/-- The program computes the task at every length — out of distribution
included.  This is the conclusion that Diversity is there to force. -/
def Computes (P : Prog V) (T : Task V) : Prop :=
  ∀ (n : ℕ) (x : Seq (n + 1) V), nextTok P x = T n x

/-- The program agrees with the task in-distribution: on the prompts the
training distribution `D` puts mass on. -/
def AgreesOn (P : Prog V) (T : Task V) (D : (n : ℕ) → Finset (Seq (n + 1) V)) : Prop :=
  ∀ (n : ℕ), ∀ x ∈ D n, nextTok P x = T n x

/-- **Realizability.**  Some map in `Rep` — the length-polymorphic maps a
single causal Transformer computes at all lengths — is the task's true
next-token function. -/
def Realizable (Rep : Set (Prog V)) (T : Task V) : Prop :=
  ∃ P ∈ Rep, Computes P T

/-- **Simplicity.**  That representation lies in the RASP-L-writable subset
`L`. -/
def Simple (Rep L : Set (Prog V)) (T : Task V) : Prop :=
  ∃ P ∈ Rep, P ∈ L ∧ Computes P T

/-- **Diversity.**  No shorter RASP-L program agrees with the task
in-distribution but not out-of-distribution: every shorter program of `L`
that agrees on `D` in fact computes the task at every length. -/
def Diverse (L : Set (Prog V)) (size : Prog V → ℕ) (P : Prog V) (T : Task V)
    (D : (n : ℕ) → Finset (Seq (n + 1) V)) : Prop :=
  ∀ Q ∈ L, size Q < size P → AgreesOn Q T D → Computes Q T

/-- The three conditions of the conjecture, on one task and one training
distribution: a single RASP-L-writable representation that a Transformer
computes at all lengths, against which no shorter RASP-L program is a
spurious in-distribution match. -/
def RASPGeneralizable (Rep L : Set (Prog V)) (size : Prog V → ℕ) (T : Task V)
    (D : (n : ℕ) → Finset (Seq (n + 1) V)) : Prop :=
  ∃ P ∈ Rep, P ∈ L ∧ Computes P T ∧ Diverse L size P T D

/-- **The RASP-Generalization Conjecture** (§3), as a predicate.

It is not written as a `theorem … := by sorry`, the way every unproved
statement in this development is, because it is not unproved: read over an
arbitrary `LengthGeneralizes` it is **false**, and
`not_forall_raspGeneralizationConjecture` below proves it so.  The conclusion
"likely to length-generalize" is empirical and has no formal content to fix
`LengthGeneralizes` with, so the conjecture is a predicate of it — and a
predicate of its arguments is a definition. -/
def RASPGeneralizationConjecture (Rep L : Set (Prog V)) (size : Prog V → ℕ)
    (D : (n : ℕ) → Finset (Seq (n + 1) V)) (LengthGeneralizes : Task V → Prop) : Prop :=
  ∀ T : Task V, RASPGeneralizable Rep L size T D → LengthGeneralizes T

/-- Simplicity implies Realizability, which is the only implication among the
three conditions: §3 presents Simplicity as a property of the representation
Realizability produces. -/
theorem Realizable.of_simple {Rep L : Set (Prog V)} {T : Task V} (h : Simple Rep L T) :
    Realizable Rep T := by
  obtain ⟨P, hP, _, hc⟩ := h
  exact ⟨P, hP, hc⟩

/-- The three conditions are satisfiable together, so the conjecture is not
vacuous.  The task is "copy the last token", its representation is the
identity program, and the training distribution is the extreme case of
diversity: every prompt of every length occurs, so agreeing in-distribution
*is* computing the task. -/
example :
    RASPGeneralizable (V := Bool) Set.univ Set.univ (fun _ => 0)
      (fun n x => x (Fin.last n)) (fun n => (Finset.univ : Finset (Seq (n + 1) Bool))) :=
  ⟨fun _ x => x, Set.mem_univ _, Set.mem_univ _, fun _ _ => rfl,
    fun _ _ _ h n x => h n x (Finset.mem_univ x)⟩

/-- Simplicity is satisfiable on its own too. -/
example : Simple (V := Bool) Set.univ Set.univ (fun n x => x (Fin.last n)) :=
  ⟨fun _ x => x, Set.mem_univ _, Set.mem_univ _, fun _ _ => rfl⟩

/-- **The conjecture is about one `LengthGeneralizes`, not about all of
them.**

Taking `LengthGeneralizes` to be the constantly false predicate, the
conjecture asserts that no task is RASP-generalizable; the copy task of the
example above is one.  So `RASPGeneralizationConjecture` quantified over its
last argument is not an open statement but a refuted one, which is why it is
a definition here and not a `theorem … := by sorry`. -/
theorem not_forall_raspGeneralizationConjecture :
    ¬ ∀ LengthGeneralizes : Task Bool → Prop,
        RASPGeneralizationConjecture (V := Bool) Set.univ Set.univ (fun _ => 0)
          (fun n => (Finset.univ : Finset (Seq (n + 1) Bool))) LengthGeneralizes := by
  intro h
  exact h (fun _ => False) (fun n x => x (Fin.last n))
    ⟨fun _ x => x, Set.mem_univ _, Set.mem_univ _, fun _ _ => rfl,
      fun _ _ _ hQ n x => hQ n x (Finset.mem_univ x)⟩

end RASPL
end Transformer
