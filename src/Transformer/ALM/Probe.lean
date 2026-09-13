/-
# Algorithms that can only evaluate the formula

`Transformer.ALM.SETH` carries the Strong Exponential Time Hypothesis as an
assumption, and `Transformer.ALM.Independence` shows it cannot be discharged
from the `SATModel` interface.  Both are statements about *unrestricted*
algorithms, where the hypothesis is open.

Restrict the algorithm and the hypothesis becomes a theorem.  This file builds
the restriction: a `Probe` is an adaptive decision tree whose only operation is
to pick an assignment and ask whether it satisfies the formula.  Nothing else
about the formula is visible — no reading of the clauses, no structure to
exploit.

Two facts are proved here, and `Transformer.ALM.QueryModel` turns them into the
lower bound.  `run_eq_noVal` is the adversary: if none of the assignments the
tree examines along its all-"no" path satisfies the formula, the tree cannot
tell that formula from any other with the same property, and answers whatever
that path answers.  `run_scan` is the converse, that exhaustive search is a
legitimate probe algorithm, which is what keeps the lower bound from being a
statement about an empty model.

The argument is the classical half of the black-box search bound:

* C. Bennett, E. Bernstein, G. Brassard, U. Vazirani, *Strengths and weaknesses
  of quantum computing*, SIAM J. Comput. 26 (1997), §3.
-/

import Transformer.ALM.SAT

namespace Transformer
namespace ALM

variable {n m : ℕ}

/-- Satisfaction as a `Bool`, so a decision tree can branch on it. -/
def satB (φ : CNF n m) (α β : Fin n → Bool) : Bool :=
  decide (∀ c, ((φ c).any fun l => lsat α l || rsat β l) = true)

lemma satB_eq_true_iff (φ : CNF n m) (α β : Fin n → Bool) :
    satB φ α β = true ↔ Sat φ α β := by
  simp [satB, Sat]

lemma satB_eq_false_iff (φ : CNF n m) (α β : Fin n → Bool) :
    satB φ α β = false ↔ ¬ Sat φ α β := by
  rw [← satB_eq_true_iff]
  simp

/-- **A black-box algorithm.**  It may name an assignment and learn one bit —
whether that assignment satisfies the formula — and branch on the answer.  The
formula itself is never inspected. -/
inductive Probe (n : ℕ) : Type
  /-- Stop and answer. -/
  | done : Bool → Probe n
  /-- Evaluate the formula on one assignment, then continue either way. -/
  | ask : (Fin n → Bool) → (Fin n → Bool) → (Bool → Probe n) → Probe n

/-- Running the tree on a formula. -/
def run (φ : CNF n m) : Probe n → Bool
  | .done b => b
  | .ask α β k => run φ (k (satB φ α β))

/-- Worst-case number of evaluations: the depth of the tree. -/
def depth : Probe n → ℕ
  | .done _ => 0
  | .ask _ _ k => 1 + max (depth (k true)) (depth (k false))

/-! ### The adversary

Answering "no" to every question traces one path through the tree.  The
assignments on that path are the only ones the tree ever learns about, so any
two formulas unsatisfied by all of them receive the same answer.
-/

/-- The assignments examined when every answer is "no". -/
def noPath : Probe n → List ((Fin n → Bool) × (Fin n → Bool))
  | .done _ => []
  | .ask α β k => (α, β) :: noPath (k false)

/-- The answer given when every answer is "no". -/
def noVal : Probe n → Bool
  | .done b => b
  | .ask _ _ k => noVal (k false)

/-- **The adversary argument.**  A formula satisfied by none of the
assignments on the all-"no" path drives the tree down exactly that path, so
the tree answers `noVal` — whatever the formula actually is. -/
theorem run_eq_noVal (φ : CNF n m) :
    ∀ t : Probe n, (∀ p ∈ noPath t, ¬ Sat φ p.1 p.2) → run φ t = noVal t := by
  intro t
  induction t with
  | done b => intro _; rfl
  | ask α β k ih =>
      intro h
      have hfalse : satB φ α β = false :=
        (satB_eq_false_iff φ α β).mpr (h (α, β) (by simp [noPath]))
      rw [run, hfalse, noVal]
      exact ih false fun p hp => h p (by simp [noPath, hp])

/-- The path is no longer than the tree is deep, so it bounds the cost. -/
theorem noPath_length_le_depth : ∀ t : Probe n, (noPath t).length ≤ depth t := by
  intro t
  induction t with
  | done b => exact le_refl 0
  | ask α β k ih =>
      show (noPath (k false)).length + 1 ≤ 1 + max (depth (k true)) (depth (k false))
      have := ih false
      omega

/-! ### Exhaustive search is such an algorithm -/

/-- Try the listed assignments in order, stopping at the first that works. -/
def scan : List ((Fin n → Bool) × (Fin n → Bool)) → Probe n
  | [] => .done false
  | p :: t => .ask p.1 p.2 fun b => bif b then .done true else scan t

/-- And it is correct on the assignments it lists. -/
theorem run_scan (φ : CNF n m) :
    ∀ L : List ((Fin n → Bool) × (Fin n → Bool)),
      run φ (scan L) = true ↔ ∃ p ∈ L, Sat φ p.1 p.2 := by
  intro L
  induction L with
  | nil => simp [scan, run]
  | cons p t ih =>
      rw [scan, run]
      by_cases hp : Sat φ p.1 p.2
      · rw [(satB_eq_true_iff φ p.1 p.2).mpr hp]
        exact iff_of_true rfl ⟨p, List.mem_cons_self, hp⟩
      · rw [(satB_eq_false_iff φ p.1 p.2).mpr hp]
        show run φ (scan t) = true ↔ _
        rw [ih]
        constructor
        · rintro ⟨q, hq, hsat⟩
          exact ⟨q, List.mem_cons_of_mem _ hq, hsat⟩
        · rintro ⟨q, hq, hsat⟩
          rcases List.mem_cons.mp hq with rfl | hq'
          · exact absurd hsat hp
          · exact ⟨q, hq', hsat⟩

end ALM
end Transformer
