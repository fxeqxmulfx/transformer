module

public import Transformer.Phasor.InclusiveTransitions

/-! DiffGPT 5a37c515, synthetic/crasp.py, inclusive counting fragment used
for the fixed-length transfer sets. Terms have signed integer constants,
negation and addition. Each count includes the position currently scored.
The mask discards BOS outside the formula. All positions below index the
payload starting at zero. -/
@[expose] public section

namespace Transformer.Phasor.CraspLengthScore
open scoped BigOperators
inductive Expr (α : Type) : Type → Type 1 where
  | symbol (v : α) : Expr α Bool
  | literal (v : ℤ) : Expr α ℤ
  | count (p : Expr α Bool) : Expr α ℤ
  | neg (p : Expr α ℤ) : Expr α ℤ
  | add (p q : Expr α ℤ) : Expr α ℤ
  | lt (p q : Expr α ℤ) : Expr α Bool
  | inv (p : Expr α Bool) : Expr α Bool
  | conj (p q : Expr α Bool) : Expr α Bool

variable {α β : Type} [DecidableEq α]

def Expr.eval : {γ : Type} → Expr α γ → (ℕ → α) → ℕ → γ
  | _, .symbol a, x, i => decide (x i = a)
  | _, .literal z, _, _ => z
  | _, .count p, x, i => ∑ j ∈ Finset.range (i+1), if p.eval x j then 1 else 0
  | _, .neg p, x, i => -p.eval x i
  | _, .add p q, x, i => p.eval x i+q.eval x i
  | _, .lt p q, x, i => decide (p.eval x i<q.eval x i)
  | _, .inv p, x, i => !(p.eval x i)
  | _, .conj p q, x, i => p.eval x i && q.eval x i

/-- Recounting a source formula on a longer sequence preserves every
previously supervised prefix, through arbitrarily nested signed terms. -/
theorem Expr.prefix_eq (e : Expr α β) (x y : ℕ → α) (i : ℕ)
    (same : ∀ j ≤ i, x j=y j) : e.eval x i=e.eval y i := by
  induction e generalizing i with
  | symbol a => simp [eval,same i (by omega)]
  | literal z => rfl
  | count p ih =>
    apply Finset.sum_congr rfl
    intro j hj
    have le : j ≤ i := by simpa using hj
    rw [ih j (fun k hk => same k (hk.trans le))]
  | neg p ih | inv p ih => simp only [eval,ih i same]
  | add p q ih ik | lt p q ih ik | conj p q ih ik => simp only [eval,ih i same,ik i same]

/-- The model's exact-sequence scorer checks every one of the first n
payload positions. The correctness of a prefix depends only on that part
of both the output and the source input. -/
theorem exact_prefix (e : Expr α Bool) (x y : ℕ → α) (p q : ℕ → Bool) (n : ℕ)
    (inputs : ∀ j<n, x j=y j) (outputs : ∀ j<n, p j=q j) :
    (∀ j<n, p j=e.eval x j) ↔ (∀ j<n, q j=e.eval y j) := by
  have he (j : ℕ) (hj : j<n) : p j=q j ∧ e.eval x j=e.eval y j :=
    ⟨outputs j hj, e.prefix_eq x y j (fun k hk => inputs k (by omega))⟩
  constructor <;> intro h j hj
  · rw [← (he j hj).1,← (he j hj).2,h j hj]
  · rw [(he j hj).1,(he j hj).2,h j hj]

private def formula : Expr Bool Bool := .lt (.count (.symbol true)) (.literal 2)
private def x (i : ℕ) := decide (i%2=0)
private def y (i : ℕ) := decide (i=0)

example : formula.eval x 1=formula.eval y 1 :=
  formula.prefix_eq x y 1 (by intro k hk; interval_cases k <;> decide)

example : formula.eval x 2=false ∧ formula.eval y 2=true := by decide

example : (∀ j<2, true=formula.eval x j) ↔ (∀ j<2, true=formula.eval y j) :=
  exact_prefix formula x y (fun _=>true) (fun _=>true) 2
    (by intro j hj; interval_cases j <;> decide) (by intros; rfl)
end Transformer.Phasor.CraspLengthScore
