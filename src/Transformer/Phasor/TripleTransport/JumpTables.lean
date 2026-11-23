module
public import Transformer.Phasor.TripleTransport.ReadHorizon
public import Batteries.Data.Vector.Lemmas

/-! # Materialized powers of a causal-prefix transition

The Phasor path lookup takes its first edge at the query key and then
advances a cursor. Each eligible directed write can be assigned a finite
index. A missing continuation keeps the current edge, including its write
time. The index is constructed after consuming the prefix being queried.
-/
@[expose] public section
namespace Transformer.Phasor.TripleTransport.JumpTables
open ReadHorizon

/-- At level zero store the one-step successor. Each additional level
performs two reads of the previous vector for each state in parallel. -/
def level {n : ℕ} (f : Fin n → Fin n) : ℕ → Vector (Fin n) n
  | 0 => Vector.ofFn f
  | j+1 => let t := level f j; Vector.ofFn (fun i => t.get (t.get i))

/-- Looking up a materialized level takes a single index operation and
returns the same state as the corresponding power of the transition. -/
theorem get_level {n : ℕ} (f : Fin n → Fin n) (j : ℕ) (i : Fin n) :
    (level f j).get i = double f j i := by
  induction j generalizing i with
  | zero => simp [level, double]
  | succ j ih => simp only [level, Vector.get_ofFn, ih, double]

/-- With levels three and five available, two dependent table reads
reach the forty-step state, including any stored cursor or terminal flag. -/
theorem forty {n : ℕ} (f : Fin n → Fin n) (i : Fin n) :
    (level f 5).get ((level f 3).get i) = f^[40] i := by
  rw [get_level, get_level, forty_from_two_jumps]

/-- A query's first read can select its initial edge differently from
the continuation rule for an edge already carrying a write time. -/
def step {Q E : Type*} (first : Q → E) (next : E → E) : Sum Q E → Sum Q E
  | .inl q => .inr (first q)
  | .inr e => .inr (next e)

/-- A budget of n+1 includes the first read. The remaining n steps
advance the cursor that the original lookup uses to select its terminal. -/
theorem from_query {Q E : Type*} (first : Q → E) (next : E → E) (q : Q) (n : ℕ) :
    (step first next)^[n+1] (.inl q) = .inr (next^[n] (first q)) := by
  induction n with
  | zero => rfl
  | succ n ih => rw [Function.iterate_succ_apply', ih, Function.iterate_succ_apply']; rfl

example : (level (fun x : Fin 41 => x + 1) 5).get ((level (fun x => x + 1) 3).get 0) = 40 := by
  decide

end Transformer.Phasor.TripleTransport.JumpTables
