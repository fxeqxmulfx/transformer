module

public import Transformer.Phasor.TripleTransport.ScoreTiles

/-! Phasor frozen scoring, DiffGPT at 5a37c515, Engine::forward and forward_mhc.
A forward operation records its ordered read addresses and its destination.
Some training buffers also serve forward kernels. The live set must contain
every dependency of a live write, including such buffers. -/
@[expose] public section

namespace Transformer.Phasor.TripleTransport.AdjointStorage

variable {I V : Type*} [DecidableEq I]

abbrev Memory (I V : Type*) := I → V

structure Op (I V : Type*) where
  toCell : I
  fromCells : List I
  calculate : List V → V

def applyOp (op : Op I V) (m : Memory I V) (i : I) : V :=
  if i = op.toCell then op.calculate (op.fromCells.map m) else m i

def keepOp (live : Finset I) (op : Op I V) (m : Memory I V) : Memory I V :=
  if op.toCell ∈ live then applyOp op m else m

def agrees (live : Finset I) (m n : Memory I V) : Prop :=
  ∀ i ∈ live, m i = n i

def safe (live : Finset I) (op : Op I V) : Prop :=
  op.toCell ∈ live → ∀ j ∈ op.fromCells, j ∈ live

/-- A retained computation reads the same ordered inputs, including repeated
reads. The two executions may have arbitrary initial dead contents. -/
theorem step (live : Finset I) (op : Op I V) (m n : Memory I V) (old : agrees live m n)
    (covered : safe live op) : agrees live (applyOp op m) (keepOp live op n) := by
  intro i hi
  by_cases writes : op.toCell ∈ live
  · simp only [keepOp, writes, ite_true, applyOp]
    by_cases here : i = op.toCell
    · simp only [here, ite_true]
      congr 1
      exact List.map_congr_left fun j hj => old j (covered writes j hj)
    · simp [here, old i hi]
  · have away : i ≠ op.toCell := fun e => writes (e ▸ hi)
    simp [keepOp, writes, applyOp, away, old i hi]

def run (apply : Op I V → Memory I V → Memory I V) :
    List (Op I V) → Memory I V → Memory I V
  | [], m => m
  | op :: ops, m => run apply ops (apply op m)

/-- Omitting adjoint-only writes preserves every live output of an entire
frozen forward whose live destinations depend only on live sources. -/
theorem forward (live : Finset I) (ops : List (Op I V)) (m n : Memory I V)
    (init : agrees live m n) (closed : ∀ op ∈ ops, safe live op) :
    agrees live (run applyOp ops m) (run (keepOp live) ops n) := by
  induction ops generalizing m n with
  | nil => exact init
  | cons op ops ih =>
    apply ih
    · exact step live op m n init (closed _ List.mem_cons_self)
    · exact fun later h => closed _ (List.mem_cons_of_mem _ h)

private def sumInto (dest : ℕ) (args : List ℕ) : Op ℕ ℕ := ⟨dest, args, List.sum⟩
private def fresh (i : ℕ) := i + 1

example : agrees {0, 1} (applyOp (sumInto 1 [0, 1, 0]) fresh)
    (keepOp {0, 1} (sumInto 1 [0, 1, 0]) fresh) := by
  apply step
  · intro i _
    rfl
  · intro _ j hj
    simp [sumInto] at hj ⊢
    tauto

example : run applyOp [sumInto 7 [0, 1], sumInto 1 [0, 1, 0]] fresh 1 =
    run (keepOp {0, 1}) [sumInto 7 [0, 1], sumInto 1 [0, 1, 0]] fresh 1 := by
  apply forward (live := {0, 1})
  · intro i _
    rfl
  · intro op h
    simp only [List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with rfl | rfl
    · simp [safe, sumInto]
    · intro _ j hj
      simp [sumInto] at hj ⊢
      tauto
  · simp

/-- Dropping a scratch buffer consumed by a later forward operation can
change the final prediction. -/
example : run applyOp [sumInto 7 [0, 1], sumInto 1 [7]] fresh 1 ≠
    run (keepOp {0, 1}) [sumInto 7 [0, 1], sumInto 1 [7]] fresh 1 := by decide

end Transformer.Phasor.TripleTransport.AdjointStorage
