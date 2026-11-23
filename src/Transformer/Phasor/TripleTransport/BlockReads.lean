module

public import Transformer.Phasor.TripleTransport.TypedLookupPaths

/-!
# Read-only blocks of a causal associative scan

A worker seeds the complete write state at its left boundary. Each row uses
the state through its own position and writes its output to that same index.
This applies to directed relation histories and to stores with overwrites.
-/
@[expose] public section
namespace Transformer.Phasor.TripleTransport.BlockReads
variable {A S V : Type*}

/-- A local result after replaying the previous writes and then the requested
part of the block. The read may depend on the absolute row and its full state. -/
def blockRead (write : S → A → S) (read : ℕ → S → V) (initial : S)
    (xs : List A) (start offset : ℕ) : V :=
  let base := (xs.take start).foldl write initial
  read (start+offset) (((xs.drop start).take (offset+1)).foldl write base)

/-- A block may start anywhere, including the second half of the same
sequence. Even an imperfect choice of block widths leaves each emitted
position and every observable of its causal store unchanged. -/
theorem agrees (write : S → A → S) (read : ℕ → S → V) (initial : S)
    (xs : List A) (start offset : ℕ) :
    blockRead write read initial xs start offset =
      read (start+offset) ((xs.take (start+offset+1)).foldl write initial) := by
  unfold blockRead
  rw [Nat.add_assoc, List.take_add (l := xs) (i := start) (j := offset+1), List.foldl_append]

/-- Blocks cannot reuse the final state of a previous packed sequence.
Resetting to the incoming sequence's initial state restores its prefix. -/
example : blockRead (fun s a : Nat => 7*s+a) Prod.mk 0 [3,1,2,0,4] 2 1 = (3,1092) ∧
    blockRead (fun s a : Nat => 7*s+a) Prod.mk 0 [4,1] 0 1 = (1,29) := by decide

end Transformer.Phasor.TripleTransport.BlockReads
