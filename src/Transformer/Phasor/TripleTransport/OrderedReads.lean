module

public import Transformer.Phasor.TripleTransport.TypedLookupPaths

/-!
# A shared read with a cursor over append-only history

The Phasor deep-loop experiment retains a chronological history for each dense
key and relation. The state carries the earliest permitted write time. A
successful read advances past the selected write. Repeated keys may therefore
have different successors on the same trajectory.
-/

@[expose] public section
namespace Transformer.Phasor.TripleTransport.OrderedReads
variable {K R : Type*} [DecidableEq K] [DecidableEq R]

structure Edge (K R : Type*) where
  key : K
  relation : R
  dest : K
  time : ℕ

def live (r : R) (s : K × ℕ) (e : Edge K R) : Bool :=
  decide (e.key = s.1 ∧ e.relation = r ∧ s.2 ≤ e.time)

def lookup (history : List (Edge K R)) (r : R) (s : K × ℕ) : Option (K × ℕ) :=
  (history.find? (live r s)).map fun e => (e.dest, e.time+1)

/-- The first still-eligible matching write determines the successor and the
new time. All later writes may disagree at exactly the same address. -/
theorem first_live (before after : List (Edge K R)) (e : Edge K R) (r : R) (s : K × ℕ)
    (skip : ∀ b ∈ before, live r s b = false) (hit : live r s e = true) :
    lookup (before ++ e::after) r s = some (e.dest,e.time+1) := by
  have empty : before.find? (live r s) = none :=
    List.find?_eq_none.mpr (fun b hb => by simp [skip b hb])
  simp [lookup, List.find?_append, empty, hit]

/-- Per-state relation choice and a strictly chronological cursor suffice to
recover paths with rebound keys. Any excess read budget preserves a path that
ends with no eligible edge of its chosen relation. -/
theorem follows (history : List (Edge K R)) (choose : K × ℕ → R)
    (path : ℕ → K × ℕ) (n extra : ℕ)
    (edges : ∀ i<n, ∃ before e after, history=before++e::after ∧
      (∀ b ∈ before, live (choose (path i)) (path i) b=false) ∧
      live (choose (path i)) (path i) e=true ∧ path (i+1)=(e.dest,e.time+1))
    (ends : ∀ b ∈ history, live (choose (path n)) (path n) b=false) :
    TypedLookupPaths.follow (fun s => lookup history (choose s) s) (path 0) (n+extra) = path n := by
  apply TypedLookupPaths.saturates
  · intro i hi
    obtain ⟨before,e,after,whole,skip,hit,next⟩ := edges i hi
    rw [whole,next,first_live before after e _ _ skip hit]
  · have empty := List.find?_eq_none.mpr (fun b hb => by
        have h := ends b hb
        exact Bool.not_eq_true _ |>.mpr h)
    unfold lookup
    rw [empty]
    rfl

private def exampleHistory : List (Edge ℕ Bool) := [⟨0,true,1,0⟩,⟨1,false,0,1⟩,⟨0,true,2,2⟩]
private def examplePath (i : ℕ) : ℕ × ℕ := [(0,0),(1,1),(0,2),(2,3)][i]?.getD (2,3)

example : lookup exampleHistory true (0,2) = some (2,3) := by
  apply first_live [⟨0,true,1,0⟩,⟨1,false,0,1⟩] [] ⟨0,true,2,2⟩
  · intro b hb
    simp at hb
    rcases hb with rfl|rfl <;> decide
  · decide

example : TypedLookupPaths.follow (fun s => lookup exampleHistory (decide (s.1=0)) s) (0,0) (3+40) =
    (2,3) := by
  apply follows _ (fun s => decide (s.1=0)) examplePath 3 40
  · intro i hi
    interval_cases i
    · exact ⟨[],⟨0,true,1,0⟩,[⟨1,false,0,1⟩,⟨0,true,2,2⟩],rfl,by decide,rfl,rfl⟩
    · exact ⟨[⟨0,true,1,0⟩],⟨1,false,0,1⟩,[⟨0,true,2,2⟩],rfl,by decide,rfl,rfl⟩
    · exact ⟨[⟨0,true,1,0⟩,⟨1,false,0,1⟩],⟨0,true,2,2⟩,[],rfl,by decide,rfl,rfl⟩
  · decide

end Transformer.Phasor.TripleTransport.OrderedReads
