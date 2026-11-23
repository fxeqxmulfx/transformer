module

public import Transformer.Phasor.TripleTransport.ScoreScratch

/-! Phasor's frozen scorer, DiffGPT at 5a37c515, phase_score_forward::forward.
Each batch/query/head row uses only its own causal prefix. The score's
imaginary component is stored by the original kernel for later backward.
The softmax normalizer and value sum use the scalar score in causal order. -/
@[expose] public section

namespace Transformer.Phasor.TripleTransport.ScoreTiles

variable {S I V : Type*}

/-- Flatten only the current tile. Read its scalar scores independently of
the old scratch contents. Padding after an active prefix is left untouched. -/
def tile (origin rows stride : ℕ) (width : ℕ → ℕ) (score : ℕ → ℕ → S × I)
    (pad : ℕ → S) (address : ℕ) : S :=
  let localRow := address / stride
  let j := address % stride
  if localRow < rows ∧ j < width (origin + localRow) then
    (score (origin + localRow) j).1
  else pad address

/-- Every causal slot in a live tile recovers the same row and column in
the original buffer, including a short final tile. -/
theorem read_eq (origin rows stride localRow j : ℕ) (width : ℕ → ℕ)
    (score : ℕ → ℕ → S × I) (pad : ℕ → S) (live : localRow < rows)
    (active : j < width (origin + localRow)) (fits : width (origin + localRow) ≤ stride) :
    tile origin rows stride width score pad (localRow * stride + j) =
    (score (origin + localRow) j).1 := by
  have small : j < stride := lt_of_lt_of_le active fits
  have pos : 0 < stride := by omega
  have quotient : (localRow * stride + j) / stride = localRow := by
    rw [Nat.mul_comm localRow, Nat.mul_add_div pos, Nat.div_eq_of_lt small, Nat.add_zero]
  have remainder : (localRow * stride + j) % stride = j := by
    rw [Nat.mul_comm localRow, Nat.mul_add_mod, Nat.mod_eq_of_lt small]
  simp [tile, quotient, remainder, live, active]

/-- The terminal operation on a row can include a prescribed floating-point
reduction order. It observes precisely the same indexed scalar inputs in
both executions, including the last incomplete group of rows. -/
theorem forward_eq (n block stride i : ℕ) (positive : 0 < block) (live : i < n)
    (width : ℕ → ℕ) (fits : ∀ r < n, width r ≤ stride)
    (score : ℕ → ℕ → S × I) (pad : ℕ → S) (finish : (r : ℕ) → (Fin (width r) → S) → V) :
    let origin := block * (i / block)
    let localRow := i % block
    finish i (fun j => tile origin (min block (n-origin)) stride width score pad
      (localRow * stride + j)) = finish i (fun j => (score i j).1) := by
  dsimp only
  have id := Nat.div_add_mod i block
  have tail : i % block < min block (n - block * (i / block)) :=
    lt_min (Nat.mod_lt i positive) (by omega)
  congr 1
  funext j
  have bounded : width (block * (i / block) + i % block) ≤ stride := by rw [id]; exact fits i live
  have active : (j : ℕ) < width (block * (i / block) + i % block) := by rw [id]; exact j.isLt
  rw [read_eq _ _ _ _ _ width score pad tail active bounded, id]

example : tile 8 2 7 (fun _ => 4) (fun i j => (10*i+j, i+j)) (fun _ => 999) (1*7+3) = 93 :=
  read_eq 8 2 7 1 3 _ _ _ (by decide) (by decide) (by decide)
example : tile 8 2 7 (fun _ => 4) (fun i j => (10*i+j, i+j)) (fun _ => 999) (1*7+5) = 999 := by decide

/-- The final tile has two active rows. Its second row uses scratch slots
7 and 10 for two widely separated live columns, in the specified order. -/
example :
    let result := fun (row : Fin 4 → ℕ) => row 3 - row 0
    result (fun j => tile 8 2 7 (fun _ => 4) (fun i k => (10*i+k,i+k))
      (fun _ => 999) (7 + j)) = result (fun j => 10*9+j) := by
  exact forward_eq 10 4 7 9 (by decide) (by decide) (fun _ => 4)
    (by intros; decide) (fun i k => (10*i+k,i+k)) (fun _ => 999)
    (fun _ row => row 3 - row 0)

end Transformer.Phasor.TripleTransport.ScoreTiles
