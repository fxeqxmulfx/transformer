module

public import Transformer.Phasor.TripleTransport.DenseInduction

/-! The Phasor dense-input contract applies to actual prompt truncation, as
well as a different continuation of the same length. A length specialization
may skip a read at short lengths exactly when the read already returns its
neutral contribution on every observed prefix at those lengths. -/
@[expose] public section

namespace Transformer.Phasor.TripleTransport.PrefixTruncation

variable {X V : Type*}

def readAt (read : List X → V) (xs : List X) (seen : ℕ) : V := read (xs.take seen)

/-- Every cut beyond the observed prefix leaves the result unchanged. -/
theorem truncate (read : List X → V) (xs : List X) (seen cut : ℕ) (within : seen ≤ cut) :
    readAt read (xs.take cut) seen = readAt read xs seen := by
  unfold readAt
  rw [List.take_take, Nat.min_eq_left within]

/-- A whole-prompt fast path is safe across all truncations of a fixed
sequence only if the short branch returns its actual prefix result. -/
theorem guard_iff (read : List X → V) (neutral : V) (xs : List X) (seen cutoff : ℕ)
    (short : seen ≤ cutoff) (extended : cutoff < xs.length) :
    (if (xs.take seen).length ≤ cutoff then neutral else readAt read (xs.take seen) seen) =
      (if xs.length ≤ cutoff then neutral else readAt read xs seen) ↔
    neutral = readAt read xs seen := by
  have h : (xs.take seen).length ≤ cutoff := by simp [List.length_take]; omega
  rw [ite_eq_left h, ite_eq_right (Nat.not_le_of_lt extended)]

example : readAt List.sum ([2,3,7,8].take 3) 2 = readAt List.sum ([2,3,7,8] : List ℕ) 2 :=
  truncate _ _ _ _ (by decide)

example :
    (if (([3,9,7] : List ℕ).take 1).length ≤ 2 then 0 else readAt List.sum ([3,9,7].take 1) 1) ≠
    (if ([3,9,7] : List ℕ).length ≤ 2 then 0 else readAt List.sum [3,9,7] 1) := by
  intro equal
  have bad := (guard_iff List.sum 0 ([3,9,7] : List ℕ) 1 2 (by decide) (by decide)).mp equal
  exact (by decide : ¬ (0:ℕ)=readAt List.sum [3,9,7] 1) bad

/-- The dense-induction fast path in the native Phasor runner is valid
at every observed position. A key, its completed successor and the later
query require at least three input positions. -/
theorem induction_shortcut (accept : X → X → Bool) (input : ℕ → X) (size t : ℕ)
    (observed : t < size) : (if size ≤ 2 then none else DenseInduction.donor accept input t) =
    DenseInduction.donor accept input t := by
  by_cases short : size ≤ 2
  · have empty : t-1 = 0 := by omega
    simp [short, DenseInduction.donor, empty, DenseInduction.findDonor]
  · simp [short]

example : (if 2 ≤ 2 then none else DenseInduction.donor (· == ·) (fun i => i % 2) 1) =
    DenseInduction.donor (· == ·) (fun i => i % 2) 1 :=
  induction_shortcut _ _ 2 1 (by decide)

/-- Phasor's directed associative store in `typed_lookup/paths.rs`. A write
requires three consecutive geometric addresses. The prefix ends at position
`t`, so a window beginning at `i` can enter only when `i + 2 ≤ t`. -/
def completed (addr : ℕ → Option X) (t : ℕ) : List (X × X × X) :=
  (List.range (t - 1)).filterMap fun i =>
    match addr i, addr (i+1), addr (i+2) with
    | some a, some r, some b => some (a,r,b)
    | _, _, _ => none

/-- Characterization of an empty write history in the native associative
read. It also covers long prefixes with gaps in the learned addresses. -/
theorem completed_eq_nil (addr : ℕ → Option X) (t : ℕ) :
    completed addr t = [] ↔
      ∀ i, i+2 ≤ t → addr i = none ∨ addr (i+1) = none ∨ addr (i+2) = none := by
  simp only [completed, List.filterMap_eq_nil_iff, List.mem_range]
  constructor
  · intro h i hi
    have miss := h i (by omega)
    cases a : addr i <;> cases b : addr (i+1) <;> cases c : addr (i+2) <;>
      simp_all
  · intro h i hi
    rcases h i (by omega) with a | b | c
    · rw [a]
    · cases a : addr i <;> simp [b]
    · cases a : addr i <;> cases b : addr (i+1) <;> simp [c]

/-- Every path vote starts with a stored directed write. With no complete
window, enumerating continuations changes neither the forward accumulator nor
its reverse sweep, for any learned weights and any number of repeated reads.
The two folds permit arbitrary update operations and accumulation order. -/
theorem empty_path_sweeps {W S G : Type*} (addr : ℕ → Option X) (t : ℕ)
    (paths : X × X × X → List W) (emit : S → W → S) (pull : W → G → G)
    (state : S) (adjoint : G)
    (gap : ∀ i, i+2 ≤ t → addr i = none ∨ addr (i+1) = none ∨ addr (i+2) = none) :
    let ws := (completed addr t).flatMap paths
    ws.foldl emit state = state ∧ ws.foldr pull adjoint = adjoint := by
  rw [(completed_eq_nil addr t).mpr gap]
  exact ⟨rfl, rfl⟩

example (paths : ℕ × ℕ × ℕ → List ℕ) :
    let ws := (completed (fun i => if i=1 ∨ i=4 then none else some i) 5).flatMap paths
    ws.foldl (·+·) 7 = 7 ∧ ws.foldr (·*·) 3 = 3 := by
  apply empty_path_sweeps
  intro i hi
  have bound : i ≤ 3 := by omega
  interval_cases i <;> decide

/-- The same test can run before the addresses have been computed whenever
the requested position cannot yet finish a triple. This holds for an early
request inside an arbitrarily long sequence. -/
theorem before_first_write (addr : ℕ → Option X) (t : ℕ) (early : t < 2) :
    completed addr t = [] := by
  apply (completed_eq_nil addr t).mpr
  intro i hi
  omega

example : completed (fun i => some (3*i+7)) 1 = [] := before_first_write _ _ (by decide)
example : completed (fun i => some (3*i+7)) 2 = [(7,10,13)] := by decide

end Transformer.Phasor.TripleTransport.PrefixTruncation
