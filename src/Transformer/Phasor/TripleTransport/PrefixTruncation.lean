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

end Transformer.Phasor.TripleTransport.PrefixTruncation
