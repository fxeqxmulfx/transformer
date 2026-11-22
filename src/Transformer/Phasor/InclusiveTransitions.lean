module

public import Transformer.Phasor.SaturatedTransitions
public import Transformer.Grokking.FlatSpectrum.BlockCut.Basic

/-!
# Including the current edge in a causal transition read

Source: the inclusive transition-table trial in `experiments/phasor_value_transport/README.md`
and the binary block languages of Yang, Cadilhac and Chiang, §2.4, Equation
`eq:altplus`. The same counter can process any token stream. On a binary
word preceded by a distinct start token, its inclusive value counts all
blocks, including the block of the current letter. Below the absorbing
bucket, adding two categorical scores suffices to recognize the language.
-/

@[expose] public section

namespace Transformer.Phasor

open Transformer.CRASP Transformer.Grokking.FlatSpectrum.BlockCut

/-- Encode a stream with a start symbol distinct from every letter. -/
def markedLetters (xs : ℕ → Bool) : ℕ → Option Bool
  | 0 => none
  | n+1 => some (xs n)

/-- The first `n` letters in input order, excluding the start symbol. -/
def letterPrefix (xs : ℕ → Bool) : ℕ → List Bool
  | 0 => []
  | n+1 => letterPrefix xs n ++ [xs n]

/-- At the current letter the number of token changes from the start
symbol is exactly the number of maximal constant blocks in the prefix. -/
theorem inclusiveCount_eq_blocks (xs : ℕ → Bool) (n : ℕ) :
    transitionCount (markedLetters xs) n = blocks (letterPrefix xs n) := by
  induction n with
  | zero => rfl
  | succ n ih =>
    cases n with
    | zero => simp [transitionCount, markedLetters, letterPrefix]
    | succ m =>
      have join := blocks_append (letterPrefix xs (m+1)) (v := [xs (m+1)]) (by simp)
      have last : (letterPrefix xs (m+1)).getLast? = some (xs m) :=
        List.getLast?_concat
      rw [last, List.head?_cons, blocks_singleton] at join
      rw [transitionCount, ih]
      change blocks (letterPrefix xs (m+1)) + (if some (xs m) = some (xs (m+1)) then 0 else 1)
        = blocks (letterPrefix xs (m+1) ++ [xs (m+1)])
      split <;> rename_i h <;> simp only [h, ite_true, ite_false] at join <;> omega

/-- The endpoint of a nonempty binary word equals its first letter at an
odd number of blocks and is the other letter at an even number. -/
theorem lastLetter_of_blockParity (first : Bool) (tail : List Bool) :
    (first :: tail).getLast? =
      some (if blocks (first :: tail) % 2 = 1 then first else !first) := by
  induction tail generalizing first with
  | nil => simp [blocks_singleton]
  | cons b tail ih =>
    rw [List.getLast?_cons_cons, ih b, blocks_cons first (b :: tail), List.head?_cons]
    generalize blocks (b :: tail) = n
    cases first <;> cases b <;>
      simp only [Option.some.injEq, reduceCtorEq, ite_false, ite_true, Nat.zero_add,
        Bool.not_false, Bool.not_true]
    all_goals split <;> split <;> first | rfl | omega

/-- A target below the absorbing bucket is recognized by matching its
capped count and final letter, at every length and every number of changes. -/
theorem mem_blocks_iff_inclusive_read (p k : ℕ) [NeZero p] (fits : k+1 < p-1)
    (start : Bool) (w : List Bool) :
    w ∈ altPlus start (k+1) ↔
      saturatedChanges p (blocks w) = saturatedChanges p (k+1) ∧
      w.getLast? = some (if (k+1) % 2 = 1 then start else !start) := by
  rw [mem_altPlus_succ_iff, saturatedChanges_eq_iff]
  cases w with
  | nil => simp
  | cons a t =>
    rw [List.head?_cons, lastLetter_of_blockParity]
    constructor
    · rintro ⟨ha, hn⟩
      obtain rfl : a = start := by simpa using ha
      simp [hn]
    · rintro ⟨h, last⟩
      have hn : blocks (a :: t) = k+1 := by omega
      rw [hn] at last
      have ha : a = start := by
        split at last <;> cases a <;> cases start <;> simp_all
      exact ⟨congrArg some ha, hn⟩

/-- One additive scalar score per count and one per current token suffice
for every binary block language with its target below saturation. Neither
table depends on length or on the number of previously seen prefixes. -/
theorem exists_additive_block_read (p k : ℕ) [NeZero p] (fits : k+1 < p-1) (start : Bool) :
    ∃ countWeight : Fin p → ℤ, ∃ tokenWeight : Option Bool → ℤ, ∀ w : List Bool,
      w ∈ altPlus start (k+1) ↔
        1 < countWeight (saturatedChanges p (blocks w)) + tokenWeight w.getLast? := by
  refine ⟨fun b => if b = saturatedChanges p (k+1) then 1 else 0,
    fun e => if e = some (if (k+1) % 2 = 1 then start else !start) then 1 else 0, ?_⟩
  intro w
  rw [mem_blocks_iff_inclusive_read p k fits]
  by_cases hc : saturatedChanges p (blocks w) = saturatedChanges p (k+1) <;>
    by_cases ht : w.getLast? = some (if (k+1) % 2 = 1 then start else !start) <;>
    simp [hc, ht]

example : transitionCount (markedLetters (fun i => decide (i % 2 = 1))) 5 = 5 := by decide

example : [false, false, true, false] ∈ altPlus false 3 :=
  (mem_blocks_iff_inclusive_read 5 2 (by decide) _ _).2 (by decide)

example : ∃ c : Fin 36 → ℤ, ∃ t : Option Bool → ℤ, ∀ w : List Bool,
    w ∈ altPlus true 6 ↔ 1 < c (saturatedChanges 36 (blocks w)) + t w.getLast? :=
  exists_additive_block_read 36 5 (by decide) true

/-- The second-change trial in the experiment README. Its residual starts
at the second observed change and then reads the inclusive capped count. -/
def secondChangeRead {E : Type*} [Zero E] {p : ℕ} [NeZero p] (table : Fin p → E) (n : ℕ) : E :=
  if 2 ≤ n then table (saturatedChanges p n) else 0

/-- A prefix with one possible change of identity has exactly one edge
when it spans that boundary and the two identities differ. The alphabet
and both run lengths are arbitrary. -/
theorem transitionCount_one_boundary {α : Type*} [DecidableEq α] (a b : α) (cut n : ℕ) :
    transitionCount (fun j => if j < cut then a else b) n =
      if a = b ∨ cut = 0 ∨ n < cut then 0 else 1 := by
  induction n with
  | zero =>
    by_cases h : cut = 0
    · simp [transitionCount, h]
    · simp [transitionCount, Nat.pos_of_ne_zero h]
  | succ n ih =>
    rw [transitionCount, ih]
    by_cases eq : a = b
    · subst b; simp
    · by_cases empty : cut = 0
      · subst cut; simp
      · by_cases next : n+1 < cut
        · have prev : n < cut := by omega
          simp [eq, empty, next, prev]
        · by_cases prev : n < cut <;> simp [eq, empty, next, prev]

/-- The second-change trial in the experiment README. Single-boundary
streams of any length leave every coordinate of the backbone residual
unchanged at every position, for arbitrary learned table values. -/
theorem secondChangeRead_one_boundary {α E : Type*} [DecidableEq α] [Zero E] {p : ℕ} [NeZero p]
    (table : Fin p → E) (a b : α) (cut n : ℕ) :
    secondChangeRead table (transitionCount (fun j => if j < cut then a else b) n) = 0 := by
  rw [transitionCount_one_boundary]
  split <;> simp [secondChangeRead]

/-- The second-change trial in the experiment README. For every target from
two blocks up to the last distinguishable count, a learned count score and
a current-token score separate the binary language at every word length. -/
theorem exists_secondChange_block_read (p k : ℕ) [NeZero p] (fits : k+2 < p-1) (start : Bool) :
    ∃ countWeight : Fin p → ℤ, ∃ tokenWeight : Option Bool → ℤ, ∀ w : List Bool,
      w ∈ altPlus start (k+2) ↔ 1 < secondChangeRead countWeight (blocks w) + tokenWeight w.getLast? := by
  let target := saturatedChanges p (k+2)
  let last := some (if (k+2) % 2 = 1 then start else !start)
  refine ⟨fun b => if b = target then 1 else 0, fun e => if e = last then 1 else 0, ?_⟩
  intro w
  rw [mem_blocks_iff_inclusive_read p (k+1) fits]
  change (saturatedChanges p (blocks w) = target ∧ w.getLast? = last) ↔ _
  by_cases hit : saturatedChanges p (blocks w) = target
  · have high : 2 ≤ blocks w := by
      rcases (saturatedChanges_eq_iff p _ _).mp hit with h | h <;> omega
    by_cases endHit : w.getLast? = last <;> simp [secondChangeRead, hit, high, endHit]
  · by_cases endHit : w.getLast? = last <;> by_cases high : 2 ≤ blocks w <;>
      simp [secondChangeRead, hit, high, endHit]

/-- The second-change trial in the experiment README. Its batch adjoint
uses exactly the rows with two or more changes. Every coordinate receives
the sum of upstream contributions in its own capped bucket. -/
theorem secondChangeRead_adjoint {ι : Type*} {p : ℕ} [NeZero p] (rows : Finset ι)
    (count : ι → ℕ) (upstream : ι → ℝ) (delta : Fin p → ℝ) :
    (∑ i ∈ rows, upstream i * secondChangeRead delta (count i)) =
      ∑ b, (∑ i ∈ rows.filter (fun j => 2 ≤ count j) with saturatedChanges p (count i) = b,
        upstream i) * delta b := by
  let active := rows.filter (fun i => 2 ≤ count i)
  let key := fun i => saturatedChanges p (count i)
  calc
    _ = ∑ i ∈ active, upstream i * delta (key i) := by
      rw [Finset.sum_filter]
      apply Finset.sum_congr rfl
      intro i _
      by_cases h : 2 ≤ count i <;> simp [secondChangeRead, key, h]
    _ = ∑ b, ∑ i ∈ active with key i = b, upstream i * delta (key i) :=
      (Finset.sum_fiberwise active key _).symm
    _ = _ := by
      apply Finset.sum_congr rfl
      intro b _
      rw [Finset.sum_mul]
      apply Finset.sum_congr rfl
      intro i hi
      rw [(Finset.mem_filter.mp hi).2]

example : ∃ c : Fin 12 → ℤ, ∃ t : Option Bool → ℤ, ∀ w : List Bool,
    w ∈ altPlus false 4 ↔ 1 < secondChangeRead c (blocks w) + t w.getLast? :=
  exists_secondChange_block_read 12 2 (by decide) false

end Transformer.Phasor
