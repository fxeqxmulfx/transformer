import Transformer.GPTMini.Sparsemax.MemoryExampleGrams
import Mathlib.Logic.Equiv.Basic

/-!
# Genuine permutation atoms for a local learned-memory Gram

Derived architecture before arXiv:1602.02068v2, Eq. (1), with the common
value operation at `73f8a0b`. An atom uses ordinary basis queries and a
permuted basis of keys. Its Gram is PSD, its memory scores are a permutation
matrix, and both same-family blocks are identity. These are explicit
embedding constructions, not assumed factorizations of desired attention.

The local family mixes identity with swaps of adjacent dictionary indices.
Each swap moves a query by at most one slot. Subsequent modules prove that
nonnegative learned mixing weights with a self-weight budget preserve a
convex structural inverse domain. This restricts the unrestricted Gram
architecture to a fixed path of possible connections, while allowing the
actual support to change. No claim about the meaning of dictionary order
or output-only identifiability follows from this structural construction.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open scoped BigOperators

/-- Basis queries and permuted basis keys supply genuine finite embedding coordinates.
Source: the derived permutation construction before arXiv:1602.02068v2, Eq. (1). -/
def permutationMemoryFeatures {N : ℕ} (perm : Equiv.Perm (Fin (N + 1)))
    (d : Fin (N + 1)) : Sum (Fin (N + 1)) (Fin (N + 1)) → ℝ
  | Sum.inl i => if d = i then 1 else 0
  | Sum.inr j => if d = perm.symm j then 1 else 0

/-- The full query/key Gram of the explicit permutation feature table.
Source: the derived embedding lift before arXiv:1602.02068v2, Eq. (1). -/
def permutationMemoryGram {N : ℕ} (perm : Equiv.Perm (Fin (N + 1))) :
    EmbeddingGram (N + 1) := featureGram (permutationMemoryFeatures perm)

/-- Atom entries are computed from feature products, including both embedding families.
Source: the explicit derived feature construction for arXiv:1602.02068v2, Eq. (1). -/
theorem permutationMemoryGram_entries {N : ℕ} (perm : Equiv.Perm (Fin (N + 1)))
    (x y : Sum (Fin (N + 1)) (Fin (N + 1))) :
    permutationMemoryGram perm x y =
      if Sum.elim id perm.symm x = Sum.elim id perm.symm y then 1 else 0 := by
  unfold permutationMemoryGram
  rw [featureGram_apply]
  rcases x with i | i <;> rcases y with j | j <;>
    change (∑ d : Fin (N + 1), (if d = _ then (1 : ℝ) else 0) *
      (if d = _ then 1 else 0)) = if _ = _ then 1 else 0
  all_goals
    simp only [ite_mul, one_mul, zero_mul]
    exact Fintype.sum_ite_eq' _ _

/-- Genuine scalar products give the indicated permutation scores.
Source: the cross block of the derived Gram preceding arXiv:1602.02068v2, Eq. (1). -/
theorem permutationMemoryGram_scores {N : ℕ} (perm : Equiv.Perm (Fin (N + 1))) :
    memoryGramScores (permutationMemoryGram perm) =
      Matrix.of (fun i j => if perm i = j then 1 else 0) := by
  ext i j
  change permutationMemoryGram perm (Sum.inl i) (Sum.inr j) =
    if perm i = j then 1 else 0
  simp only [permutationMemoryGram_entries, Sum.elim_inl, Sum.elim_inr, id_eq,
    perm.eq_symm_apply]

/-- Query and key same-family blocks are both identity for every permutation atom.
Source: the orthonormal feature columns in the derived arXiv:1602.02068v2, Eq. (1) lift. -/
theorem permutationMemoryGram_sameSide {N : ℕ} (perm : Equiv.Perm (Fin (N + 1)))
    (i j : Fin (N + 1)) :
    permutationMemoryGram perm (Sum.inl i) (Sum.inl j) = (if i = j then 1 else 0) ∧
    permutationMemoryGram perm (Sum.inr i) (Sum.inr j) = (if i = j then 1 else 0) := by
  constructor
  · by_cases h : i = j <;>
      simp only [permutationMemoryGram_entries, Sum.elim_inl, id_eq, h, ite_true, ite_false]
  · rw [permutationMemoryGram_entries, Sum.elim_inr, Sum.elim_inr]
    have he : perm.symm i = perm.symm j ↔ i = j :=
      ⟨fun h => perm.symm.injective h, fun h => congrArg perm.symm h⟩
    simp only [he]

/-- Every explicit query and key column in an atom has squared norm one.
Source: the derived finite embedding table for arXiv:1602.02068v2, Eq. (1). -/
theorem permutationMemoryGram_diagonal {N : ℕ} (perm : Equiv.Perm (Fin (N + 1)))
    (x : Sum (Fin (N + 1)) (Fin (N + 1))) : permutationMemoryGram perm x x = 1 := by
  rw [permutationMemoryGram_entries]
  exact ite_eq_left rfl

/-- All permutation atoms are bounded PSD memories with normalized scores and floor zero.
Source: actual feature products before arXiv:1602.02068v2, Eq. (1).
Strict invertibility will be obtained by a learned identity-weight budget. -/
theorem permutationMemoryGram_mem {N : ℕ} (perm : Equiv.Perm (Fin (N + 1))) :
    permutationMemoryGram perm ∈ memoryGramDomain N 1 0 := by
  refine ⟨⟨featureGram_posSemidef _, ?_⟩, ?_, ?_⟩
  · intro x y
    rw [permutationMemoryGram_entries]
    split_ifs <;> norm_num
  · intro i
    rw [permutationMemoryGram_scores]
    have he : (Matrix.of (fun i j => if perm i = j then (1 : ℝ) else 0)) i =
        basis (perm i) := by
      funext j
      by_cases h : perm i = j <;> simp [Matrix.of_apply, basis, h, eq_comm]
    rw [he]
    exact basis_mem_simplex Set.univ (perm i) (Set.mem_univ _)
  · intro i
    rw [permutationMemoryGram_scores]
    change 0 ≤ if perm i = i then (1 : ℝ) else 0
    split_ifs <;> norm_num

/-- Identity and one adjacent swap for each edge of the dictionary path.
Source: the local architectural restriction before arXiv:1602.02068v2, Eq. (1). -/
def localMemoryPermutation {N : ℕ} : Option (Fin N) → Equiv.Perm (Fin (N + 1))
  | none => Equiv.refl _
  | some e => Equiv.swap e.castSucc e.succ

/-- A finite path dictionary of genuine Gram atoms.
Source: the explicit permutation embeddings preceding arXiv:1602.02068v2, Eq. (1). -/
def localMemoryAtom {N : ℕ} (e : Option (Fin N)) : EmbeddingGram (N + 1) :=
  permutationMemoryGram (localMemoryPermutation e)

/-- Every local atom is feasible before imposing a strict self-weight budget.
Source: the derived permutation Gram domain for arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryAtom_mem {N : ℕ} (e : Option (Fin N)) :
    localMemoryAtom e ∈ memoryGramDomain N 1 0 :=
  permutationMemoryGram_mem _

/-- Same-family blocks do not change across the path atoms.
Source: orthonormality of the derived atom embeddings for arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryAtom_sameSide {N : ℕ} (e : Option (Fin N)) (i j : Fin (N + 1)) :
    localMemoryAtom e (Sum.inl i) (Sum.inl j) = (if i = j then 1 else 0) ∧
    localMemoryAtom e (Sum.inr i) (Sum.inr j) = (if i = j then 1 else 0) :=
  permutationMemoryGram_sameSide _ _ _

/-- Each path atom leaves a slot fixed or moves it exactly one step.
Source: the adjacent-swap restriction before arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryPermutation_moves {N : ℕ} (e : Option (Fin N)) (i : Fin (N + 1)) :
    localMemoryPermutation e i = i ∨
      i.val + 1 = (localMemoryPermutation e i).val ∨
      (localMemoryPermutation e i).val + 1 = i.val := by
  rcases e with _ | e
  · exact Or.inl rfl
  · by_cases hl : i = e.castSucc
    · right
      left
      rw [hl]
      simp only [localMemoryPermutation, Equiv.swap_apply_left, Fin.val_castSucc, Fin.val_succ]
    · by_cases hr : i = e.succ
      · right
        right
        rw [hr]
        simp only [localMemoryPermutation, Equiv.swap_apply_right, Fin.val_castSucc, Fin.val_succ]
      · left
        exact Equiv.swap_apply_of_ne_of_ne hl hr

end Transformer.GPTMini.Sparsemax
