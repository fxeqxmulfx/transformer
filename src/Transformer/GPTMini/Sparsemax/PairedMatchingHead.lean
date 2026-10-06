import Transformer.GPTMini.Sparsemax.AtomicMatchingContinuity
import Mathlib.Logic.Equiv.Fin.Basic

/-!
# Causal neighboring-token keys with freely learned original values

New encoder repair for the genuine atomic model following arXiv:2211.11052v1,
§3.1 and Appendix A.4. A content-only head loses the association between a
written key and its next value under value permutations. Here a key has two
learned roles: the current token and its predecessor. Queries and independent
original values still use the current token. The first position repeats itself.

The lift below expresses the actual forward using existing variational causal
sparsemax, arXiv:1602.02068v2, Eq. (1). Its expanded pair vocabulary is a proof
representation: physical storage remains V-by-2H Q/K and V-by-D values, with
no V-squared table. All coordinates are free inside the same numerical box.
This preserves causality and continuity across attention support changes.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.GPTMini.Convex
open scoped BigOperators

/-- Causal predecessor, with the first token repeated at the left boundary.
Source: the new binding encoder after §3.1; there is no future-token lookup. -/
def pairedPrevious {T : ℕ} (j : Fin T) : Fin T :=
  ⟨j.val - 1, Nat.lt_of_le_of_lt (Nat.sub_le _ _) j.isLt⟩

/-- Every predecessor is visible whenever its token is visible.
Source: the boundary convention of the new neighboring-token encoder. -/
theorem pairedPrevious_le {T : ℕ} (j : Fin T) : pairedPrevious j ≤ j :=
  Nat.sub_le _ _

/-- Proof-level encoding of the previous/current pair; physical tables stay factorized.
Source: the new contextual input encoder following Appendix A.4. -/
def pairedHeadTokens {V T : ℕ} (tokens : Fin T → Fin V) : Fin T → Fin (V * V) :=
  fun j => finProdFinEquiv (tokens (pairedPrevious j), tokens j)

/-- Lift the original tables without adding independent pair parameters.
Source: §3.1's dot products with separate current and predecessor key roles. -/
def pairedHeadLift {V H D : ℕ} (h : MatchingHead V (2 * H) D) : MatchingHead (V * V) (2 * H) D :=
  (((fun v d => h.1.1 (finProdFinEquiv.symm v).2 d),
    (fun v d => if d.val < H then h.1.2 (finProdFinEquiv.symm v).2 d
      else h.1.2 (finProdFinEquiv.symm v).1 d)),
    fun v d => h.2 (finProdFinEquiv.symm v).2 d)

/-- Actual scores of the factorized contextual head, with all learned coordinates retained.
Source: the new input encoder before the genuine sparsemax Eq. (1). -/
def pairedHeadScores {V H D T : ℕ} (h : MatchingHead V (2 * H) D)
    (tokens : Fin T → Fin V) (row j : Fin T) : ℝ :=
  matchingHeadScores (pairedHeadLift h) (pairedHeadTokens tokens) row j

/-- Actual causal sparsemax times independent original current-token values.
Source: sparsemax Eq. (1) inside the repaired §3.1 head. -/
def pairedHeadOutput {V H D T : ℕ} (h : MatchingHead V (2 * H) D)
    (tokens : Fin T → Fin V) (row : Fin T) (channel : Fin D) : ℝ :=
  matchingHeadOutput (pairedHeadLift h) (pairedHeadTokens tokens) row channel

/-- Scores use the current key in the first half and its predecessor in the second.
Source: explicit physical formula of the new binding encoder after §3.1. -/
theorem pairedHeadScores_formula {V H D T : ℕ} (h : MatchingHead V (2 * H) D)
    (tokens : Fin T → Fin V) (row j : Fin T) :
    pairedHeadScores h tokens row j = ∑ d, h.1.1 (tokens row) d *
      (if d.val < H then h.1.2 (tokens j) d else h.1.2 (tokens (pairedPrevious j)) d) := by
  simp only [pairedHeadScores, matchingHeadScores, pairedHeadLift, pairedHeadTokens,
    Equiv.symm_apply_apply]

/-- The lift reads the original values, with no contextual value table or inverse decoder.
Source: genuine attention/value multiplication, sparsemax Eq. (1) after §3.1. -/
theorem pairedHeadOutput_formula {V H D T : ℕ} (h : MatchingHead V (2 * H) D)
    (tokens : Fin T → Fin V) (row : Fin T) (channel : Fin D) :
    pairedHeadOutput h tokens row channel =
      ∑ j, sparseWeights (pairedHeadScores h tokens row) row j * h.2 (tokens j) channel := by
  change (∑ j, sparseWeights (pairedHeadScores h tokens row) row j *
    h.2 (finProdFinEquiv.symm (pairedHeadTokens tokens j)).2 channel) = _
  simp only [pairedHeadTokens, Equiv.symm_apply_apply]

/-- Every bounded physical contextual head lifts to a bounded genuine sparsemax head.
Source: the coordinate box after Appendix A.4, unchanged by factorized lookups. -/
theorem pairedHeadLift_mem {V H D : ℕ} (cap : ℝ) (h : MatchingHead V (2 * H) D)
    (hh : h ∈ matchingHeadBox V (2 * H) D cap) :
    pairedHeadLift h ∈ matchingHeadBox (V * V) (2 * H) D cap := by
  refine ⟨?_, ?_, ?_⟩
  · intro v d
    exact hh.1 (finProdFinEquiv.symm v).2 d
  · intro v d
    change -cap ≤ (if d.val < H then _ else _) ∧ (if d.val < H then _ else _) ≤ cap
    split_ifs
    · exact hh.2.1 (finProdFinEquiv.symm v).2 d
    · exact hh.2.1 (finProdFinEquiv.symm v).1 d
  · intro v d
    exact hh.2.2 (finProdFinEquiv.symm v).2 d

/-- Nonempty original-token tables satisfy the lift's bound premise. -/
example : pairedHeadLift (H := 1) (0 : MatchingHead 5 2 1) ∈ matchingHeadBox 25 2 1 1 :=
  pairedHeadLift_mem (H := 1) 1 (0 : MatchingHead 5 2 1)
    (matchingHeadBox_zero_mem 5 2 1 1 (by norm_num))

/-- Contextual encoding remains causal even when future keys and values change.
Source: causal sparsemax Eq. (1) and the predecessor rule after §3.1. -/
theorem pairedHeadOutput_causal {V H D T : ℕ} (h : MatchingHead V (2 * H) D)
    (tokens other : Fin T → Fin V) (row : Fin T) (channel : Fin D)
    (ht : ∀ j, j ≤ row → tokens j = other j) :
    pairedHeadOutput h tokens row channel = pairedHeadOutput h other row channel := by
  apply matchingHeadOutput_causal
  intro j hj
  have hp : pairedPrevious j ≤ row := by
    have he := pairedPrevious_le j
    omega
  simp only [pairedHeadTokens, ht j hj, ht (pairedPrevious j) hp]

/-- Distinct future continuations inhabit the contextual causality premise. -/
example (h : MatchingHead 2 2 1) : pairedHeadOutput (H := 1) h (fun _ : Fin 2 => 0) 0 0 =
    pairedHeadOutput (H := 1) h (fun j : Fin 2 => if j = 0 then 0 else 1) 0 0 := by
  apply pairedHeadOutput_causal
  intro j hj
  have he : j = 0 := by omega
  subst j
  rfl

/-- Factorized role lookup is continuous in all original independent tables.
Source: the learned-table lift of the new §3.1 binding encoder. -/
theorem pairedHeadLift_continuous (V H D : ℕ) :
    Continuous (pairedHeadLift (V := V) (H := H) (D := D)) := by
  unfold pairedHeadLift
  apply Continuous.prodMk
  · apply Continuous.prodMk
    · fun_prop
    · apply continuous_pi
      intro v
      apply continuous_pi
      intro d
      by_cases hd : d.val < H <;> simp only [hd, ite_true, ite_false] <;> fun_prop
  · fun_prop

/-- True contextual outputs are continuous without fixing attention support or key roles.
Source: sparsemax Eq. (1) with the new factorized contextual lift. -/
theorem pairedHeadOutput_continuous {V H D T : ℕ} (tokens : Fin T → Fin V)
    (row : Fin T) (channel : Fin D) :
    Continuous (fun h : MatchingHead V (2 * H) D => pairedHeadOutput h tokens row channel) :=
  (matchingHeadOutput_continuous (pairedHeadTokens tokens) row channel).comp
    (pairedHeadLift_continuous V H D)

/-- Contextual heads retain the universal output box needed by numerical certificates.
Source: original-value bounds and causal simplex weights, sparsemax Eq. (1). -/
theorem pairedHeadOutput_bounds {V H D T : ℕ} (cap : ℝ) (h : MatchingHead V (2 * H) D)
    (hh : h ∈ matchingHeadBox V (2 * H) D cap) (tokens : Fin T → Fin V)
    (row : Fin T) (channel : Fin D) :
    -cap ≤ pairedHeadOutput h tokens row channel ∧ pairedHeadOutput h tokens row channel ≤ cap :=
  matchingHeadOutput_bounds cap _ (pairedHeadLift_mem cap h hh) _ _ _

/-- A zero free head inhabits the contextual output-bound premise. -/
example : -1 ≤ pairedHeadOutput (H := 1) (0 : MatchingHead 2 2 1) id 1 0 ∧
    pairedHeadOutput (H := 1) (0 : MatchingHead 2 2 1) id 1 0 ≤ 1 :=
  pairedHeadOutput_bounds (H := 1) _ _ (matchingHeadBox_zero_mem _ _ _ _ (by norm_num)) _ _ _

end Transformer.GPTMini.Sparsemax
