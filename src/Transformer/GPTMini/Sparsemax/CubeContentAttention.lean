import Transformer.GPTMini.Sparsemax.CubeContentQK

/-!
# Genuine variational sparsemax on generated content neighbors

New attention after arXiv:1602.02068v2, Eq. (1) and §2.2,
Proposition 1. Original dot-product scores are retained on the query
and its one-bit neighbors; all other scores are minus one. Feasible
content routing is nonnegative and sums to one. Consequently clipping
these actual scores at threshold zero recovers the generated routing
matrix, including when learned neighbor supports acquire exact zeros.

The existing finite causal sparsemax is evaluated at its last slot, so
all virtual memory destinations are eligible. Text causality is enforced
by the observed content encoder, not by treating virtual destinations
as future input-token positions. Possible support has at most r+1 slots
for r content bits, independent of the 2^r virtual dictionary size.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.GPTMini.Convex
open scoped BigOperators

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- Query and generated flip destinations form the structural content mask.
Source: the new local-memory restriction before sparsemax Eq. (1). -/
def cubeContentNeighbours (x : CubeContentState ι) : Finset (CubeContentState ι) :=
  insert x (Finset.univ.image (fun i => cubeContentFlip i x))

/-- All possible local destinations are generated in linear size, with no duplicate slots.
Source: the structural sparse mask before sparsemax Eq. (1). -/
theorem cubeContentNeighbours_card (x : CubeContentState ι) :
    (cubeContentNeighbours x).card = Fintype.card ι + 1 := by
  have hn : x ∉ Finset.univ.image (fun i => cubeContentFlip i x) := by
    simp only [Finset.mem_image, Finset.mem_univ, true_and, not_exists]
    exact fun i => cubeContentFlip_ne i x
  unfold cubeContentNeighbours
  rw [Finset.card_insert_of_notMem hn, Finset.card_image_of_injective _ (cubeContentFlip_index_injective x)]
  rfl

/-- Original dot products realize generated routing on every possible local destination.
Source: the physical content Q/K restriction before sparsemax Eq. (1). -/
theorem cubeContentScore_local (t : ι → ℝ) (x y : CubeContentState ι)
    (hy : y ∈ cubeContentNeighbours x) : cubeContentScore t x y = cubeContentRouting t x y := by
  simp only [cubeContentNeighbours, Finset.mem_insert, Finset.mem_image, Finset.mem_univ, true_and] at hy
  rcases hy with h | ⟨i, h⟩
  · subst y
    rw [cubeContentScore_self, cubeContentRouting_self]
  · subst y
    rw [cubeContentScore_flip, cubeContentRouting_flip]

/-- A positive one-bit neighbor inhabits the actual local-dot-product premise. -/
example : cubeContentScore (fun _ : Fin 1 => (1 / 8 : ℝ)) (fun _ => false)
    (cubeContentFlip 0 (fun _ => false)) =
    cubeContentRouting (fun _ : Fin 1 => (1 / 8 : ℝ)) (fun _ => false) (cubeContentFlip 0 (fun _ => false)) := by
  apply cubeContentScore_local
  simp [cubeContentNeighbours]

/-- Generated routing vanishes outside the actual structural score mask.
Source: the sparse content construction before sparsemax Eq. (1). -/
theorem cubeContentRouting_zero_outside (t : ι → ℝ) (x y : CubeContentState ι)
    (hy : y ∉ cubeContentNeighbours x) : cubeContentRouting t x y = 0 := by
  apply cubeContentRouting_zero
  · intro h
    exact hy (by simp [cubeContentNeighbours, ← h])
  · intro i h
    exact hy (by simp [cubeContentNeighbours, ← h])

/-- Two-bit changes are actual excluded slots, including at positive routing weights. -/
example : cubeContentRouting (fun _ : Fin 2 => (1 / 8 : ℝ))
    (fun _ => false) (fun _ => true) = 0 := by
  apply cubeContentRouting_zero_outside
  simp only [cubeContentNeighbours, Finset.mem_insert, Finset.mem_image, Finset.mem_univ, true_and, not_or, not_exists]
  constructor
  · intro h
    have hb := congrFun h 0
    contradiction
  · intro i h
    fin_cases i
    · have hb := congrFun h 1
      norm_num [cubeContentFlip] at hb
    · have hb := congrFun h 0
      norm_num [cubeContentFlip] at hb

/-- Actual finite physical scores retain the local dot product and mask all other virtual states.
Source: structural masking before arXiv:1602.02068v2, Eq. (1). -/
def cubeContentMaskedScores (t : ι → ℝ) (x : CubeContentState ι)
    (j : Fin (cubeContentMemoryN ι + 1)) : ℝ :=
  if cubeContentIndex.symm j ∈ cubeContentNeighbours x then
    cubeContentScore t x (cubeContentIndex.symm j) else -1

/-- Original variational sparsemax applied to the actual generated Q/K row.
Source: arXiv:1602.02068v2, Eq. (1), with the explicit local content mask. -/
def cubeContentAttention (t : ι → ℝ) (x y : CubeContentState ι) : ℝ :=
  sparseWeights (cubeContentMaskedScores t x) (Fin.last (cubeContentMemoryN ι)) (cubeContentIndex y)

/-- The genuine finite threshold clips the physical scores to generated probabilities.
Source: sparsemax §2.2, Proposition 1, verified at threshold zero. -/
theorem cubeContentThreshold_eq (floor : ℝ) (t : ι → ℝ) (hf : 0 ≤ floor)
    (ht : t ∈ cubeContentRouteDomain ι floor) (x : CubeContentState ι) :
    thresholdWeights (cubeContentMaskedScores t x) (Fin.last (cubeContentMemoryN ι)) 0 =
      fun j => cubeContentRouting t x (cubeContentIndex.symm j) := by
  funext j
  rw [thresholdWeights, ite_eq_left (Fin.le_last j), sub_zero]
  by_cases hj : cubeContentIndex.symm j ∈ cubeContentNeighbours x
  · rw [cubeContentMaskedScores, ite_eq_left hj, cubeContentScore_local t x _ hj]
    exact max_eq_left (cubeContentRouting_nonneg floor t hf ht x _)
  · rw [cubeContentMaskedScores, ite_eq_right hj, cubeContentRouting_zero_outside t x _ hj]
    norm_num

/-- Actual physical two-bit scores have the asserted nonidentity variational threshold. -/
example : thresholdWeights (cubeContentMaskedScores (fun _ : Fin 2 => (1 / 8 : ℝ)) (fun _ => false))
    (Fin.last (cubeContentMemoryN (Fin 2))) 0 =
    (fun j => cubeContentRouting (fun _ : Fin 2 => (1 / 8 : ℝ)) (fun _ => false) (cubeContentIndex.symm j)) :=
  cubeContentThreshold_eq (3 / 4) _ (by norm_num)
    (by constructor; intro i; norm_num; norm_num [Fin.sum_univ_two]) _

/-- Actual sparsemax equals content diffusion, with no fixed positive support premise.
Source: original Eq. (1) and §2.2, Proposition 1, in the new physical memory chart. -/
theorem cubeContentAttention_eq (floor : ℝ) (t : ι → ℝ) (hf : 0 ≤ floor)
    (ht : t ∈ cubeContentRouteDomain ι floor) : cubeContentAttention t = cubeContentRouting t := by
  ext x y
  have he := cubeContentThreshold_eq floor t hf ht x
  have hs : (∑ j, thresholdWeights (cubeContentMaskedScores t x) (Fin.last (cubeContentMemoryN ι)) 0 j) = 1 := by
    rw [he, cubeContentIndex.symm.sum_comp]
    exact cubeContentRouting_rowSum t x
  have hp := thresholdWeights_eq_sparseWeights _ _ 0 hs
  rw [he] at hp
  have hy := congrFun hp (cubeContentIndex y)
  simpa only [cubeContentAttention, Equiv.symm_apply_apply] using hy.symm

/-- A genuine sparse row with two positive flips inhabits every threshold and normalization premise. -/
example : cubeContentAttention (fun _ : Fin 2 => (1 / 8 : ℝ)) =
    cubeContentRouting (fun _ : Fin 2 => (1 / 8 : ℝ)) :=
  cubeContentAttention_eq (3 / 4) _ (by norm_num)
    (by constructor; intro i; norm_num; norm_num [Fin.sum_univ_two])

/-- True sparsemax support has at most one plus the bit count actual destinations.
Source: the proved structural mask and Eq. (1), rather than an assumed sparse output. -/
theorem cubeContentAttention_support_card (floor : ℝ) (t : ι → ℝ) (hf : 0 ≤ floor)
    (ht : t ∈ cubeContentRouteDomain ι floor) (x : CubeContentState ι) :
    (Finset.univ.filter (fun y => cubeContentAttention t x y ≠ 0)).card ≤ Fintype.card ι + 1 := by
  classical
  rw [← cubeContentNeighbours_card x]
  apply Finset.card_le_card
  intro y hy
  have ha := (Finset.mem_filter.mp hy).2
  by_contra hn
  rw [cubeContentAttention_eq floor t hf ht] at ha
  exact ha (cubeContentRouting_zero_outside t x y hn)

/-- Two positive content flips still use at most three of the four virtual memory states. -/
example : (Finset.univ.filter (fun y : CubeContentState (Fin 2) =>
    cubeContentAttention (fun _ : Fin 2 => (1 / 8 : ℝ)) (fun _ => false) y ≠ 0)).card ≤ 3 := by
  have h := cubeContentAttention_support_card (3 / 4) (fun _ : Fin 2 => (1 / 8 : ℝ)) (by norm_num)
    (by constructor; intro i; norm_num; norm_num [Fin.sum_univ_two]) (fun _ => false)
  simpa only [Fintype.card_fin] using h

end Transformer.GPTMini.Sparsemax
