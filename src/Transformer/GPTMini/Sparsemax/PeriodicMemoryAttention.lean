import Transformer.GPTMini.Sparsemax.PeriodicMemoryQueries
import Transformer.GPTMini.Sparsemax.ClosedForm

/-!
# Actual sparsemax attention from bounded width-three Q/K

New restricted architecture for arXiv:1602.02068v2, Eq. (1). QK products
are kept on the existing possible local destinations, and other scores
receive -1. On the feasible path domain the clipped threshold is exactly
zero. The original variational sparsemax therefore gives the desired path
matrix, including zeros and changes of support.

The real Q/K Gram has width three and bounded norms, rather than the old
identity same-family blocks. The old affine lift is used only to reuse
the proved inverse guarantee; it is not the physical embedding Gram.
The fixed mask is essential to this architecture and is stated explicitly.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.GPTMini.Convex
open scoped BigOperators

/-- A structural local mask on genuine learned three-coordinate dot products.
Source: the new bounded-width restriction before arXiv:1602.02068v2, Eq. (1).
The score -1 outside the mask has zero clipped mass at the proved threshold. -/
def periodicMemoryMaskedScores {N : ℕ} (t : Fin N → ℝ) (i j : Fin (N + 1)) : ℝ :=
  if j ∈ localMemoryNeighbours i then periodicMemoryScore t i j else -1

/-- Actual variational sparsemax applied to the genuinely computed masked QK scores.
Source: arXiv:1602.02068v2, Eq. (1), in the new local width-three architecture. -/
def periodicMemoryAttention {N : ℕ} (t : Fin N → ℝ) :
    Matrix (Fin (N + 1)) (Fin (N + 1)) ℝ :=
  Matrix.of (fun i => sparseWeights (periodicMemoryMaskedScores t i) (Fin.last N))

/-- Clipping the actual masked QK scores at zero recovers every desired probability row.
Source: the new restricted use of arXiv:1602.02068v2, §2.2, Proposition 1. -/
theorem periodicMemoryThreshold_eq {N : ℕ} (floor : ℝ) (t : Fin N → ℝ)
    (hf : 0 ≤ floor) (ht : t ∈ incidentMemoryWeightDomain N floor) (i : Fin (N + 1)) :
    thresholdWeights (periodicMemoryMaskedScores t i) (Fin.last N) 0 =
      memoryGramScores (localMemoryCore t) i := by
  funext j
  rw [thresholdWeights, ite_eq_left (Fin.le_last j), sub_zero]
  by_cases hj : j ∈ localMemoryNeighbours i
  · rw [periodicMemoryMaskedScores, ite_eq_left hj, periodicMemoryScore_local t ht.1 i j hj]
    exact max_eq_left (incidentMemory_scores_nonneg floor t hf ht i j)
  · rw [periodicMemoryMaskedScores, ite_eq_right hj, localMemoryCore_zero_of_not_mem t i j hj]
    norm_num

/-- Four prototypes with simultaneous positive edges inhabit the threshold premises. -/
example : thresholdWeights (periodicMemoryMaskedScores (fun _ : Fin 3 => (1 / 8 : ℝ)) 1)
    (Fin.last 3) 0 = memoryGramScores (localMemoryCore (fun _ : Fin 3 => (1 / 8 : ℝ))) 1 :=
  periodicMemoryThreshold_eq _ _ (by norm_num) incidentMemoryExampleWeights_mem _

/-- Actual sparsemax, with width-three learned Q/K, is exactly the affine path attention.
Source: the new fixed-width realization of arXiv:1602.02068v2, Eq. (1).
This conclusion is proved for the existing variational projection, including support boundaries. -/
theorem periodicMemoryAttention_normalized {N : ℕ} (floor : ℝ) (t : Fin N → ℝ)
    (hf : 0 ≤ floor) (ht : t ∈ incidentMemoryWeightDomain N floor) :
    periodicMemoryAttention t = memoryGramScores (localMemoryCore t) := by
  ext i j
  have htau := periodicMemoryThreshold_eq floor t hf ht i
  have hsum : (∑ j, thresholdWeights (periodicMemoryMaskedScores t i) (Fin.last N) 0 j) = 1 := by
    rw [htau]
    exact localMemoryCore_scores_rowSum t i
  have he := thresholdWeights_eq_sparseWeights (periodicMemoryMaskedScores t i) (Fin.last N) 0 hsum
  rw [htau] at he
  exact congrFun he.symm j

/-- Nonidentity actual attention inhabits the normalization premises. -/
example : periodicMemoryAttention (fun _ : Fin 3 => (1 / 8 : ℝ)) =
    memoryGramScores (localMemoryCore (fun _ : Fin 3 => (1 / 8 : ℝ))) :=
  periodicMemoryAttention_normalized _ _ (by norm_num) incidentMemoryExampleWeights_mem

/-- Width three preserves the genuine inverse guarantee for a self-weight floor above one half.
Source: the derived Gershgorin memory guarantee after arXiv:1602.02068v2, Eq. (1). -/
theorem periodicMemoryAttention_det_unit {N : ℕ} (floor : ℝ) (t : Fin N → ℝ)
    (hf : 1 / 2 < floor) (ht : t ∈ incidentMemoryWeightDomain N floor) :
    IsUnit (periodicMemoryAttention t).det := by
  have hp : (t, (0 : Sum (Fin (N + 1)) (Fin (N + 1)) → ℝ)) ∈
      incidentMemoryParameterDomain N 1 floor := by
    refine ⟨ht, ?_⟩
    intro x
    norm_num
  have hi := incidentMemoryAttention_det_unit 1 floor (t, 0) hf hp
  rw [incidentMemoryAttention_normalized 1 floor _ (by linarith) hp] at hi
  rw [periodicMemoryAttention_normalized floor t (by linarith) ht]
  exact hi

/-- Simultaneous nonzero edges satisfy all proved inverse premises. -/
example : IsUnit (periodicMemoryAttention (fun _ : Fin 3 => (1 / 8 : ℝ))).det :=
  periodicMemoryAttention_det_unit _ _ (by norm_num) incidentMemoryExampleWeights_mem

/-- Actual fixed-width attention is affine in all learned path weights on the convex domain.
Source: the new masked chart before arXiv:1602.02068v2, Eq. (1). -/
theorem periodicMemoryAttention_affine {N : ℕ} (floor : ℝ) (t s : Fin N → ℝ)
    (hf : 0 ≤ floor) (ht : t ∈ incidentMemoryWeightDomain N floor)
    (hs : s ∈ incidentMemoryWeightDomain N floor) (a b : ℝ)
    (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b = 1) :
    periodicMemoryAttention (a • t + b • s) =
      a • periodicMemoryAttention t + b • periodicMemoryAttention s := by
  have hm := incidentMemoryWeightDomain_convex N floor ht hs ha hb hab
  rw [periodicMemoryAttention_normalized floor _ hf hm,
    periodicMemoryAttention_normalized floor t hf ht, periodicMemoryAttention_normalized floor s hf hs]
  have hg := localMemoryCore_affine t s a b hab
  ext i j
  change localMemoryCore (a • t + b • s) (Sum.inl i) (Sum.inr j) = _
  rw [hg]
  rfl

/-- Different edge allocations satisfy the actual midpoint-affinity premises. -/
example : periodicMemoryAttention ((1 / 2 : ℝ) • (0 : Fin 3 → ℝ) +
    (1 / 2 : ℝ) • (fun _ : Fin 3 => (1 / 8 : ℝ))) =
      (1 / 2 : ℝ) • periodicMemoryAttention (0 : Fin 3 → ℝ) +
        (1 / 2 : ℝ) • periodicMemoryAttention (fun _ : Fin 3 => (1 / 8 : ℝ)) :=
  periodicMemoryAttention_affine _ _ _ (by norm_num)
    (zero_mem_incidentMemoryWeightDomain _ _ (by norm_num)) incidentMemoryExampleWeights_mem
    _ _ (by norm_num) (by norm_num) (by norm_num)

/-- The explicit real feature Gram has exactly three coordinates for every prototype count.
Source: the new genuine bounded Q/K families before arXiv:1602.02068v2, Eq. (1). -/
def periodicMemoryFeatures {N : ℕ} (t : Fin N → ℝ) (d : Fin 3) :
    Sum (Fin (N + 1)) (Fin (N + 1)) → ℝ :=
  Sum.elim (fun i => periodicMemoryQuery t i d) (fun j => periodicMemoryKey t j d)

/-- The physical Q/K Gram is PSD at fixed width three, without an assumed factorization.
Source: the new explicit score construction before arXiv:1602.02068v2, Eq. (1). -/
theorem periodicMemoryFeatures_posSemidef {N : ℕ} (t : Fin N → ℝ) :
    (featureGram (periodicMemoryFeatures t)).PosSemidef :=
  featureGram_posSemidef _

/-- The real width-three Q/K Gram has uniformly bounded squared norms at every prototype.
Source: the new bounded physical embedding chart before arXiv:1602.02068v2, Eq. (1).
Neither the head dimension nor this norm cap depends on the number of prototypes. -/
theorem periodicMemoryFeatures_sq_bound {N : ℕ} (floor : ℝ) (t : Fin N → ℝ)
    (hf : 0 ≤ floor) (ht : t ∈ incidentMemoryWeightDomain N floor)
    (x : Sum (Fin (N + 1)) (Fin (N + 1))) :
    (∑ d, (periodicMemoryFeatures t d x) ^ 2) ≤ 4 := by
  cases x with
  | inl i =>
    have h := periodicMemoryQuery_sq_bound floor t hf ht i
    change (∑ d, (periodicMemoryQuery t i d) ^ 2) ≤ 4
    linarith
  | inr j => exact periodicMemoryKey_sq_bound floor t hf ht j

/-- A nonzero physical embedding satisfies the uniform cap at a changed interior slot. -/
example : (∑ d, (periodicMemoryFeatures (fun _ : Fin 3 => (1 / 8 : ℝ)) d (Sum.inl 1)) ^ 2) ≤ 4 :=
  periodicMemoryFeatures_sq_bound _ _ (by norm_num) incidentMemoryExampleWeights_mem _

end Transformer.GPTMini.Sparsemax
