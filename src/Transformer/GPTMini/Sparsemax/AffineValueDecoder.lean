import Transformer.GPTMini.Sparsemax.AffineValueBasis

/-!
# Original common values decoded with two coefficients per channel

New compact-value construction after arXiv:1602.02068v2, Eq. (1).
The learned quadratic edge profile preserves the two generated positional
features. A strict self-weight floor bounds its nonconstant eigenvalue
below by `2*floor-1`. The original inverse therefore divides only the
slope coefficient by `1-2*c`, while the constant coefficient is unchanged.

The actual common value at a slot is evaluated from these two coefficient
rows and its normalized position. No per-prototype value rows, feature
matrix or dictionary-sized inverse are needed to evaluate this formula.
The equality to the original global nonsingular value decoder is proved.
This is an exact restricted response family, not a loss of sparsity or an
approximation to the former arbitrary-output table.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- Four prototypes have a genuine nonzero generated profile with strict diagonal floor.
Source: a concrete invariant-feature witness before sparsemax Eq. (1). -/
theorem affineValueFourEdges_mem :
    affineValueEdges 2 (1 / 48) ∈ incidentMemoryWeightDomain 3 (3 / 4) := by
  constructor
  · intro e
    fin_cases e <;> norm_num [affineValueEdges]
  · intro i
    fin_cases i <;>
      norm_num [localIncidentWeight, affineValueEdges, Fin.sum_univ_succ]

/-- Local feasibility makes the single generated edge coefficient nonnegative.
Source: the left boundary edge of the compact profile before arXiv:1602.02068v2, Eq. (1). -/
theorem affineValueMix_nonneg (N : ℕ) (floor c : ℝ)
    (ht : affineValueEdges N c ∈ incidentMemoryWeightDomain (N + 1) floor) : 0 ≤ c := by
  have h := ht.1 0
  simp only [affineValueEdges, Fin.val_zero, Nat.cast_zero, zero_add, mul_one, sub_zero] at h
  exact (mul_nonneg_iff_of_pos_right (Nat.cast_add_one_pos N)).mp h

/-- Nonidentity four-slot weights inhabit every scalar nonnegativity premise. -/
example : (0 : ℝ) ≤ 1 / 48 :=
  affineValueMix_nonneg 2 (3 / 4) (1 / 48) affineValueFourEdges_mem

/-- A strict self-weight floor bounds the actual inverse coefficient away from zero uniformly in size.
Source: the boundary edge budget and invariant feature before sparsemax Eq. (1). -/
theorem affineValueMix_eigenvalue (N : ℕ) (floor c : ℝ)
    (ht : affineValueEdges N c ∈ incidentMemoryWeightDomain (N + 1) floor) :
    2 * floor - 1 ≤ 1 - 2 * c := by
  have hc := affineValueMix_nonneg N floor c ht
  have hb := (incidentMemoryWeightDomain_coordinateBound floor _ ht 0).2
  simp only [affineValueEdges, Fin.val_zero, Nat.cast_zero, zero_add, mul_one, sub_zero] at hb
  have hn : 0 ≤ (N : ℝ) := Nat.cast_nonneg N
  have hp := mul_nonneg hc hn
  nlinarith

/-- A four-slot nonidentity profile inhabits every uniform inverse-bound premise. -/
example : 2 * (3 / 4 : ℝ) - 1 ≤ 1 - 2 * (1 / 48 : ℝ) :=
  affineValueMix_eigenvalue 2 (3 / 4) (1 / 48) affineValueFourEdges_mem

/-- Two learned original value coefficient rows: unchanged constant and inverse-adjusted slope.
Source: the proved invariant-feature inverse after arXiv:1602.02068v2, Eq. (1). -/
def affineValueCoefficients {D : ℕ} (c : ℝ) (W : Matrix (Fin 2) (Fin D) ℝ) :
    Matrix (Fin 2) (Fin D) ℝ :=
  fun k d => if k = 0 then W 0 d else W 1 d / (1 - 2 * c)

/-- Applying the genuine path scores to compact values acts diagonally on their two features.
Source: the invariant-feature algebra before variational sparsemax Eq. (1). -/
theorem affineValueOutput_action (N : ℕ) {D : ℕ} (c : ℝ)
    (W : Matrix (Fin 2) (Fin D) ℝ) (i : Fin (N + 2)) (d : Fin D) :
    (∑ j, memoryGramScores (localMemoryCore (affineValueEdges N c)) i j *
      affineValueOutput N W j d) = W 0 d + (1 - 2 * c) * affineValuePosition N i * W 1 d := by
  simp only [affineValueOutput, mul_add, ← mul_assoc]
  rw [Finset.sum_add_distrib, ← Finset.sum_mul, ← Finset.sum_mul,
    localMemoryCore_scores_rowSum, affineValuePosition_action, one_mul]

/-- Genuine categorical masked sparsemax multiplies values generated from the small coefficient table.
Source: the new constant-value-storage `attn @ v` construction after sparsemax Eq. (1). -/
def affineValueForward {R N D : ℕ} (code : Fin R → Fin (N + 2)) (c : ℝ)
    (W : Matrix (Fin 2) (Fin D) ℝ) : Matrix (Fin R) (Fin D) ℝ :=
  Matrix.of (fun r => periodicMemoryAttention (affineValueEdges N c) (code r)) *
    affineValueOutput N (affineValueCoefficients c W)

/-- The actual variational sparsemax forward gives the affine positional response exactly.
Source: proved probability-score projection and compact original inverse after Eq. (1). -/
theorem affineValueForward_eq {R N D : ℕ} (floor : ℝ) (code : Fin R → Fin (N + 2))
    (c : ℝ) (W : Matrix (Fin 2) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (ht : affineValueEdges N c ∈ incidentMemoryWeightDomain (N + 1) floor) :
    affineValueForward code c W = Matrix.of (fun r => affineValueOutput N W (code r)) := by
  have hl : 0 < 1 - 2 * c := by
    have h := affineValueMix_eigenvalue N floor c ht
    linarith
  unfold affineValueForward
  rw [periodicMemoryAttention_normalized floor _ (by linarith) ht]
  ext r d
  change (∑ j, memoryGramScores (localMemoryCore (affineValueEdges N c)) (code r) j *
    affineValueOutput N (affineValueCoefficients c W) j d) = affineValueOutput N W (code r) d
  rw [affineValueOutput_action]
  simp only [affineValueCoefficients, ite_true, eq_self, Fin.isValue,
    show (1 : Fin 2) ≠ 0 by decide, ite_false]
  unfold affineValueOutput
  have hc := mul_div_cancel₀ (W 1 d) (ne_of_gt hl)
  nlinarith [congrArg (fun z : ℝ => affineValuePosition N (code r) * z) hc]

/-- A nonconstant four-slot forward inhabits all actual compact-decoder premises. -/
example : affineValueForward (fun j : Fin 4 => j) (1 / 48)
    (Matrix.of (fun k : Fin 2 => fun _ : Fin 1 => (k.val : ℝ))) =
      affineValueOutput 2 (Matrix.of (fun k : Fin 2 => fun _ : Fin 1 => (k.val : ℝ))) :=
  affineValueForward_eq (3 / 4) _ _ _ (by norm_num) affineValueFourEdges_mem

/-- The original global inverse decoder is exactly the compact generated common values.
Source: true nonsingular sparsemax uniqueness after arXiv:1602.02068v2, Eq. (1). -/
theorem affineValueDecoder_original (N : ℕ) {D : ℕ} (floor c : ℝ)
    (W : Matrix (Fin 2) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (ht : affineValueEdges N c ∈ incidentMemoryWeightDomain (N + 1) floor) :
    periodicMemoryValues (affineValueEdges N c) (affineValueOutput N W) =
      affineValueOutput N (affineValueCoefficients c W) := by
  have he := affineValueForward_eq floor (fun j : Fin (N + 2) => j) c W hf ht
  change periodicMemoryAttention (affineValueEdges N c) *
    affineValueOutput N (affineValueCoefficients c W) = affineValueOutput N W at he
  unfold periodicMemoryValues
  rw [← he, Matrix.nonsing_inv_mul_cancel_left _ _
    (periodicMemoryAttention_det_unit floor _ hf ht)]

/-- Learned nonconstant original values satisfy every compact inverse-equality premise. -/
example : periodicMemoryValues (affineValueEdges 2 (1 / 48))
    (affineValueOutput 2 (Matrix.of (fun k : Fin 2 => fun _ : Fin 1 => (k.val : ℝ)))) =
      affineValueOutput 2 (affineValueCoefficients (1 / 48)
        (Matrix.of (fun k : Fin 2 => fun _ : Fin 1 => (k.val : ℝ)))) :=
  affineValueDecoder_original 2 (3 / 4) (1 / 48) _ (by norm_num) affineValueFourEdges_mem

/-- The four original value rows are generated by two genuinely inverse-adjusted coefficients. -/
example : affineValueOutput 2 (affineValueCoefficients (1 / 48)
    (Matrix.of (fun k : Fin 2 => fun _ : Fin 1 => (k.val : ℝ)))) 0 0 = -24 / 23 := by
  norm_num [affineValueOutput, affineValuePosition, affineValueCoefficients, Matrix.of_apply]

/-- Actual original common values admit exactly two stored coefficient rows at every memory size.
Source: proved compact inverse equality after sparsemax Eq. (1), with the parameter count explicit. -/
theorem affineValueDecoder_constant_storage (N : ℕ) {D : ℕ} (floor c : ℝ)
    (W : Matrix (Fin 2) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (ht : affineValueEdges N c ∈ incidentMemoryWeightDomain (N + 1) floor) :
    ∃ C : Matrix (Fin 2) (Fin D) ℝ,
      periodicMemoryValues (affineValueEdges N c) (affineValueOutput N W) = affineValueOutput N C ∧
      Fintype.card (Fin 2 × Fin D) = 2 * D := by
  exact ⟨affineValueCoefficients c W, affineValueDecoder_original N floor c W hf ht,
    affineValueCoefficient_count D⟩

/-- Four nonconstant original value rows inhabit all compact-storage premises. -/
example : ∃ C : Matrix (Fin 2) (Fin 1) ℝ,
    periodicMemoryValues (affineValueEdges 2 (1 / 48))
      (affineValueOutput 2 (Matrix.of (fun k : Fin 2 => fun _ : Fin 1 => (k.val : ℝ)))) =
        affineValueOutput 2 C ∧ Fintype.card (Fin 2 × Fin 1) = 2 * 1 :=
  affineValueDecoder_constant_storage 2 (3 / 4) (1 / 48) _ (by norm_num) affineValueFourEdges_mem

end Transformer.GPTMini.Sparsemax
