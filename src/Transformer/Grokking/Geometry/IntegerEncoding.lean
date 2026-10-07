import Transformer.Grokking.Geometry.RowAlignment

/-!
# Exact integer encoding of observed-logit cleanup errors

Source: geometry_certificates.py, whose common-denominator computation
implements the sufficient margin criterion derived from the restricted
logits of Nanda et al., arXiv:2301.05217v1, section 5.1.

Represent the observed logits by an integer matrix divided by a positive
common scale. For n cell observations and m classes, the aligned error
numerator is n*m*a - n*row_sum - m*column_sum + total_sum. Its denominator
is n*m*scale. These equations derive that formula from the actual averages
and connect its squared integer energy to the real-valued certificate.

The existence of this representation for Python's observed binary floats
is not identified with a theorem about Lean Float or the transformer's
exact-real forward map. All claims here take the integer matrix and scale
as inputs, with nonempty averaging sets and positive scale explicit.

All squared energies are sums over retained class coordinates. Replacing
one by its mean without the class count would change the sufficient
inequality. The observation and class counts remain in the numerator
until they cancel against the corresponding reference-margin factors.
-/

namespace Transformer.Grokking.Geometry

open scoped BigOperators

variable {I C : Type*}

/-- Observed logits on an integer grid. Source: exact_cell in
geometry_certificates.py; the margin criterion comes from the
restricted-logit interpretation of arXiv:2301.05217v1, section 5.1. -/
noncomputable def gridLogits (a : I → C → ℤ) (scale : ℝ) (d : I) (c : C) : ℝ :=
  (a d c : ℝ) / scale

/-- Integer numerator of the actual aligned error. Source: exact_cell
in geometry_certificates.py, derived from restricted-logit averaging
in the adaptation of arXiv:2301.05217v1, section 5.1. -/
def integerAlignedNumerator (s : Finset I) (cs : Finset C) (a : I → C → ℤ)
    (d : I) (c : C) : ℤ :=
  (s.card : ℤ) * (cs.card : ℤ) * a d c - (s.card : ℤ) * (∑ k ∈ cs, a d k) -
    (cs.card : ℤ) * (∑ i ∈ s, a i c) + ∑ i ∈ s, ∑ k ∈ cs, a i k

/-- The raw cell mean is its actual integer column sum divided by
observation count and scale. Source: exact_cell in geometry_certificates.py,
adapting restricted logits of arXiv:2301.05217v1, section 5.1. -/
theorem grid_column_mean (s : Finset I) (a : I → C → ℤ) (scale : ℝ) (c : C) :
    columnMean s (gridLogits a scale) c =
      ((∑ i ∈ s, a i c : ℤ) : ℝ) / ((s.card : ℝ) * scale) := by
  unfold columnMean meanOver gridLogits
  rw [← Finset.sum_div, ← Int.cast_sum, div_div, mul_comm scale]

/-- Raw reference differences use column-sum differences, retaining
class bias. Source: exact_cell in geometry_certificates.py, adapting
restricted-logit correctness in arXiv:2301.05217v1, section 5.1. -/
theorem grid_target_gap (s : Finset I) (a : I → C → ℤ) (scale : ℝ) (y k : C) :
    columnMean s (gridLogits a scale) y - columnMean s (gridLogits a scale) k =
      (((∑ i ∈ s, a i y) - (∑ i ∈ s, a i k) : ℤ) : ℝ) / ((s.card : ℝ) * scale) := by
  rw [grid_column_mean, grid_column_mean, div_sub_div_same, Int.cast_sub]

/-- Integer sums and the error polynomial embed exactly into the real
formula. Source: exact_cell in geometry_certificates.py; this algebra
supports the adaptation of arXiv:2301.05217v1, section 5.1. -/
theorem integer_aligned_numerator_cast (s : Finset I) (cs : Finset C)
    (a : I → C → ℤ) (d : I) (c : C) :
    (integerAlignedNumerator s cs a d c : ℝ) =
      (s.card : ℝ) * (cs.card : ℝ) * (a d c : ℝ) -
        (s.card : ℝ) * (∑ k ∈ cs, (a d k : ℝ)) -
        (cs.card : ℝ) * (∑ i ∈ s, (a i c : ℝ)) + ∑ i ∈ s, ∑ k ∈ cs, (a i k : ℝ) := by
  unfold integerAlignedNumerator
  push_cast
  ring

/-- The coded numerator divided by n*m*scale is exactly the aligned
error derived from raw means. Source: exact_cell in geometry_certificates.py,
adapting restricted logits in arXiv:2301.05217v1, section 5.1. -/
theorem grid_aligned_error (s : Finset I) (cs : Finset C) (a : I → C → ℤ)
    (scale : ℝ) (d : I) (c : C) (hs : s.Nonempty) (hc : cs.Nonempty) (hp : 0 < scale) :
    alignedLogits s cs (gridLogits a scale) d c - columnMean s (gridLogits a scale) c =
      (integerAlignedNumerator s cs a d c : ℝ) /
        ((s.card : ℝ) * (cs.card : ℝ) * scale) := by
  have hn : (s.card : ℝ) ≠ 0 := by
    have hpos : (0 : ℝ) < s.card := by exact_mod_cast Finset.card_pos.mpr hs
    linarith
  have hm : (cs.card : ℝ) ≠ 0 := by
    have hpos : (0 : ℝ) < cs.card := by exact_mod_cast Finset.card_pos.mpr hc
    linarith
  have hscale : scale ≠ 0 := by linarith
  rw [integer_aligned_numerator_cast]
  unfold alignedLogits columnMean meanOver gridLogits
  simp_rw [← Finset.sum_div]
  field_simp
  ring

example : ({0, 1} : Finset ℕ).Nonempty ∧ ({false, true} : Finset Bool).Nonempty ∧
    0 < (2 : ℝ) := by
  refine ⟨⟨0, by norm_num⟩, ⟨false, by norm_num⟩, by norm_num⟩

/-- Squared aligned error is the integer squared-numerator sum divided
by the common squared denominator. Source: exact_cell in
geometry_certificates.py, adapting arXiv:2301.05217v1, section 5.1. -/
theorem grid_aligned_energy (s : Finset I) (cs : Finset C) (a : I → C → ℤ)
    (scale : ℝ) (d : I) (hs : s.Nonempty) (hc : cs.Nonempty) (hp : 0 < scale) :
    energyOver cs (fun c => alignedLogits s cs (gridLogits a scale) d c -
      columnMean s (gridLogits a scale) c) =
      ((∑ c ∈ cs, (integerAlignedNumerator s cs a d c) ^ 2 : ℤ) : ℝ) /
        ((s.card : ℝ) * (cs.card : ℝ) * scale) ^ 2 := by
  unfold energyOver
  simp_rw [grid_aligned_error s cs a scale d _ hs hc hp, div_pow]
  rw [← Finset.sum_div]
  simp only [Int.cast_sum, Int.cast_pow]

example : ({false, true} : Finset Bool).Nonempty ∧ ({0, 1} : Finset ℕ).Nonempty ∧
    0 < (1 : ℝ) := by
  refine ⟨⟨true, by norm_num⟩, ⟨1, by norm_num⟩, by norm_num⟩

example : integerAlignedNumerator ({0, 1} : Finset ℕ) ({false, true} : Finset Bool)
    (fun i b => if b then (2 : ℤ) * (i + 1) else -2 * (i + 1)) 0 true = -4 := by
  norm_num [integerAlignedNumerator]

-- The two observed rows are (2, -2) and (4, -4). Their raw reference
-- is (3, -3). This nonzero-residual example checks the denominator and
-- the class/observation cardinality factors in the exact encoding.
example : columnMean ({0, 1} : Finset ℕ)
    (gridLogits (fun i b => if b then (2 : ℤ) * (i + 1) else -2 * (i + 1)) 1) true = 3 := by
  norm_num [columnMean, meanOver, gridLogits]

example : alignedLogits ({0, 1} : Finset ℕ) ({false, true} : Finset Bool)
    (gridLogits (fun i b => if b then (2 : ℤ) * (i + 1) else -2 * (i + 1)) 1) 0 true -
    columnMean ({0, 1} : Finset ℕ)
      (gridLogits (fun i b => if b then (2 : ℤ) * (i + 1) else -2 * (i + 1)) 1) true = -1 := by
  norm_num [alignedLogits, columnMean, meanOver, gridLogits]

example : energyOver ({false, true} : Finset Bool) (fun b =>
    alignedLogits ({0, 1} : Finset ℕ) ({false, true} : Finset Bool)
      (gridLogits (fun i b => if b then (2 : ℤ) * (i + 1) else -2 * (i + 1)) 1) 0 b -
    columnMean ({0, 1} : Finset ℕ)
      (gridLogits (fun i b => if b then (2 : ℤ) * (i + 1) else -2 * (i + 1)) 1) b) = 2 := by
  norm_num [energyOver, alignedLogits, columnMean, meanOver, gridLogits]

example : columnMean ({0, 1} : Finset ℕ)
    (gridLogits (fun i b => if b then (2 : ℤ) * (i + 1) else -2 * (i + 1)) 1) true -
    columnMean ({0, 1} : Finset ℕ)
      (gridLogits (fun i b => if b then (2 : ℤ) * (i + 1) else -2 * (i + 1)) 1) false = 6 := by
  norm_num [columnMean, meanOver, gridLogits]

end Transformer.Grokking.Geometry
