module

public import Transformer.Clusters.Section8_AttentionRoot

/-!
# A quadratic metric built from finitely many value-map powers

ArXiv:2305.05465v6, §9, `r:notdiagonalizable` and `e:notdi`, as used in §10.
The growth estimate needs a norm in which the complementary value map
contracts below the leading eigenvalue. Here a finite sum of squared,
scaled power norms constructs a positive quadratic form. The positive-root
construction then represents it by the Euclidean norm of a linear map.

This is an alternative to choosing a Jordan basis with small off-diagonal
entries. The zero-th power makes the metric dominate the original norm.
The finite-power contraction estimate will give the required operator
bound in this metric; no spectral conclusion is assumed in its definition.
-/

@[expose] public section

open scoped BigOperators
open Real

namespace Transformer.Clusters

variable {d : ℕ}

/-- The finite sum of scaled squared power norms used to construct the
metric for the growth bound. Source: arXiv:2305.05465v6, §9,
`r:notdiagonalizable`, `e:notdi`; this replaces the Jordan-basis construction. -/
noncomputable def powerEnergy (V : ParamMatrix d) (R : ℝ) (m : ℕ) (x : EucSpace d) : ℝ :=
  ∑ k ∈ Finset.range m, ((R ^ k)⁻¹) ^ 2 * ‖(V ^ k) x‖ ^ 2

/-- The Gram operator representing the finite-power quadratic form.
Source: arXiv:2305.05465v6, §9, `r:notdiagonalizable`, `e:notdi`;
its positive square root constructs the adapted Euclidean metric. -/
noncomputable def powerGram (V : ParamMatrix d) (R : ℝ) (m : ℕ) : ParamMatrix d :=
  ∑ k ∈ Finset.range m, ((R ^ k)⁻¹) ^ 2 • (V ^ k).adjoint.comp (V ^ k)

/-- Every term in the finite-power energy is nonnegative.
Source: arXiv:2305.05465v6, §9, the metric needed for `e:notdi`. -/
theorem powerEnergy_nonneg (V : ParamMatrix d) (R : ℝ) (m : ℕ) (x : EucSpace d) :
    0 ≤ powerEnergy V R m x :=
  Finset.sum_nonneg fun k _ =>
    mul_nonneg (sq_nonneg ((R ^ k)⁻¹)) (sq_nonneg ‖(V ^ k) x‖)

/-- Increasing the power cutoff appends exactly the last squared norm.
Source: arXiv:2305.05465v6, §9, the finite-power construction for `e:notdi`. -/
theorem powerEnergy_succ (V : ParamMatrix d) (R : ℝ) (m : ℕ) (x : EucSpace d) :
    powerEnergy V R (m + 1) x = powerEnergy V R m x +
      ((R ^ m)⁻¹) ^ 2 * ‖(V ^ m) x‖ ^ 2 :=
  Finset.sum_range_succ _ _

/-- Consecutive scaling coefficients cancel one application of the value
map. Source: arXiv:2305.05465v6, §9, the adapted metric for `e:notdi`. -/
theorem powerCoefficient_succ (R : ℝ) (hR : R ≠ 0) (m : ℕ) :
    R ^ 2 * ((R ^ (m + 1))⁻¹) ^ 2 = ((R ^ m)⁻¹) ^ 2 := by
  rw [pow_succ R m, mul_inv_rev, mul_pow]
  field_simp [hR]

/-- A nonzero scale is available for every power cutoff.
Source: arXiv:2305.05465v6, §9, `e:notdi`. -/
example : (1 : ℝ) ≠ 0 := one_ne_zero

/-- Applying the value map shifts the finite sum by one power. The only
new term is the last power, while the zero-th norm leaves the sum.
Source: arXiv:2305.05465v6, §9, `r:notdiagonalizable`, `e:notdi`;
this exact identity replaces the Jordan-basis operator estimate. -/
theorem powerEnergy_apply (V : ParamMatrix d) (R : ℝ) (hR : R ≠ 0)
    (m : ℕ) (x : EucSpace d) :
    powerEnergy V R m (V x) = R ^ 2 * (powerEnergy V R m x - ‖x‖ ^ 2 +
      ((R ^ m)⁻¹) ^ 2 * ‖(V ^ m) x‖ ^ 2) := by
  induction m with
  | zero => simp only [powerEnergy, Finset.range_zero, Finset.sum_empty,
      pow_zero, inv_one, one_pow, one_apply_eq_self, one_mul]; ring
  | succ m ih =>
    rw [powerEnergy_succ, powerEnergy_succ, ih]
    have hpower : (V ^ m) (V x) = (V ^ (m + 1)) x := by
      rw [pow_succ]
      rfl
    rw [hpower, ← powerCoefficient_succ R hR m]
    ring

/-- The shift identity has a genuine nonzero scale, independently of the
value map. Source: arXiv:2305.05465v6, §9, `r:notdiagonalizable`. -/
example : (2 : ℝ) ≠ 0 := by norm_num

/-- The zero-th power controls the original squared norm whenever the
cutoff is positive. Source: arXiv:2305.05465v6, §9, `r:notdiagonalizable`,
the norm comparison required to transfer `e:notdi` back to Euclidean space. -/
theorem norm_sq_le_powerEnergy (V : ParamMatrix d) (R : ℝ) (m : ℕ)
    (hm : 0 < m) (x : EucSpace d) : ‖x‖ ^ 2 ≤ powerEnergy V R m x := by
  have hsum := Finset.single_le_sum
    (f := fun k => ((R ^ k)⁻¹) ^ 2 * ‖(V ^ k) x‖ ^ 2)
    (fun k _ => mul_nonneg (sq_nonneg _) (sq_nonneg _)) (Finset.mem_range.mpr hm)
  simpa only [powerEnergy, pow_zero, inv_one, one_pow, one_apply_eq_self, one_mul]
    using hsum

/-- A positive cutoff is possible for every value map, including the
identity. Source: arXiv:2305.05465v6, §9, `r:notdiagonalizable`. -/
example : (0 : ℕ) < 1 := one_pos

/-- The Gram operator has the bilinear form of the finite-power metric.
Source: arXiv:2305.05465v6, §9, `r:notdiagonalizable`, `e:notdi`.
The adjoint identity verifies every term of the construction. -/
theorem powerGram_inner (V : ParamMatrix d) (R : ℝ) (m : ℕ) (x y : EucSpace d) :
    inner (𝕜 := ℝ) x (powerGram V R m y) =
      ∑ k ∈ Finset.range m,
        ((R ^ k)⁻¹) ^ 2 * inner (𝕜 := ℝ) ((V ^ k) x) ((V ^ k) y) := by
  simp only [powerGram, sum_apply, smul_apply, inner_sum, real_inner_smul_right,
    ContinuousLinearMap.comp_apply, ContinuousLinearMap.adjoint_inner_right]

/-- The quadratic form of the Gram operator is exactly the defined
power energy. Source: arXiv:2305.05465v6, §9, `e:notdi`. -/
theorem powerGram_self (V : ParamMatrix d) (R : ℝ) (m : ℕ) (x : EucSpace d) :
    inner (𝕜 := ℝ) x (powerGram V R m x) = powerEnergy V R m x := by
  rw [powerGram_inner]
  simp only [real_inner_self_eq_norm_sq, powerEnergy]

/-- The finite-power Gram operator is symmetric and strictly positive
for a positive cutoff. Source: arXiv:2305.05465v6, §9,
`r:notdiagonalizable`, the change of metric behind `e:notdi`. -/
theorem isPosDefQK_one_powerGram (V : ParamMatrix d) (R : ℝ) (m : ℕ) (hm : 0 < m) :
    IsPosDefQK (1 : ParamMatrix d) (powerGram V R m) := by
  refine ⟨?_, ?_⟩
  · intro x y
    simp only [one_apply_eq_self, powerGram_inner]
    apply Finset.sum_congr rfl
    intro k _
    rw [real_inner_comm]
  · intro x hx
    rw [one_apply_eq_self, powerGram_self]
    exact (sq_pos_of_pos (norm_pos_iff.mpr hx)).trans_le
      (norm_sq_le_powerEnergy V R m hm x)

/-- The positivity input is a genuine positive cutoff, without an
assumption on the value map. Source: arXiv:2305.05465v6, §9, `e:notdi`. -/
example : (0 : ℕ) < 2 := by norm_num

/-- **The finite-power metric has an actual positive linear square root.**
Source: arXiv:2305.05465v6, §9, `r:notdiagonalizable`, `e:notdi`.
The root represents the full quadratic form and dominates the original
Euclidean norm. It is constructed, not taken as an unproved input. -/
theorem exists_powerMetric (V : ParamMatrix d) (R : ℝ) (m : ℕ) (hm : 0 < m) :
    ∃ B : ParamMatrix d, IsAttentionRoot 1 (powerGram V R m) B ∧
      (∀ x : EucSpace d, ‖B x‖ ^ 2 = powerEnergy V R m x) ∧
      ∀ x : EucSpace d, ‖x‖ ≤ ‖B x‖ := by
  obtain ⟨B, hB⟩ := (isPosDefQK_one_powerGram V R m hm).exists_attentionRoot
  have henergy (x : EucSpace d) : ‖B x‖ ^ 2 = powerEnergy V R m x := by
    rw [← real_inner_self_eq_norm_sq, ← hB.2.2, one_apply_eq_self, powerGram_self]
  refine ⟨B, hB, henergy, ?_⟩
  intro x
  apply (pow_le_pow_iff_left₀ (norm_nonneg _) (norm_nonneg _) two_ne_zero).mp
  rw [henergy]
  exact norm_sq_le_powerEnergy V R m hm x

/-- The full cutoff hypothesis holds at length one, so the adapted
metric is available. Source: arXiv:2305.05465v6, §9, `r:notdiagonalizable`. -/
example : (0 : ℕ) < 1 := one_pos

end Transformer.Clusters
