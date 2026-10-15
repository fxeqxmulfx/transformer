/-
# The actual original last frozen-token force and critical derivative

ArXiv:2411.04990v2, §5, proof of `thm:fixed_centers`.
Fix an original preceding angular tuple and vary only the original
last token. Its original partition is one plus the original
predecessor exponentials and all original weighted frozen-center
exponentials. Its original velocity is minus the sum of original
`h` interactions divided by that exact original partition.

The derivative of the original force sum is precisely the original
sum of `g=h'` terms. At an actual original critical point, the force
vanishes, so the partition derivative cancels from the quotient
rule. The actual original last-coordinate velocity derivative is
therefore minus that original curvature sum divided by the original
positive partition. This is the diagonal stability calculation
used by the manuscript, with every original predecessor and
every original frozen-center mass retained.
-/

import Transformer.Causal.Section5_FrozenField

open scoped BigOperators
open Real

namespace Transformer.Causal

variable {n m : ℕ}

/-- The actual original last-row partition, retaining its original self exponential one.
Source: arXiv:2411.04990v2, §5, the last-row normalization in `thm:fixed_centers`. -/
noncomputable def frozenLastPartition (β : ℝ) (θ a : Idx m → ℝ) (p : Idx n → ℝ) (u : ℝ) : ℝ :=
  1 + (∑ j : Idx n, Real.exp (β * (Real.cos (u - p j) - 1))) +
    ∑ j : Idx m, a j * Real.exp (β * (Real.cos (u - θ j) - 1))

/-- The original last-token force with the actual original predecessor and frozen-center interactions.
Source: arXiv:2411.04990v2, §5, `sum h` in the proof of `thm:fixed_centers`. -/
noncomputable def frozenLastForce (β : ℝ) (θ a : Idx m → ℝ) (p : Idx n → ℝ) (u : ℝ) : ℝ :=
  (∑ j : Idx n, h_pot β (u - p j)) + ∑ j : Idx m, a j * h_pot β (u - θ j)

/-- The actual original last-token curvature sum in the original diagonal Jacobian.
Source: arXiv:2411.04990v2, §5, `sum g` in the proof of `thm:fixed_centers`. -/
noncomputable def frozenLastCurvature (β : ℝ) (θ a : Idx m → ℝ) (p : Idx n → ℝ) (u : ℝ) : ℝ :=
  (∑ j : Idx n, g_pot β (u - p j)) + ∑ j : Idx m, a j * g_pot β (u - θ j)

/-- The displayed original last-row partition equals the actual original full normalization exactly.
Source: arXiv:2411.04990v2, §5, the partition in `thm:fixed_centers`. -/
theorem frozenZ_last_eq_partition (β : ℝ) (θ a : Idx m → ℝ) (p : Idx n → ℝ) (u : ℝ) :
    frozenZ (n + 1) m β θ a (Fin.snoc p u) (Fin.last n) = frozenLastPartition β θ a p u := by
  unfold frozenZ frozenLastPartition
  simp only [Finset.sum_filter, Fin.le_last, ite_true]
  rw [Fin.sum_univ_castSucc]
  simp only [Fin.snoc_last, Fin.snoc_castSucc, sub_self, Real.cos_zero, mul_zero, Real.exp_zero]
  ring

/-- The actual original last row is exactly the negative original interaction force over its original partition.
Source: arXiv:2411.04990v2, §5, the scalar field in the proof of `thm:fixed_centers`. -/
theorem frozenVelocity_last_eq_force (β : ℝ) (θ a : Idx m → ℝ) (p : Idx n → ℝ) (u : ℝ) :
    frozenVelocity (n + 1) m β θ a (Fin.snoc p u) (Fin.last n) =
      -frozenLastForce β θ a p u / frozenLastPartition β θ a p u := by
  unfold frozenVelocity
  rw [frozenZ_last_eq_partition]
  simp only [Finset.sum_filter, Fin.le_last, ite_true]
  rw [Fin.sum_univ_castSucc]
  simp only [Fin.snoc_last, Fin.snoc_castSucc, sub_self, Real.sin_zero, mul_zero, add_zero]
  have hflip (v : ℝ) : Real.sin (v - u) = -Real.sin (u - v) := by
    rw [← Real.sin_neg]
    congr 1
    ring
  simp_rw [hflip]
  simp only [mul_neg, Finset.sum_neg_distrib, frozenLastForce, h_pot, div_eq_mul_inv, mul_assoc]
  ring

/-- The actual original last-row partition stays positive for the original nonnegative center masses.
Source: arXiv:2411.04990v2, §5, the positive original normalization in `thm:fixed_centers`. -/
theorem frozenLastPartition_pos (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (p : Idx n → ℝ) (u : ℝ) : 0 < frozenLastPartition β θ a p u := by
  rw [← frozenZ_last_eq_partition]
  exact frozenZ_pos β θ a ha (Fin.snoc p u) (Fin.last n)

/-- The actual original last-row partition is smooth as the original last angle varies.
Source: arXiv:2411.04990v2, §5, the smooth original scalar field in `thm:fixed_centers`. -/
theorem contDiff_frozenLastPartition (β : ℝ) (θ a : Idx m → ℝ) (p : Idx n → ℝ) :
    ContDiff ℝ (⊤ : ℕ∞) (frozenLastPartition β θ a p) := by
  unfold frozenLastPartition
  fun_prop

/-- The derivative of the original last force is exactly the original curvature sum, with every original center mass.
Source: arXiv:2411.04990v2, §5, differentiating `sum h` to `sum g` in `thm:fixed_centers`. -/
theorem hasDerivAt_frozenLastForce (β : ℝ) (θ a : Idx m → ℝ) (p : Idx n → ℝ) (u : ℝ) :
    HasDerivAt (frozenLastForce β θ a p) (frozenLastCurvature β θ a p u) u := by
  unfold frozenLastForce frozenLastCurvature
  apply HasDerivAt.fun_add
  · apply HasDerivAt.fun_sum
    intro j hj
    simpa! only [Function.comp_def, id_eq, mul_one] using
      (hasDerivAt_h_pot β (u - p j)).comp u ((hasDerivAt_id u).sub_const (p j))
  · apply HasDerivAt.fun_sum
    intro j hj
    have h := (hasDerivAt_h_pot β (u - θ j)).comp u ((hasDerivAt_id u).sub_const (θ j))
    simpa! only [Function.comp_def, id_eq, mul_one] using h.const_mul (a j)

/-- An actual original last-row critical point has exactly zero original interaction force.
Source: arXiv:2411.04990v2, §5, the criticality equation in `thm:fixed_centers`. -/
theorem frozenVelocity_last_eq_zero_iff (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (p : Idx n → ℝ) (u : ℝ) :
    frozenVelocity (n + 1) m β θ a (Fin.snoc p u) (Fin.last n) = 0 ↔ frozenLastForce β θ a p u = 0 := by
  rw [frozenVelocity_last_eq_force, div_eq_zero_iff]
  simp only [(frozenLastPartition_pos β θ a ha p u).ne', or_false, neg_eq_zero]

/-- At an actual original critical last token, the original diagonal velocity derivative is minus curvature over partition.
Source: arXiv:2411.04990v2, §5, the diagonal Jacobian display in the proof of `thm:fixed_centers`. -/
theorem hasDerivAt_frozenLastVelocity_of_critical (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (p : Idx n → ℝ) (u : ℝ)
    (hcrit : frozenVelocity (n + 1) m β θ a (Fin.snoc p u) (Fin.last n) = 0) :
    HasDerivAt (fun v => frozenVelocity (n + 1) m β θ a (Fin.snoc p v) (Fin.last n))
      (-frozenLastCurvature β θ a p u / frozenLastPartition β θ a p u) u := by
  have hZ := ((contDiff_frozenLastPartition β θ a p).differentiable (by simp) u).hasDerivAt
  have hZ0 := (frozenLastPartition_pos β θ a ha p u).ne'
  have hF0 := (frozenVelocity_last_eq_zero_iff β θ a ha p u).mp hcrit
  have h := (hasDerivAt_frozenLastForce β θ a p u).fun_neg.fun_div hZ hZ0
  have hfun : (fun v => frozenVelocity (n + 1) m β θ a (Fin.snoc p v) (Fin.last n)) =
      (fun v => -frozenLastForce β θ a p v / frozenLastPartition β θ a p v) :=
    funext (frozenVelocity_last_eq_force β θ a p)
  rw [hfun]
  apply h.congr_deriv
  rw [hF0]
  field_simp
  ring

/-- The actual original last-coordinate derivative has the manuscript's exact critical curvature formula.
Source: arXiv:2411.04990v2, §5, the original diagonal stability calculation in `thm:fixed_centers`. -/
theorem deriv_frozenLastVelocity_of_critical (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (p : Idx n → ℝ) (u : ℝ)
    (hcrit : frozenVelocity (n + 1) m β θ a (Fin.snoc p u) (Fin.last n) = 0) :
    deriv (fun v => frozenVelocity (n + 1) m β θ a (Fin.snoc p v) (Fin.last n)) u =
      -frozenLastCurvature β θ a p u / frozenLastPartition β θ a p u :=
  (hasDerivAt_frozenLastVelocity_of_critical β θ a ha p u hcrit).deriv

/-- A nonpositive actual original critical diagonal derivative is equivalent to a nonnegative original curvature sum.
Source: arXiv:2411.04990v2, §5, the first diagonal stability implication in `thm:fixed_centers`. -/
theorem deriv_frozenLastVelocity_nonpos_iff (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (p : Idx n → ℝ) (u : ℝ)
    (hcrit : frozenVelocity (n + 1) m β θ a (Fin.snoc p u) (Fin.last n) = 0) :
    deriv (fun v => frozenVelocity (n + 1) m β θ a (Fin.snoc p v) (Fin.last n)) u ≤ 0 ↔
      0 ≤ frozenLastCurvature β θ a p u := by
  rw [deriv_frozenLastVelocity_of_critical β θ a ha p u hcrit,
    div_le_iff₀ (frozenLastPartition_pos β θ a ha p u), zero_mul, neg_nonpos]

/-- A negative actual original critical diagonal derivative is equivalent to a positive original curvature sum.
Source: arXiv:2411.04990v2, §5, the strictly stable diagonal criterion in `thm:fixed_centers`. -/
theorem deriv_frozenLastVelocity_neg_iff (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (p : Idx n → ℝ) (u : ℝ)
    (hcrit : frozenVelocity (n + 1) m β θ a (Fin.snoc p u) (Fin.last n) = 0) :
    deriv (fun v => frozenVelocity (n + 1) m β θ a (Fin.snoc p v) (Fin.last n)) u < 0 ↔
      0 < frozenLastCurvature β θ a p u := by
  rw [deriv_frozenLastVelocity_of_critical β θ a ha p u hcrit,
    div_lt_iff₀ (frozenLastPartition_pos β θ a ha p u), zero_mul, neg_lt_zero]

/-- Actual positive frozen mass and actual coincident angular tokens jointly satisfy the original critical-point hypotheses.
Source: arXiv:2411.04990v2, §5, the one-center consensus equilibrium in `thm:fixed_centers`. -/
example (β : ℝ) : (∀ j : Idx 1, (0 : ℝ) ≤ (fun _ : Idx 1 => (1 : ℝ)) j) ∧
    frozenVelocity 2 1 β (fun _ => 0) (fun _ => 1)
      (Fin.snoc (fun _ : Idx 1 => 0) 0) (Fin.last 1) = 0 := by
  refine ⟨fun _ => zero_le_one, ?_⟩
  rw [frozenVelocity_last_eq_force]
  simp [frozenLastForce, h_pot]

end Transformer.Causal
