module

public import Transformer.Causal.Section5_FrozenDefs

/-!
# The original frozen-token normalized interactions

ArXiv:2411.04990v2, §5, Theorem `thm:fixed_centers` and its proof.
The original last-row normalization includes its current-token
exponential, all actual causal predecessors, and all original frozen
centers with their original weights. Its current-token exponential
is one, so nonnegative frozen-center weights make every original
partition at least one for every lifted angular configuration.

The original active particle weights and frozen-center weights are
nonnegative and together sum exactly to one. The original frozen
velocity is their actual weighted sum of the original signed sine
interactions, and consequently every original angular velocity has
absolute value at most one. The scores, mask, center weights and
normalization are those of the original theorem. These estimates
do not assume any convergence or stability conclusion.
-/

@[expose] public section

open scoped BigOperators
open Real

namespace Transformer.Causal

variable {n m : ℕ}

/-- The actual original causal particle weight in the original frozen-token equation.
Source: arXiv:2411.04990v2, §5, the displayed dynamics of `thm:fixed_centers`. -/
noncomputable def frozenParticleWeight (β : ℝ) (θ a : Idx m → ℝ) (φ : Idx n → ℝ)
    (k j : Idx n) : ℝ :=
  (if j ≤ k then Real.exp (β * (Real.cos (φ k - φ j) - 1)) else 0) / frozenZ n m β θ a φ k

/-- The actual original frozen-center weight, including the original center mass and partition.
Source: arXiv:2411.04990v2, §5, the frozen terms of `thm:fixed_centers`. -/
noncomputable def frozenCenterWeight (β : ℝ) (θ a : Idx m → ℝ) (φ : Idx n → ℝ)
    (k : Idx n) (j : Idx m) : ℝ :=
  (a j * Real.exp (β * (Real.cos (φ k - θ j) - 1))) / frozenZ n m β θ a φ k

/-- The actual original self exponential makes the original partition at least one at every lifted angular state.
Source: arXiv:2411.04990v2, §5, the normalizing factor in `thm:fixed_centers`. -/
theorem one_le_frozenZ (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (φ : Idx n → ℝ) (k : Idx n) : 1 ≤ frozenZ n m β θ a φ k := by
  have hs := Finset.single_le_sum
    (s := Finset.univ.filter (fun j : Idx n => j ≤ k))
    (f := fun j => Real.exp (β * (Real.cos (φ k - φ j) - 1)))
    (fun j hj => (Real.exp_pos _).le) (by simp : k ∈ Finset.univ.filter (fun j : Idx n => j ≤ k))
  have he : Real.exp (β * (Real.cos (φ k - φ k) - 1)) = 1 := by simp
  rw [he] at hs
  have hc : 0 ≤ ∑ j : Idx m, a j * Real.exp (β * (Real.cos (φ k - θ j) - 1)) :=
    Finset.sum_nonneg (fun j hj => mul_nonneg (ha j) (Real.exp_pos _).le)
  unfold frozenZ
  linarith

/-- Every original frozen-token partition is strictly positive under the original nonnegative center masses.
Source: arXiv:2411.04990v2, §5, the original normalized field in `thm:fixed_centers`. -/
theorem frozenZ_pos (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (φ : Idx n → ℝ) (k : Idx n) : 0 < frozenZ n m β θ a φ k :=
  lt_of_lt_of_le one_pos (one_le_frozenZ β θ a ha φ k)

/-- The original masked particle weights are nonnegative at every original angular state.
Source: arXiv:2411.04990v2, §5, the actual causal interactions of `thm:fixed_centers`. -/
theorem frozenParticleWeight_nonneg (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (φ : Idx n → ℝ) (k j : Idx n) : 0 ≤ frozenParticleWeight β θ a φ k j := by
  apply div_nonneg _ (frozenZ_pos β θ a ha φ k).le
  split_ifs <;> positivity

/-- The original normalized frozen-center weights retain the nonnegativity of their original center masses.
Source: arXiv:2411.04990v2, §5, the actual frozen-center interactions of `thm:fixed_centers`. -/
theorem frozenCenterWeight_nonneg (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (φ : Idx n → ℝ) (k : Idx n) (j : Idx m) : 0 ≤ frozenCenterWeight β θ a φ k j :=
  div_nonneg (mul_nonneg (ha j) (Real.exp_pos _).le) (frozenZ_pos β θ a ha φ k).le

/-- The actual original particle and frozen-center masses jointly form one normalized probability row.
Source: arXiv:2411.04990v2, §5, the exact normalization in `thm:fixed_centers`. -/
theorem sum_frozenWeights (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (φ : Idx n → ℝ) (k : Idx n) :
    (∑ j : Idx n, frozenParticleWeight β θ a φ k j) +
      (∑ j : Idx m, frozenCenterWeight β θ a φ k j) = 1 := by
  unfold frozenParticleWeight frozenCenterWeight
  rw [← Finset.sum_div, ← Finset.sum_div, ← add_div, ← Finset.sum_filter]
  exact div_self (frozenZ_pos β θ a ha φ k).ne'

/-- The actual original frozen velocity is exactly the actual original weighted sine interaction sum.
Source: arXiv:2411.04990v2, §5, the equation for `thm:fixed_centers`.
The original mask and both kinds of original normalized weights are retained. -/
theorem frozenVelocity_eq_weights (β : ℝ) (θ a : Idx m → ℝ) (φ : Idx n → ℝ) (k : Idx n) :
    frozenVelocity n m β θ a φ k =
      (∑ j : Idx n, frozenParticleWeight β θ a φ k j * Real.sin (φ j - φ k)) +
        (∑ j : Idx m, frozenCenterWeight β θ a φ k j * Real.sin (θ j - φ k)) := by
  unfold frozenVelocity frozenParticleWeight frozenCenterWeight
  rw [Finset.sum_filter, mul_add, Finset.mul_sum, Finset.mul_sum]
  congr 1
  · apply Finset.sum_congr rfl
    intro j hj
    by_cases h : j ≤ k
    · simp only [ite_eq_left h, div_eq_mul_inv]
      ring
    · simp only [ite_eq_right h, zero_mul, zero_div, mul_zero]
  · apply Finset.sum_congr rfl
    intro j hj
    rw [div_eq_mul_inv]
    ring

/-- Every original frozen angular velocity has absolute value at most one at every lifted state.
Source: arXiv:2411.04990v2, §5, the bounded original normalized field of `thm:fixed_centers`. -/
theorem abs_frozenVelocity_le_one (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (φ : Idx n → ℝ) (k : Idx n) : |frozenVelocity n m β θ a φ k| ≤ 1 := by
  rw [frozenVelocity_eq_weights]
  calc
    _ ≤ (∑ j : Idx n, |frozenParticleWeight β θ a φ k j * Real.sin (φ j - φ k)|) +
        (∑ j : Idx m, |frozenCenterWeight β θ a φ k j * Real.sin (θ j - φ k)|) :=
      (abs_add_le _ _).trans (add_le_add (Finset.abs_sum_le_sum_abs _ _) (Finset.abs_sum_le_sum_abs _ _))
    _ ≤ (∑ j : Idx n, frozenParticleWeight β θ a φ k j) +
        (∑ j : Idx m, frozenCenterWeight β θ a φ k j) := by
      apply add_le_add
      · apply Finset.sum_le_sum
        intro j hj
        rw [abs_mul, abs_of_nonneg (frozenParticleWeight_nonneg β θ a ha φ k j)]
        simpa only [mul_one] using mul_le_mul_of_nonneg_left (Real.abs_sin_le_one (x := φ j - φ k))
          (frozenParticleWeight_nonneg β θ a ha φ k j)
      · apply Finset.sum_le_sum
        intro j hj
        rw [abs_mul, abs_of_nonneg (frozenCenterWeight_nonneg β θ a ha φ k j)]
        simpa only [mul_one] using mul_le_mul_of_nonneg_left (Real.abs_sin_le_one (x := θ j - φ k))
          (frozenCenterWeight_nonneg β θ a ha φ k j)
    _ = 1 := sum_frozenWeights β θ a ha φ k

/-- The original full frozen velocity is uniformly bounded in the original angular maximum norm.
Source: arXiv:2411.04990v2, §5, the bounded original flow of `thm:fixed_centers`. -/
theorem norm_frozenVelocity_le_one (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (φ : Idx n → ℝ) : ‖frozenVelocity n m β θ a φ‖ ≤ 1 := by
  apply (pi_norm_le_iff_of_nonneg zero_le_one).mpr
  intro k
  simpa only [Real.norm_eq_abs] using abs_frozenVelocity_le_one β θ a ha φ k

/-- At the original nonnegative temperature, the actual partition is at most the original total mass.
Source: arXiv:2411.04990v2, §5, `N = n + sum a` in `thm:fixed_centers`. -/
theorem frozenZ_le_total_mass (β : ℝ) (hβ : 0 ≤ β) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (φ : Idx n → ℝ) (k : Idx n) : frozenZ n m β θ a φ k ≤ (n : ℝ) + ∑ j, a j := by
  have he (s : ℝ) : Real.exp (β * (Real.cos s - 1)) ≤ 1 :=
    Real.exp_le_one_iff.mpr (mul_nonpos_of_nonneg_of_nonpos hβ (sub_nonpos.mpr (Real.cos_le_one s)))
  unfold frozenZ
  rw [Finset.sum_filter]
  calc
    _ ≤ (∑ _j : Idx n, (1 : ℝ)) + ∑ j : Idx m, a j := by
      apply add_le_add
      · apply Finset.sum_le_sum
        intro j hj
        split_ifs
        · exact he _
        · exact zero_le_one
      · apply Finset.sum_le_sum
        intro j hj
        simpa only [mul_one] using mul_le_mul_of_nonneg_left (he (φ k - θ j)) (ha j)
    _ = _ := by simp

/-- Nonzero frozen-center masses and an actual active original particle row jointly realize every mass hypothesis.
Source: arXiv:2411.04990v2, §5, the two frozen antipodal centers in `thm:fixed_centers`. -/
example : (∀ j : Idx 2, (0 : ℝ) ≤ (fun _ : Idx 2 => (1 : ℝ)) j) ∧ (0 : ℝ) ≤ 11 ∧
    (0 : Idx 1) ≤ 0 ∧ (0 : ℝ) < frozenZ 1 2 11 (![0, Real.pi]) (fun _ => 1) (fun _ => 0) 0 := by
  refine ⟨fun _ => zero_le_one, by norm_num, le_rfl, ?_⟩
  exact frozenZ_pos 11 (![0, Real.pi]) (fun _ => 1) (fun _ => zero_le_one) (fun _ => 0) 0

end Transformer.Causal
