module

public import Transformer.Causal.Section5_InteractionDecay
public import Mathlib.Analysis.InnerProductSpace.Calculus
public import Mathlib.Data.Fintype.Fin

/-!
# The actual normalized circle vector field

ArXiv:2411.04990v2, §5, proof of `lemma:meta`, `eq:bddotphi`.
Dividing the numerator and denominator of causal attention by `exp β`
gives the source's normalization `Z_k ≥ 1`. The self term contributes
one to the denominator and zero to the projected velocity.
Indices are zero-based: the index `k` has exactly `k` strict predecessors.
-/

@[expose] public section

open scoped BigOperators
open Real

namespace Transformer.Causal

variable {n : ℕ}

/-- The shifted normalization in the proof of `lemma:meta`.
Source: arXiv:2411.04990v2, §5, the display `Z_k ≥ 1`. -/
noncomputable def circleZ (β : ℝ) (x : SphereTuple 2 n) (k : Idx n) : ℝ :=
  ∑ j : Idx n, if j ≤ k then
    Real.exp (β * (inner (𝕜 := ℝ) (x k : EucSpace 2) (x j : EucSpace 2) - 1)) else 0

/-- The actual causal softmax weight, after cancelling `exp β`.
Source: arXiv:2411.04990v2, §5, proof of `lemma:meta`. -/
noncomputable def circleWeight (β : ℝ) (x : SphereTuple 2 n) (k j : Idx n) : ℝ :=
  if j ≤ k then (circleZ β x k)⁻¹ *
    Real.exp (β * (inner (𝕜 := ℝ) (x k : EucSpace 2) (x j : EucSpace 2) - 1)) else 0

/-- The projected vector velocity of the actual circle tokens.
Source: arXiv:2411.04990v2, `eqn:csa`, at `Q = K = V = I₂`. -/
noncomputable def circleVelocity (β : ℝ) (x : SphereTuple 2 n) (k : Idx n) : EucSpace 2 :=
  ∑ j : Idx n, circleWeight β x k j • proj 2 (x k : EucSpace 2) (x j : EucSpace 2)

/-- The self-attention term makes the shifted denominator at least one.
Source: arXiv:2411.04990v2, §5, proof of `lemma:meta`, `Z_k ≥ 1`. -/
theorem one_le_circleZ (β : ℝ) (x : SphereTuple 2 n) (k : Idx n) :
    1 ≤ circleZ β x k := by
  have h := Finset.single_le_sum (s := Finset.univ)
    (f := fun j : Idx n => if j ≤ k then
      Real.exp (β * (inner (𝕜 := ℝ) (x k : EucSpace 2) (x j : EucSpace 2) - 1)) else 0)
    (fun j _ => by split_ifs <;> positivity) (Finset.mem_univ k)
  simpa only [circleZ, le_refl, ite_true, real_inner_self_eq_norm_sq,
    mem_sphere_zero_iff_norm.mp (x k).2, one_pow, sub_self, mul_zero, Real.exp_zero] using h

/-- Every actual causal weight is nonnegative.
Source: arXiv:2411.04990v2, `eqn:csa`, the softmax weights. -/
theorem circleWeight_nonneg (β : ℝ) (x : SphereTuple 2 n) (k j : Idx n) :
    0 ≤ circleWeight β x k j := by
  have hZ : 0 < circleZ β x k := lt_of_lt_of_le one_pos (one_le_circleZ β x k)
  unfold circleWeight
  split_ifs <;> positivity

/-- Normalization can only reduce the shifted interaction weight.
Source: arXiv:2411.04990v2, §5, `eq:bddotphi`, using `Z_k ≥ 1`. -/
theorem circleWeight_le_exp (β : ℝ) (x : SphereTuple 2 n) (k j : Idx n) :
    circleWeight β x k j ≤
      Real.exp (β * (inner (𝕜 := ℝ) (x k : EucSpace 2) (x j : EucSpace 2) - 1)) := by
  have hZ := one_le_circleZ β x k
  unfold circleWeight
  split_ifs
  · exact mul_le_of_le_one_left (Real.exp_pos _).le
      ((inv_le_one₀ (lt_of_lt_of_le one_pos hZ)).mpr hZ)
  · exact (Real.exp_pos _).le

/-- The self projection vanishes, so only strict predecessors contribute.
Source: arXiv:2411.04990v2, §5, the angular equation preceding `eq:bddotphi`. -/
theorem circleVelocity_eq_predecessors (β : ℝ) (x : SphereTuple 2 n) (k : Idx n) :
    circleVelocity β x k = ∑ j ∈ Finset.univ.filter (fun j : Idx n => j < k),
      circleWeight β x k j • proj 2 (x k : EucSpace 2) (x j : EucSpace 2) := by
  rw [Finset.sum_filter]
  apply Finset.sum_congr rfl
  intro j _
  by_cases hj : j < k
  · simp only [hj, ite_true]
  · rcases eq_or_lt_of_le (le_of_not_gt hj) with rfl | hkj
    · have hp : proj 2 (x k : EucSpace 2) (x k : EucSpace 2) = 0 := by
        simpa using proj_smul_self (mem_sphere_zero_iff_norm.mp (x k).2) 1
      simp [hp]
    · simp [circleWeight, not_le.mpr hkj, hj]

/-- The shifted vector field is exactly the derivative in `CSA`.
Source: arXiv:2411.04990v2, `eqn:csa` and §5, proof of `lemma:meta`.
This identifies the actual dynamics rather than assuming the angular form. -/
theorem CSA.hasDerivAt_circleVelocity (β : ℝ) (X : ℝ → SphereTuple 2 n)
    (hX : CSA 2 n β (ContinuousLinearMap.id ℝ (EucSpace 2))
      (ContinuousLinearMap.id ℝ (EucSpace 2))
      (ContinuousLinearMap.id ℝ (EucSpace 2)) X) (t : ℝ) (k : Idx n) :
    HasDerivAt (fun s => (X s k : EucSpace 2)) (circleVelocity β (X t) k) t := by
  have hd := hX t k
  simp only [ContinuousLinearMap.id_apply] at hd
  have he (j : Idx n) :
      Real.exp (β * inner (𝕜 := ℝ) (X t k : EucSpace 2) (X t j : EucSpace 2)) =
        Real.exp β * Real.exp
          (β * (inner (𝕜 := ℝ) (X t k : EucSpace 2) (X t j : EucSpace 2) - 1)) := by
    rw [← Real.exp_add]
    congr 1
    ring
  have hZ : (∑ j : Idx n, if (j : ℕ) ≤ (k : ℕ) then
      Real.exp (β * inner (𝕜 := ℝ) (X t k : EucSpace 2) (X t j : EucSpace 2)) else 0) =
        Real.exp β * circleZ β (X t) k := by
    rw [circleZ, Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro j _
    by_cases hj : j ≤ k
    · simp [hj, show (j : ℕ) ≤ (k : ℕ) from hj, he]
    · simp [hj, show ¬(j : ℕ) ≤ (k : ℕ) from hj]
  rw [hZ] at hd
  convert hd using 1
  unfold circleVelocity circleWeight proj
  rw [inner_smul_right, inner_sum, ← smul_smul, Finset.sum_smul,
    Finset.smul_sum, Finset.smul_sum,
    ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro j _
  by_cases hj : j ≤ k
  · simp only [hj, ite_true, show (j : ℕ) ≤ (k : ℕ) from hj, he,
      inner_smul_right, smul_smul, mul_inv_rev]
    have hcancel : (circleZ β (X t) k)⁻¹ * (Real.exp β)⁻¹ *
        (Real.exp β * Real.exp
          (β * (inner (𝕜 := ℝ) (X t k : EucSpace 2) (X t j : EucSpace 2) - 1))) =
      (circleZ β (X t) k)⁻¹ * Real.exp
        (β * (inner (𝕜 := ℝ) (X t k : EucSpace 2) (X t j : EucSpace 2) - 1)) := by
      field_simp
    rw [hcancel]
    simp only [smul_sub, smul_smul]
    congr 1
    congr 1
    field_simp
  · simp [hj, show ¬(j : ℕ) ≤ (k : ℕ) from hj]

/-- The identification hypotheses hold for the stationary one-token trajectory.
Source: arXiv:2411.04990v2, `eqn:csa`. -/
example (β : ℝ) (x : SSphere 2) :
    CSA 2 1 β (ContinuousLinearMap.id ℝ (EucSpace 2))
      (ContinuousLinearMap.id ℝ (EucSpace 2))
      (ContinuousLinearMap.id ℝ (EucSpace 2)) (fun _ _ => x) := by
  intro t k
  have hp : proj 2 (x : EucSpace 2) (x : EucSpace 2) = 0 := by
    simpa using proj_smul_self (mem_sphere_zero_iff_norm.mp x.2) 1
  have he : Real.exp (β * inner (𝕜 := ℝ) (x : EucSpace 2) (x : EucSpace 2)) ≠ 0 :=
    Real.exp_ne_zero _
  simpa [Subsingleton.elim k (0 : Idx 1), smul_smul, he, hp] using
    hasDerivAt_const t (x : EucSpace 2)

end Transformer.Causal
