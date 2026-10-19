module

public import Transformer.Causal.Section5_InteractionDecay
public import Mathlib.Order.Interval.Finset.Fin
public import Mathlib.Algebra.Order.BigOperators.Group.LocallyFinite

/-!
# The actual normalized causal velocity and its separated-token bound

ArXiv:2411.04990v2, §5, `lemma:meta` and `eq:bddotphi`.
The self-attention term is one after division by `exp β`. Hence the
normalized partition function is at least one, and the self-projection
vanishes. The remaining `k` projected interactions give the speed bound
`k h(r)` whenever the token is chord-separated by `r` from its predecessors.

The field below is identified with the original spherical equation `CSA`.
The change of normalization is proved by cancellation, and no speed or
separation assumption is placed in the definition of the field.
-/

@[expose] public section

open scoped BigOperators
open Real

namespace Transformer.Causal

variable {n : ℕ}

/-- The normalized causal partition function, including the unit self term.
Source: arXiv:2411.04990v2, §5, `eqn:csa-2d` and proof of `lemma:meta`. -/
noncomputable def causalNormalizer (β : ℝ) (X : SphereTuple 2 n) (k : Idx n) : ℝ :=
  1 + ∑ j ∈ Finset.Iio k,
    Real.exp (β * (inner ℝ (X k : EucSpace 2) (X j : EucSpace 2) - 1))

/-- The actual causal velocity after cancellation of `exp β` and removal
of the zero self-projection. Source: arXiv:2411.04990v2, §5, `eq:bddotphi`. -/
noncomputable def causalDrift (β : ℝ) (X : SphereTuple 2 n) (k : Idx n) : EucSpace 2 :=
  (causalNormalizer β X k)⁻¹ • ∑ j ∈ Finset.Iio k,
    Real.exp (β * (inner ℝ (X k : EucSpace 2) (X j : EucSpace 2) - 1)) •
      proj 2 (X k : EucSpace 2) (X j : EucSpace 2)

/-- **The self term bounds the normalized partition function below by one.**
Source: arXiv:2411.04990v2, §5, proof of `lemma:meta`, before `eq:bddotphi`. -/
theorem one_le_causalNormalizer (β : ℝ) (X : SphereTuple 2 n) (k : Idx n) :
    1 ≤ causalNormalizer β X k := by
  have hs : 0 ≤ ∑ j ∈ Finset.Iio k,
      Real.exp (β * (inner ℝ (X k : EucSpace 2) (X j : EucSpace 2) - 1)) :=
    Finset.sum_nonneg fun _ _ => (Real.exp_pos _).le
  unfold causalNormalizer
  linarith

/-- The inverse normalized partition function lies in `[0,1]`.
Source: arXiv:2411.04990v2, §5, proof of `lemma:meta`, `eq:bddotphi`. -/
theorem inv_causalNormalizer_bounds (β : ℝ) (X : SphereTuple 2 n) (k : Idx n) :
    0 ≤ (causalNormalizer β X k)⁻¹ ∧ (causalNormalizer β X k)⁻¹ ≤ 1 := by
  have hZ := one_le_causalNormalizer β X k
  have hp : 0 < causalNormalizer β X k := by linarith
  exact ⟨inv_nonneg.mpr hp.le, (inv_le_one₀ hp).mpr hZ⟩

/-- **The normalized vector field is the original `CSA` field.**
Every omitted self term has zero tangent projection. Source:
arXiv:2411.04990v2, §5, `lemma:meta`, and §2, `eqn:csa`, at `Q=K=V=I₂`. -/
theorem causalDrift_eq_original (β : ℝ) (X : SphereTuple 2 n) (k : Idx n) :
    proj 2 (X k : EucSpace 2)
      ((∑ j : Idx n, if (j : ℕ) ≤ (k : ℕ) then
          Real.exp (β * inner ℝ (X k : EucSpace 2) (X j : EucSpace 2)) else 0)⁻¹ •
        ∑ j : Idx n, if (j : ℕ) ≤ (k : ℕ) then
          Real.exp (β * inner ℝ (X k : EucSpace 2) (X j : EucSpace 2)) •
            (X j : EucSpace 2) else 0) = causalDrift β X k := by
  let q (j : Idx n) := Real.exp
    (β * (inner ℝ (X k : EucSpace 2) (X j : EucSpace 2) - 1))
  have hself : q k = 1 := by
    dsimp [q]
    rw [real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp (X k).2]
    norm_num
  have hweight (j : Idx n) :
      Real.exp (β * inner ℝ (X k : EucSpace 2) (X j : EucSpace 2)) = Real.exp β * q j := by
    dsimp only [q]
    rw [← Real.exp_add]
    congr 1
    ring
  have hfilter : Finset.univ.filter (fun j : Idx n => (j : ℕ) ≤ (k : ℕ)) = Finset.Iic k := by
    ext j
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_Iic, Fin.le_def]
  rw [← Finset.sum_filter, ← Finset.sum_filter, hfilter]
  simp_rw [hweight]
  rw [← Finset.mul_sum]
  have hsum : ∑ j ∈ Finset.Iic k, q j = causalNormalizer β X k := by
    rw [← Finset.sum_Iio_add_eq_sum_Iic k, hself, causalNormalizer]
    dsimp [q]
    ring
  rw [hsum]
  have hvec : ∑ j ∈ Finset.Iic k, (Real.exp β * q j) • (X j : EucSpace 2) =
      Real.exp β • ∑ j ∈ Finset.Iic k, q j • (X j : EucSpace 2) := by
    simp only [← smul_smul, Finset.smul_sum]
  rw [hvec, mul_inv, smul_smul]
  have hcancel : (Real.exp β)⁻¹ * (causalNormalizer β X k)⁻¹ * Real.exp β =
      (causalNormalizer β X k)⁻¹ := by
    field_simp
  rw [hcancel]
  have hproj_smul (a : ℝ) (x y : EucSpace 2) : proj 2 x (a • y) = a • proj 2 x y := by
    simp only [proj, real_inner_smul_right, smul_sub, smul_smul]
  rw [hproj_smul]
  have hproj_sum : proj 2 (X k : EucSpace 2) (∑ j ∈ Finset.Iic k, q j • (X j : EucSpace 2)) =
      ∑ j ∈ Finset.Iic k, q j • proj 2 (X k : EucSpace 2) (X j : EucSpace 2) := by
    simp only [proj, inner_sum, real_inner_smul_right, Finset.sum_smul, smul_sub,
      smul_smul, Finset.sum_sub_distrib]
  rw [hproj_sum, ← Finset.sum_Iio_add_eq_sum_Iic k, hself]
  have hzero : proj 2 (X k : EucSpace 2) (X k : EucSpace 2) = 0 := by
    simpa only [one_smul] using proj_smul_self (mem_sphere_zero_iff_norm.mp (X k).2) 1
  simp only [hzero, smul_zero, add_zero, causalDrift, q]

/-- Every actual spherical `CSA` trajectory has this normalized derivative.
Source: arXiv:2411.04990v2, §5, `lemma:meta` and `eq:bddotphi`. -/
theorem hasDerivAt_causalDrift (β : ℝ) (X : ℝ → SphereTuple 2 n)
    (hX : CSA 2 n β (ContinuousLinearMap.id ℝ _) (ContinuousLinearMap.id ℝ _)
      (ContinuousLinearMap.id ℝ _) X) (t : ℝ) (k : Idx n) :
    HasDerivAt (fun s => (X s k : EucSpace 2)) (causalDrift β (X t) k) t := by
  have h := hX t k
  simp only [ContinuousLinearMap.id_apply] at h
  rwa [causalDrift_eq_original] at h

/-- A single actual token is stationary: the predecessor sum is empty.
Source: arXiv:2411.04990v2, §5, `lemma:meta`, with its first causal token. -/
theorem csa_const_one (β : ℝ) (x₀ : SSphere 2) :
    CSA 2 1 β (ContinuousLinearMap.id ℝ _) (ContinuousLinearMap.id ℝ _)
      (ContinuousLinearMap.id ℝ _) (fun _ _ => x₀) := by
  intro t k
  have hzero : causalDrift β (fun _ : Idx 1 => x₀) k = 0 := by
    rw [Subsingleton.elim k (0 : Idx 1)]
    simp [causalDrift]
  have hfield := causalDrift_eq_original β (fun _ : Idx 1 => x₀) k
  rw [hzero] at hfield
  apply (hasDerivAt_const t (x₀ : EucSpace 2)).congr_deriv
  simpa only [ContinuousLinearMap.id_apply] using hfield.symm

/-- The normalized derivative theorem's full dynamical input is supplied
by a nonzero stationary token on the actual circle. -/
example (β : ℝ) :
    CSA 2 1 β (ContinuousLinearMap.id ℝ _) (ContinuousLinearMap.id ℝ _)
      (ContinuousLinearMap.id ℝ _) (fun _ _ => basePoint 1) :=
  csa_const_one β (basePoint 1)

/-- **The actual center speed is at most `k h(r)` under predecessor separation.**
Source: arXiv:2411.04990v2, §5, `eq:bddotphi`. The lower gap is a chord
distance, and the peak and temperature hypotheses are exactly those used there. -/
theorem norm_causalDrift_le (β : ℝ) (hβ : 0 < β) (r : ℝ) (hrπ : r ≤ π)
    (hrpeak : Real.arccos ((-1 + Real.sqrt (4 * β ^ 2 + 1)) / (2 * β)) ≤ r)
    (X : SphereTuple 2 n) (k : Idx n)
    (hgap : ∀ j : Idx n, j < k → r ≤ ‖(X k : EucSpace 2) - (X j : EucSpace 2)‖) :
    ‖causalDrift β X k‖ ≤ (k : ℝ) * h_pot β r := by
  have hnum : ‖∑ j ∈ Finset.Iio k,
      Real.exp (β * (inner ℝ (X k : EucSpace 2) (X j : EucSpace 2) - 1)) •
        proj 2 (X k : EucSpace 2) (X j : EucSpace 2)‖ ≤ (k : ℝ) * h_pot β r := by
    refine (norm_sum_le _ _).trans ?_
    calc
      _ ≤ ∑ _j ∈ Finset.Iio k, h_pot β r := Finset.sum_le_sum fun j hj => by
        rw [norm_smul, Real.norm_eq_abs, abs_of_pos (Real.exp_pos _)]
        exact circle_weighted_projection_le β hβ r hrπ hrpeak (X k) (X j)
          (hgap j (Finset.mem_Iio.mp hj))
      _ = (k : ℝ) * h_pot β r := by
        rw [Finset.sum_const, Fin.card_Iio, nsmul_eq_mul]
  have hZ := inv_causalNormalizer_bounds β X k
  rw [causalDrift, norm_smul, Real.norm_eq_abs, abs_of_nonneg hZ.1]
  exact (mul_le_mul_of_nonneg_right hZ.2 (norm_nonneg _)).trans (by simpa using hnum)

/-- The speed hypotheses are satisfiable at the first actual token, where
the predecessor family is empty and the normalized velocity is zero. -/
example : (0 : ℝ) < 1 ∧ (2 : ℝ) ≤ π ∧
    Real.arccos ((-1 + Real.sqrt (4 * (1 : ℝ) ^ 2 + 1)) / (2 * 1)) ≤ 2 ∧
    (∀ j : Idx 1, j < 0 → (2 : ℝ) ≤
      ‖(basePoint 1 : EucSpace 2) - (basePoint 1 : EucSpace 2)‖) := by
  obtain ⟨τ, hτ0, hτπ, hcos, _, hτ1, _⟩ := h_pot_unimodal 1 le_rfl
  refine ⟨one_pos, by linarith [Real.pi_gt_three], ?_, fun j hj => ?_⟩
  · rw [← hcos, Real.arccos_cos hτ0.le hτπ.le, Real.one_rpow] at *
    linarith
  · change (j : ℕ) < 0 at hj
    omega

end Transformer.Causal
