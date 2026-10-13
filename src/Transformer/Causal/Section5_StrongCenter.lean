/-
# The strong-center displacement comparison

ArXiv:2411.04990v2, §5, `lemma:meta`, `eq:renyi`, and `eq:time_bd`.
The finite minimum of the actual squared chord gaps stays above the
interaction radius. The center velocity is therefore bounded throughout
the interval, and the mean-value inequality bounds its displacement.
-/

import Transformer.Causal.Section5_MinCurve
import Transformer.Causal.Section5_VelocityBounds

open scoped BigOperators
open Real

namespace Transformer.Causal

variable {n : ℕ}

/-- **The displacement comparison for a strong center with predecessors.**
The separation and time conditions are `eq:renyi` and `eq:time_bd`, with
`r = c β⁻¹ᐟ²`. Source: arXiv:2411.04990v2, §5, `lemma:meta` and its proof.
The conclusion retains the intermediate bound `k h(r) t`. -/
theorem strong_center_displacement_le (β : ℝ) (hβ : 0 < β) (r ε : ℝ)
    (hr : 0 < r) (hε : 0 < ε) (hrπ : r ≤ π)
    (hrpeak : Real.arccos ((-1 + Real.sqrt (4 * β ^ 2 + 1)) / (2 * β)) ≤ r)
    (X : ℝ → SphereTuple 2 n)
    (hX : CSA 2 n β (ContinuousLinearMap.id ℝ (EucSpace 2))
      (ContinuousLinearMap.id ℝ (EucSpace 2))
      (ContinuousLinearMap.id ℝ (EucSpace 2)) X)
    (k : Idx n) (hk : 0 < (k : ℕ))
    (hsep : ∀ i : Idx n, i < k →
      r * (1 + 2 * ε) < ‖(X 0 k : EucSpace 2) - (X 0 i : EucSpace 2)‖)
    (T : ℝ) (hT : 0 < T) (hbudget : T * (k : ℝ) * h_pot β r < ε * r) :
    ∀ t ∈ Set.Icc (0 : ℝ) T,
      ‖(X t k : EucSpace 2) - (X 0 k : EucSpace 2)‖ ≤ (k : ℝ) * h_pot β r * t := by
  let : Nonempty (Fin (k : ℕ)) := ⟨⟨0, hk⟩⟩
  let p (i : Fin (k : ℕ)) : Idx n := ⟨i, i.isLt.trans k.isLt⟩
  let f (i : Fin (k : ℕ)) (t : ℝ) := ‖(X t k : EucSpace 2) - (X t (p i) : EucSpace 2)‖ ^ 2
  let f' (i : Fin (k : ℕ)) (t : ℝ) := 2 *
    inner (𝕜 := ℝ) ((X t k : EucSpace 2) - (X t (p i) : EucSpace 2))
      (circleVelocity β (X t) k - circleVelocity β (X t) (p i))
  have hd (i : Fin (k : ℕ)) (t : ℝ) : HasDerivAt (f i) (f' i t) t :=
    ((hX.hasDerivAt_circleVelocity β X t k).sub
      (hX.hasDerivAt_circleVelocity β X t (p i))).norm_sq
  have hR : r < r * (1 + 2 * ε) := by nlinarith
  have htime : 2 * ((k : ℝ) * h_pot β r) * T < r * (1 + 2 * ε) - r := by
    nlinarith
  have hinit (i : Fin (k : ℕ)) : (r * (1 + 2 * ε)) ^ 2 ≤ f i 0 := by
    have hi : p i < k := i.isLt
    have hs := hsep (p i) hi
    dsimp only [f]
    nlinarith [norm_nonneg ((X 0 k : EucSpace 2) - (X 0 (p i) : EucSpace 2))]
  have hbound : ∀ t ∈ Set.Ico (0 : ℝ) T, ∀ i,
      (∀ j, f i t ≤ f j t) → r ^ 2 ≤ f i t →
        -4 * ((k : ℝ) * h_pot β r) * Real.sqrt (f i t) ≤ f' i t := by
    intro t _ i hmin hgap
    have hg : r ≤ ‖(X t k : EucSpace 2) - (X t (p i) : EucSpace 2)‖ := by
      exact (sq_le_sq₀ hr.le (norm_nonneg _)).mp hgap
    have hnearest : ∀ j : Idx n, j < k →
        ‖(X t k : EucSpace 2) - (X t (p i) : EucSpace 2)‖ ≤
          ‖(X t k : EucSpace 2) - (X t j : EucSpace 2)‖ := by
      intro j hj
      have hp : p ⟨j, hj⟩ = j := rfl
      have h := hmin ⟨j, hj⟩
      dsimp only [f] at h
      rw [hp] at h
      exact (sq_le_sq₀ (norm_nonneg _) (norm_nonneg _)).mp h
    dsimp only [f, f']
    rw [Real.sqrt_sq (norm_nonneg _)]
    exact circle_squared_gap_deriv_lower β hβ r hrπ hrpeak (X t) k (p i) i.isLt hg hnearest
  have hgaps := min_sq_curve_ge f f' r (r * (1 + 2 * ε)) ((k : ℝ) * h_pot β r) T
    hr hR hT htime hd hinit hbound
  have hgap : ∀ t ∈ Set.Icc (0 : ℝ) T, ∀ j : Idx n, j < k →
      r ≤ ‖(X t k : EucSpace 2) - (X t j : EucSpace 2)‖ := by
    intro t ht j hj
    have h := hgaps t ht ⟨j, hj⟩
    exact (sq_le_sq₀ hr.le (norm_nonneg _)).mp h
  have hdisplacement := norm_image_sub_le_of_norm_deriv_le_segment'
    (fun t (_ : t ∈ Set.Icc (0 : ℝ) T) =>
      (hX.hasDerivAt_circleVelocity β X t k).hasDerivWithinAt)
    (fun t ht => norm_circleVelocity_le β hβ r hrπ hrpeak (X t) k
      (hgap t ⟨ht.1, ht.2.le⟩))
  simpa only [sub_zero] using hdisplacement

/-- **Lemma `lemma:meta` for one actual token index.** The interaction
radius is `c β⁻¹ᐟ²`, exactly as printed. The zero-predecessor case is
stationary; otherwise separation itself places the radius below `2 < π`.
Source: arXiv:2411.04990v2, §5, `lemma:meta`, `eq:c`, `eq:renyi`, and
`eq:time_bd`. The hypotheses are witnessed in `Section5_StrongCenterWitness`. -/
theorem strong_center_displacement_at_scale (β c ε : ℝ) (hβ : 0 < β) (hc : 0 < c) (hε : 0 < ε)
    (X : ℝ → SphereTuple 2 n)
    (hX : CSA 2 n β (ContinuousLinearMap.id ℝ (EucSpace 2))
      (ContinuousLinearMap.id ℝ (EucSpace 2))
      (ContinuousLinearMap.id ℝ (EucSpace 2)) X)
    (k : Idx n)
    (hstrong : ∀ i : Idx n, i < k → c * (1 + 2 * ε) * β ^ (-(1 / 2 : ℝ)) <
      ‖(X 0 k : EucSpace 2) - (X 0 i : EucSpace 2)‖)
    (h_c : Real.sqrt β * Real.arccos ((-1 + Real.sqrt (4 * β ^ 2 + 1)) / (2 * β)) < c)
    (T : ℝ) (hT : T * (k : ℝ) * h_pot β (c * β ^ (-(1 / 2 : ℝ))) <
      ε * c * β ^ (-(1 / 2 : ℝ))) :
    ∀ t : ℝ, 0 ≤ t → t ≤ T →
      ‖(X t k : EucSpace 2) - (X 0 k : EucSpace 2)‖ < ε * c * β ^ (-(1 / 2 : ℝ)) := by
  let r := c * β ^ (-(1 / 2 : ℝ))
  have hr : 0 < r := mul_pos hc (Real.rpow_pos_of_pos hβ _)
  have hεr : 0 < ε * r := mul_pos hε hr
  have hT' : T * (k : ℝ) * h_pot β r < ε * r := by simpa only [r, mul_assoc] using hT
  intro t ht htT
  rw [mul_assoc]
  change ‖(X t k : EucSpace 2) - (X 0 k : EucSpace 2)‖ < ε * r
  by_cases ht0 : t = 0
  · simpa [ht0] using hεr
  have hTpos : 0 < T := (lt_of_le_of_ne ht (Ne.symm ht0)).trans_le htT
  by_cases hk0 : (k : ℕ) = 0
  · have hv (u : ℝ) : circleVelocity β (X u) k = 0 := by
      rw [circleVelocity_eq_predecessors]
      have hempty : (Finset.univ.filter (fun j : Idx n => j < k)) = ∅ := by
        apply Finset.filter_eq_empty_iff.mpr
        intro j _
        simp [Fin.lt_def, hk0]
      rw [hempty, Finset.sum_empty]
    have hn := norm_image_sub_le_of_norm_deriv_le_segment'
      (fun u (_ : u ∈ Set.Icc (0 : ℝ) T) =>
        (hX.hasDerivAt_circleVelocity β X u k).hasDerivWithinAt)
      (C := 0) (fun u _ => by rw [hv u, norm_zero]) t ⟨ht, htT⟩
    exact (by simpa using hn : ‖(X t k : EucSpace 2) - (X 0 k : EucSpace 2)‖ ≤ 0).trans_lt hεr
  have hk : 0 < (k : ℕ) := Nat.pos_of_ne_zero hk0
  have hsep : ∀ i : Idx n, i < k →
      r * (1 + 2 * ε) < ‖(X 0 k : EucSpace 2) - (X 0 i : EucSpace 2)‖ := by
    intro i hi
    simpa only [r, mul_right_comm] using hstrong i hi
  let i₀ : Idx n := ⟨0, hk.trans k.isLt⟩
  have hsep0 := hsep i₀ hk
  have hd2 : ‖(X 0 k : EucSpace 2) - (X 0 i₀ : EucSpace 2)‖ ≤ 2 := by
    simpa only [mem_sphere_zero_iff_norm.mp (X 0 k).2,
      mem_sphere_zero_iff_norm.mp (X 0 i₀).2, one_add_one_eq_two] using
        norm_sub_le (X 0 k : EucSpace 2) (X 0 i₀ : EucSpace 2)
  have hrπ : r ≤ π := by nlinarith [Real.pi_gt_three]
  have hrrepr : r = c / Real.sqrt β := by
    dsimp only [r]
    rw [Real.rpow_neg hβ.le, ← Real.sqrt_eq_rpow, div_eq_mul_inv]
  have hrpeak : Real.arccos ((-1 + Real.sqrt (4 * β ^ 2 + 1)) / (2 * β)) ≤ r := by
    rw [hrrepr]
    apply le_of_lt
    apply (lt_div_iff₀ (Real.sqrt_pos.mpr hβ)).mpr
    simpa only [mul_comm] using h_c
  have hdisp := strong_center_displacement_le β hβ r ε hr hε hrπ hrpeak X hX k hk hsep T
    hTpos hT' t ⟨ht, htT⟩
  have hK : 0 ≤ (k : ℝ) * h_pot β r := by
    apply mul_nonneg (Nat.cast_nonneg _)
    exact mul_nonneg (Real.exp_pos _).le
      (Real.sin_nonneg_of_mem_Icc ⟨hr.le, hrπ⟩)
  have htime := mul_le_mul_of_nonneg_left htT hK
  exact hdisp.trans_lt (by nlinarith)

end Transformer.Causal
