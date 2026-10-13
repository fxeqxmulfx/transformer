/-
# Bounds for the actual center and nearest-predecessor velocities

ArXiv:2411.04990v2, §5, proof of `lemma:meta`, `eq:bddotphi`.
The actual vector norm of a center's velocity is at most `k h(r)`
when every preceding chord gap is at least `r` beyond the interaction
peak. For a nearest predecessor, only inward pulls matter, and circle
geometry puts those interactions beyond the same gap.
-/

import Transformer.Causal.Section5_VectorField

open scoped BigOperators
open Real

namespace Transformer.Causal

variable {n : ℕ}

/-- Each normalized projected interaction is bounded by `h(r)`.
Source: arXiv:2411.04990v2, §5, `eq:bddotphi`. -/
theorem circleWeight_mul_norm_proj_le (β : ℝ) (hβ : 0 < β) (r : ℝ)
    (hrπ : r ≤ π)
    (hrpeak : Real.arccos ((-1 + Real.sqrt (4 * β ^ 2 + 1)) / (2 * β)) ≤ r)
    (x : SphereTuple 2 n) (k j : Idx n)
    (hgap : r ≤ ‖(x k : EucSpace 2) - (x j : EucSpace 2)‖) :
    circleWeight β x k j * ‖proj 2 (x k : EucSpace 2) (x j : EucSpace 2)‖ ≤ h_pot β r := by
  exact (mul_le_mul_of_nonneg_right (circleWeight_le_exp β x k j) (norm_nonneg _)).trans
    (circle_weighted_projection_le β hβ r hrπ hrpeak (x k) (x j) hgap)

/-- **The actual center speed bound.** Only the `k` strict predecessors
contribute. Source: arXiv:2411.04990v2, §5, `eq:bddotphi`. -/
theorem norm_circleVelocity_le (β : ℝ) (hβ : 0 < β) (r : ℝ)
    (hrπ : r ≤ π)
    (hrpeak : Real.arccos ((-1 + Real.sqrt (4 * β ^ 2 + 1)) / (2 * β)) ≤ r)
    (x : SphereTuple 2 n) (k : Idx n)
    (hgap : ∀ j : Idx n, j < k → r ≤ ‖(x k : EucSpace 2) - (x j : EucSpace 2)‖) :
    ‖circleVelocity β x k‖ ≤ (k : ℝ) * h_pot β r := by
  rw [circleVelocity_eq_predecessors]
  calc
    ‖∑ j ∈ Finset.univ.filter (fun j : Idx n => j < k),
        circleWeight β x k j • proj 2 (x k : EucSpace 2) (x j : EucSpace 2)‖
      ≤ ∑ j ∈ Finset.univ.filter (fun j : Idx n => j < k),
        ‖circleWeight β x k j • proj 2 (x k : EucSpace 2) (x j : EucSpace 2)‖ := norm_sum_le _ _
    _ ≤ ∑ j ∈ Finset.univ.filter (fun j : Idx n => j < k), h_pot β r := by
      apply Finset.sum_le_sum
      intro j hj
      rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg (circleWeight_nonneg β x k j)]
      exact circleWeight_mul_norm_proj_le β hβ r hrπ hrpeak x k j
        (hgap j (Finset.mem_filter.mp hj).2)
    _ = (k : ℝ) * h_pot β r := by
      simp only [Finset.sum_const, Fin.lt_def, Fin.card_filter_val_lt, Nat.min_eq_right k.isLt.le,
        nsmul_eq_mul]

/-- **A nearest predecessor cannot move inward faster than `i h(r)`.**
Every negative radial summand has a chord gap at least `r`, by the actual
two-dimensional Gram identity. Source: arXiv:2411.04990v2, §5, proof of
`lemma:meta`, the bound on `-φ̇_{r*}` following `eq:bddotphi`. -/
theorem inner_circleVelocity_lower_of_nearest (β : ℝ) (hβ : 0 < β) (r : ℝ)
    (hrπ : r ≤ π)
    (hrpeak : Real.arccos ((-1 + Real.sqrt (4 * β ^ 2 + 1)) / (2 * β)) ≤ r)
    (x : SphereTuple 2 n) (k i : Idx n) (hi : i < k)
    (hgap : r ≤ ‖(x k : EucSpace 2) - (x i : EucSpace 2)‖)
    (hnearest : ∀ j : Idx n, j < k →
      ‖(x k : EucSpace 2) - (x i : EucSpace 2)‖ ≤
        ‖(x k : EucSpace 2) - (x j : EucSpace 2)‖) :
    -‖(x i : EucSpace 2) - (x k : EucSpace 2)‖ * ((i : ℝ) * h_pot β r) ≤
      inner (𝕜 := ℝ) ((x i : EucSpace 2) - (x k : EucSpace 2)) (circleVelocity β x i) := by
  have hr0 : 0 ≤ r := (Real.arccos_nonneg _).trans hrpeak
  have hh : 0 ≤ h_pot β r := mul_nonneg (Real.exp_pos _).le
    (Real.sin_nonneg_of_mem_Icc ⟨hr0, hrπ⟩)
  rw [circleVelocity_eq_predecessors, inner_sum]
  have hterm : ∀ j : Idx n, j < i →
      -‖(x i : EucSpace 2) - (x k : EucSpace 2)‖ * h_pot β r ≤
        inner (𝕜 := ℝ) ((x i : EucSpace 2) - (x k : EucSpace 2))
          (circleWeight β x i j • proj 2 (x i : EucSpace 2) (x j : EucSpace 2)) := by
    intro j hj
    let q := inner (𝕜 := ℝ) ((x i : EucSpace 2) - (x k : EucSpace 2))
      (proj 2 (x i : EucSpace 2) (x j : EucSpace 2))
    rw [real_inner_smul_right]
    by_cases hq : 0 ≤ q
    · exact (mul_nonpos_of_nonpos_of_nonneg (neg_nonpos.mpr (norm_nonneg _)) hh).trans
        (mul_nonneg (circleWeight_nonneg β x i j) hq)
    · have hinner : inner (𝕜 := ℝ) (x k : EucSpace 2) (x j : EucSpace 2) ≤
          inner (𝕜 := ℝ) (x k : EucSpace 2) (x i : EucSpace 2) := by
        have hn := hnearest j (hj.trans hi)
        have h1 := norm_sub_sq_real (x k : EucSpace 2) (x i : EucSpace 2)
        have h2 := norm_sub_sq_real (x k : EucSpace 2) (x j : EucSpace 2)
        rw [mem_sphere_zero_iff_norm.mp (x k).2,
          mem_sphere_zero_iff_norm.mp (x i).2] at h1
        rw [mem_sphere_zero_iff_norm.mp (x k).2,
          mem_sphere_zero_iff_norm.mp (x j).2] at h2
        nlinarith [sq_le_sq₀ (norm_nonneg _) (norm_nonneg _) |>.mpr hn]
      have hinward : inner (𝕜 := ℝ) (x k : EucSpace 2) (x i : EucSpace 2) *
          inner (𝕜 := ℝ) (x i : EucSpace 2) (x j : EucSpace 2) <
          inner (𝕜 := ℝ) (x k : EucSpace 2) (x j : EucSpace 2) := by
        have hq' : q < 0 := lt_of_not_ge hq
        dsimp [q] at hq'
        rw [circle_chord_inner_proj] at hq'
        linarith
      have hji := inner_le_of_circle_inward (x k) (x i) (x j) hinner hinward
      have hgap' := hgap.trans (circle_chord_le_of_inner_le (x k) (x i) (x j) hji)
      have hb := circleWeight_mul_norm_proj_le β hβ r hrπ hrpeak x i j hgap'
      have hc := (abs_le.mp (abs_real_inner_le_norm
        ((x i : EucSpace 2) - (x k : EucSpace 2))
        (proj 2 (x i : EucSpace 2) (x j : EucSpace 2)))).1
      have hw := circleWeight_nonneg β x i j
      have hmul := mul_le_mul_of_nonneg_left hc hw
      have hbound := mul_le_mul_of_nonneg_left hb
        (norm_nonneg ((x i : EucSpace 2) - (x k : EucSpace 2)))
      dsimp [q] at *
      nlinarith
  calc
    -‖(x i : EucSpace 2) - (x k : EucSpace 2)‖ * ((i : ℝ) * h_pot β r)
      = ∑ j ∈ Finset.univ.filter (fun j : Idx n => j < i),
        -‖(x i : EucSpace 2) - (x k : EucSpace 2)‖ * h_pot β r := by
          simp only [Finset.sum_const, Fin.lt_def, Fin.card_filter_val_lt, Nat.min_eq_right i.isLt.le,
            nsmul_eq_mul]
          ring
    _ ≤ _ := Finset.sum_le_sum fun j hj => hterm j (Finset.mem_filter.mp hj).2

/-- **The nearest squared gap has the source's decay bound.**
Combining the center speed and inward predecessor speed gives
`(gap²)' ≥ -4 k h(r) gap`. Source: arXiv:2411.04990v2, §5, proof of
`lemma:meta`, the bound on `-Δ̇` following `eq:bddotphi`. -/
theorem circle_squared_gap_deriv_lower (β : ℝ) (hβ : 0 < β) (r : ℝ)
    (hrπ : r ≤ π)
    (hrpeak : Real.arccos ((-1 + Real.sqrt (4 * β ^ 2 + 1)) / (2 * β)) ≤ r)
    (x : SphereTuple 2 n) (k i : Idx n) (hi : i < k)
    (hgap : r ≤ ‖(x k : EucSpace 2) - (x i : EucSpace 2)‖)
    (hnearest : ∀ j : Idx n, j < k →
      ‖(x k : EucSpace 2) - (x i : EucSpace 2)‖ ≤
        ‖(x k : EucSpace 2) - (x j : EucSpace 2)‖) :
    -4 * ((k : ℝ) * h_pot β r) * ‖(x k : EucSpace 2) - (x i : EucSpace 2)‖ ≤
      2 * inner (𝕜 := ℝ) ((x k : EucSpace 2) - (x i : EucSpace 2))
        (circleVelocity β x k - circleVelocity β x i) := by
  have hv := norm_circleVelocity_le β hβ r hrπ hrpeak x k
    (fun j hj => hgap.trans (hnearest j hj))
  have hp := inner_circleVelocity_lower_of_nearest β hβ r hrπ hrpeak x k i hi hgap hnearest
  rw [norm_sub_rev (x i : EucSpace 2) (x k : EucSpace 2), inner_sub_left] at hp
  have hc := (abs_le.mp (abs_real_inner_le_norm
    ((x k : EucSpace 2) - (x i : EucSpace 2)) (circleVelocity β x k))).1
  have hm := mul_le_mul_of_nonneg_left hv
    (norm_nonneg ((x k : EucSpace 2) - (x i : EucSpace 2)))
  have hh : 0 ≤ h_pot β r := mul_nonneg (Real.exp_pos _).le
    (Real.sin_nonneg_of_mem_Icc ⟨(Real.arccos_nonneg _).trans hrpeak, hrπ⟩)
  have hik : (i : ℝ) ≤ (k : ℝ) := by exact_mod_cast hi.le
  have hcoeff := mul_le_mul_of_nonneg_right hik hh
  have hscale := mul_le_mul_of_nonneg_left hcoeff
    (norm_nonneg ((x k : EucSpace 2) - (x i : EucSpace 2)))
  simp only [inner_sub_right, inner_sub_left]
  rw [inner_sub_left] at hc
  nlinarith

/-- All velocity-bound hypotheses hold simultaneously for two antipodal
actual tokens, with `β = 1`, `r = 2`, center `k = 1` and predecessor `i = 0`.
Source: arXiv:2411.04990v2, §5, proof of `lemma:meta`. -/
example : ∃ x : SphereTuple 2 2, (0 : ℝ) < 1 ∧ (2 : ℝ) ≤ π ∧
    Real.arccos ((-1 + Real.sqrt (4 * (1 : ℝ) ^ 2 + 1)) / (2 * 1)) ≤ 2 ∧
    (0 : Idx 2) < 1 ∧
    (∀ j : Idx 2, j < 1 → (2 : ℝ) ≤ ‖(x 1 : EucSpace 2) - (x j : EucSpace 2)‖) ∧
    (∀ j : Idx 2, j < 1 → ‖(x 1 : EucSpace 2) - (x 0 : EucSpace 2)‖ ≤
      ‖(x 1 : EucSpace 2) - (x j : EucSpace 2)‖) := by
  let u : SSphere 2 := basePoint 1
  let v : SSphere 2 := ⟨-(u : EucSpace 2), by
    simp only [mem_sphere_zero_iff_norm, norm_neg, mem_sphere_zero_iff_norm.mp u.2]⟩
  have hd : ‖(v : EucSpace 2) - (u : EucSpace 2)‖ = 2 := by
    change ‖-(u : EucSpace 2) - (u : EucSpace 2)‖ = 2
    have heq : -(u : EucSpace 2) - (u : EucSpace 2) = (-2 : ℝ) • (u : EucSpace 2) := by
      module
    rw [heq, norm_smul, mem_sphere_zero_iff_norm.mp u.2]
    norm_num
  obtain ⟨τ, hτ0, hτπ, hcos, _, hτ1, _⟩ := h_pot_unimodal 1 le_rfl
  refine ⟨![u, v], one_pos, by linarith [Real.pi_gt_three], ?_, by decide, ?_, ?_⟩
  · rw [← hcos, Real.arccos_cos hτ0.le hτπ.le, Real.one_rpow] at *
    linarith
  · intro j hj
    have hj0 : j = 0 := by fin_cases j <;> simp_all
    simpa [hj0] using hd.symm.le
  · intro j hj
    have hj0 : j = 0 := by fin_cases j <;> simp_all
    rw [hj0]

end Transformer.Causal
