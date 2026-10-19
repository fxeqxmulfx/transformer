module

public import Transformer.Causal.Section5_ClosestDrift
public import Transformer.Causal.Section5_GapBarrier
public import Mathlib.Analysis.Calculus.MeanValue

/-!
# Strong causal centers retain their gaps and barely move

ArXiv:2411.04990v2, §5, proof of `lemma:meta`, the comparison after
`-Δ̇ ≤ 2s_k h(Δ)`. The proof uses squared chord distances of the actual
circle trajectory. At a first crossing the active gap is a closest gap,
so the verified relative-velocity estimate applies.

An affine barrier with slope strictly larger than `2k h(r)` avoids
assuming that a chosen closest predecessor is differentiable in time.
Its initial height fits below the strict strong-center separation.
-/

@[expose] public section

open Set Real

namespace Transformer.Causal

variable {n : ℕ}

/-- **Strong-center predecessor gaps stay above the interaction threshold.**
Source: arXiv:2411.04990v2, §5, proof of `lemma:meta`, the comparison
following `-Δ̇ ≤ 2s_k h(Δ)`. Here `k` is a zero-based index, hence exactly
the number of predecessors. We use chord distances throughout, with
the weighted-projection estimate proved in `Section5_InteractionDecay`. -/
theorem strong_center_gaps (β : ℝ) (hβ : 0 < β) (r ε T : ℝ)
    (hr : 0 < r) (hε : 0 < ε) (hT : 0 < T) (hrπ : r ≤ π)
    (hrpeak : Real.arccos ((-1 + Real.sqrt (4 * β ^ 2 + 1)) / (2 * β)) ≤ r)
    (X : ℝ → SphereTuple 2 n)
    (hX : CSA 2 n β (ContinuousLinearMap.id ℝ _) (ContinuousLinearMap.id ℝ _)
      (ContinuousLinearMap.id ℝ _) X) (k : Idx n)
    (hinit : ∀ i : Idx n, i < k →
      (1 + 2 * ε) * r < ‖(X 0 k : EucSpace 2) - (X 0 i : EucSpace 2)‖)
    (htime : T * (k : ℝ) * h_pot β r < ε * r) :
    ∀ t ∈ Icc (0 : ℝ) T, ∀ i : Idx n, i < k →
      r < ‖(X t k : EucSpace 2) - (X t i : EucSpace 2)‖ := by
  let K := (k : ℝ) * h_pot β r
  have hK : 0 ≤ K := mul_nonneg (by positivity) (h_pot_nonneg β ⟨hr.le, hrπ⟩)
  have hTK : T * K < ε * r := by simpa only [K, mul_assoc] using htime
  let L := K + ε * r / T
  have hLgt : 2 * K < L := by
    have hdiv : K < ε * r / T := (lt_div_iff₀ hT).mpr (by nlinarith [hTK])
    dsimp only [L]
    linarith
  have hL : 0 ≤ L := add_nonneg hK (div_nonneg (mul_pos hε hr).le hT.le)
  have hLT : L * T = K * T + ε * r := by
    dsimp only [L]
    field_simp
  have hb0 : r + L * T < (1 + 2 * ε) * r := by
    rw [hLT]
    nlinarith [hTK]
  let g (i : Idx n) (t : ℝ) := ‖(X t i : EucSpace 2) - (X t k : EucSpace 2)‖ ^ 2
  let dg (i : Idx n) (t : ℝ) := deriv (g i) t
  have hg (i : Idx n) (t : ℝ) : HasDerivAt (g i) (dg i t) t :=
    ((hasDerivAt_causalDrift β X hX t i).sub
      (hasDerivAt_causalDrift β X hX t k)).norm_sq.differentiableAt.hasDerivAt
  have hstart : ∀ i ∈ Finset.Iio k, (r + L * T) ^ 2 < g i 0 := by
    intro i hi
    have hsep := hb0.trans (hinit i (Finset.mem_Iio.mp hi))
    rw [norm_sub_rev] at hsep
    have hb : 0 ≤ r + L * T := by positivity
    dsimp only [g]
    nlinarith [norm_nonneg ((X 0 i : EucSpace 2) - (X 0 k : EucSpace 2))]
  have hbar := finite_affine_gap_barrier (Finset.Iio k) g dg T r L hr.le hL
    (fun i _ t => hg i t) hstart ?_
  · intro t ht i hik
    have hs := hbar t ht i (Finset.mem_Iio.mpr hik)
    dsimp only [g] at hs
    rw [norm_sub_rev] at hs
    nlinarith [norm_nonneg ((X t k : EucSpace 2) - (X t i : EucSpace 2))]
  · intro t ht hall i hi heq
    let b := r + L * (T - t)
    have hb : 0 < b := by
      dsimp only [b]
      nlinarith [mul_nonneg hL (sub_nonneg.mpr ht.2)]
    have hbr : r ≤ b := by
      dsimp only [b]
      linarith [mul_nonneg hL (sub_nonneg.mpr ht.2)]
    have hD : ‖(X t i : EucSpace 2) - (X t k : EucSpace 2)‖ = b := by
      change ‖(X t i : EucSpace 2) - (X t k : EucSpace 2)‖ ^ 2 = b ^ 2 at heq
      nlinarith [norm_nonneg ((X t i : EucSpace 2) - (X t k : EucSpace 2))]
    have hgap : ∀ j : Idx n, j < k →
        r ≤ ‖(X t k : EucSpace 2) - (X t j : EucSpace 2)‖ := by
      intro j hj
      have hs := hall j (Finset.mem_Iio.mpr hj)
      change b ^ 2 ≤ ‖(X t j : EucSpace 2) - (X t k : EucSpace 2)‖ ^ 2 at hs
      rw [norm_sub_rev] at hs
      nlinarith [norm_nonneg ((X t k : EucSpace 2) - (X t j : EucSpace 2))]
    have hclosest : ∀ j : Idx n, j < k →
        inner ℝ (X t k : EucSpace 2) (X t j : EucSpace 2) ≤
          inner ℝ (X t k : EucSpace 2) (X t i : EucSpace 2) := by
      intro j hj
      have hs := hall j (Finset.mem_Iio.mpr hj)
      change b ^ 2 ≤ ‖(X t j : EucSpace 2) - (X t k : EucSpace 2)‖ ^ 2 at hs
      have hiSq := heq
      dsimp only [g] at hiSq
      rw [norm_sub_sq_real, mem_sphere_zero_iff_norm.mp (X t j).2,
        mem_sphere_zero_iff_norm.mp (X t k).2, real_inner_comm] at hs
      rw [norm_sub_sq_real, mem_sphere_zero_iff_norm.mp (X t i).2,
        mem_sphere_zero_iff_norm.mp (X t k).2, real_inner_comm] at hiSq
      change _ = b ^ 2 at hiSq
      nlinarith
    obtain ⟨a, ha, hbound⟩ := squared_chord_derivative_lower β hβ r hrπ hrpeak
      X hX t k i (Finset.mem_Iio.mp hi) hclosest hgap
    have haeq : dg i t = a := ha.deriv
    rw [hD] at hbound
    have hstrict := mul_pos (sub_pos.mpr hLgt) hb
    change -2 * L * b < dg i t
    dsimp only [K] at hstrict
    nlinarith

/-- **The strong center's displacement is below `εr` throughout the horizon.**
Retained predecessor gaps bound its actual speed by `k h(r)`, and the mean
value inequality integrates that bound. Source: arXiv:2411.04990v2, §5,
proof of `lemma:meta`, the final displacement estimate. -/
theorem strong_center_displacement (β : ℝ) (hβ : 0 < β) (r ε T : ℝ)
    (hr : 0 < r) (hε : 0 < ε) (hT : 0 < T) (hrπ : r ≤ π)
    (hrpeak : Real.arccos ((-1 + Real.sqrt (4 * β ^ 2 + 1)) / (2 * β)) ≤ r)
    (X : ℝ → SphereTuple 2 n)
    (hX : CSA 2 n β (ContinuousLinearMap.id ℝ _) (ContinuousLinearMap.id ℝ _)
      (ContinuousLinearMap.id ℝ _) X) (k : Idx n)
    (hinit : ∀ i : Idx n, i < k →
      (1 + 2 * ε) * r < ‖(X 0 k : EucSpace 2) - (X 0 i : EucSpace 2)‖)
    (htime : T * (k : ℝ) * h_pot β r < ε * r) :
    ∀ t ∈ Icc (0 : ℝ) T,
      ‖(X t k : EucSpace 2) - (X 0 k : EucSpace 2)‖ < ε * r := by
  have hgap := strong_center_gaps β hβ r ε T hr hε hT hrπ hrpeak X hX k hinit htime
  have hK : 0 ≤ (k : ℝ) * h_pot β r :=
    mul_nonneg (by positivity) (h_pot_nonneg β ⟨hr.le, hrπ⟩)
  have hbound := norm_image_sub_le_of_norm_deriv_le_segment'
    (fun s (_ : s ∈ Icc (0 : ℝ) T) => (hasDerivAt_causalDrift β X hX s k).hasDerivWithinAt)
    (fun s (hs : s ∈ Ico (0 : ℝ) T) =>
      norm_causalDrift_le β hβ r hrπ hrpeak (X s) k
        (fun i hi => (hgap s ⟨hs.1, hs.2.le⟩ i hi).le))
  intro t ht
  have h := hbound t ht
  have hKT : (k : ℝ) * h_pot β r * T < ε * r := by nlinarith [htime]
  exact (h.trans (by nlinarith [mul_le_mul_of_nonneg_left ht.2 hK])).trans_lt hKT

/-- The complete hypotheses of both strong-center theorems hold for an actual stationary circle token,
with `β=1`, `r=2`, `ε=1`, and `T=1`. All positivity and peak assumptions
hold, and the first token has no predecessors. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    (2 : ℝ) ≤ π ∧
    Real.arccos ((-1 + Real.sqrt (4 * (1 : ℝ) ^ 2 + 1)) / (2 * 1)) ≤ 2 ∧
    CSA 2 1 1 (ContinuousLinearMap.id ℝ _) (ContinuousLinearMap.id ℝ _)
      (ContinuousLinearMap.id ℝ _) (fun _ _ => basePoint 1) ∧
    (∀ i : Idx 1, i < 0 → (1 + 2 * 1) * (2 : ℝ) <
      ‖(basePoint 1 : EucSpace 2) - (basePoint 1 : EucSpace 2)‖) ∧
    (1 : ℝ) * (0 : ℝ) * h_pot 1 2 < 1 * 2 := by
  obtain ⟨τ, hτ0, hτπ, hcos, _, hτ1, _⟩ := h_pot_unimodal 1 le_rfl
  refine ⟨one_pos, by norm_num, one_pos, one_pos, by linarith [pi_gt_three], ?_,
    csa_const_one 1 (basePoint 1), fun i hi => ?_, by norm_num⟩
  · rw [← hcos, Real.arccos_cos hτ0.le hτπ.le, Real.one_rpow] at *
    linarith
  · change (i : ℕ) < 0 at hi
    omega

end Transformer.Causal
