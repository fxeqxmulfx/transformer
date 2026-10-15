/-
# Actual cluster curvature with its original nonzero force retained

ArXiv:2411.04990v2, §5, the same-cluster stability estimate
in the proof of `thm:fixed_centers`.
Other clusters need not exert zero force, even at a full critical
point. We therefore keep the actual cluster force explicitly.
Weighted Cauchy-Schwarz gives a convex tangent bound for the
original curvature as a function of its actual cosine moment.
The original cap geometry controls the loss in that moment by
the magnitude of its actual sine moment, with coefficient
`epsilon/cos(epsilon)`.

Consequently the original cluster curvature is at least its
positive central value minus an explicit multiple of the actual
cluster force. The final theorem uses the original masses and
original exponentials, so this error is precisely a multiple of
the actual original `sum a*h`, rather than an unproved zero-force
premise. A genuine positive-force example checks that the error
term is realized.
-/

import Transformer.Causal.Section5_ClusterCurvature

open scoped BigOperators
open Real Set

namespace Transformer.Causal

variable {ι : Type*}

/-- Weighted Cauchy-Schwarz bounds the original curvature below by its actual cosine-moment tangent at any prescribed original radius.
Source: arXiv:2411.04990v2, §5, auxiliary aggregate calculation for the same-cluster terms in `thm:fixed_centers`. -/
theorem weighted_curvature_tangent_lower (s : Finset ι) (w v : ι → ℝ) (hw : ∀ i ∈ s, 0 ≤ w i)
    (hW : 0 < ∑ i ∈ s, w i) (β e : ℝ) (hβ : 0 ≤ β) :
    (∑ i ∈ s, w i) * (Real.cos e - β * Real.sin e ^ 2) +
      (1 + 2 * β * Real.cos e) * ((∑ i ∈ s, w i * Real.cos (v i)) - (∑ i ∈ s, w i) * Real.cos e) ≤
        ∑ i ∈ s, w i * (Real.cos (v i) - β * Real.sin (v i) ^ 2) := by
  let W := ∑ i ∈ s, w i
  let C := ∑ i ∈ s, w i * Real.cos (v i)
  let S₂ := ∑ i ∈ s, w i * Real.sin (v i) ^ 2
  let Q := ∑ i ∈ s, w i * Real.cos (v i) ^ 2
  have hCS : C ^ 2 ≤ W * Q := weighted_cosine_moment_sq_le s w v hw
  have hunit : S₂ + Q = W := weighted_trig_square_moments s w v
  have hmul : β * W * S₂ + β * W * Q = β * W ^ 2 := by linear_combination β * W * hunit
  have hCSβ := mul_le_mul_of_nonneg_left hCS hβ
  have hsq := mul_nonneg hβ (sq_nonneg (C - W * Real.cos e))
  have htrig : W ^ 2 * Real.sin e ^ 2 + W ^ 2 * Real.cos e ^ 2 = W ^ 2 := by
    linear_combination W ^ 2 * Real.sin_sq_add_cos_sq e
  have hid : (∑ i ∈ s, w i * (Real.cos (v i) - β * Real.sin (v i) ^ 2)) = C - β * S₂ := by
    dsimp [C, S₂]
    simp_rw [mul_sub, Finset.sum_sub_distrib, mul_left_comm (w _), ← Finset.mul_sum]
  rw [hid]
  apply le_of_mul_le_mul_left (a := W) ?_ hW
  change W * (W * (Real.cos e - β * Real.sin e ^ 2) +
    (1 + 2 * β * Real.cos e) * (C - W * Real.cos e)) ≤ W * (C - β * S₂)
  have htrigβ := congrArg (fun x : ℝ => β * x) htrig
  nlinarith only [hmul, hCSβ, hsq, htrigβ]

/-- The original cluster cap bounds the actual cosine-moment loss by its actual original force magnitude.
Source: arXiv:2411.04990v2, §5, aggregate circle geometry for the same-cluster contribution in `thm:fixed_centers`. -/
theorem weighted_cluster_cosine_force_error (s : Finset ι) (w z : ι → ℝ) (hw : ∀ i ∈ s, 0 ≤ w i)
    (hW : 0 < ∑ i ∈ s, w i) (β : ℝ) (hβ : 0 < β) (e θ u : ℝ)
    (he : e < interactionPeak β) (hu : circDist (u - θ) ≤ e) (hz : ∀ i ∈ s, circDist (z i - θ) ≤ e) :
    (∑ i ∈ s, w i) * Real.cos e - e / Real.cos e * |∑ i ∈ s, w i * Real.sin (u - z i)| ≤
      ∑ i ∈ s, w i * Real.cos (u - z i) := by
  let W := ∑ i ∈ s, w i
  let C := ∑ i ∈ s, w i * Real.cos (u - z i)
  let S := ∑ i ∈ s, w i * Real.sin (u - z i)
  let B := e / Real.cos e * |S|
  have he0 : 0 ≤ e := (circDist_nonneg _).trans hu
  have hepi : e < Real.pi := by linarith [(interactionPeak_properties β hβ).2.1, Real.pi_pos]
  have hge := (g_pot_pos_iff_lt_peak β hβ e ⟨he0, hepi.le⟩).mpr he
  have hD : 0 < Real.cos e - β * Real.sin e ^ 2 := (mul_pos_iff_of_pos_left (Real.exp_pos _)).mp hge
  have hce : 0 < Real.cos e := by nlinarith [mul_nonneg hβ.le (sq_nonneg (Real.sin e))]
  have hcδ := Real.cos_le_cos_of_nonneg_of_le_pi (circDist_nonneg (u - θ)) hepi.le hu
  rw [cos_circDist] at hcδ
  have hcδ0 := hce.trans_le hcδ
  have hB0 : 0 ≤ B := mul_nonneg (div_nonneg he0 hce.le) (abs_nonneg _)
  have hBc : B * Real.cos e = e * |S| := by dsimp [B]; field_simp
  have hBδ := mul_le_mul_of_nonneg_left hcδ hB0
  rw [hBc] at hBδ
  have hsin : |Real.sin (u - θ)| ≤ e := by
    rw [abs_sin_eq_sin_circDist]
    exact (Real.sin_le (circDist_nonneg _)).trans hu
  have hSsin : Real.sin (u - θ) * S ≤ e * |S| := by
    have ha := le_abs_self (Real.sin (u - θ) * S)
    rw [abs_mul] at ha
    exact ha.trans (mul_le_mul_of_nonneg_right hsin (abs_nonneg _))
  have hcap := weighted_cluster_cap_lower s w z hw e θ u ⟨he0, hepi.le⟩ hz
  change W * Real.cos e ≤ Real.cos (u - θ) * C + Real.sin (u - θ) * S at hcap
  have hWδ := mul_le_mul_of_nonneg_left (Real.cos_le_one (u - θ)) (mul_pos hW hce).le
  apply le_of_mul_le_mul_left (a := Real.cos (u - θ)) ?_ hcδ0
  change Real.cos (u - θ) * (W * Real.cos e - B) ≤ Real.cos (u - θ) * C
  nlinarith only [hcap, hWδ, hSsin, hBδ]

/-- The actual original weighted cluster curvature has an explicit error proportional to its actual original nonzero force.
Source: arXiv:2411.04990v2, §5, the same-cluster stability estimate for `thm:fixed_centers`, with the force term retained. -/
theorem weighted_cluster_curvature_force_error (s : Finset ι) (w z : ι → ℝ) (hw : ∀ i ∈ s, 0 ≤ w i)
    (hW : 0 < ∑ i ∈ s, w i) (β : ℝ) (hβ : 0 < β) (e θ u : ℝ)
    (he : e < interactionPeak β) (hu : circDist (u - θ) ≤ e) (hz : ∀ i ∈ s, circDist (z i - θ) ≤ e) :
    (∑ i ∈ s, w i) * (Real.cos e - β * Real.sin e ^ 2) -
      ((1 + 2 * β * Real.cos e) * e / Real.cos e) * |∑ i ∈ s, w i * Real.sin (u - z i)| ≤
        ∑ i ∈ s, w i * (Real.cos (u - z i) - β * Real.sin (u - z i) ^ 2) := by
  have hc := weighted_cluster_cosine_force_error s w z hw hW β hβ e θ u he hu hz
  have he0 := (circDist_nonneg (u - θ)).trans hu
  have hge := (g_pot_pos_iff_lt_peak β hβ e ⟨he0, by
    linarith [(interactionPeak_properties β hβ).2.1, Real.pi_pos]⟩).mpr he
  have hD : 0 < Real.cos e - β * Real.sin e ^ 2 := (mul_pos_iff_of_pos_left (Real.exp_pos _)).mp hge
  have hce : 0 < Real.cos e := by nlinarith [mul_nonneg hβ.le (sq_nonneg (Real.sin e))]
  have hA : 0 ≤ 1 + 2 * β * Real.cos e := by positivity
  have hcm := mul_le_mul_of_nonneg_left hc hA
  have ht := weighted_curvature_tangent_lower s w (fun i => u - z i) hw hW β e hβ.le
  simp only [div_eq_mul_inv] at hcm ⊢
  nlinarith only [hcm, ht]

/-- Original masses and original exponentials retain precisely the actual original cluster force in the curvature lower bound.
Source: arXiv:2411.04990v2, §5, the original `h,g` cluster sums in `thm:fixed_centers`; no zero-force assumption is added. -/
theorem original_cluster_curvature_force_error (s : Finset ι) (a z : ι → ℝ) (ha : ∀ i ∈ s, 0 ≤ a i)
    (hapos : ∃ i ∈ s, 0 < a i) (β : ℝ) (hβ : 0 < β) (e θ u : ℝ)
    (he : e < interactionPeak β) (hu : circDist (u - θ) ≤ e) (hz : ∀ i ∈ s, circDist (z i - θ) ≤ e) :
    (∑ i ∈ s, a i * Real.exp (β * (Real.cos (u - z i) - 1))) * (Real.cos e - β * Real.sin e ^ 2) -
      ((1 + 2 * β * Real.cos e) * e / Real.cos e) * |∑ i ∈ s, a i * h_pot β (u - z i)| ≤
        ∑ i ∈ s, a i * g_pot β (u - z i) := by
  let w := fun i => a i * Real.exp (β * (Real.cos (u - z i) - 1))
  have hw : ∀ i ∈ s, 0 ≤ w i := fun i hi => mul_nonneg (ha i hi) (Real.exp_pos _).le
  obtain ⟨j, hj, haj⟩ := hapos
  have hW : 0 < ∑ i ∈ s, w i := Finset.sum_pos' hw ⟨j, hj, mul_pos haj (Real.exp_pos _)⟩
  have hb := weighted_cluster_curvature_force_error s w z hw hW β hβ e θ u he hu hz
  simpa only [w, h_pot, g_pot, mul_assoc] using hb

/-- A genuine original cluster with positive actual force satisfies the curvature estimate with its nonzero error term retained.
Source: arXiv:2411.04990v2, §5, the actual `h,g` same-cluster contribution in `thm:fixed_centers`. -/
example : ∃ e u : ℝ, ∃ a z : Idx 2 → ℝ,
    0 < e ∧ e < interactionPeak 1 ∧ circDist (u - 0) ≤ e ∧
    (∀ i, circDist (z i - 0) ≤ e) ∧ (∀ i, 0 < a i) ∧
    (0 < ∑ i, a i * Real.exp (Real.cos (u - z i) - 1)) ∧ (0 < ∑ i, a i * h_pot 1 (u - z i)) ∧
    (∑ i, a i * Real.exp (Real.cos (u - z i) - 1)) * (Real.cos e - Real.sin e ^ 2) -
      ((1 + 2 * Real.cos e) * e / Real.cos e) * |∑ i, a i * h_pot 1 (u - z i)| ≤
        ∑ i, a i * g_pot 1 (u - z i) := by
  let e := interactionPeak 1 / 4
  let z : Idx 2 → ℝ := ![-e, e]
  let a : Idx 2 → ℝ := fun _ => 1
  have ht := interactionPeak_properties 1 one_pos
  have he0 : 0 < e := by dsimp [e]; linarith
  have he : e < interactionPeak 1 := by dsimp [e]; linarith
  have hlarge : 3 * e / 2 < interactionPeak 1 := by dsimp [e]; linarith
  have hcire : circDist e = e := circDist_eq_self _ ⟨he0.le, by linarith [Real.pi_pos]⟩
  have hcirhalf : circDist (e / 2) = e / 2 := circDist_eq_self _ ⟨by positivity, by linarith [Real.pi_pos]⟩
  have hz : ∀ i ∈ (Finset.univ : Finset (Idx 2)), circDist (z i - 0) ≤ e := by
    intro i hi
    fin_cases i <;> simp [z, circDist_neg, hcire]
  have hu : circDist (e / 2 - 0) ≤ e := by rw [sub_zero, hcirhalf]; linarith
  have ha : ∀ i, 0 < a i := fun _ => one_pos
  have hW : 0 < ∑ i, a i * Real.exp (Real.cos (e / 2 - z i) - 1) :=
    Finset.sum_pos' (fun i _ => mul_nonneg (ha i).le (Real.exp_pos _).le)
      ⟨0, Finset.mem_univ _, mul_pos (ha 0) (Real.exp_pos _)⟩
  have hmono := (h_pot_monotone_peak 1 one_pos).1
    (show e / 2 ∈ Icc 0 (interactionPeak 1) from ⟨by positivity, by linarith⟩)
    (show 3 * e / 2 ∈ Icc 0 (interactionPeak 1) from ⟨by positivity, hlarge.le⟩)
    (show e / 2 < 3 * e / 2 by linarith)
  refine ⟨e, e / 2, a, z, he0, he, hu, (fun i => hz i (Finset.mem_univ _)), ha, hW, ?_, ?_⟩
  · simp only [Fin.sum_univ_two, a, z, one_mul, Matrix.cons_val_zero, Matrix.cons_val_one,
      show e / 2 - -e = 3 * e / 2 by ring, show e / 2 - e = -(e / 2) by ring, h_pot_odd]
    linarith
  · simpa only [one_mul, mul_one] using original_cluster_curvature_force_error Finset.univ a z
      (fun i _ => (ha i).le) ⟨0, Finset.mem_univ _, ha 0⟩ 1 one_pos e 0 (e / 2) he hu hz

end Transformer.Causal
