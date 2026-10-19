module

public import Transformer.Causal.Section5_WeightedInteraction

/-!
# Aggregate curvature of an original frozen interaction cluster

ArXiv:2411.04990v2, §5, the same-cluster contribution in the
proof of `thm:fixed_centers`, with Appendix B's original `h,g`.
Individual same-cluster curvatures need not be positive: two
tokens each within epsilon of a center can be twice epsilon
apart, beyond the actual interaction peak. We therefore estimate
the actual weighted cluster as a whole at zero cluster force.

For original nonnegative kernel weights, let W be their total,
C their cosine moment, and S their sine moment. The original
cluster cap gives `C cos(delta)+S sin(delta)>=W cos(epsilon)`.
At zero cluster force, this implies `C>=W cos(epsilon)`.
Weighted Cauchy-Schwarz then bounds the actual sine-square
moment by `W sin²(epsilon)`. The actual aggregate curvature is
at least `W (cos(epsilon)-beta sin²(epsilon))`, strictly positive
below the original peak. The zero-cluster-force premise is
explicit; interactions with other clusters cannot be dropped
from the original full criticality equation.
-/

@[expose] public section

open scoped BigOperators
open Real Set

namespace Transformer.Causal

variable {ι : Type*}

/-- Original nonnegative weights obey the exact weighted cosine-moment Cauchy-Schwarz inequality.
Source: arXiv:2411.04990v2, §5, auxiliary aggregate estimate for the actual same-cluster curvature in `thm:fixed_centers`. -/
theorem weighted_cosine_moment_sq_le (s : Finset ι) (w v : ι → ℝ) (hw : ∀ i ∈ s, 0 ≤ w i) :
    (∑ i ∈ s, w i * Real.cos (v i)) ^ 2 ≤ (∑ i ∈ s, w i) * ∑ i ∈ s, w i * Real.cos (v i) ^ 2 := by
  apply Finset.sum_sq_le_sum_mul_sum_of_sq_le_mul s hw
    (fun i hi => mul_nonneg (hw i hi) (sq_nonneg _))
  intro i hi
  exact le_of_eq (by ring)

/-- The original weighted sine-square and cosine-square moments sum to the actual original total weight.
Source: arXiv:2411.04990v2, §5, the original unit-circle identity used in the aggregate curvature calculation. -/
theorem weighted_trig_square_moments (s : Finset ι) (w v : ι → ℝ) :
    (∑ i ∈ s, w i * Real.sin (v i) ^ 2) + (∑ i ∈ s, w i * Real.cos (v i) ^ 2) = ∑ i ∈ s, w i := by
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i hi
  linear_combination w i * Real.sin_sq_add_cos_sq (v i)

/-- An actual original cluster cap bounds its weighted rotated cosine moment without assuming zero force.
Source: arXiv:2411.04990v2, §5, aggregate circle geometry for the actual same-cluster terms in `thm:fixed_centers`. -/
theorem weighted_cluster_cap_lower (s : Finset ι) (w z : ι → ℝ) (hw : ∀ i ∈ s, 0 ≤ w i)
    (e θ u : ℝ) (he : e ∈ Icc 0 Real.pi) (hz : ∀ i ∈ s, circDist (z i - θ) ≤ e) :
    (∑ i ∈ s, w i) * Real.cos e ≤
      Real.cos (u - θ) * (∑ i ∈ s, w i * Real.cos (u - z i)) +
        Real.sin (u - θ) * (∑ i ∈ s, w i * Real.sin (u - z i)) := by
  have hc : ∀ i ∈ s, Real.cos e ≤ Real.cos (z i - θ) := by
    intro i hi
    rw [← cos_circDist (z i - θ)]
    exact Real.cos_le_cos_of_nonneg_of_le_pi (circDist_nonneg _) he.2 (hz i hi)
  have hm := Finset.sum_le_sum fun i hi => mul_le_mul_of_nonneg_left (hc i hi) (hw i hi)
  rw [← Finset.sum_mul] at hm
  have hid : ∀ i : ι, Real.cos (z i - θ) = Real.cos (u - z i) * Real.cos (u - θ) +
      Real.sin (u - z i) * Real.sin (u - θ) := by
    intro i
    rw [show z i - θ = (u - θ) - (u - z i) by ring, Real.cos_sub]
    ring
  have hs : (∑ i ∈ s, w i * Real.cos (z i - θ)) =
      Real.cos (u - θ) * (∑ i ∈ s, w i * Real.cos (u - z i)) +
        Real.sin (u - θ) * (∑ i ∈ s, w i * Real.sin (u - z i)) := by
    simp_rw [hid, mul_add, ← mul_assoc, Finset.sum_add_distrib, ← Finset.sum_mul]
    ring
  exact hm.trans_eq hs

/-- At zero actual cluster force, an original subpeak cap controls the whole weighted curvature, even when individual pair curvatures are negative.
Source: arXiv:2411.04990v2, §5, an aggregate replacement for the individual same-cluster sign step in `thm:fixed_centers`. -/
theorem weighted_cluster_curvature_lower (s : Finset ι) (w z : ι → ℝ) (hw : ∀ i ∈ s, 0 ≤ w i)
    (hW : 0 < ∑ i ∈ s, w i) (β : ℝ) (hβ : 0 < β) (e θ u : ℝ)
    (he : e < interactionPeak β) (hu : circDist (u - θ) ≤ e) (hz : ∀ i ∈ s, circDist (z i - θ) ≤ e)
    (hforce : (∑ i ∈ s, w i * Real.sin (u - z i)) = 0) :
    (∑ i ∈ s, w i) * (Real.cos e - β * Real.sin e ^ 2) ≤
      ∑ i ∈ s, w i * (Real.cos (u - z i) - β * Real.sin (u - z i) ^ 2) := by
  let W := ∑ i ∈ s, w i
  let C := ∑ i ∈ s, w i * Real.cos (u - z i)
  let S₂ := ∑ i ∈ s, w i * Real.sin (u - z i) ^ 2
  let Q := ∑ i ∈ s, w i * Real.cos (u - z i) ^ 2
  have he0 : 0 ≤ e := (circDist_nonneg _).trans hu
  have hepi : e < Real.pi := by linarith [(interactionPeak_properties β hβ).2.1, Real.pi_pos]
  have hge := (g_pot_pos_iff_lt_peak β hβ e ⟨he0, hepi.le⟩).mpr he
  have hD : 0 < Real.cos e - β * Real.sin e ^ 2 := (mul_pos_iff_of_pos_left (Real.exp_pos _)).mp hge
  have hce : 0 < Real.cos e := by nlinarith [mul_nonneg hβ.le (sq_nonneg (Real.sin e))]
  have hcδ : 0 < Real.cos (u - θ) := by
    have hc := Real.cos_le_cos_of_nonneg_of_le_pi (circDist_nonneg (u - θ)) hepi.le hu
    rw [cos_circDist] at hc
    exact hce.trans_le hc
  have hcap := weighted_cluster_cap_lower s w z hw e θ u ⟨he0, hepi.le⟩ hz
  rw [hforce, mul_zero, add_zero] at hcap
  change W * Real.cos e ≤ Real.cos (u - θ) * C at hcap
  have hC0 : 0 < C := (mul_pos_iff_of_pos_left hcδ).mp ((mul_pos hW hce).trans_le hcap)
  have hC : W * Real.cos e ≤ C := by nlinarith [Real.cos_le_one (u - θ)]
  have hCS : C ^ 2 ≤ W * Q := weighted_cosine_moment_sq_le s w (fun i => u - z i) hw
  have hunit : S₂ + Q = W := weighted_trig_square_moments s w (fun i => u - z i)
  have hsq := mul_self_le_mul_self (mul_pos hW hce).le hC
  change (W * Real.cos e) * (W * Real.cos e) ≤ C * C at hsq
  have hWunit : W * S₂ + W * Q = W ^ 2 := by linear_combination W * hunit
  have hsqtrig : W ^ 2 * Real.sin e ^ 2 + W ^ 2 * Real.cos e ^ 2 = W ^ 2 := by
    linear_combination W ^ 2 * Real.sin_sq_add_cos_sq e
  have hsin : S₂ ≤ W * Real.sin e ^ 2 := by
    apply le_of_mul_le_mul_left (a := W) ?_ hW
    nlinarith only [hCS, hWunit, hsq, hsqtrig]
  have hβsin := mul_le_mul_of_nonneg_left hsin hβ.le
  have hid : (∑ i ∈ s, w i * (Real.cos (u - z i) - β * Real.sin (u - z i) ^ 2)) = C - β * S₂ := by
    dsimp [C, S₂]
    simp_rw [mul_sub, Finset.sum_sub_distrib, mul_left_comm (w _), ← Finset.mul_sum]
  change W * (Real.cos e - β * Real.sin e ^ 2) ≤ _
  rw [hid]
  nlinarith only [hC, hβsin]

/-- Actual original masses and exponentials give positive aggregate curvature at zero original same-cluster force inside an actual original subpeak cap.
Source: arXiv:2411.04990v2, §5, the original `h,g` same-cluster calculation in `thm:fixed_centers`, with zero cluster force explicit. -/
theorem original_cluster_curvature_pos (s : Finset ι) (a z : ι → ℝ) (ha : ∀ i ∈ s, 0 ≤ a i)
    (hapos : ∃ i ∈ s, 0 < a i) (β : ℝ) (hβ : 0 < β) (e θ u : ℝ)
    (he : e < interactionPeak β) (hu : circDist (u - θ) ≤ e) (hz : ∀ i ∈ s, circDist (z i - θ) ≤ e)
    (hforce : (∑ i ∈ s, a i * h_pot β (u - z i)) = 0) : 0 < ∑ i ∈ s, a i * g_pot β (u - z i) := by
  let w := fun i => a i * Real.exp (β * (Real.cos (u - z i) - 1))
  have hw : ∀ i ∈ s, 0 ≤ w i := fun i hi => mul_nonneg (ha i hi) (Real.exp_pos _).le
  obtain ⟨j, hj, haj⟩ := hapos
  have hW : 0 < ∑ i ∈ s, w i := Finset.sum_pos' hw ⟨j, hj, mul_pos haj (Real.exp_pos _)⟩
  have hF : (∑ i ∈ s, w i * Real.sin (u - z i)) = 0 := by
    simpa only [w, h_pot, mul_assoc] using hforce
  have hb := weighted_cluster_curvature_lower s w z hw hW β hβ e θ u he hu hz hF
  have he0 := (circDist_nonneg (u - θ)).trans hu
  have hge := (g_pot_pos_iff_lt_peak β hβ e ⟨he0, by
    linarith [(interactionPeak_properties β hβ).2.1, Real.pi_pos]⟩).mpr he
  have hD : 0 < Real.cos e - β * Real.sin e ^ 2 := (mul_pos_iff_of_pos_left (Real.exp_pos _)).mp hge
  have hp := (mul_pos hW hD).trans_le hb
  simpa only [w, g_pot, mul_assoc] using hp

/-- A genuine original zero-force cluster has positive aggregate curvature despite one actual original interaction having negative curvature.
Source: arXiv:2411.04990v2, §5, the same-cluster curvature step in `thm:fixed_centers`; this witness checks the aggregate replacement. -/
example : ∃ e u : ℝ, ∃ a z : Idx 2 → ℝ,
    0 < e ∧ e < interactionPeak 1 ∧ circDist (u - 0) ≤ e ∧
    (∀ i, circDist (z i - 0) ≤ e) ∧ (∀ i, 0 < a i) ∧
    (0 < ∑ i, a i * Real.exp (Real.cos (u - z i) - 1)) ∧
    (∑ i, a i * h_pot 1 (u - z i)) = 0 ∧ g_pot 1 (u - z 0) < 0 ∧
    0 < ∑ i, a i * g_pot 1 (u - z i) := by
  let e := 3 * interactionPeak 1 / 4
  let z : Idx 2 → ℝ := ![-e, e]
  let a : Idx 2 → ℝ := ![h_pot 1 (e / 2), h_pot 1 (3 * e / 2)]
  have ht := interactionPeak_properties 1 one_pos
  have he0 : 0 < e := by dsimp [e]; linarith
  have he : e < interactionPeak 1 := by dsimp [e]; linarith
  have hlarge : 3 * e / 2 < Real.pi := by dsimp [e]; linarith [Real.pi_pos]
  have hcire : circDist e = e := circDist_eq_self _ ⟨he0.le, by linarith⟩
  have hcirhalf : circDist (e / 2) = e / 2 := circDist_eq_self _ ⟨by positivity, by linarith⟩
  have hcirlarge : circDist (3 * e / 2) = 3 * e / 2 := circDist_eq_self _ ⟨by positivity, hlarge.le⟩
  have hsmall : 0 < h_pot 1 (e / 2) :=
    mul_pos (Real.exp_pos _) (Real.sin_pos_of_pos_of_lt_pi (by positivity) (by linarith))
  have hbig : 0 < h_pot 1 (3 * e / 2) :=
    mul_pos (Real.exp_pos _) (Real.sin_pos_of_pos_of_lt_pi (by positivity) hlarge)
  have ha : ∀ i, 0 < a i := by intro i; fin_cases i; exact hsmall; exact hbig
  have hz : ∀ i ∈ (Finset.univ : Finset (Idx 2)), circDist (z i - 0) ≤ e := by
    intro i hi
    fin_cases i <;> simp [z, circDist_neg, hcire]
  have hu : circDist (e / 2 - 0) ≤ e := by rw [sub_zero, hcirhalf]; linarith
  have hF : (∑ i, a i * h_pot 1 (e / 2 - z i)) = 0 := by
    simp only [Fin.sum_univ_two, a, z, Matrix.cons_val_zero, Matrix.cons_val_one,
      show e / 2 - -e = 3 * e / 2 by ring, show e / 2 - e = -(e / 2) by ring, h_pot_odd]
    ring
  have hW : 0 < ∑ i, a i * Real.exp (Real.cos (e / 2 - z i) - 1) :=
    Finset.sum_pos' (fun i _ => mul_nonneg (ha i).le (Real.exp_pos _).le)
      ⟨0, Finset.mem_univ _, mul_pos (ha 0) (Real.exp_pos _)⟩
  refine ⟨e, e / 2, a, z, he0, he, hu, (fun i => hz i (Finset.mem_univ _)), ha, hW, hF, ?_, ?_⟩
  · change g_pot 1 (e / 2 - -e) < 0
    rw [show e / 2 - -e = 3 * e / 2 by ring]
    apply (g_pot_neg_iff_peak_lt_circDist 1 one_pos _).mpr
    rw [hcirlarge]
    dsimp [e]
    linarith
  · exact original_cluster_curvature_pos Finset.univ a z (fun i _ => (ha i).le)
      ⟨0, Finset.mem_univ _, ha 0⟩ 1 one_pos e 0 (e / 2) he hu hz hF

end Transformer.Causal
