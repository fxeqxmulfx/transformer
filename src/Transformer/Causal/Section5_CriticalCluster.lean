/-
# The actual full critical-point curvature split around one frozen center

ArXiv:2411.04990v2, §5, the central-versus-far stability
calculation in the proof of `thm:fixed_centers`.
An actual original frozen center of mass at least one gives the
cluster at least the original central kernel mass. The proved
aggregate estimate therefore retains at least `g(epsilon)`
before its explicit force error.

At a full original critical point, the actual cluster force is
the negative of the actual complementary force. Substituting that
identity and the finite-sum triangle inequality gives the exact
lower bound `g(epsilon)+sum_other a*(g-A*abs(h))` for the full
actual curvature, where `A=(1+2 beta cos(epsilon))*epsilon/cos(epsilon)`.
The complementary forces are retained even though their sum
cancels the cluster force. The example has strictly positive
cluster force canceled by a genuine second frozen center.
-/

import Transformer.Causal.Section5_ClusterForceError

open scoped BigOperators
open Real Set

namespace Transformer.Causal

variable {ι : Type*}

/-- An actual original frozen center of mass at least one supplies at least the original central-edge kernel mass to its actual cluster.
Source: arXiv:2411.04990v2, §5, the distinguished frozen-center contribution in `thm:fixed_centers`. -/
theorem original_cluster_mass_lower (s : Finset ι) (a z : ι → ℝ) (ha : ∀ j ∈ s, 0 ≤ a j)
    (β e θ u : ℝ) (hβ : 0 ≤ β) (he : e ≤ Real.pi) (hu : circDist (u - θ) ≤ e)
    (i : ι) (hi : i ∈ s) (hzi : z i = θ) (hai : 1 ≤ a i) :
    Real.exp (β * (Real.cos e - 1)) ≤ ∑ j ∈ s, a j * Real.exp (β * (Real.cos (u - z j) - 1)) := by
  have hc := Real.cos_le_cos_of_nonneg_of_le_pi (circDist_nonneg (u - θ)) he hu
  rw [cos_circDist] at hc
  have heq : Real.exp (β * (Real.cos e - 1)) ≤ Real.exp (β * (Real.cos (u - z i) - 1)) := by
    rw [hzi]
    exact Real.exp_le_exp.mpr (mul_le_mul_of_nonneg_left (sub_le_sub_right hc 1) hβ)
  have hm := mul_le_mul_of_nonneg_right hai (Real.exp_pos (β * (Real.cos (u - z i) - 1))).le
  simp only [one_mul] at hm
  exact heq.trans (hm.trans (Finset.single_le_sum
    (fun j hj => mul_nonneg (ha j hj) (Real.exp_pos (β * (Real.cos (u - z j) - 1))).le) hi))

/-- An actual original frozen center retains the original edge curvature in the aggregate cluster bound, with the actual nonzero force error explicit.
Source: arXiv:2411.04990v2, §5, the central contribution in the stability estimate for `thm:fixed_centers`. -/
theorem original_cluster_curvature_with_center (s : Finset ι) (a z : ι → ℝ) (ha : ∀ j ∈ s, 0 ≤ a j)
    (β : ℝ) (hβ : 0 < β) (e θ u : ℝ) (he : e < interactionPeak β)
    (hu : circDist (u - θ) ≤ e) (hz : ∀ j ∈ s, circDist (z j - θ) ≤ e)
    (i : ι) (hi : i ∈ s) (hzi : z i = θ) (hai : 1 ≤ a i) :
    g_pot β e - ((1 + 2 * β * Real.cos e) * e / Real.cos e) * |∑ j ∈ s, a j * h_pot β (u - z j)| ≤
      ∑ j ∈ s, a j * g_pot β (u - z j) := by
  have he0 := (circDist_nonneg (u - θ)).trans hu
  have hepi : e ≤ Real.pi := by linarith [(interactionPeak_properties β hβ).2.1, Real.pi_pos]
  have hge := (g_pot_pos_iff_lt_peak β hβ e ⟨he0, hepi⟩).mpr he
  have hD : 0 < Real.cos e - β * Real.sin e ^ 2 := (mul_pos_iff_of_pos_left (Real.exp_pos _)).mp hge
  have hm := original_cluster_mass_lower s a z ha β e θ u hβ.le hepi hu i hi hzi hai
  have hcentral := mul_le_mul_of_nonneg_right hm hD.le
  have hb := original_cluster_curvature_force_error s a z ha ⟨i, hi, one_pos.trans_le hai⟩ β hβ e θ u he hu hz
  change Real.exp (β * (Real.cos e - 1)) * (Real.cos e - β * Real.sin e ^ 2) ≤ _ at hcentral
  rw [← g_pot] at hcentral
  linarith

/-- At a full actual original critical point, one actual frozen cluster controls the total curvature with every complementary force and mass retained.
Source: arXiv:2411.04990v2, §5, the full critical-force split and diagonal stability calculation in `thm:fixed_centers`. -/
theorem original_critical_cluster_curvature_lower (s t : Finset ι) (a z : ι → ℝ)
    (ha : ∀ j ∈ s, 0 ≤ a j) (hat : ∀ j ∈ t, 0 ≤ a j)
    (β : ℝ) (hβ : 0 < β) (e θ u : ℝ) (he : e < interactionPeak β)
    (hu : circDist (u - θ) ≤ e) (hz : ∀ j ∈ s, circDist (z j - θ) ≤ e)
    (i : ι) (hi : i ∈ s) (hzi : z i = θ) (hai : 1 ≤ a i)
    (hcrit : (∑ j ∈ s, a j * h_pot β (u - z j)) + (∑ j ∈ t, a j * h_pot β (u - z j)) = 0) :
    g_pot β e + (∑ j ∈ t, a j * (g_pot β (u - z j) -
      ((1 + 2 * β * Real.cos e) * e / Real.cos e) * |h_pot β (u - z j)|)) ≤
        (∑ j ∈ s, a j * g_pot β (u - z j)) + (∑ j ∈ t, a j * g_pot β (u - z j)) := by
  let A := (1 + 2 * β * Real.cos e) * e / Real.cos e
  have he0 := (circDist_nonneg (u - θ)).trans hu
  have hge := (g_pot_pos_iff_lt_peak β hβ e ⟨he0, by
    linarith [(interactionPeak_properties β hβ).2.1, Real.pi_pos]⟩).mpr he
  have hD : 0 < Real.cos e - β * Real.sin e ^ 2 := (mul_pos_iff_of_pos_left (Real.exp_pos _)).mp hge
  have hce : 0 < Real.cos e := by nlinarith [mul_nonneg hβ.le (sq_nonneg (Real.sin e))]
  have hA : 0 ≤ A := div_nonneg (mul_nonneg (by positivity) he0) hce.le
  have hc := original_cluster_curvature_with_center s a z ha β hβ e θ u he hu hz i hi hzi hai
  have hF : (∑ j ∈ s, a j * h_pot β (u - z j)) = -(∑ j ∈ t, a j * h_pot β (u - z j)) := by linarith
  rw [hF, abs_neg] at hc
  have habs : |∑ j ∈ t, a j * h_pot β (u - z j)| ≤ ∑ j ∈ t, a j * |h_pot β (u - z j)| := by
    calc
      _ ≤ ∑ j ∈ t, |a j * h_pot β (u - z j)| := Finset.abs_sum_le_sum_abs _ _
      _ = _ := Finset.sum_congr rfl fun j hj => by rw [abs_mul, abs_of_nonneg (hat j hj)]
  have hm := mul_le_mul_of_nonneg_left habs hA
  have hid : (∑ j ∈ t, a j * (g_pot β (u - z j) - A * |h_pot β (u - z j)|)) =
      (∑ j ∈ t, a j * g_pot β (u - z j)) - A * (∑ j ∈ t, a j * |h_pot β (u - z j)|) := by
    rw [Finset.mul_sum, ← Finset.sum_sub_distrib]
    exact Finset.sum_congr rfl fun j hj => by ring
  change g_pot β e + (∑ j ∈ t, a j * (g_pot β (u - z j) - A * |h_pot β (u - z j)|)) ≤ _
  rw [hid]
  change g_pot β e - A * |∑ j ∈ t, a j * h_pot β (u - z j)| ≤ _ at hc
  linarith

/-- Two actual original frozen centers realize a full critical point with strictly positive cluster force canceled by the second actual frozen force.
Source: arXiv:2411.04990v2, §5, the actual weighted critical-force split in `thm:fixed_centers`. -/
example : ∃ e u : ℝ, ∃ a z : Idx 2 → ℝ,
    0 < e ∧ e < interactionPeak 1 ∧ circDist (u - 0) ≤ e ∧ (∀ j, 1 ≤ a j) ∧ z 0 = 0 ∧
    (0 < ∑ j ∈ ({0} : Finset (Idx 2)), a j * h_pot 1 (u - z j)) ∧
    ((∑ j ∈ ({0} : Finset (Idx 2)), a j * h_pot 1 (u - z j)) +
      (∑ j ∈ ({1} : Finset (Idx 2)), a j * h_pot 1 (u - z j)) = 0) ∧
    g_pot 1 e + (∑ j ∈ ({1} : Finset (Idx 2)), a j * (g_pot 1 (u - z j) -
      ((1 + 2 * Real.cos e) * e / Real.cos e) * |h_pot 1 (u - z j)|)) ≤
        (∑ j ∈ ({0} : Finset (Idx 2)), a j * g_pot 1 (u - z j)) +
          (∑ j ∈ ({1} : Finset (Idx 2)), a j * g_pot 1 (u - z j)) := by
  let e := interactionPeak 1 / 4
  let u := e / 2
  let z : Idx 2 → ℝ := ![0, Real.pi]
  let a : Idx 2 → ℝ := ![1, Real.exp (2 * Real.cos u)]
  have ht := interactionPeak_properties 1 one_pos
  have he0 : 0 < e := by dsimp [e]; linarith
  have he : e < interactionPeak 1 := by dsimp [e]; linarith
  have hu0 : 0 < u := by dsimp [u]; positivity
  have hupi : u < Real.pi / 2 := by dsimp [u, e]; linarith [Real.pi_pos]
  have hcos : 0 < Real.cos u := Real.cos_pos_of_mem_Ioo ⟨by linarith [Real.pi_pos], hupi⟩
  have ha : ∀ j, 1 ≤ a j := by
    intro j
    fin_cases j
    · exact le_rfl
    · exact Real.one_le_exp_iff.mpr (by positivity)
  have hu : circDist (u - 0) ≤ e := by
    rw [sub_zero, circDist_eq_self u ⟨hu0.le, by linarith [Real.pi_pos]⟩]
    dsimp [u]
    linarith
  have hz : ∀ j ∈ ({0} : Finset (Idx 2)), circDist (z j - 0) ≤ e := by
    intro j hj
    have hj0 := Finset.mem_singleton.mp hj
    subst j
    simpa [z, circDist_zero] using he0.le
  have hsmall : 0 < h_pot 1 u :=
    mul_pos (Real.exp_pos _) (Real.sin_pos_of_pos_of_lt_pi hu0 (by linarith [Real.pi_pos]))
  have hexp : Real.exp (2 * Real.cos u) * Real.exp (-Real.cos u - 1) = Real.exp (Real.cos u - 1) := by
    rw [← Real.exp_add]
    congr 1
    ring
  have hright : Real.exp (2 * Real.cos u) * h_pot 1 (u - Real.pi) = -h_pot 1 u := by
    simp only [h_pot, Real.cos_sub_pi, Real.sin_sub_pi, one_mul]
    rw [← mul_assoc, hexp]
    ring
  have hbalance : ((∑ j ∈ ({0} : Finset (Idx 2)), a j * h_pot 1 (u - z j)) +
      (∑ j ∈ ({1} : Finset (Idx 2)), a j * h_pot 1 (u - z j))) = 0 := by
    simp only [Finset.sum_singleton, a, z, Matrix.cons_val_zero, Matrix.cons_val_one, one_mul, sub_zero,
      hright, add_neg_cancel]
  refine ⟨e, u, a, z, he0, he, hu, ha, by simp [z], ?_, hbalance, ?_⟩
  · simpa only [Finset.sum_singleton, a, z, Matrix.cons_val_zero, one_mul, sub_zero] using hsmall
  · simpa only [one_mul, mul_one] using original_critical_cluster_curvature_lower {0} {1} a z
      (fun j _ => le_trans zero_le_one (ha j)) (fun j _ => le_trans zero_le_one (ha j))
      1 one_pos e 0 u he hu hz 0 (by simp) (by simp [z]) (ha 0) hbalance

end Transformer.Causal
