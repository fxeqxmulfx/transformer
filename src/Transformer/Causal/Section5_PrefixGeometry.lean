module

public import Transformer.Causal.Section5_FrozenLastRow
public import Transformer.Causal.Section5_WeightedInteraction
public import Transformer.Causal.Section5_ParameterGeometry

/-!
# Locating the next original frozen token from its curvature

ArXiv:2411.04990v2, §5, the induction in the proof of
`thm:fixed_centers`.
The actual last-row curvature includes every preceding moving
token and every original weighted frozen center. Two separated
centers ensure at least one strictly negative summand. Hence a
nonnegative total has a strictly positive original interaction,
even when that total itself is zero.

If every preceding token already lies near a frozen center,
this positive interaction puts the current token within one
peak plus the cluster radius of a center. We also prove the
actual far-distance bounds for atoms assigned to other centers,
both in this larger neighborhood and in the narrow cluster.
These statements retain circular distance and arbitrary lifts,
so the induction does not assume a common real-angle chart.
-/

@[expose] public section

open scoped BigOperators
open Real Set

namespace Transformer.Causal

variable {n m : ℕ}

/-- Nonnegative actual last-row curvature with a strictly negative frozen-center interaction forces a positive interaction with a predecessor or frozen center.
Source: arXiv:2411.04990v2, §5, the positive-summand step in the proof of `thm:fixed_centers`, including the zero-total case. -/
theorem frozenLastCurvature_exists_pos (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 < a j)
    (p : Idx n → ℝ) (u : ℝ) (hcurv : 0 ≤ frozenLastCurvature β θ a p u)
    (hneg : ∃ j, g_pot β (u - θ j) < 0) :
    (∃ j, 0 < g_pot β (u - p j)) ∨ ∃ j, 0 < g_pot β (u - θ j) := by
  by_contra h
  push Not at h
  have hp : (∑ j, g_pot β (u - p j)) ≤ 0 := Finset.sum_nonpos (fun j _ => h.1 j)
  have hc : (∑ j, a j * g_pot β (u - θ j)) < 0 := by
    apply Finset.sum_neg' (fun j _ => mul_nonpos_of_nonneg_of_nonpos (ha j).le (h.2 j))
    obtain ⟨j, hj⟩ := hneg
    exact ⟨j, Finset.mem_univ _, mul_neg_of_pos_of_neg (ha j) hj⟩
  unfold frozenLastCurvature at hcurv
  linarith

/-- If every actual predecessor is near an original center, nonnegative last-row curvature locates the current token within one peak plus that radius of an original center.
Source: arXiv:2411.04990v2, §5, the localization step of the induction for `thm:fixed_centers`; separation supplies the required strictly negative term. -/
theorem frozenLastCurvature_localizes (β : ℝ) (hβ : 0 < β) (θ a : Idx m → ℝ)
    (ha : ∀ j, 0 < a j) (p : Idx n → ℝ) (e : ℝ) (he : 0 ≤ e)
    (hpre : ∀ j, ∃ i, circDist (p j - θ i) ≤ e)
    (hsep : ∃ i j, 2 * interactionPeak β < circDist (θ i - θ j))
    (u : ℝ) (hcurv : 0 ≤ frozenLastCurvature β θ a p u) :
    ∃ i, circDist (u - θ i) < interactionPeak β + e := by
  have hneg : ∃ j, g_pot β (u - θ j) < 0 := by
    obtain ⟨i, j, hij⟩ := hsep
    obtain ⟨k, hk, hg⟩ := separated_centers_exists_neg_g Finset.univ θ β hβ u
      ⟨i, Finset.mem_univ _, j, Finset.mem_univ _, hij⟩
    exact ⟨k, hg⟩
  rcases frozenLastCurvature_exists_pos β θ a ha p u hcurv hneg with ⟨j, hj⟩ | ⟨j, hj⟩
  · obtain ⟨i, hi⟩ := hpre j
    have hnear := (g_pot_pos_iff_circDist_lt_peak β hβ _).mp hj
    exact ⟨i, lt_of_le_of_lt (circDist_triangle u (p j) (θ i)) (by linarith)⟩
  · exact ⟨j, (g_pot_pos_iff_circDist_lt_peak β hβ _).mp hj |>.trans_le (by linarith)⟩

/-- The unchanged original separation parameters give peak-plus-neighborhood localization of the next token whenever all its actual predecessors are clustered.
Source: arXiv:2411.04990v2, §5, the induction-localization assertion in `thm:fixed_centers`. -/
theorem frozenLastCurvature_localizes_of_parameters (β ε c : ℝ) (hβ : 0 < β) (hε : 0 < ε)
    (hc : 2 + 2 * ε < c) (θ a : Idx m → ℝ) (ha : ∀ j, 1 ≤ a j)
    (hsep : ∀ i j, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j))
    (hpair : ∃ i j : Idx m, i ≠ j) (p : Idx n → ℝ)
    (hpre : ∀ j, ∃ i, circDist (p j - θ i) ≤ ε * β ^ (-(1 / 2 : ℝ)))
    (u : ℝ) (hcurv : 0 ≤ frozenLastCurvature β θ a p u) :
    ∃ i, circDist (u - θ i) < interactionPeak β + ε * β ^ (-(1 / 2 : ℝ)) := by
  apply frozenLastCurvature_localizes β hβ θ a (fun j => lt_of_lt_of_le one_pos (ha j)) p _
    (mul_pos hε (Real.rpow_pos_of_pos hβ _)).le hpre _ u hcurv
  obtain ⟨i, j, hij⟩ := hpair
  refine ⟨i, j, ?_⟩
  have h := frozen_parameter_peak_separation β ε c (θ i) (θ j) hβ hc (hsep i j hij)
  have he := mul_pos hε (Real.rpow_pos_of_pos hβ (-(1 / 2 : ℝ)))
  linarith

/-- Any actual atom assigned to another original center is beyond the original force comparison radius when the current token is within the peak-plus-cluster neighborhood.
Source: arXiv:2411.04990v2, §5, the far-force distance estimate in the induction for `thm:fixed_centers`. -/
theorem frozen_atom_far_from_peak_cluster (β ε c : ℝ) (hβ : 0 < β) (θ ψ u v : ℝ)
    (hsep : c * β ^ (-(1 / 2 : ℝ)) < circDist (θ - ψ))
    (hu : circDist (u - θ) ≤ interactionPeak β + ε * β ^ (-(1 / 2 : ℝ)))
    (hv : circDist (v - ψ) ≤ ε * β ^ (-(1 / 2 : ℝ))) :
    (c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ)) < circDist (u - v) := by
  have h := circDist_between_neighborhoods θ ψ u v _ _ _ hsep hu hv
  have ht := interactionPeak_lt_scale β hβ
  linarith

/-- Any actual atom assigned to another original center is beyond the original curvature comparison radius when both atoms lie in their original clusters.
Source: arXiv:2411.04990v2, §5, the far-curvature distance estimate in the induction for `thm:fixed_centers`. -/
theorem frozen_atom_far_from_cluster (β ε c θ ψ u v : ℝ)
    (hsep : c * β ^ (-(1 / 2 : ℝ)) < circDist (θ - ψ))
    (hu : circDist (u - θ) ≤ ε * β ^ (-(1 / 2 : ℝ)))
    (hv : circDist (v - ψ) ≤ ε * β ^ (-(1 / 2 : ℝ))) :
    (c - 2 * ε) * β ^ (-(1 / 2 : ℝ)) < circDist (u - v) := by
  have h := circDist_between_neighborhoods θ ψ u v _ _ _ hsep hu hv
  linarith

/-- Actual positive center masses and a genuine predecessor jointly realize nonnegative original last-row curvature, a negative center term, and the predecessor-cluster premises.
Source: arXiv:2411.04990v2, §5, the two-center configuration in the induction for `thm:fixed_centers`. -/
example : (0 : ℝ) < 1 ∧ (∀ j : Idx 2, (0 : ℝ) < (fun _ : Idx 2 => (1 : ℝ)) j) ∧
    0 ≤ frozenLastCurvature 1 (![0, Real.pi] : Idx 2 → ℝ) (fun _ => 1) (fun _ : Idx 1 => 0) 0 ∧
    (∃ j : Idx 2, g_pot 1 (0 - (![0, Real.pi] : Idx 2 → ℝ) j) < 0) ∧
    (0 : ℝ) ≤ interactionPeak 1 / 4 ∧
    (∀ j : Idx 1, ∃ i : Idx 2, circDist ((fun _ : Idx 1 => 0) j - (![0, Real.pi] : Idx 2 → ℝ) i) ≤ interactionPeak 1 / 4) ∧
    ∃ i j : Idx 2, 2 * interactionPeak 1 < circDist ((![0, Real.pi] : Idx 2 → ℝ) i - (![0, Real.pi] : Idx 2 → ℝ) j) := by
  have ht := interactionPeak_properties 1 one_pos
  have hpi : circDist Real.pi = Real.pi := circDist_eq_self _ ⟨Real.pi_pos.le, le_rfl⟩
  have hexp : Real.exp (-2 : ℝ) ≤ 1 := Real.exp_le_one_iff.mpr (by norm_num)
  refine ⟨one_pos, fun _ => one_pos, ?_, ?_, by linarith [ht.1], ?_, 0, 1, ?_⟩
  · norm_num [frozenLastCurvature, Fin.sum_univ_one, Fin.sum_univ_two, g_pot]
    linarith
  · refine ⟨1, ?_⟩
    norm_num [g_pot]
    exact Real.exp_pos _
  · intro j
    refine ⟨0, ?_⟩
    simp only [Matrix.cons_val_zero, sub_self, circDist_zero]
    linarith [ht.1]
  · simp only [Matrix.cons_val_zero, Matrix.cons_val_one, zero_sub, circDist_neg, hpi]
    linarith [ht.2.1]

/-- Distinct original antipodal centers and actual tokens at their centers jointly satisfy both far-distance comparison hypotheses.
Source: arXiv:2411.04990v2, §5, the original circular distance estimates for `thm:fixed_centers`. -/
example : (0 : ℝ) < 1 ∧ (3 : ℝ) * (1 : ℝ) ^ (-(1 / 2 : ℝ)) < circDist (0 - Real.pi) ∧
    circDist (0 - 0) ≤ interactionPeak 1 + (1 / 4 : ℝ) * (1 : ℝ) ^ (-(1 / 2 : ℝ)) ∧
    circDist (0 - 0) ≤ (1 / 4 : ℝ) * (1 : ℝ) ^ (-(1 / 2 : ℝ)) ∧
    circDist (Real.pi - Real.pi) ≤ (1 / 4 : ℝ) * (1 : ℝ) ^ (-(1 / 2 : ℝ)) := by
  have ht := (interactionPeak_properties 1 one_pos).1
  have hpi : circDist Real.pi = Real.pi := circDist_eq_self _ ⟨Real.pi_pos.le, le_rfl⟩
  simp only [Real.one_rpow, mul_one, zero_sub, sub_self, circDist_zero, circDist_neg, hpi]
  exact ⟨one_pos, Real.pi_gt_three, by linarith, by norm_num, by norm_num⟩

/-- Two actual antipodal centers and a genuine predecessor at the first center jointly realize the original separation and prefix-assignment hypotheses.
Source: arXiv:2411.04990v2, §5, the actual circular induction configuration for `thm:fixed_centers`. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 4 ∧ 2 + 2 * (1 / 4 : ℝ) < 3 ∧
    (∀ j : Idx 2, (1 : ℝ) ≤ (fun _ : Idx 2 => (1 : ℝ)) j) ∧
    (∀ i j : Idx 2, i ≠ j → (3 : ℝ) * (1 : ℝ) ^ (-(1 / 2 : ℝ)) <
      circDist ((![0, Real.pi] : Idx 2 → ℝ) i - (![0, Real.pi] : Idx 2 → ℝ) j)) ∧
    (∃ i j : Idx 2, i ≠ j) ∧
    ∀ j : Idx 1, ∃ i : Idx 2, circDist ((fun _ : Idx 1 => 0) j - (![0, Real.pi] : Idx 2 → ℝ) i) ≤
      (1 / 4 : ℝ) * (1 : ℝ) ^ (-(1 / 2 : ℝ)) := by
  have hpi : circDist Real.pi = Real.pi := circDist_eq_self _ ⟨Real.pi_pos.le, le_rfl⟩
  refine ⟨one_pos, by norm_num, by norm_num, fun _ => le_rfl, ?_, ⟨0, 1, by decide⟩, ?_⟩
  · intro i j hij
    fin_cases i <;> fin_cases j <;> simp_all [circDist_neg, Real.pi_gt_three]
  · intro j
    refine ⟨0, ?_⟩
    norm_num [circDist_zero]

end Transformer.Causal
