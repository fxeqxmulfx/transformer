module

public import Transformer.Causal.Section5_PrefixGeometry
public import Transformer.Causal.Section5_AlignedForce
public import Transformer.Causal.Section5_NarrowParameters

/-!
# The next nonrepelling original critical token is clustered

ArXiv:2411.04990v2, §5, the critical-point induction in the
proof of `thm:fixed_centers`.
Assume every actual predecessor is already near an original
frozen center, at a radius no larger than half the original
interaction peak. Nonnegative last-row curvature first puts
the current token within one peak plus that radius of a center.

If it were outside the smaller ball, the center and all its
assigned predecessors would exert forces with the same sign.
Their magnitude is at least the original force at the radius.
All other actual atoms lie in the far regime, and their full
weighted force is at most the original total mass times the
original comparison force. The original strict force condition
contradicts criticality. Every predecessor and frozen mass is
retained, and all distances use the original circular metric.
-/

@[expose] public section

open scoped BigOperators
open Real Set

namespace Transformer.Causal

variable {n m : ℕ}

/-- Every actual nonrepelling last-token critical point is clustered if its actual predecessors are clustered, under the original force condition at a half-peak radius.
Source: arXiv:2411.04990v2, §5, the induction step in `thm:fixed_centers`; the sufficient narrow-radius premise is explicit and is derived from the unchanged original parameters in `frozen_parameters_narrow`. -/
theorem frozen_last_critical_clusters_narrow (N β ε c : ℝ) (hβ : 0 < β) (hε : 0 < ε)
    (hc : 2 + 2 * ε < c) (hnarrow : ε * β ^ (-(1 / 2 : ℝ)) ≤ interactionPeak β / 2)
    (θ a : Idx m → ℝ) (ha : ∀ j, 1 ≤ a j)
    (hsep : ∀ i j, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j))
    (hpair : ∃ i j : Idx m, i ≠ j) (hN : (n : ℝ) + (∑ j, a j) ≤ N)
    (hh : N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < h_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (p : Idx n → ℝ) (hpre : ∀ j, ∃ i, circDist (p j - θ i) ≤ ε * β ^ (-(1 / 2 : ℝ)))
    (u : ℝ) (hcrit : frozenLastForce β θ a p u = 0) (hcurv : 0 ≤ frozenLastCurvature β θ a p u) :
    ∃ i, circDist (u - θ i) < ε * β ^ (-(1 / 2 : ℝ)) := by
  classical
  have hs := Real.rpow_pos_of_pos hβ (-(1 / 2 : ℝ))
  have he := mul_pos hε hs
  have hr := frozen_parameter_radii β ε c hβ hε hc
  obtain ⟨q, r, hqr⟩ := hpair
  have hcpi := frozen_separation_scale_lt_pi β c (θ q) (θ r) (hsep q r hqr)
  have hrpi := (frozen_parameter_radii_lt_pi β ε c hβ hε hcpi).1
  have hnonneg := h_pot_nonneg β (show (c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ)) ∈ Icc 0 Real.pi from
    ⟨(interactionPeak_properties β hβ).1.le.trans hr.2.1.le, hrpi.le⟩)
  obtain ⟨i, hi⟩ := frozenLastCurvature_localizes_of_parameters β ε c hβ hε hc θ a ha hsep
    ⟨q, r, hqr⟩ p hpre u hcurv
  by_contra h
  push Not at h
  choose σ hσ using hpre
  let S := Finset.univ.filter (fun j : Idx n => σ j = i)
  let T := Finset.univ.filter (fun j : Idx n => σ j ≠ i)
  have hcore := weighted_h_pot_narrow_core_lower S (fun _ => 1) p (fun _ _ => zero_le_one)
    β hβ _ (θ i) u (a i) (ha i) hnarrow (h i) hi.le (by
      intro j hj
      have hjS := (Finset.mem_filter.mp hj).2
      simpa only [hjS] using hσ j)
  simp only [one_mul] at hcore
  have htp := weighted_h_pot_far_le T (fun _ => 1) p (fun _ _ => zero_le_one) β hβ _ u hr.2.1.le (by
    intro j hj
    have hji := (Finset.mem_filter.mp hj).2
    exact (frozen_atom_far_from_peak_cluster β ε c hβ (θ i) (θ (σ j)) u (p j)
      (hsep i (σ j) hji.symm) hi.le (hσ j)).le)
  simp only [one_mul] at htp
  have htc := weighted_h_pot_far_le (Finset.univ.erase i) a θ
    (fun j _ => le_trans zero_le_one (ha j)) β hβ _ u hr.2.1.le (by
      intro j hj
      exact (frozen_atom_far_from_peak_cluster β ε c hβ (θ i) (θ j) u (θ j)
        (hsep i j (Finset.mem_erase.mp hj).1.symm) hi.le (by rw [sub_self, circDist_zero]; exact he.le)).le)
  have hpmass : (∑ j ∈ T, (1 : ℝ)) ≤ (n : ℝ) := by
    calc
      _ ≤ ∑ j : Idx n, (1 : ℝ) := Finset.sum_le_sum_of_subset_of_nonneg
        (Finset.filter_subset (fun j : Idx n => σ j ≠ i) Finset.univ) (fun _ _ _ => zero_le_one)
      _ = _ := by simp [Idx]
  have hcmass : (∑ j ∈ Finset.univ.erase i, a j) ≤ ∑ j, a j := by
    rw [Finset.sum_erase_eq_sub (Finset.mem_univ i)]
    linarith [ha i]
  have hmass : (∑ j ∈ T, (1 : ℝ)) + (∑ j ∈ Finset.univ.erase i, a j) ≤ N := by linarith
  have htail : |(∑ j ∈ T, h_pot β (u - p j)) + (∑ j ∈ Finset.univ.erase i, a j * h_pot β (u - θ j))| ≤
      N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) := by
    calc
      _ ≤ |∑ j ∈ T, h_pot β (u - p j)| + |∑ j ∈ Finset.univ.erase i, a j * h_pot β (u - θ j)| := abs_add_le _ _
      _ ≤ (∑ j ∈ T, (1 : ℝ)) * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) +
          (∑ j ∈ Finset.univ.erase i, a j) * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) := add_le_add htp htc
      _ ≤ _ := by
        have hm := mul_le_mul_of_nonneg_right hmass hnonneg
        nlinarith only [hm]
  have hsplitp := Finset.sum_filter_add_sum_filter_not Finset.univ (fun j : Idx n => σ j = i)
    (fun j => h_pot β (u - p j))
  have hsplitc := Finset.add_sum_erase Finset.univ (fun j => a j * h_pot β (u - θ j)) (Finset.mem_univ i)
  have hforce : a i * h_pot β (u - θ i) + (∑ j ∈ S, h_pot β (u - p j)) =
      -((∑ j ∈ T, h_pot β (u - p j)) + (∑ j ∈ Finset.univ.erase i, a j * h_pot β (u - θ j))) := by
    unfold frozenLastForce at hcrit
    change (∑ j ∈ S, h_pot β (u - p j)) + (∑ j ∈ T, h_pot β (u - p j)) = _ at hsplitp
    linarith
  rw [hforce, abs_neg] at hcore
  have hlower := h_pot_narrow_window_lower β hβ _ he.le hnarrow (circDist (u - θ i)) ⟨h i, hi.le⟩
  rw [← abs_h_pot_eq_h_pot_circDist β (u - θ i)] at hlower
  linarith

/-- Under the unchanged original parameters, every nonrepelling actual critical last token joins the derived narrow clusters of its actual predecessors.
Source: arXiv:2411.04990v2, §5, the induction step for `thm:fixed_centers`; the half-peak radius is derived, and the original temperature, separation, and total mass are retained. -/
theorem frozen_last_critical_clusters (N β ε c : ℝ) (hN3 : 3 ≤ N) (hβ : 0 < β) (hε : 0 < ε)
    (hc : 2 + 2 * ε < c) (θ a : Idx m → ℝ) (ha : ∀ j, 1 ≤ a j)
    (hsep : ∀ i j, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j))
    (hpair : ∃ i j : Idx m, i ≠ j) (hN : (n : ℝ) + (∑ j, a j) ≤ N)
    (hh : N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < h_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (hg : -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ)))) < g_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (p : Idx n → ℝ)
    (hpre : ∀ j, ∃ i, circDist (p j - θ i) ≤ frozenNarrowRadius β ε * β ^ (-(1 / 2 : ℝ)))
    (u : ℝ) (hcrit : frozenLastForce β θ a p u = 0) (hcurv : 0 ≤ frozenLastCurvature β θ a p u) :
    ∃ i, circDist (u - θ i) < frozenNarrowRadius β ε * β ^ (-(1 / 2 : ℝ)) := by
  obtain ⟨i, j, hij⟩ := hpair
  have hcpi := frozen_separation_scale_lt_pi β c (θ i) (θ j) (hsep i j hij)
  rcases frozen_parameters_narrow N β ε c hN3 hβ hε hc hcpi hh hg with
    ⟨hη, hηε, hcap, hcη, hhη, hgη⟩
  exact frozen_last_critical_clusters_narrow N β (frozenNarrowRadius β ε) c hβ hη hcη hcap θ a ha
    hsep ⟨i, j, hij⟩ hN hhη p hpre u hcrit hcurv

/-- The original nontrivial parameter example yields a genuine positive narrow radius, a clustered predecessor, and an actual nonrepelling critical last token.
Source: arXiv:2411.04990v2, §5, the original antipodal-center configuration for `thm:fixed_centers`, with both parameter inequalities preserved. -/
theorem frozen_sample_narrow_prefix_data : ∃ ε : ℝ, (0 : ℝ) < 11 ∧ 0 < ε ∧ 2 + 2 * ε < 142 / 25 ∧
    ε * (11 : ℝ) ^ (-(1 / 2 : ℝ)) ≤ interactionPeak 11 / 2 ∧
    (∀ j : Idx 2, (1 : ℝ) ≤ (fun _ : Idx 2 => (1 : ℝ)) j) ∧
    (∀ i j : Idx 2, i ≠ j → (142 / 25 : ℝ) * (11 : ℝ) ^ (-(1 / 2 : ℝ)) <
      circDist ((![0, Real.pi] : Idx 2 → ℝ) i - (![0, Real.pi] : Idx 2 → ℝ) j)) ∧
    (∃ i j : Idx 2, i ≠ j) ∧
    (1 : ℝ) + (∑ j : Idx 2, (fun _ : Idx 2 => (1 : ℝ)) j) ≤ 3 ∧
    3 * h_pot 11 ((142 / 25 - 1 - 2 * ε) * (11 : ℝ) ^ (-(1 / 2 : ℝ))) < h_pot 11 (ε * (11 : ℝ) ^ (-(1 / 2 : ℝ))) ∧
    -(3 * g_pot 11 ((142 / 25 - 2 * ε) * (11 : ℝ) ^ (-(1 / 2 : ℝ)))) < g_pot 11 (ε * (11 : ℝ) ^ (-(1 / 2 : ℝ))) ∧
    (∀ j : Idx 1, ∃ i : Idx 2, circDist ((fun _ : Idx 1 => 0) j - (![0, Real.pi] : Idx 2 → ℝ) i) ≤ ε * (11 : ℝ) ^ (-(1 / 2 : ℝ))) ∧
    frozenLastForce 11 (![0, Real.pi] : Idx 2 → ℝ) (fun _ => 1) (fun _ : Idx 1 => 0) 0 = 0 ∧
    0 ≤ frozenLastCurvature 11 (![0, Real.pi] : Idx 2 → ℝ) (fun _ => 1) (fun _ : Idx 1 => 0) 0 := by
  rcases frozen_sample_parameters with ⟨hN, hβ, hε, hc, hcpi, hh, hg⟩
  rcases frozen_parameters_narrow 3 11 (9 / 100) (142 / 25) le_rfl hβ hε hc hcpi hh hg with
    ⟨hη, hηε, hcap, hcη, hhη, hgη⟩
  refine ⟨frozenNarrowRadius 11 (9 / 100), hβ, hη, hcη, hcap, fun _ => le_rfl, ?_,
    ⟨0, 1, by decide⟩, by norm_num, hhη, hgη, ?_, ?_, ?_⟩
  · have hpi : circDist Real.pi = Real.pi := circDist_eq_self _ ⟨Real.pi_pos.le, le_rfl⟩
    intro i j hij
    fin_cases i <;> fin_cases j <;> simp_all [circDist_neg]
  · intro j
    refine ⟨0, ?_⟩
    simp only [Matrix.cons_val_zero, sub_self, circDist_zero]
    exact (mul_pos hη (Real.rpow_pos_of_pos hβ _)).le
  · norm_num [frozenLastForce, Fin.sum_univ_one, Fin.sum_univ_two, h_pot]
  · have hexp : Real.exp (-22 : ℝ) ≤ 1 := Real.exp_le_one_iff.mpr (by norm_num)
    norm_num [frozenLastCurvature, Fin.sum_univ_one, Fin.sum_univ_two, g_pot]
    linarith

/-- A genuine predecessor and the next actual token at a frozen center realize the original-parameter induction conclusion, with every original premise supplied.
Source: arXiv:2411.04990v2, §5, the original two-center configuration for `thm:fixed_centers`. -/
example : ∃ i : Idx 2, circDist (0 - (![0, Real.pi] : Idx 2 → ℝ) i) <
    frozenNarrowRadius 11 (9 / 100) * (11 : ℝ) ^ (-(1 / 2 : ℝ)) := by
  rcases frozen_sample_parameters with ⟨hN, hβ, hε, hc, hcpi, hh, hg⟩
  apply frozen_last_critical_clusters 3 11 (9 / 100) (142 / 25) le_rfl hβ hε hc
    (![0, Real.pi]) (fun _ => 1) (fun _ => le_rfl) _ ⟨0, 1, by decide⟩
    (by norm_num : ((1 : ℕ) : ℝ) + (∑ j : Idx 2, (1 : ℝ)) ≤ 3) hh hg (fun _ : Idx 1 => 0) _ 0 _ _
  · have hpi : circDist Real.pi = Real.pi := circDist_eq_self _ ⟨Real.pi_pos.le, le_rfl⟩
    intro i j hij
    fin_cases i <;> fin_cases j <;> simp_all [circDist_neg]
  · intro j
    refine ⟨0, ?_⟩
    simp only [Matrix.cons_val_zero, sub_self, circDist_zero]
    exact (mul_pos (frozenNarrowRadius_properties 11 _ hβ hε).1 (Real.rpow_pos_of_pos hβ _)).le
  · norm_num [frozenLastForce, Fin.sum_univ_one, Fin.sum_univ_two, h_pot]
  · have hexp : Real.exp (-22 : ℝ) ≤ 1 := Real.exp_le_one_iff.mpr (by norm_num)
    norm_num [frozenLastCurvature, Fin.sum_univ_one, Fin.sum_univ_two, g_pot]
    linarith

end Transformer.Causal
