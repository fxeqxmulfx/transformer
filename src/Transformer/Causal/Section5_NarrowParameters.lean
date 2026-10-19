module

public import Transformer.Causal.Section5_ForceRatio
public import Transformer.Causal.Section5_NarrowWindow

/-!
# A narrow neighborhood derived from the original parameters

ArXiv:2411.04990v2, §5, `thm:fixed_centers` and its proof.
The printed critical-point argument uses a force lower bound
across the peak and nonnegative curvature within a cluster.
Both estimates have now been proved for neighborhoods of
physical radius at most half the actual interaction peak.

We reduce the original normalized radius to the minimum of
itself and that half-peak radius. Under the original nontrivial
separated parameter hypotheses, the original force and signed
curvature inequalities both hold at the smaller radius, with
the same temperature, separation constant, and total mass.
Thus the narrow radius is derived from those hypotheses.
It is not an extra hypothesis on the original theorem.

The force comparison uses the proved decrease of the original
force ratio; curvature preservation uses the original core and
far monotonicity. The final estimates apply directly to the
original force and curvature of the critical-point induction.
-/

@[expose] public section

open Real Set

namespace Transformer.Causal

/-- The original normalized radius reduced, when needed, to half the actual interaction peak.
Source: arXiv:2411.04990v2, §5, a derived smaller neighborhood for the proof of `thm:fixed_centers`. -/
noncomputable def frozenNarrowRadius (β ε : ℝ) : ℝ :=
  min ε (interactionPeak β / (2 * β ^ (-(1 / 2 : ℝ))))

/-- The reduced original radius is positive, no larger than the original radius, and physically at most half the original peak.
Source: arXiv:2411.04990v2, §5, the radius needed for the force and same-cluster curvature estimates in `thm:fixed_centers`. -/
theorem frozenNarrowRadius_properties (β ε : ℝ) (hβ : 0 < β) (hε : 0 < ε) :
    0 < frozenNarrowRadius β ε ∧ frozenNarrowRadius β ε ≤ ε ∧
      frozenNarrowRadius β ε * β ^ (-(1 / 2 : ℝ)) ≤ interactionPeak β / 2 := by
  have ht := (interactionPeak_properties β hβ).1
  have hs := Real.rpow_pos_of_pos hβ (-(1 / 2 : ℝ))
  have hcap : interactionPeak β / (2 * β ^ (-(1 / 2 : ℝ))) * β ^ (-(1 / 2 : ℝ)) = interactionPeak β / 2 := by
    field_simp
  refine ⟨lt_min hε (div_pos ht (by positivity)), min_le_left _ _, ?_⟩
  have h := mul_le_mul_of_nonneg_right (min_le_right ε (interactionPeak β / (2 * β ^ (-(1 / 2 : ℝ))))) hs.le
  simpa! only [frozenNarrowRadius, hcap] using h

/-- An original radius already physically below half the peak is unchanged by the reduction.
Source: arXiv:2411.04990v2, §5, the original neighborhood in the narrow case of the force and curvature comparisons for `thm:fixed_centers`. -/
theorem frozenNarrowRadius_eq_self (β ε : ℝ) (hβ : 0 < β)
    (hnarrow : ε * β ^ (-(1 / 2 : ℝ)) ≤ interactionPeak β / 2) : frozenNarrowRadius β ε = ε := by
  unfold frozenNarrowRadius
  apply min_eq_left
  rw [le_div_iff₀ (show 0 < 2 * β ^ (-(1 / 2 : ℝ)) by positivity)]
  nlinarith only [hnarrow]

/-- The original force inequality is preserved at the derived narrow radius under the unchanged original nontrivial separated parameter assumptions.
Source: arXiv:2411.04990v2, §5, the original `h` condition of `thm:fixed_centers`; the signed `g` condition supplies the derived subpeak original radius. -/
theorem frozen_parameter_force_narrow (N β ε c : ℝ) (hN : 3 ≤ N) (hβ : 0 < β)
    (hε : 0 < ε) (hc : 2 + 2 * ε < c) (hcpi : c * β ^ (-(1 / 2 : ℝ)) < Real.pi)
    (hh : N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < h_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (hg : -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ)))) < g_pot β (ε * β ^ (-(1 / 2 : ℝ)))) :
    N * h_pot β ((c - 1 - 2 * frozenNarrowRadius β ε) * β ^ (-(1 / 2 : ℝ))) <
      h_pot β (frozenNarrowRadius β ε * β ^ (-(1 / 2 : ℝ))) := by
  let s := β ^ (-(1 / 2 : ℝ))
  have hs : 0 < s := Real.rpow_pos_of_pos hβ _
  by_cases hsmall : ε ≤ interactionPeak β / (2 * s)
  · change ε ≤ interactionPeak β / (2 * β ^ (-(1 / 2 : ℝ))) at hsmall
    rw [frozenNarrowRadius, min_eq_left hsmall]
    exact hh
  have hcap : interactionPeak β / (2 * s) * s = interactionPeak β / 2 := by field_simp
  have hηε : interactionPeak β / (2 * s) ≤ ε := (lt_of_not_ge hsmall).le
  have he : interactionPeak β / 2 ≤ ε * s := by
    have h := mul_le_mul_of_nonneg_right hηε hs.le
    rwa [hcap] at h
  have hcore := frozen_parameter_core_lt_peak N β ε c (by linarith) hβ hε hc hcpi hg
  have hfar := frozen_force_radius_gt_double_peak N β ε c hN hβ hε hc hcpi hh
  have horig : (c - 1) * s - 2 * (ε * s) = (c - 1 - 2 * ε) * s := by ring
  have hbound := frozen_force_window_half_peak N β ((c - 1) * s) (ε * s) hβ ⟨he, hcore.le⟩
    (by rw [horig]; exact hfar.le) (by change c * s < Real.pi at hcpi; linarith)
    (by rw [horig]; exact hh)
  have harg : (c - 1 - 2 * (interactionPeak β / (2 * s))) * s = (c - 1) * s - interactionPeak β := by
    calc
      _ = (c - 1) * s - 2 * (interactionPeak β / (2 * s) * s) := by ring
      _ = _ := by rw [hcap]; ring
  change N * h_pot β ((c - 1 - 2 * min ε (interactionPeak β / (2 * s))) * s) <
    h_pot β (min ε (interactionPeak β / (2 * s)) * s)
  rw [min_eq_right hηε, harg, hcap]
  exact hbound

/-- The derived narrow radius preserves all original parameter inequalities, without changing the temperature, separation, or total mass.
Source: arXiv:2411.04990v2, §5, the original `h` and corrected-sign `g` conditions in `thm:fixed_centers`. -/
theorem frozen_parameters_narrow (N β ε c : ℝ) (hN : 3 ≤ N) (hβ : 0 < β)
    (hε : 0 < ε) (hc : 2 + 2 * ε < c) (hcpi : c * β ^ (-(1 / 2 : ℝ)) < Real.pi)
    (hh : N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < h_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (hg : -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ)))) < g_pot β (ε * β ^ (-(1 / 2 : ℝ)))) :
    0 < frozenNarrowRadius β ε ∧ frozenNarrowRadius β ε ≤ ε ∧
      frozenNarrowRadius β ε * β ^ (-(1 / 2 : ℝ)) ≤ interactionPeak β / 2 ∧
      2 + 2 * frozenNarrowRadius β ε < c ∧
      N * h_pot β ((c - 1 - 2 * frozenNarrowRadius β ε) * β ^ (-(1 / 2 : ℝ))) <
        h_pot β (frozenNarrowRadius β ε * β ^ (-(1 / 2 : ℝ))) ∧
      -(N * g_pot β ((c - 2 * frozenNarrowRadius β ε) * β ^ (-(1 / 2 : ℝ)))) <
        g_pot β (frozenNarrowRadius β ε * β ^ (-(1 / 2 : ℝ))) := by
  have hη := frozenNarrowRadius_properties β ε hβ hε
  exact ⟨hη.1, hη.2.1, hη.2.2, by linarith [hη.2.1],
    frozen_parameter_force_narrow N β ε c hN hβ hε hc hcpi hh hg,
    frozen_parameter_curvature_window_mono N β ε (frozenNarrowRadius β ε) c
      (by linarith) hβ hε hη.1.le hη.2.1 hc hcpi hg⟩

/-- The original force lower bound holds across the peak at the derived radius, directly from the original positive temperature and radius.
Source: arXiv:2411.04990v2, §5, the force lower bound in the critical-point induction for `thm:fixed_centers`, now justified on the derived narrow window. -/
theorem h_pot_frozenNarrowRadius_lower (β ε : ℝ) (hβ : 0 < β) (hε : 0 < ε) (x : ℝ)
    (hx : x ∈ Icc (frozenNarrowRadius β ε * β ^ (-(1 / 2 : ℝ)))
      (interactionPeak β + frozenNarrowRadius β ε * β ^ (-(1 / 2 : ℝ)))) :
    h_pot β (frozenNarrowRadius β ε * β ^ (-(1 / 2 : ℝ))) ≤ h_pot β x := by
  have hη := frozenNarrowRadius_properties β ε hβ hε
  exact h_pot_narrow_window_lower β hβ _
    (mul_pos hη.1 (Real.rpow_pos_of_pos hβ _)).le hη.2.2 x hx

/-- Two original particles in the derived neighborhood have nonnegative original pair curvature.
Source: arXiv:2411.04990v2, §5, the same-cluster sign step in `thm:fixed_centers`, proved at the radius derived from its unchanged assumptions. -/
theorem g_pot_frozenNarrowRadius_nonneg (β ε : ℝ) (hβ : 0 < β) (hε : 0 < ε) (θ x y : ℝ)
    (hx : circDist (x - θ) ≤ frozenNarrowRadius β ε * β ^ (-(1 / 2 : ℝ)))
    (hy : circDist (y - θ) ≤ frozenNarrowRadius β ε * β ^ (-(1 / 2 : ℝ))) : 0 ≤ g_pot β (x - y) :=
  g_pot_nonneg_in_narrow_cluster β hβ _ θ x y (frozenNarrowRadius_properties β ε hβ hε).2.2 hx hy

/-- The original nontrivial numerical parameters jointly satisfy every original premise used to derive the smaller valid neighborhood.
Source: arXiv:2411.04990v2, §5, the original parameter window in `thm:fixed_centers`, with the corrected curvature sign. -/
example : (3 : ℝ) ≤ 3 ∧ (0 : ℝ) < 11 ∧ (0 : ℝ) < 9 / 100 ∧
    2 + 2 * (9 / 100 : ℝ) < 142 / 25 ∧ (142 / 25 : ℝ) * (11 : ℝ) ^ (-(1 / 2 : ℝ)) < Real.pi ∧
    3 * h_pot 11 ((142 / 25 - 1 - 2 * (9 / 100)) * (11 : ℝ) ^ (-(1 / 2 : ℝ))) <
      h_pot 11 ((9 / 100) * (11 : ℝ) ^ (-(1 / 2 : ℝ))) ∧
    -(3 * g_pot 11 ((142 / 25 - 2 * (9 / 100)) * (11 : ℝ) ^ (-(1 / 2 : ℝ)))) <
      g_pot 11 ((9 / 100) * (11 : ℝ) ^ (-(1 / 2 : ℝ))) := by
  rcases frozen_sample_parameters with ⟨hN, hβ, hε, hc, hcpi, hh, hg⟩
  exact ⟨le_rfl, hβ, hε, hc, hcpi, hh, hg⟩

/-- A genuine narrow-window endpoint and two actual tokens at their common center jointly realize the force and curvature comparison hypotheses.
Source: arXiv:2411.04990v2, §5, the original force and pair-curvature steps in `thm:fixed_centers`. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    frozenNarrowRadius 1 1 * (1 : ℝ) ^ (-(1 / 2 : ℝ)) ∈
      Icc (frozenNarrowRadius 1 1 * (1 : ℝ) ^ (-(1 / 2 : ℝ)))
        (interactionPeak 1 + frozenNarrowRadius 1 1 * (1 : ℝ) ^ (-(1 / 2 : ℝ))) ∧
    circDist (0 - 0) ≤ frozenNarrowRadius 1 1 * (1 : ℝ) ^ (-(1 / 2 : ℝ)) := by
  have ht := (interactionPeak_properties 1 one_pos).1
  have hη := (frozenNarrowRadius_properties 1 1 one_pos one_pos).1
  rw [sub_self, circDist_zero]
  exact ⟨one_pos, one_pos, ⟨le_rfl, by linarith⟩, by positivity⟩

/-- One quarter of the actual unit-temperature peak is a positive original radius that is unchanged by the narrow-radius construction.
Source: arXiv:2411.04990v2, Appendix B and §5, the original interaction scale and narrow neighborhood. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) < interactionPeak 1 / 4 ∧
    interactionPeak 1 / 4 * (1 : ℝ) ^ (-(1 / 2 : ℝ)) ≤ interactionPeak 1 / 2 ∧
    frozenNarrowRadius 1 (interactionPeak 1 / 4) = interactionPeak 1 / 4 := by
  have ht := (interactionPeak_properties 1 one_pos).1
  have hn : interactionPeak 1 / 4 * (1 : ℝ) ^ (-(1 / 2 : ℝ)) ≤ interactionPeak 1 / 2 := by
    rw [Real.one_rpow, mul_one]
    linarith
  exact ⟨one_pos, by linarith, hn, frozenNarrowRadius_eq_self 1 _ one_pos hn⟩

end Transformer.Causal
