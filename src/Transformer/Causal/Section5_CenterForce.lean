module

public import Transformer.Causal.Section5_WeightedInteraction
public import Transformer.Causal.Section5_ParameterGeometry

/-!
# The first original frozen token clusters at every nonrepelling critical point

ArXiv:2411.04990v2, §5, induction base in the proof of
`thm:fixed_centers`. The first moving token has no preceding moving
tokens. For at least two separated original centers, nonnegative
original curvature therefore puts it strictly inside the original
peak neighborhood of one frozen center.

At an actual critical point, isolating that center's actual force
equates its magnitude with the remaining original weighted force.
If its distance were at least the original epsilon radius, the
increasing original central branch would bound that force below
by `h(epsilon beta^(-1/2))`. All other original centers are far
enough to apply the manuscript's original total-mass tail bound,
contradicting its original strict `h` parameter inequality.

The proof retains the full original temperature range, original
frozen masses, circular separation, and strict clustering radius.
Its scope is the first token; no condition on later predecessors
is assumed or hidden in the original dynamics.
-/

@[expose] public section

open scoped BigOperators
open Real Set

namespace Transformer.Causal

variable {ι : Type*} {m : ℕ}

/-- At an actual original critical point, one original center of mass at least one has force magnitude bounded by the actual remaining far mass.
Source: arXiv:2411.04990v2, §5, the critical-force splitting in `thm:fixed_centers`. -/
theorem weighted_h_pot_critical_tail_bound (s : Finset ι) (a z : ι → ℝ) (ha : ∀ j ∈ s, 0 ≤ a j)
    (β : ℝ) (hβ : 0 < β) (r u : ℝ) (hr : interactionPeak β ≤ r) (i : ι) (hi : i ∈ s)
    (hai : 1 ≤ a i) (hcrit : ∑ j ∈ s, a j * h_pot β (u - z j) = 0)
    (hfar : ∀ j ∈ s, j ≠ i → r ≤ circDist (u - z j)) :
    |h_pot β (u - z i)| ≤ ((∑ j ∈ s, a j) - a i) * h_pot β r := by
  classical
  have ht := weighted_h_pot_far_le (s.erase i) a z
    (fun j hj => ha j (Finset.mem_erase.mp hj).2) β hβ r u hr
    (fun j hj => hfar j (Finset.mem_erase.mp hj).2 (Finset.mem_erase.mp hj).1)
  rw [Finset.sum_erase_eq_sub (f := a) hi] at ht
  have hs := Finset.add_sum_erase s (fun j => a j * h_pot β (u - z j)) hi
  rw [hcrit] at hs
  have heq : a i * h_pot β (u - z i) = -(∑ j ∈ s.erase i, a j * h_pot β (u - z j)) := by linarith
  have habs := congrArg abs heq
  rw [abs_mul, abs_of_nonneg (show 0 ≤ a i by linarith), abs_neg] at habs
  have hm := mul_le_mul_of_nonneg_right hai (abs_nonneg (h_pot β (u - z i)))
  simp only [one_mul] at hm
  linarith

/-- Every actual nonrepelling first-token critical point lies in an original epsilon neighborhood when there are two or more separated original frozen centers.
Source: arXiv:2411.04990v2, §5, the first-token induction base of `thm:fixed_centers`; this helper retains the original `h` condition. -/
theorem frozen_center_critical_clusters (N β ε c : ℝ) (hβ : 0 < β) (hε : 0 < ε)
    (hc : 2 + 2 * ε < c) (θ a : Idx m → ℝ) (ha : ∀ j, 1 ≤ a j)
    (hsep : ∀ i j : Idx m, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j))
    (hpair : ∃ i j : Idx m, i ≠ j) (hN : (∑ j, a j) ≤ N)
    (hh : N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) <
      h_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (u : ℝ) (hcrit : ∑ j, a j * h_pot β (u - θ j) = 0)
    (hcurv : 0 ≤ ∑ j, a j * g_pot β (u - θ j)) :
    ∃ i : Idx m, circDist (u - θ i) < ε * β ^ (-(1 / 2 : ℝ)) := by
  have hs := Real.rpow_pos_of_pos hβ (-(1 / 2 : ℝ))
  have hr := frozen_parameter_radii β ε c hβ hε hc
  obtain ⟨p, q, hpq⟩ := hpair
  have hcpi := frozen_separation_scale_lt_pi β c (θ p) (θ q) (hsep p q hpq)
  have hrpi := (frozen_parameter_radii_lt_pi β ε c hβ hε hcpi).1
  have hseppeak : ∃ i ∈ (Finset.univ : Finset (Idx m)), ∃ j ∈ (Finset.univ : Finset (Idx m)),
      2 * interactionPeak β < circDist (θ i - θ j) := by
    refine ⟨p, Finset.mem_univ _, q, Finset.mem_univ _, ?_⟩
    have hp := frozen_parameter_peak_separation β ε c (θ p) (θ q) hβ hc (hsep p q hpq)
    nlinarith [mul_pos hε hs]
  obtain ⟨i, hi, hpeak⟩ := weighted_center_curvature_localizes Finset.univ a θ
    (fun j _ => lt_of_lt_of_le one_pos (ha j)) β hβ u hcurv hseppeak
  by_contra h
  push Not at h
  have he := h i
  have hfar : ∀ j ∈ (Finset.univ : Finset (Idx m)), j ≠ i →
      (c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ)) ≤ circDist (u - θ j) := by
    intro j hj hji
    have hd := hsep i j (Ne.symm hji)
    have ht := circDist_triangle (θ i) u (θ j)
    rw [show θ i - u = -(u - θ i) by ring, circDist_neg] at ht
    nlinarith [interactionPeak_lt_scale β hβ, mul_pos hε hs]
  have htail := weighted_h_pot_critical_tail_bound Finset.univ a θ
    (fun j _ => le_trans zero_le_one (ha j)) β hβ _ u hr.2.1.le i hi (ha i) hcrit hfar
  have hcore := (h_pot_monotone_peak β hβ).1.monotoneOn
    (show ε * β ^ (-(1 / 2 : ℝ)) ∈ Icc 0 (interactionPeak β) from ⟨hr.1.le, he.trans hpeak.le⟩)
    (show circDist (u - θ i) ∈ Icc 0 (interactionPeak β) from ⟨circDist_nonneg _, hpeak.le⟩) he
  rw [← abs_h_pot_eq_h_pot_circDist β (u - θ i)] at hcore
  have hnonneg := h_pot_nonneg β (show (c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ)) ∈ Icc 0 Real.pi from
    ⟨(interactionPeak_properties β hβ).1.le.trans hr.2.1.le, hrpi.le⟩)
  have hmass : (∑ j, a j) - a i ≤ N := by linarith [ha i]
  have hm := mul_le_mul_of_nonneg_right hmass hnonneg
  linarith

/-- Every actual original first-token critical point outside all original epsilon balls has strictly negative total curvature and is therefore repelling.
Source: arXiv:2411.04990v2, §5, the first-token stable-versus-unstable classification in `thm:fixed_centers`. -/
theorem frozen_center_curvature_neg_outside (N β ε c : ℝ) (hβ : 0 < β) (hε : 0 < ε)
    (hc : 2 + 2 * ε < c) (θ a : Idx m → ℝ) (ha : ∀ j, 1 ≤ a j)
    (hsep : ∀ i j : Idx m, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j))
    (hpair : ∃ i j : Idx m, i ≠ j) (hN : (∑ j, a j) ≤ N)
    (hh : N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) <
      h_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (u : ℝ) (hcrit : ∑ j, a j * h_pot β (u - θ j) = 0)
    (hfar : ∀ j, ε * β ^ (-(1 / 2 : ℝ)) ≤ circDist (u - θ j)) :
    (∑ j, a j * g_pot β (u - θ j)) < 0 := by
  by_contra h
  obtain ⟨i, hi⟩ := frozen_center_critical_clusters N β ε c hβ hε hc θ a ha hsep hpair hN hh u hcrit
    (le_of_not_gt h)
  exact not_lt_of_ge (hfar i) hi

/-- The actual original two-center example realizes every first-token critical-point and parameter hypothesis jointly.
Source: arXiv:2411.04990v2, §5, `thm:fixed_centers`, using Appendix B's original parameter window. -/
theorem frozen_sample_center_critical_data : ∃ N β ε c : ℝ, ∃ θ a : Idx 2 → ℝ,
    0 < β ∧ 0 < ε ∧ 2 + 2 * ε < c ∧ (∀ j, 1 ≤ a j) ∧
    (∀ i j : Idx 2, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j)) ∧
    (∃ i j : Idx 2, i ≠ j) ∧ (∑ j, a j) ≤ N ∧
    N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < h_pot β (ε * β ^ (-(1 / 2 : ℝ))) ∧
    -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ)))) < g_pot β (ε * β ^ (-(1 / 2 : ℝ))) ∧
    (∑ j, a j * h_pot β (0 - θ j)) = 0 ∧ (0 ≤ ∑ j, a j * g_pot β (0 - θ j)) ∧
    (∑ j, a j * h_pot β (Real.pi / 2 - θ j)) = 0 ∧
    (∀ j, ε * β ^ (-(1 / 2 : ℝ)) ≤ circDist (Real.pi / 2 - θ j)) := by
  rcases frozen_sample_parameters with ⟨hN, hβ, hε, hc, hcpi, hh, hg⟩
  refine ⟨3, 11, 9 / 100, 142 / 25, ![0, Real.pi], (fun _ => 1), hβ, hε, hc,
    (fun _ => le_rfl), ?_, ⟨0, 1, by decide⟩, by norm_num, hh, hg, ?_, ?_, ?_, ?_⟩
  · have hpi : circDist Real.pi = Real.pi := circDist_eq_self _ ⟨Real.pi_pos.le, le_rfl⟩
    intro i j hij
    fin_cases i <;> fin_cases j <;> simp_all [circDist_neg]
  · norm_num [Fin.sum_univ_two, h_pot]
  · norm_num [Fin.sum_univ_two, g_pot]
  · simp [Fin.sum_univ_two, h_pot]
  · have hrad := (frozen_parameter_core_lt_peak 3 11 (9 / 100) (142 / 25) hN hβ hε hc hcpi hg).trans
      (interactionPeak_properties 11 hβ).2.1
    have hhalf : circDist (Real.pi / 2) = Real.pi / 2 :=
      circDist_eq_self _ ⟨by linarith [Real.pi_pos], by linarith [Real.pi_pos]⟩
    have hle := hrad.le
    intro j
    fin_cases j <;> simp_all [show Real.pi / 2 - Real.pi = -(Real.pi / 2) by ring, circDist_neg]

/-- Actual original antipodal centers at unit temperature jointly satisfy the isolated critical-force and nonempty far-tail hypotheses.
Source: arXiv:2411.04990v2, §5, the original critical-force splitting in `thm:fixed_centers`. -/
example : (∀ j ∈ (Finset.univ : Finset (Idx 2)), (0 : ℝ) ≤ (fun _ => 1 : Idx 2 → ℝ) j) ∧
    (0 : ℝ) < 1 ∧ interactionPeak 1 ≤ 2 ∧ (0 : Idx 2) ∈ Finset.univ ∧ (1 : ℝ) ≤ 1 ∧
    (∑ j : Idx 2, (1 : ℝ) * h_pot 1 (0 - (![0, Real.pi] : Idx 2 → ℝ) j)) = 0 ∧
    ∀ j ∈ (Finset.univ : Finset (Idx 2)), j ≠ 0 → (2 : ℝ) ≤ circDist (0 - (![0, Real.pi] : Idx 2 → ℝ) j) := by
  have ht := interactionPeak_lt_scale 1 one_pos
  norm_num at ht
  refine ⟨fun _ _ => zero_le_one, one_pos, by linarith, Finset.mem_univ _, le_rfl,
    by norm_num [Fin.sum_univ_two, h_pot], ?_⟩
  have hpi : circDist Real.pi = Real.pi := circDist_eq_self _ ⟨Real.pi_pos.le, le_rfl⟩
  intro j hj hj0
  fin_cases j <;> simp_all [circDist_neg]
  linarith [Real.pi_gt_three]

/-- The original numerical configuration jointly realizes every clustering and outside-repulsion premise, with both actual conclusions checked.
Source: arXiv:2411.04990v2, §5, the first-token induction base of `thm:fixed_centers`. -/
example : ∃ N β ε c : ℝ, ∃ θ a : Idx 2 → ℝ,
    0 < β ∧ 0 < ε ∧ 2 + 2 * ε < c ∧ (∀ j, 1 ≤ a j) ∧
    (∀ i j : Idx 2, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j)) ∧
    (∃ i j : Idx 2, i ≠ j) ∧ (∑ j, a j) ≤ N ∧
    N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < h_pot β (ε * β ^ (-(1 / 2 : ℝ))) ∧
    (∑ j, a j * h_pot β (0 - θ j)) = 0 ∧ (0 ≤ ∑ j, a j * g_pot β (0 - θ j)) ∧
    (∃ i, circDist (0 - θ i) < ε * β ^ (-(1 / 2 : ℝ))) ∧
    (∑ j, a j * h_pot β (Real.pi / 2 - θ j)) = 0 ∧
    (∀ j, ε * β ^ (-(1 / 2 : ℝ)) ≤ circDist (Real.pi / 2 - θ j)) ∧
    (∑ j, a j * g_pot β (Real.pi / 2 - θ j)) < 0 := by
  obtain ⟨N, β, ε, c, θ, a, hβ, hε, hc, ha, hsep, hpair, hN, hh, hg, hF, hG, hM, hfar⟩ :=
    frozen_sample_center_critical_data
  refine ⟨N, β, ε, c, θ, a, hβ, hε, hc, ha, hsep, hpair, hN, hh, hF, hG, ?_, hM, hfar, ?_⟩
  · exact frozen_center_critical_clusters N β ε c hβ hε hc θ a ha hsep hpair hN hh 0 hF hG
  · exact frozen_center_curvature_neg_outside N β ε c hβ hε hc θ a ha hsep hpair hN hh (Real.pi / 2) hM hfar

end Transformer.Causal
