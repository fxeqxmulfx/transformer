module

public import Transformer.Causal.Section5_CenterForce
public import Transformer.Causal.Section5_FrozenLastRow

/-!
# Nondegenerate critical points of the first original frozen token

ArXiv:2411.04990v2, §5, induction base and diagonal stability
calculation in the proof of `thm:fixed_centers`.
Inside an original epsilon neighborhood, its original frozen
center contributes at least the original edge curvature. Every
other actual center is in the original far regime. The original
signed `g` inequality makes the actual weighted curvature strictly
positive, with the original full total-mass bound retained.

Combining this with the proved original critical-force localization
classifies every first-token critical point for two or more actual
separated centers. Its actual curvature is strictly negative or
strictly positive; in the latter case it lies within an original
epsilon neighborhood. The actual original last-row derivative
formula then gives a strictly repelling or strictly attracting
scalar critical point. No zero derivative is discarded, and the
original normalization and every actual frozen mass remain present.
-/

@[expose] public section

open scoped BigOperators
open Real Set

namespace Transformer.Causal

variable {m : ℕ}

/-- The original weighted center curvature is strictly positive throughout each actual original epsilon neighborhood under the original signed curvature condition.
Source: arXiv:2411.04990v2, §5, the first-token diagonal stability estimate in `thm:fixed_centers`. -/
theorem frozen_center_curvature_pos_near (N β ε c : ℝ) (hβ : 0 < β) (hε : 0 < ε)
    (hc : 2 + 2 * ε < c) (θ a : Idx m → ℝ) (ha : ∀ j, 1 ≤ a j)
    (hsep : ∀ i j : Idx m, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j))
    (hpair : ∃ i j : Idx m, i ≠ j) (hN : (∑ j, a j) ≤ N)
    (hg : -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ)))) <
      g_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (u : ℝ) (i : Idx m) (hi : circDist (u - θ i) ≤ ε * β ^ (-(1 / 2 : ℝ))) :
    0 < ∑ j, a j * g_pot β (u - θ j) := by
  have hs := Real.rpow_pos_of_pos hβ (-(1 / 2 : ℝ))
  have hr := frozen_parameter_radii β ε c hβ hε hc
  obtain ⟨p, q, hpq⟩ := hpair
  have hcpi := frozen_separation_scale_lt_pi β c (θ p) (θ q) (hsep p q hpq)
  have hrpi := (frozen_parameter_radii_lt_pi β ε c hβ hε hcpi).2
  have hsingle := Finset.single_le_sum (s := Finset.univ) (f := a)
    (fun j _ => le_trans zero_le_one (ha j)) (Finset.mem_univ i)
  have hN0 : 0 < N := by linarith [ha i]
  have he := frozen_parameter_core_lt_peak N β ε c hN0 hβ hε hc hcpi hg
  have hfar : ∀ j ∈ (Finset.univ : Finset (Idx m)), j ≠ i →
      (c - 2 * ε) * β ^ (-(1 / 2 : ℝ)) ≤ circDist (u - θ j) := by
    intro j hj hji
    have hd := hsep i j (Ne.symm hji)
    have ht := circDist_triangle (θ i) u (θ j)
    rw [show θ i - u = -(u - θ i) by ring, circDist_neg] at ht
    nlinarith [mul_pos hε hs]
  have hb := weighted_g_pot_core_tail_lower Finset.univ a θ
    (fun j _ => le_trans zero_le_one (ha j)) β hβ u _ _ i (Finset.mem_univ _) (ha i) hi he hr.2.2.2.le hfar
  have hneg := g_pot_far_neg β hβ _ hrpi.le hr.2.2.2.le
  have hm : (∑ j, a j) - a i ≤ N := by linarith [ha i]
  have hp := mul_le_mul_of_nonpos_right hm hneg.le
  linarith

/-- Every actual original first-token critical point is strictly repelling or has strictly positive original curvature inside an original epsilon neighborhood.
Source: arXiv:2411.04990v2, §5, the first-token nondegeneracy and clustering steps of `thm:fixed_centers`. -/
theorem frozen_center_critical_dichotomy (N β ε c : ℝ) (hβ : 0 < β) (hε : 0 < ε)
    (hc : 2 + 2 * ε < c) (θ a : Idx m → ℝ) (ha : ∀ j, 1 ≤ a j)
    (hsep : ∀ i j : Idx m, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j))
    (hpair : ∃ i j : Idx m, i ≠ j) (hN : (∑ j, a j) ≤ N)
    (hh : N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < h_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (hg : -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ)))) < g_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (u : ℝ) (hcrit : ∑ j, a j * h_pot β (u - θ j) = 0) :
    (∑ j, a j * g_pot β (u - θ j)) < 0 ∨
      (0 < ∑ j, a j * g_pot β (u - θ j)) ∧ ∃ i, circDist (u - θ i) < ε * β ^ (-(1 / 2 : ℝ)) := by
  by_cases hcurv : 0 ≤ ∑ j, a j * g_pot β (u - θ j)
  · obtain ⟨i, hi⟩ := frozen_center_critical_clusters N β ε c hβ hε hc θ a ha hsep hpair hN hh u hcrit hcurv
    exact Or.inr ⟨frozen_center_curvature_pos_near N β ε c hβ hε hc θ a ha hsep hpair hN hg u i hi.le, i, hi⟩
  · exact Or.inl (lt_of_not_ge hcurv)

/-- The actual original first-row velocity has a strictly signed derivative at every critical point; every attracting one is in an original epsilon neighborhood.
Source: arXiv:2411.04990v2, §5, the actual original first-token diagonal Jacobian in `thm:fixed_centers`. -/
theorem frozen_first_velocity_critical_dichotomy (N β ε c : ℝ) (hβ : 0 < β) (hε : 0 < ε)
    (hc : 2 + 2 * ε < c) (θ a : Idx m → ℝ) (ha : ∀ j, 1 ≤ a j)
    (hsep : ∀ i j : Idx m, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j))
    (hpair : ∃ i j : Idx m, i ≠ j) (hN : (∑ j, a j) ≤ N)
    (hh : N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < h_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (hg : -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ)))) < g_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (u : ℝ) (hcrit : frozenVelocity 1 m β θ a (Fin.snoc (fun _ : Idx 0 => 0) u) (Fin.last 0) = 0) :
    0 < deriv (fun v => frozenVelocity 1 m β θ a (Fin.snoc (fun _ : Idx 0 => 0) v) (Fin.last 0)) u ∨
      (deriv (fun v => frozenVelocity 1 m β θ a (Fin.snoc (fun _ : Idx 0 => 0) v) (Fin.last 0)) u < 0 ∧
        ∃ i, circDist (u - θ i) < ε * β ^ (-(1 / 2 : ℝ))) := by
  have ha0 : ∀ j, 0 ≤ a j := fun j => le_trans zero_le_one (ha j)
  have hforce := (frozenVelocity_last_eq_zero_iff β θ a ha0 (fun _ : Idx 0 => 0) u).mp hcrit
  have hf : (∑ j, a j * h_pot β (u - θ j)) = 0 := by simpa [frozenLastForce, Idx] using hforce
  have hcurv : frozenLastCurvature β θ a (fun _ : Idx 0 => 0) u = ∑ j, a j * g_pot β (u - θ j) := by
    simp [frozenLastCurvature, Idx]
  have hd := deriv_frozenLastVelocity_of_critical β θ a ha0 (fun _ : Idx 0 => 0) u hcrit
  rw [hcurv] at hd
  rcases frozen_center_critical_dichotomy N β ε c hβ hε hc θ a ha hsep hpair hN hh hg u hf with hn | ⟨hp, hi⟩
  · left
    rw [hd]
    exact div_pos (neg_pos.mpr hn) (frozenLastPartition_pos β θ a ha0 (fun _ : Idx 0 => 0) u)
  · right
    refine ⟨?_, hi⟩
    apply (deriv_frozenLastVelocity_neg_iff β θ a ha0 (fun _ : Idx 0 => 0) u hcrit).mpr
    rw [hcurv]
    exact hp

/-- Every actual original first-row critical point has nonzero actual derivative under the original frozen-center parameters.
Source: arXiv:2411.04990v2, §5, the first-token nondegeneracy required for `thm:fixed_centers`. -/
theorem frozen_first_velocity_critical_nondegenerate (N β ε c : ℝ) (hβ : 0 < β) (hε : 0 < ε)
    (hc : 2 + 2 * ε < c) (θ a : Idx m → ℝ) (ha : ∀ j, 1 ≤ a j)
    (hsep : ∀ i j : Idx m, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j))
    (hpair : ∃ i j : Idx m, i ≠ j) (hN : (∑ j, a j) ≤ N)
    (hh : N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < h_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (hg : -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ)))) < g_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (u : ℝ) (hcrit : frozenVelocity 1 m β θ a (Fin.snoc (fun _ : Idx 0 => 0) u) (Fin.last 0) = 0) :
    deriv (fun v => frozenVelocity 1 m β θ a (Fin.snoc (fun _ : Idx 0 => 0) v) (Fin.last 0)) u ≠ 0 := by
  rcases frozen_first_velocity_critical_dichotomy N β ε c hβ hε hc θ a ha hsep hpair hN hh hg u hcrit with
    hp | ⟨hn, hi⟩
  · exact hp.ne'
  · exact hn.ne

/-- Actual separated original centers jointly realize both original parameter inequalities and genuine actual first-row critical points with both strict curvature signs.
Source: arXiv:2411.04990v2, §5, `thm:fixed_centers`, with the original Appendix B parameter window. -/
example : ∃ N β ε c : ℝ, ∃ θ a : Idx 2 → ℝ,
    0 < β ∧ 0 < ε ∧ 2 + 2 * ε < c ∧ (∀ j, 1 ≤ a j) ∧
    (∀ i j : Idx 2, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j)) ∧
    (∃ i j : Idx 2, i ≠ j) ∧ (∑ j, a j) ≤ N ∧
    N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < h_pot β (ε * β ^ (-(1 / 2 : ℝ))) ∧
    -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ)))) < g_pot β (ε * β ^ (-(1 / 2 : ℝ))) ∧
    (∃ i, circDist (0 - θ i) ≤ ε * β ^ (-(1 / 2 : ℝ))) ∧
    frozenVelocity 1 2 β θ a (Fin.snoc (fun _ : Idx 0 => 0) 0) (Fin.last 0) = 0 ∧
    frozenVelocity 1 2 β θ a (Fin.snoc (fun _ : Idx 0 => 0) (Real.pi / 2)) (Fin.last 0) = 0 ∧
    (0 < ∑ j, a j * g_pot β (0 - θ j)) ∧ (∑ j, a j * g_pot β (Real.pi / 2 - θ j)) < 0 := by
  obtain ⟨N, β, ε, c, θ, a, hβ, hε, hc, ha, hsep, hpair, hN, hh, hg, hF, hG, hM, hfar⟩ :=
    frozen_sample_center_critical_data
  have ha0 : ∀ j, 0 ≤ a j := fun j => le_trans zero_le_one (ha j)
  obtain ⟨i, hi⟩ := frozen_center_critical_clusters N β ε c hβ hε hc θ a ha hsep hpair hN hh 0 hF hG
  refine ⟨N, β, ε, c, θ, a, hβ, hε, hc, ha, hsep, hpair, hN, hh, hg, ⟨i, hi.le⟩, ?_, ?_, ?_, ?_⟩
  · apply (frozenVelocity_last_eq_zero_iff β θ a ha0 (fun _ : Idx 0 => 0) 0).mpr
    simpa [frozenLastForce, Idx] using hF
  · apply (frozenVelocity_last_eq_zero_iff β θ a ha0 (fun _ : Idx 0 => 0) (Real.pi / 2)).mpr
    simpa [frozenLastForce, Idx] using hM
  · exact frozen_center_curvature_pos_near N β ε c hβ hε hc θ a ha hsep hpair hN hg 0 i hi.le
  · exact frozen_center_curvature_neg_outside N β ε c hβ hε hc θ a ha hsep hpair hN hh
      (Real.pi / 2) hM hfar

end Transformer.Causal
