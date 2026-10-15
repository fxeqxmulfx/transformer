/-
# Strict curvature at the next clustered original critical token

ArXiv:2411.04990v2, §5, the final diagonal-curvature estimate
in the proof of `thm:fixed_centers`.
For the derived half-peak radius, every same-cluster predecessor
has nonnegative original curvature. Every other predecessor or
center is in the original far regime. Retaining the original
center's positive contribution and the original full mass bound
therefore makes the actual last-row curvature strictly positive
throughout each neighborhood.

Combined with the critical-force localization, this classifies
every last-token critical point once its actual predecessors
are clustered. Its curvature is strictly negative, or strictly
positive inside a derived narrow neighborhood. In particular,
zero curvature is excluded. The final statement uses the
unchanged original parameters; the narrow radius is derived.
-/

import Transformer.Causal.Section5_PrefixForce

open scoped BigOperators
open Real Set

namespace Transformer.Causal

variable {n m : ℕ}

/-- At a narrow-cluster token with clustered actual predecessors, the original signed curvature condition makes the full original last-row curvature strictly positive.
Source: arXiv:2411.04990v2, §5, the final signed `g` estimate in `thm:fixed_centers`, with the narrow-radius premise explicit. -/
theorem frozen_last_curvature_pos_near_narrow (N β ε c : ℝ) (hβ : 0 < β) (hε : 0 < ε)
    (hc : 2 + 2 * ε < c) (hnarrow : ε * β ^ (-(1 / 2 : ℝ)) ≤ interactionPeak β / 2)
    (θ a : Idx m → ℝ) (ha : ∀ j, 1 ≤ a j)
    (hsep : ∀ i j, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j))
    (hpair : ∃ i j : Idx m, i ≠ j) (hN : (n : ℝ) + (∑ j, a j) ≤ N)
    (hg : -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ)))) < g_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (p : Idx n → ℝ) (hpre : ∀ j, ∃ i, circDist (p j - θ i) ≤ ε * β ^ (-(1 / 2 : ℝ)))
    (u : ℝ) (i : Idx m) (hi : circDist (u - θ i) ≤ ε * β ^ (-(1 / 2 : ℝ))) :
    0 < frozenLastCurvature β θ a p u := by
  have hs := Real.rpow_pos_of_pos hβ (-(1 / 2 : ℝ))
  have he := mul_pos hε hs
  have ht := interactionPeak_properties β hβ
  have hr := frozen_parameter_radii β ε c hβ hε hc
  obtain ⟨q, r, hqr⟩ := hpair
  have hcpi := frozen_separation_scale_lt_pi β c (θ q) (θ r) (hsep q r hqr)
  have hrpi := (frozen_parameter_radii_lt_pi β ε c hβ hε hcpi).2
  have hneg := g_pot_far_neg β hβ _ hrpi.le hr.2.2.2.le
  have hcentral := weighted_g_pot_core_tail_lower Finset.univ a θ
    (fun j _ => le_trans zero_le_one (ha j)) β hβ u _ _ i (Finset.mem_univ _) (ha i) hi
    (by linarith [hnarrow, ht.1]) hr.2.2.2.le (by
      intro j hj hji
      exact (frozen_atom_far_from_cluster β ε c (θ i) (θ j) u (θ j) (hsep i j hji.symm) hi
        (by rw [sub_self, circDist_zero]; exact he.le)).le)
  have hp : ∀ j, g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ))) ≤ g_pot β (u - p j) := by
    intro j
    obtain ⟨k, hk⟩ := hpre j
    by_cases hki : k = i
    · subst k
      exact hneg.le.trans (g_pot_nonneg_in_narrow_cluster β hβ _ (θ i) u (p j) hnarrow hi hk)
    · exact (g_pot_circular_far_bounds β hβ _ _ hr.2.2.2.le
        (frozen_atom_far_from_cluster β ε c (θ i) (θ k) u (p j) (hsep i k (Ne.symm hki)) hi hk).le).1
  have hsum : (n : ℝ) * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ))) ≤ ∑ j, g_pot β (u - p j) := by
    calc
      _ = ∑ j : Idx n, g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ))) := by simp [Idx]
      _ ≤ _ := Finset.sum_le_sum (fun j _ => hp j)
  have hmass : (n : ℝ) + (∑ j, a j) - a i ≤ N := by linarith [ha i]
  have hm := mul_le_mul_of_nonpos_right hmass hneg.le
  unfold frozenLastCurvature
  nlinarith only [hcentral, hsum, hm, hg]

/-- With clustered actual predecessors, every original last-token critical point has strictly signed curvature, and every nonrepelling one is clustered.
Source: arXiv:2411.04990v2, §5, the nondegeneracy and clustering induction for `thm:fixed_centers`, at the explicit half-peak radius. -/
theorem frozen_last_critical_dichotomy_narrow (N β ε c : ℝ) (hβ : 0 < β) (hε : 0 < ε)
    (hc : 2 + 2 * ε < c) (hnarrow : ε * β ^ (-(1 / 2 : ℝ)) ≤ interactionPeak β / 2)
    (θ a : Idx m → ℝ) (ha : ∀ j, 1 ≤ a j)
    (hsep : ∀ i j, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j))
    (hpair : ∃ i j : Idx m, i ≠ j) (hN : (n : ℝ) + (∑ j, a j) ≤ N)
    (hh : N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < h_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (hg : -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ)))) < g_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (p : Idx n → ℝ) (hpre : ∀ j, ∃ i, circDist (p j - θ i) ≤ ε * β ^ (-(1 / 2 : ℝ)))
    (u : ℝ) (hcrit : frozenLastForce β θ a p u = 0) :
    frozenLastCurvature β θ a p u < 0 ∨
      0 < frozenLastCurvature β θ a p u ∧ ∃ i, circDist (u - θ i) < ε * β ^ (-(1 / 2 : ℝ)) := by
  by_cases hcurv : 0 ≤ frozenLastCurvature β θ a p u
  · obtain ⟨i, hi⟩ := frozen_last_critical_clusters_narrow N β ε c hβ hε hc hnarrow θ a ha
      hsep hpair hN hh p hpre u hcrit hcurv
    exact Or.inr ⟨frozen_last_curvature_pos_near_narrow N β ε c hβ hε hc hnarrow θ a ha
      hsep hpair hN hg p hpre u i hi.le, i, hi⟩
  · exact Or.inl (lt_of_not_ge hcurv)

/-- The original unchanged parameter conditions classify the next actual critical token once its actual predecessors lie in the derived narrow neighborhoods.
Source: arXiv:2411.04990v2, §5, the critical-point induction of `thm:fixed_centers`; the narrow radius is a proved consequence of its original conditions. -/
theorem frozen_last_critical_dichotomy (N β ε c : ℝ) (hN3 : 3 ≤ N) (hβ : 0 < β) (hε : 0 < ε)
    (hc : 2 + 2 * ε < c) (θ a : Idx m → ℝ) (ha : ∀ j, 1 ≤ a j)
    (hsep : ∀ i j, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j))
    (hpair : ∃ i j : Idx m, i ≠ j) (hN : (n : ℝ) + (∑ j, a j) ≤ N)
    (hh : N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < h_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (hg : -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ)))) < g_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (p : Idx n → ℝ)
    (hpre : ∀ j, ∃ i, circDist (p j - θ i) ≤ frozenNarrowRadius β ε * β ^ (-(1 / 2 : ℝ)))
    (u : ℝ) (hcrit : frozenLastForce β θ a p u = 0) :
    frozenLastCurvature β θ a p u < 0 ∨ 0 < frozenLastCurvature β θ a p u ∧
      ∃ i, circDist (u - θ i) < frozenNarrowRadius β ε * β ^ (-(1 / 2 : ℝ)) := by
  obtain ⟨i, j, hij⟩ := hpair
  have hcpi := frozen_separation_scale_lt_pi β c (θ i) (θ j) (hsep i j hij)
  rcases frozen_parameters_narrow N β ε c hN3 hβ hε hc hcpi hh hg with
    ⟨hη, hηε, hcap, hcη, hhη, hgη⟩
  exact frozen_last_critical_dichotomy_narrow N β (frozenNarrowRadius β ε) c hβ hη hcη hcap θ a ha
    hsep ⟨i, j, hij⟩ hN hhη hgη p hpre u hcrit

/-- Every actual last-row critical velocity has nonzero derivative when its actual predecessors lie in the proved narrow clusters.
Source: arXiv:2411.04990v2, §5, the diagonal nondegeneracy needed in `thm:fixed_centers`, retaining the exact original normalization. -/
theorem frozen_last_velocity_critical_nondegenerate_narrow (N β ε c : ℝ) (hβ : 0 < β) (hε : 0 < ε)
    (hc : 2 + 2 * ε < c) (hnarrow : ε * β ^ (-(1 / 2 : ℝ)) ≤ interactionPeak β / 2)
    (θ a : Idx m → ℝ) (ha : ∀ j, 1 ≤ a j)
    (hsep : ∀ i j, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j))
    (hpair : ∃ i j : Idx m, i ≠ j) (hN : (n : ℝ) + (∑ j, a j) ≤ N)
    (hh : N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < h_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (hg : -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ)))) < g_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (p : Idx n → ℝ) (hpre : ∀ j, ∃ i, circDist (p j - θ i) ≤ ε * β ^ (-(1 / 2 : ℝ)))
    (u : ℝ) (hcrit : frozenVelocity (n + 1) m β θ a (Fin.snoc p u) (Fin.last n) = 0) :
    deriv (fun v => frozenVelocity (n + 1) m β θ a (Fin.snoc p v) (Fin.last n)) u ≠ 0 := by
  have ha0 : ∀ j, 0 ≤ a j := fun j => le_trans zero_le_one (ha j)
  have hf := (frozenVelocity_last_eq_zero_iff β θ a ha0 p u).mp hcrit
  have hG : frozenLastCurvature β θ a p u ≠ 0 := by
    rcases frozen_last_critical_dichotomy_narrow N β ε c hβ hε hc hnarrow θ a ha hsep hpair hN
      hh hg p hpre u hf with hn | ⟨hp, hi⟩
    · exact hn.ne
    · exact hp.ne'
  rw [deriv_frozenLastVelocity_of_critical β θ a ha0 p u hcrit]
  exact div_ne_zero (neg_ne_zero.mpr hG) (frozenLastPartition_pos β θ a ha0 p u).ne'

/-- A genuine predecessor and actual last critical token at an original center realize the original-parameter classification, retaining every original hypothesis.
Source: arXiv:2411.04990v2, §5, the actual two-center configuration in `thm:fixed_centers`. -/
example : frozenLastCurvature 11 (![0, Real.pi] : Idx 2 → ℝ) (fun _ => 1) (fun _ : Idx 1 => 0) 0 < 0 ∨
    0 < frozenLastCurvature 11 (![0, Real.pi] : Idx 2 → ℝ) (fun _ => 1) (fun _ : Idx 1 => 0) 0 ∧
      ∃ i : Idx 2, circDist (0 - (![0, Real.pi] : Idx 2 → ℝ) i) <
        frozenNarrowRadius 11 (9 / 100) * (11 : ℝ) ^ (-(1 / 2 : ℝ)) := by
  rcases frozen_sample_parameters with ⟨hN, hβ, hε, hc, hcpi, hh, hg⟩
  apply frozen_last_critical_dichotomy 3 11 (9 / 100) (142 / 25) le_rfl hβ hε hc
    (![0, Real.pi]) (fun _ => 1) (fun _ => le_rfl) _ ⟨0, 1, by decide⟩
    (by norm_num : ((1 : ℕ) : ℝ) + (∑ j : Idx 2, (1 : ℝ)) ≤ 3) hh hg (fun _ : Idx 1 => 0) _ 0 _
  · have hpi : circDist Real.pi = Real.pi := circDist_eq_self _ ⟨Real.pi_pos.le, le_rfl⟩
    intro i j hij
    fin_cases i <;> fin_cases j <;> simp_all [circDist_neg]
  · intro j
    refine ⟨0, ?_⟩
    simp only [Matrix.cons_val_zero, sub_self, circDist_zero]
    exact (mul_pos (frozenNarrowRadius_properties 11 _ hβ hε).1 (Real.rpow_pos_of_pos hβ _)).le
  · norm_num [frozenLastForce, Fin.sum_univ_one, Fin.sum_univ_two, h_pot]

/-- The same genuine predecessor and actual critical last token jointly realize the narrow curvature and nonzero-derivative premises.
Source: arXiv:2411.04990v2, §5, the original two-center configuration for `thm:fixed_centers`. -/
example : deriv (fun v => frozenVelocity 2 2 11 (![0, Real.pi]) (fun _ => 1)
    (Fin.snoc (fun _ : Idx 1 => 0) v) (Fin.last 1)) 0 ≠ 0 := by
  obtain ⟨ε, hβ, hε, hc, hcap, ha, hsep, hpair, hN, hh, hg, hpre, hF, hG⟩ := frozen_sample_narrow_prefix_data
  apply frozen_last_velocity_critical_nondegenerate_narrow 3 11 ε (142 / 25) hβ hε hc hcap
    (![0, Real.pi]) (fun _ => 1) ha hsep hpair (by simpa only [Nat.cast_one] using hN)
    hh hg (fun _ : Idx 1 => 0) hpre 0
  exact (frozenVelocity_last_eq_zero_iff 11 (![0, Real.pi]) (fun _ => 1) (fun _ => zero_le_one)
    (fun _ : Idx 1 => 0) 0).mpr hF

end Transformer.Causal
