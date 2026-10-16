/-
# Avoidance of a repelling original last-token limit

ArXiv:2411.04990v2, §5 and Appendix A, the causal induction
step in the proof of `thm:fixed_centers`.
For a fixed actual prefix initialization whose original flow
converges to the proved narrow clusters, every actual next
angle has a real critical limit. The actual Borel detector
therefore detects precisely the repelling alternative.

The possible limiting critical angles form a countable set.
At most one original last initialization can approach each
repelling limit while sharing the actual original prefix.
The Borel detector consequently has a countable last-angle
fiber over each such prefix. Outside that fiber, the limit
has strictly positive original force curvature and lies in
a proved narrow frozen-center neighborhood. If the prefix
limit is critical, the full original limiting tuple is also
critical. The prefix convergence remains an explicit induction
input; the separate measurable Fubini step removes this input
for almost every original full initialization inductively.
-/

import Transformer.Causal.Section5_RowRepulsion
import Transformer.Causal.Section5_RepellerDetector

open Real Set Filter MeasureTheory

namespace Transformer.Causal

variable {n m : ℕ}

/-- The actual Borel repeller detector has a countable last-angle fiber over every original prefix whose actual flow converges to the proved narrow clusters.
Source: arXiv:2411.04990v2, §5 and Appendix A, the countable conditional exceptional set for the actual smooth frozen system in `thm:fixed_centers`. -/
theorem frozenRepellerDetector_fiber_countable (N β ε c : ℝ) (hβ : 0 < β) (hε : 0 < ε)
    (hc : 2 + 2 * ε < c) (hnarrow : ε * β ^ (-(1 / 2 : ℝ)) ≤ interactionPeak β / 2)
    (θ a : Idx m → ℝ) (ha : ∀ j, 1 ≤ a j)
    (hsep : ∀ i j, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j))
    (hpair : ∃ i j : Idx m, i ≠ j) (hN : (n : ℝ) + (∑ j, a j) ≤ N)
    (hh : N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < h_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (hg : -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ)))) < g_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (p₀ pstar : Idx n → ℝ)
    (hp : Tendsto (frozenFlow β θ a (fun j => le_trans zero_le_one (ha j)) p₀) atTop (nhds pstar))
    (hpre : ∀ j, ∃ i, circDist (pstar j - θ i) ≤ ε * β ^ (-(1 / 2 : ℝ))) :
    {u : ℝ | Fin.snoc p₀ u ∈ frozenRepellerDetector β θ a (fun j => le_trans zero_le_one (ha j))}.Countable := by
  let ha0 : ∀ j, 0 ≤ a j := fun j => le_trans zero_le_one (ha j)
  have hC := frozen_last_force_zeros_countable N β ε c hβ hε hc hnarrow θ a ha hsep hpair hN hh hg pstar hpre
  apply (frozenFlow_repelling_fiber_countable β θ a ha0 p₀ pstar hp
    {l | frozenLastForce β θ a pstar l = 0} hC).mono
  intro u hu
  have hprefix : Tendsto (fun t j => frozenFlow β θ a ha0 (Fin.snoc p₀ u) t j.castSucc) atTop (nhds pstar) := by
    simpa! only [frozenFlow_castSucc, Fin.snoc_castSucc] using hp
  obtain ⟨l, hl, hF, hG⟩ := frozenODE_last_tendsto_critical N β ε c hβ hε hc hnarrow θ a ha hsep hpair hN hh hg
    _ (frozenFlow_spec β θ a ha0 (Fin.snoc p₀ u)).2 pstar hprefix hpre
  exact ⟨l, hF, (frozenRepellerDetector_iff_of_prefix_last_tendsto β θ a ha0 (Fin.snoc p₀ u) pstar l hprefix hl).mp hu, hl⟩

/-- Outside the actual Borel repeller detector, the actual next original angle converges to a strictly attracting critical angle near an original frozen center.
Source: arXiv:2411.04990v2, §5 and Appendix A, the original attracting last-token conclusion in `thm:fixed_centers`, conditional on the actual converging clustered prefix. -/
theorem frozenFlow_last_tendsto_attracting_of_not_detector (N β ε c : ℝ) (hβ : 0 < β) (hε : 0 < ε)
    (hc : 2 + 2 * ε < c) (hnarrow : ε * β ^ (-(1 / 2 : ℝ)) ≤ interactionPeak β / 2)
    (θ a : Idx m → ℝ) (ha : ∀ j, 1 ≤ a j)
    (hsep : ∀ i j, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j))
    (hpair : ∃ i j : Idx m, i ≠ j) (hN : (n : ℝ) + (∑ j, a j) ≤ N)
    (hh : N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < h_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (hg : -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ)))) < g_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (φ₀ : Idx (n + 1) → ℝ) (pstar : Idx n → ℝ)
    (hp : Tendsto (fun t j => frozenFlow β θ a (fun j => le_trans zero_le_one (ha j)) φ₀ t j.castSucc) atTop (nhds pstar))
    (hpre : ∀ j, ∃ i, circDist (pstar j - θ i) ≤ ε * β ^ (-(1 / 2 : ℝ)))
    (hnot : φ₀ ∉ frozenRepellerDetector β θ a (fun j => le_trans zero_le_one (ha j))) :
    ∃ l : ℝ, Tendsto (fun t => frozenFlow β θ a (fun j => le_trans zero_le_one (ha j)) φ₀ t (Fin.last n)) atTop (nhds l) ∧
      frozenLastForce β θ a pstar l = 0 ∧ 0 < frozenLastCurvature β θ a pstar l ∧
      ∃ i, circDist (l - θ i) < ε * β ^ (-(1 / 2 : ℝ)) := by
  let ha0 : ∀ j, 0 ≤ a j := fun j => le_trans zero_le_one (ha j)
  obtain ⟨l, hl, hF, hG⟩ := frozenODE_last_tendsto_critical N β ε c hβ hε hc hnarrow θ a ha
    hsep hpair hN hh hg _ (frozenFlow_spec β θ a ha0 φ₀).2 pstar hp hpre
  refine ⟨l, hl, hF, ?_⟩
  rcases hG with hneg | hpos
  · have hv := (frozenVelocity_last_eq_zero_iff β θ a ha0 pstar l).mpr hF
    exact False.elim (hnot ((frozenRepellerDetector_iff_of_prefix_last_tendsto β θ a ha0 φ₀ pstar l hp hl).mpr
      ((frozenLastVelocityDeriv_pos_iff β θ a ha0 pstar l hv).mpr hneg)))
  · exact hpos

/-- A critical clustered original prefix extends outside the actual Borel detector to a critical full original limit with strictly negative last derivative and a clustered last angle.
Source: arXiv:2411.04990v2, §5 and Appendix A, the actual critical and attracting extension in the causal induction for `thm:fixed_centers`. -/
theorem frozenFlow_tendsto_attracting_extension (N β ε c : ℝ) (hβ : 0 < β) (hε : 0 < ε)
    (hc : 2 + 2 * ε < c) (hnarrow : ε * β ^ (-(1 / 2 : ℝ)) ≤ interactionPeak β / 2)
    (θ a : Idx m → ℝ) (ha : ∀ j, 1 ≤ a j)
    (hsep : ∀ i j, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j))
    (hpair : ∃ i j : Idx m, i ≠ j) (hN : (n : ℝ) + (∑ j, a j) ≤ N)
    (hh : N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < h_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (hg : -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ)))) < g_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (φ₀ : Idx (n + 1) → ℝ) (pstar : Idx n → ℝ)
    (hp : Tendsto (fun t j => frozenFlow β θ a (fun j => le_trans zero_le_one (ha j)) φ₀ t j.castSucc) atTop (nhds pstar))
    (hpre : ∀ j, ∃ i, circDist (pstar j - θ i) ≤ ε * β ^ (-(1 / 2 : ℝ)))
    (hpcrit : frozenVelocity n m β θ a pstar = 0)
    (hnot : φ₀ ∉ frozenRepellerDetector β θ a (fun j => le_trans zero_le_one (ha j))) :
    ∃ l : ℝ, Tendsto (frozenFlow β θ a (fun j => le_trans zero_le_one (ha j)) φ₀) atTop (nhds (Fin.snoc pstar l)) ∧
      frozenVelocity (n + 1) m β θ a (Fin.snoc pstar l) = 0 ∧ frozenLastVelocityDeriv β θ a pstar l < 0 ∧
      ∃ i, circDist (l - θ i) < ε * β ^ (-(1 / 2 : ℝ)) := by
  let ha0 : ∀ j, 0 ≤ a j := fun j => le_trans zero_le_one (ha j)
  obtain ⟨l, hl, hF, hG, hnear⟩ := frozenFlow_last_tendsto_attracting_of_not_detector N β ε c hβ hε hc hnarrow
    θ a ha hsep hpair hN hh hg φ₀ pstar hp hpre hnot
  have hv := (frozenVelocity_last_eq_zero_iff β θ a ha0 pstar l).mpr hF
  refine ⟨l, ?_, ?_, (frozenLastVelocityDeriv_neg_iff β θ a ha0 pstar l hv).mpr hG, hnear⟩
  · apply tendsto_pi_nhds.mpr
    intro j
    refine Fin.lastCases ?_ (fun k => ?_) j
    · simpa only [Fin.snoc_last] using hl
    · simpa only [Fin.snoc_castSucc] using (tendsto_pi_nhds.mp hp) k
  · funext j
    refine Fin.lastCases hv (fun k => ?_) j
    rw [frozenVelocity_castSucc]
    simpa only [Fin.snoc_castSucc, Pi.zero_apply] using congrFun hpcrit k

/-- The actual two-center equilibrium supplies a genuine converging critical clustered prefix and a full initialization outside the original repeller detector.
Source: arXiv:2411.04990v2, §5 and Appendix A, a simultaneous original initialization witness for the conditional extension in `thm:fixed_centers`. -/
example : ∃ ε : ℝ,
    {u : ℝ | Fin.snoc (fun _ : Idx 1 => 0) u ∈ frozenRepellerDetector 11 (![0, Real.pi] : Idx 2 → ℝ)
      (fun _ => 1) (fun _ => zero_le_one)}.Countable ∧ ∃ l : ℝ,
    Tendsto (frozenFlow 11 (![0, Real.pi] : Idx 2 → ℝ) (fun _ => 1) (fun _ => zero_le_one) (fun _ : Idx 2 => 0))
      atTop (nhds (Fin.snoc (fun _ : Idx 1 => 0) l)) ∧
    frozenVelocity 2 2 11 (![0, Real.pi]) (fun _ => 1) (Fin.snoc (fun _ : Idx 1 => 0) l) = 0 ∧
    frozenLastVelocityDeriv 11 (![0, Real.pi]) (fun _ => 1) (fun _ : Idx 1 => 0) l < 0 ∧
    ∃ i, circDist (l - (![0, Real.pi] : Idx 2 → ℝ) i) < ε * (11 : ℝ) ^ (-(1 / 2 : ℝ)) := by
  obtain ⟨ε, hβ, hε, hc, hcap, ha, hsep, hpair, hN, hh, hg, hpre, hF, hG⟩ := frozen_sample_narrow_prefix_data
  have hpv : frozenVelocity 1 2 11 (![0, Real.pi]) (fun _ => 1) (fun _ => 0) = 0 := by
    funext k
    simp [frozenVelocity, Fin.sum_univ_two]
  have hXv : frozenVelocity 2 2 11 (![0, Real.pi]) (fun _ => 1) (fun _ => 0) = 0 := by
    funext k
    simp [frozenVelocity, Fin.sum_univ_two]
  have hpf : frozenFlow 11 (![0, Real.pi]) (fun _ => 1) (fun _ => zero_le_one) (fun _ : Idx 1 => 0) = fun _ _ => 0 :=
    frozenFlow_eq_of_ODE 11 _ _ (fun _ => zero_le_one) (fun _ _ => 0)
      (fun t => by simpa only [hpv] using hasDerivAt_const t (fun _ : Idx 1 => (0 : ℝ)))
  have hp : Tendsto (fun (t : ℝ) (j : Idx 1) => frozenFlow 11 (![0, Real.pi]) (fun _ => 1)
      (fun _ => zero_le_one) (fun _ : Idx 2 => 0) t j.castSucc) atTop (nhds (fun _ => 0)) := by
    simpa! only [frozenFlow_castSucc, hpf] using (tendsto_const_nhds : Tendsto (fun _ : ℝ => (fun _ : Idx 1 => (0 : ℝ))) atTop (nhds (fun _ => 0)))
  have hnot : (fun _ : Idx 2 => (0 : ℝ)) ∉ frozenRepellerDetector 11 (![0, Real.pi]) (fun _ => 1) (fun _ => zero_le_one) := by
    rw [frozenRepellerDetector_iff_of_equilibrium 11 _ _ (fun _ => zero_le_one) _ hXv]
    have hv := (frozenVelocity_last_eq_zero_iff 11 (![0, Real.pi]) (fun _ => 1) (fun _ => zero_le_one)
      (fun _ : Idx 1 => 0) 0).mpr hF
    have hle : frozenLastVelocityDeriv 11 (![0, Real.pi]) (fun _ => 1) (fun _ : Idx 1 => 0) 0 ≤ 0 := by
      rw [frozenLastVelocityDeriv_of_critical 11 _ _ (fun _ => zero_le_one) _ _ hv]
      exact div_nonpos_of_nonpos_of_nonneg (neg_nonpos.mpr hG)
        (frozenLastPartition_pos 11 _ _ (fun _ => zero_le_one) _ _).le
    exact not_lt.mpr hle
  have hp₀ : Tendsto (frozenFlow 11 (![0, Real.pi]) (fun _ => 1) (fun _ => zero_le_one)
      (fun _ : Idx 1 => 0)) atTop (nhds (fun _ => 0)) := by rw [hpf]; exact tendsto_const_nhds
  refine ⟨ε, frozenRepellerDetector_fiber_countable 3 11 ε (142 / 25) hβ hε hc hcap (![0, Real.pi]) (fun _ => 1)
    ha hsep hpair (by simpa only [Nat.cast_one] using hN) hh hg (fun _ => 0) (fun _ : Idx 1 => 0) hp₀ hpre, ?_⟩
  exact frozenFlow_tendsto_attracting_extension 3 11 ε (142 / 25) hβ hε hc hcap (![0, Real.pi]) (fun _ => 1)
    ha hsep hpair (by simpa only [Nat.cast_one] using hN) hh hg (fun _ => 0) (fun _ : Idx 1 => 0) hp hpre hpv hnot

end Transformer.Causal
