module

public import Transformer.Causal.Section5_RowField
public import Transformer.Causal.Section5_CriticalSets
public import Transformer.Causal.SectionA_ScalarConvergence

/-!
# Bounded lifted last angles for the actual frozen dynamics

ArXiv:2411.04990v2, §5 and Appendix A, the smooth frozen
specialization of `thm:fixed_centers`.
A differentiable force with a simple zero has values of both
signs: otherwise that zero would be a local extremum with
zero derivative. The actual original periodic force has such
a zero once its limiting predecessors are narrowly clustered.

Convergence of the actual predecessor tuple consequently gives
two fixed angles with eventual velocity signs directed inward.
Integer translates of these angles retain their signs by the
original periodicity. We choose translates enclosing the actual
last token at a sufficiently late time. The proved scalar
barriers then keep its lift in that fixed bounded interval.
No bounded-lift assumption is added to the original dynamics.
-/

@[expose] public section

open scoped BigOperators
open Real Set Filter

namespace Transformer.Causal

variable {n m : ℕ}

/-- An actual differentiable real force with a simple zero takes both positive and negative values.
Source: arXiv:2411.04990v2, §5 and Appendix A, auxiliary sign argument for the actual smooth frozen scalar row. -/
theorem frozen_simple_zero_both_signs (f : ℝ → ℝ) (u d : ℝ) (hzero : f u = 0)
    (hd : HasDerivAt f d u) (hdne : d ≠ 0) :
    (∃ a, 0 < f a) ∧ ∃ b, f b < 0 := by
  constructor
  · by_contra h
    push Not at h
    have hm : IsLocalMax f u := Eventually.of_forall (fun x => by rw [hzero]; exact h x)
    exact hdne (hm.hasDerivAt_eq_zero hd)
  · by_contra h
    push Not at h
    have hm : IsLocalMin f u := Eventually.of_forall (fun x => by rw [hzero]; exact h x)
    exact hdne (hm.hasDerivAt_eq_zero hd)

/-- Two eventual inward signs of a genuinely periodic scalar field bound the actual lifted scalar solution after a finite time.
Source: arXiv:2411.04990v2, §5 and Appendix A, the lifted-angle barrier argument for the actual smooth frozen system. -/
theorem frozenScalar_bounded_of_periodic_barriers (r : ℝ → ℝ) (g : ℝ → ℝ → ℝ)
    (hd : ∀ t, HasDerivAt r (g t (r t)) t) (P : ℝ) (hP : 0 < P)
    (hperiodic : ∀ t, Function.Periodic (g t) P) (a b T : ℝ)
    (ha : ∀ t, T ≤ t → 0 < g t a) (hb : ∀ t, T ≤ t → g t b < 0) :
    ∃ A B : ℝ, ∀ᶠ t in atTop, A ≤ r t ∧ r t ≤ B := by
  obtain ⟨i, hi⟩ := exists_int_lt ((r T - a) / P)
  obtain ⟨j, hj⟩ := exists_int_gt ((r T - b) / P)
  have hlo : a + (i : ℝ) * P ≤ r T := by
    rw [lt_div_iff₀ hP] at hi
    linarith
  have hhi : r T ≤ b + (j : ℝ) * P := by
    rw [div_lt_iff₀ hP] at hj
    linarith
  have hlow := csaScalar_forward_barrier r (fun t => g t (r t)) hd (a + (i : ℝ) * P) T T le_rfl hlo (by
    intro t ht he
    rw [he, (hperiodic t).int_mul i a]
    exact ha t ht)
  have hupp := csaScalar_forward_barrier (fun t => -r t) (fun t => -(g t (r t)))
    (fun t => (hd t).neg) (-(b + (j : ℝ) * P)) T T le_rfl (neg_le_neg hhi) (by
      intro t ht he
      have heq : r t = b + (j : ℝ) * P := by linarith
      rw [heq, (hperiodic t).int_mul j b]
      exact neg_pos.mpr (hb t ht))
  exact ⟨a + (i : ℝ) * P, b + (j : ℝ) * P,
    eventually_atTop.mpr ⟨T, fun t ht => ⟨hlow t ht, neg_le_neg_iff.mp (hupp t ht)⟩⟩⟩

/-- The actual original force takes both signs whenever its actual predecessors are narrowly clustered under the original force and curvature comparisons.
Source: arXiv:2411.04990v2, §5 and Appendix A, the simple original force zero used to bound lifted trajectories in the proof of `thm:fixed_centers`. -/
theorem frozen_last_force_both_signs (N β ε c : ℝ) (hβ : 0 < β) (hε : 0 < ε)
    (hc : 2 + 2 * ε < c) (hnarrow : ε * β ^ (-(1 / 2 : ℝ)) ≤ interactionPeak β / 2)
    (θ a : Idx m → ℝ) (ha : ∀ j, 1 ≤ a j)
    (hsep : ∀ i j, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j))
    (hpair : ∃ i j : Idx m, i ≠ j) (hN : (n : ℝ) + (∑ j, a j) ≤ N)
    (hh : N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < h_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (hg : -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ)))) < g_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (p : Idx n → ℝ) (hpre : ∀ j, ∃ i, circDist (p j - θ i) ≤ ε * β ^ (-(1 / 2 : ℝ))) :
    (∃ v, 0 < frozenLastForce β θ a p v) ∧ ∃ w, frozenLastForce β θ a p w < 0 := by
  obtain ⟨u, hu, hF⟩ := frozenLastForce_exists_zero β hβ.ne' θ a p
  have hG : frozenLastCurvature β θ a p u ≠ 0 := by
    rcases frozen_last_critical_dichotomy_narrow N β ε c hβ hε hc hnarrow θ a ha hsep hpair hN
      hh hg p hpre u hF with hn | ⟨hpos, hi⟩
    · exact hn.ne
    · exact hpos.ne'
  exact frozen_simple_zero_both_signs (frozenLastForce β θ a p) u (frozenLastCurvature β θ a p u)
    hF (hasDerivAt_frozenLastForce β θ a p u) hG

/-- Actual convergence of the original predecessor tuple to narrow clusters makes the actual original lifted last angle eventually bounded.
Source: arXiv:2411.04990v2, §5 and Appendix A, the bounded-lift step for `thm:fixed_centers`; periodic barriers are derived from the exact original force and normalization. -/
theorem frozen_last_bounded_of_prefix_tendsto (N β ε c : ℝ) (hβ : 0 < β) (hε : 0 < ε)
    (hc : 2 + 2 * ε < c) (hnarrow : ε * β ^ (-(1 / 2 : ℝ)) ≤ interactionPeak β / 2)
    (θ a : Idx m → ℝ) (ha : ∀ j, 1 ≤ a j)
    (hsep : ∀ i j, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j))
    (hpair : ∃ i j : Idx m, i ≠ j) (hN : (n : ℝ) + (∑ j, a j) ≤ N)
    (hh : N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < h_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (hg : -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ)))) < g_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (pstar : Idx n → ℝ)
    (hpre : ∀ j, ∃ i, circDist (pstar j - θ i) ≤ ε * β ^ (-(1 / 2 : ℝ)))
    (p : ℝ → Idx n → ℝ) (hp : Tendsto p atTop (nhds pstar)) (r : ℝ → ℝ)
    (hd : ∀ t, HasDerivAt r
      (frozenVelocity (n + 1) m β θ a (Fin.snoc (p t) (r t)) (Fin.last n)) t) :
    ∃ A B : ℝ, ∀ᶠ t in atTop, A ≤ r t ∧ r t ≤ B := by
  have ha0 : ∀ j, 0 ≤ a j := fun j => le_trans zero_le_one (ha j)
  obtain ⟨⟨v, hv⟩, ⟨w, hw⟩⟩ := frozen_last_force_both_signs N β ε c hβ hε hc hnarrow
    θ a ha hsep hpair hN hh hg pstar hpre
  have hvneg : frozenVelocity (n + 1) m β θ a (Fin.snoc pstar v) (Fin.last n) < 0 := by
    rw [frozenVelocity_last_eq_force]
    exact div_neg_of_neg_of_pos (neg_neg_of_pos hv) (frozenLastPartition_pos β θ a ha0 pstar v)
  have hwpos : 0 < frozenVelocity (n + 1) m β θ a (Fin.snoc pstar w) (Fin.last n) := by
    rw [frozenVelocity_last_eq_force]
    exact div_pos (neg_pos.mpr hw) (frozenLastPartition_pos β θ a ha0 pstar w)
  obtain ⟨Tv, hTv⟩ := eventually_atTop.mp ((frozenLastVelocity_tendsto_fixed_angle β θ a ha0 p pstar hp v).eventually_lt_const hvneg)
  obtain ⟨Tw, hTw⟩ := eventually_atTop.mp ((frozenLastVelocity_tendsto_fixed_angle β θ a ha0 p pstar hp w).eventually_const_lt hwpos)
  exact frozenScalar_bounded_of_periodic_barriers r
    (fun t x => frozenVelocity (n + 1) m β θ a (Fin.snoc (p t) x) (Fin.last n)) hd
    (2 * Real.pi) (by positivity) (fun t => frozenLastVelocity_periodic β θ a (p t)) w v (max Tw Tv)
    (fun t ht => hTw t ((le_max_left _ _).trans ht)) (fun t ht => hTv t ((le_max_right _ _).trans ht))

/-- The actual identity force has a simple zero and takes both signs, jointly realizing the scalar force-sign premises.
Source: arXiv:2411.04990v2, Appendix A and §5, the auxiliary scalar sign argument for the smooth frozen specialization. -/
example : (fun x : ℝ => x) 0 = 0 ∧ HasDerivAt (fun x : ℝ => x) 1 0 ∧ (1 : ℝ) ≠ 0 :=
  ⟨rfl, hasDerivAt_id 0, one_ne_zero⟩

/-- Genuine original clustered predecessors realize both signs of the exact original force under the checked parameter window.
Source: arXiv:2411.04990v2, §5 and Appendix A, the original two-center configuration in `thm:fixed_centers`. -/
example : (∃ v, 0 < frozenLastForce 11 (![0, Real.pi] : Idx 2 → ℝ) (fun _ => 1) (fun _ : Idx 1 => 0) v) ∧
    ∃ w, frozenLastForce 11 (![0, Real.pi] : Idx 2 → ℝ) (fun _ => 1) (fun _ : Idx 1 => 0) w < 0 := by
  obtain ⟨ε, hβ, hε, hc, hcap, ha, hsep, hpair, hN, hh, hg, hpre, hF, hG⟩ := frozen_sample_narrow_prefix_data
  exact frozen_last_force_both_signs 3 11 ε (142 / 25) hβ hε hc hcap (![0, Real.pi]) (fun _ => 1)
    ha hsep hpair (by simpa only [Nat.cast_one] using hN) hh hg (fun _ : Idx 1 => 0) hpre

/-- A genuine original clustered predecessor and actual constant critical last token realize every bounded-lift premise of the original frozen row.
Source: arXiv:2411.04990v2, §5, the original two-center equilibrium configuration in `thm:fixed_centers`. -/
example : ∃ A B : ℝ, ∀ᶠ _ : ℝ in atTop, A ≤ (0 : ℝ) ∧ (0 : ℝ) ≤ B := by
  obtain ⟨ε, hβ, hε, hc, hcap, ha, hsep, hpair, hN, hh, hg, hpre, hF, hG⟩ := frozen_sample_narrow_prefix_data
  apply frozen_last_bounded_of_prefix_tendsto 3 11 ε (142 / 25) hβ hε hc hcap
    (![0, Real.pi]) (fun _ => 1) ha hsep hpair (by simpa only [Nat.cast_one] using hN) hh hg
    (fun _ : Idx 1 => 0) hpre (fun _ _ => 0) tendsto_const_nhds (fun _ => 0)
  have hv := (frozenVelocity_last_eq_zero_iff 11 (![0, Real.pi]) (fun _ => 1) (fun _ => zero_le_one)
    (fun _ : Idx 1 => 0) 0).mpr hF
  intro t
  simpa only [hv] using hasDerivAt_const t (0 : ℝ)

end Transformer.Causal
