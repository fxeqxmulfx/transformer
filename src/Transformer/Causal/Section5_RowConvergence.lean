module

public import Transformer.Causal.Section5_RowBounds
public import Mathlib.Analysis.Calculus.Deriv.Prod

/-!
# Deterministic convergence of the next original frozen token

ArXiv:2411.04990v2, §5 and Appendix A, the smooth frozen
specialization of `thm:fixed_centers`.
Assume that the actual predecessor tuple converges and its
limit lies in the proved narrow neighborhoods. The actual
last angle is eventually bounded by the periodic barriers.
The limiting force has countably many zeros. At every other
fixed level, the actual velocity eventually has the limiting
strict sign. The scalar crossing theorem therefore gives an
actual real limit of the last angle.

Joint continuity gives the limit of its actual derivative.
Boundedness forces that derivative limit to vanish, so the
limiting angle is a true zero of the original force. The
proved curvature dichotomy classifies it as strictly repelling
or strictly attracting within a narrow neighborhood. This is
deterministic convergence; excluding the repelling alternative
for almost every initialization remains a separate step.
No general printed C1 convergence lemma or exponential rate
is assumed in this argument.
-/

@[expose] public section

open scoped BigOperators
open Real Set Filter

namespace Transformer.Causal

variable {n m : ℕ}

/-- An actual original last-token trajectory converges to a genuine classified critical angle whenever its actual predecessors converge to the proved narrow clusters.
Source: arXiv:2411.04990v2, §5 and Appendix A, the deterministic smooth frozen convergence step for `thm:fixed_centers`, with no unproved sequential convergence input. -/
theorem frozen_last_tendsto_critical (N β ε c : ℝ) (hβ : 0 < β) (hε : 0 < ε)
    (hc : 2 + 2 * ε < c) (hnarrow : ε * β ^ (-(1 / 2 : ℝ)) ≤ interactionPeak β / 2)
    (θ a : Idx m → ℝ) (ha : ∀ j, 1 ≤ a j)
    (hsep : ∀ i j, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j))
    (hpair : ∃ i j : Idx m, i ≠ j) (hN : (n : ℝ) + (∑ j, a j) ≤ N)
    (hh : N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < h_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (hg : -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ)))) < g_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (pstar : Idx n → ℝ) (hpre : ∀ j, ∃ i, circDist (pstar j - θ i) ≤ ε * β ^ (-(1 / 2 : ℝ)))
    (p : ℝ → Idx n → ℝ) (hp : Tendsto p atTop (nhds pstar)) (r : ℝ → ℝ)
    (hd : ∀ t, HasDerivAt r (frozenVelocity (n + 1) m β θ a (Fin.snoc (p t) (r t)) (Fin.last n)) t) :
    ∃ l : ℝ, Tendsto r atTop (nhds l) ∧ frozenLastForce β θ a pstar l = 0 ∧
      (frozenLastCurvature β θ a pstar l < 0 ∨ 0 < frozenLastCurvature β θ a pstar l ∧
        ∃ i, circDist (l - θ i) < ε * β ^ (-(1 / 2 : ℝ))) := by
  have ha0 : ∀ j, 0 ≤ a j := fun j => le_trans zero_le_one (ha j)
  obtain ⟨A, B, hbound⟩ := frozen_last_bounded_of_prefix_tendsto N β ε c hβ hε hc hnarrow θ a ha
    hsep hpair hN hh hg pstar hpre p hp r hd
  have hC := frozen_last_force_zeros_countable N β ε c hβ hε hc hnarrow θ a ha hsep hpair hN hh hg pstar hpre
  have hex : ∃ l : ℝ, Tendsto r atTop (nhds l) := by
    apply frozenScalar_tendsto_of_countable_levels r _ hd A B hbound {u | frozenLastForce β θ a pstar u = 0} hC
    intro u hu
    have hne : frozenVelocity (n + 1) m β θ a (Fin.snoc pstar u) (Fin.last n) ≠ 0 := by
      intro hz
      exact hu ((frozenVelocity_last_eq_zero_iff β θ a ha0 pstar u).mp hz)
    have hlim := frozenLastVelocity_tendsto_fixed_angle β θ a ha0 p pstar hp u
    rcases lt_or_gt_of_ne hne with hn | hpos
    · obtain ⟨T, hT⟩ := eventually_atTop.mp (hlim.eventually_lt_const hn)
      exact ⟨T, Or.inr (fun t ht he => by rw [he]; exact hT t ht)⟩
    · obtain ⟨T, hT⟩ := eventually_atTop.mp (hlim.eventually_const_lt hpos)
      exact ⟨T, Or.inl (fun t ht he => by rw [he]; exact hT t ht)⟩
  obtain ⟨l, hl⟩ := hex
  have hlim := frozenLastVelocity_tendsto β θ a ha0 p pstar hp r l hl
  have hv := frozenScalar_derivative_limit_eq_zero r _ hd A B hbound _ hlim
  have hF := (frozenVelocity_last_eq_zero_iff β θ a ha0 pstar l).mp hv
  exact ⟨l, hl, hF, frozen_last_critical_dichotomy_narrow N β ε c hβ hε hc hnarrow θ a ha
    hsep hpair hN hh hg pstar hpre l hF⟩

/-- The last component of every actual full original solution obeys the exact scalar row equation with its actual predecessor trajectory.
Source: arXiv:2411.04990v2, §5, the actual triangular ODE in `thm:fixed_centers`. -/
theorem frozenODE_last_scalar (β : ℝ) (θ a : Idx m → ℝ) (φ : ℝ → Idx (n + 1) → ℝ)
    (hφ : ∀ t, HasDerivAt φ (frozenVelocity (n + 1) m β θ a (φ t)) t) :
    ∀ t, HasDerivAt (fun s => φ s (Fin.last n))
      (frozenVelocity (n + 1) m β θ a (Fin.snoc (fun j => φ t j.castSucc) (φ t (Fin.last n))) (Fin.last n)) t := by
  intro t
  have heq : Fin.snoc (fun j => φ t j.castSucc) (φ t (Fin.last n)) = φ t := Fin.snoc_init_self (φ t)
  simpa! only [heq] using hasDerivAt_pi.mp (hφ t) (Fin.last n)

/-- Every actual full original frozen solution has a convergent classified last component once its actual predecessor tuple converges to narrow clusters.
Source: arXiv:2411.04990v2, §5 and Appendix A, the deterministic induction step for the actual smooth dynamics in `thm:fixed_centers`. -/
theorem frozenODE_last_tendsto_critical (N β ε c : ℝ) (hβ : 0 < β) (hε : 0 < ε)
    (hc : 2 + 2 * ε < c) (hnarrow : ε * β ^ (-(1 / 2 : ℝ)) ≤ interactionPeak β / 2)
    (θ a : Idx m → ℝ) (ha : ∀ j, 1 ≤ a j)
    (hsep : ∀ i j, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j))
    (hpair : ∃ i j : Idx m, i ≠ j) (hN : (n : ℝ) + (∑ j, a j) ≤ N)
    (hh : N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < h_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (hg : -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ)))) < g_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (φ : ℝ → Idx (n + 1) → ℝ) (hφ : ∀ t, HasDerivAt φ (frozenVelocity (n + 1) m β θ a (φ t)) t)
    (pstar : Idx n → ℝ) (hp : Tendsto (fun t j => φ t j.castSucc) atTop (nhds pstar))
    (hpre : ∀ j, ∃ i, circDist (pstar j - θ i) ≤ ε * β ^ (-(1 / 2 : ℝ))) :
    ∃ l : ℝ, Tendsto (fun t => φ t (Fin.last n)) atTop (nhds l) ∧ frozenLastForce β θ a pstar l = 0 ∧
      (frozenLastCurvature β θ a pstar l < 0 ∨ 0 < frozenLastCurvature β θ a pstar l ∧
        ∃ i, circDist (l - θ i) < ε * β ^ (-(1 / 2 : ℝ))) :=
  frozen_last_tendsto_critical N β ε c hβ hε hc hnarrow θ a ha hsep hpair hN hh hg pstar hpre
    (fun t j => φ t j.castSucc) hp (fun t => φ t (Fin.last n)) (frozenODE_last_scalar β θ a φ hφ)

/-- If the actual predecessor limit is critical and narrowly clustered, the entire actual original frozen solution converges to a genuine critical full tuple.
Source: arXiv:2411.04990v2, §5 and Appendix A, the deterministic full-tuple convergence step for the actual smooth dynamics of `thm:fixed_centers`; the prefix convergence and criticality remain explicit induction inputs. -/
theorem frozenODE_tendsto_critical_of_prefix (N β ε c : ℝ) (hβ : 0 < β) (hε : 0 < ε)
    (hc : 2 + 2 * ε < c) (hnarrow : ε * β ^ (-(1 / 2 : ℝ)) ≤ interactionPeak β / 2)
    (θ a : Idx m → ℝ) (ha : ∀ j, 1 ≤ a j)
    (hsep : ∀ i j, i ≠ j → c * β ^ (-(1 / 2 : ℝ)) < circDist (θ i - θ j))
    (hpair : ∃ i j : Idx m, i ≠ j) (hN : (n : ℝ) + (∑ j, a j) ≤ N)
    (hh : N * h_pot β ((c - 1 - 2 * ε) * β ^ (-(1 / 2 : ℝ))) < h_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (hg : -(N * g_pot β ((c - 2 * ε) * β ^ (-(1 / 2 : ℝ)))) < g_pot β (ε * β ^ (-(1 / 2 : ℝ))))
    (φ : ℝ → Idx (n + 1) → ℝ) (hφ : ∀ t, HasDerivAt φ (frozenVelocity (n + 1) m β θ a (φ t)) t)
    (pstar : Idx n → ℝ) (hp : Tendsto (fun t j => φ t j.castSucc) atTop (nhds pstar))
    (hpre : ∀ j, ∃ i, circDist (pstar j - θ i) ≤ ε * β ^ (-(1 / 2 : ℝ)))
    (hpcrit : frozenVelocity n m β θ a pstar = 0) :
    ∃ φstar : Idx (n + 1) → ℝ, Tendsto φ atTop (nhds φstar) ∧ frozenVelocity (n + 1) m β θ a φstar = 0 := by
  obtain ⟨l, hl, hF, hC⟩ := frozenODE_last_tendsto_critical N β ε c hβ hε hc hnarrow θ a ha
    hsep hpair hN hh hg φ hφ pstar hp hpre
  refine ⟨Fin.snoc pstar l, ?_, ?_⟩
  · apply tendsto_pi_nhds.mpr
    intro j
    refine Fin.lastCases ?_ (fun k => ?_) j
    · simpa only [Fin.snoc_last] using hl
    · simpa only [Fin.snoc_castSucc] using (tendsto_pi_nhds.mp hp) k
  · funext j
    refine Fin.lastCases ?_ (fun k => ?_) j
    · exact (frozenVelocity_last_eq_zero_iff β θ a (fun k => le_trans zero_le_one (ha k)) pstar l).mpr hF
    · rw [frozenVelocity_castSucc]
      simpa only [Fin.snoc_castSucc, Pi.zero_apply] using congrFun hpcrit k

/-- A genuine original clustered predecessor and actual constant last-token solution jointly realize all scalar convergence premises.
Source: arXiv:2411.04990v2, §5, the actual two-center equilibrium configuration used in `thm:fixed_centers`. -/
example : ∃ l : ℝ, Tendsto (fun _ : ℝ => (0 : ℝ)) atTop (nhds l) ∧
    frozenLastForce 11 (![0, Real.pi] : Idx 2 → ℝ) (fun _ => 1) (fun _ : Idx 1 => 0) l = 0 := by
  obtain ⟨ε, hβ, hε, hc, hcap, ha, hsep, hpair, hN, hh, hg, hpre, hF, hG⟩ := frozen_sample_narrow_prefix_data
  have hv := (frozenVelocity_last_eq_zero_iff 11 (![0, Real.pi]) (fun _ => 1) (fun _ => zero_le_one)
    (fun _ : Idx 1 => 0) 0).mpr hF
  obtain ⟨l, hl, hFl, hCl⟩ := frozen_last_tendsto_critical 3 11 ε (142 / 25) hβ hε hc hcap
    (![0, Real.pi]) (fun _ => 1) ha hsep hpair (by simpa only [Nat.cast_one] using hN) hh hg
    (fun _ : Idx 1 => 0) hpre (fun _ _ => 0) tendsto_const_nhds (fun _ => 0)
    (fun t => by simpa only [hv] using hasDerivAt_const t (0 : ℝ))
  exact ⟨l, hl, hFl⟩

/-- The genuine original two-center equilibrium realizes full-solution and critical-prefix premises together, giving actual full-tuple convergence.
Source: arXiv:2411.04990v2, §5, the original equilibrium configuration for `thm:fixed_centers`. -/
example : ∃ φstar : Idx 2 → ℝ, Tendsto (fun _ : ℝ => (fun _ : Idx 2 => (0 : ℝ))) atTop (nhds φstar) ∧
    frozenVelocity 2 2 11 (![0, Real.pi]) (fun _ => 1) φstar = 0 := by
  obtain ⟨ε, hβ, hε, hc, hcap, ha, hsep, hpair, hN, hh, hg, hpre, hF, hG⟩ := frozen_sample_narrow_prefix_data
  have hv : frozenVelocity 2 2 11 (![0, Real.pi]) (fun _ => 1) (fun _ => 0) = 0 := by
    funext k
    simp [frozenVelocity, Fin.sum_univ_two]
  apply frozenODE_tendsto_critical_of_prefix 3 11 ε (142 / 25) hβ hε hc hcap
    (![0, Real.pi]) (fun _ => 1) ha hsep hpair (by simpa only [Nat.cast_one] using hN) hh hg
    (fun _ _ => 0) (fun t => by simpa only [hv] using hasDerivAt_const t (fun _ : Idx 2 => (0 : ℝ)))
    (fun _ : Idx 1 => 0) tendsto_const_nhds hpre
  funext k
  simp [frozenVelocity, Fin.sum_univ_two]

end Transformer.Causal
