module

public import Transformer.Causal.Section5_RowRepulsion
public import Transformer.Causal.Section5_RepellerDetector
public import Transformer.Causal.Section5_OneCenterField

/-!
# Actual frozen rows with simple limiting critical angles

ArXiv:2411.04990v2, §5 and Appendix A, the scalar convergence
and repelling-limit arguments for the actual frozen system.
The separated-center proofs have already established the
needed simple limiting critical points in the narrow window.
For one center, the exact limiting interaction supplies those
properties directly, without separation parameters.

Here the same actual-row argument keeps existence of a force
zero and nonzero curvature at every force zero as explicit
inputs. A simple force zero supplies both signs, which give
periodic barriers and a bounded real lift. Countable simple
force zeros and eventual signed crossings give convergence;
the actual derivative limit then proves original criticality.
The actual Borel detector has a countable exceptional fiber,
and outside it the critical last angle is strictly attracting.
The explicit inputs concern the actual limiting force only;
no convergence or null-set assertion is included in them.
-/

@[expose] public section

open Real Set Filter MeasureTheory

namespace Transformer.Causal

variable {n m : ℕ}

/-- An actual original last trajectory with a converging actual prefix has a bounded real lift when the actual limiting force has a genuine simple zero.
Source: arXiv:2411.04990v2, §5 and Appendix A, the periodic original barriers in the smooth scalar convergence argument for `thm:fixed_centers`. -/
theorem frozen_last_bounded_of_simple_limit (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (pstar : Idx n → ℝ) (hzero : ∃ u, frozenLastForce β θ a pstar u = 0 ∧ frozenLastCurvature β θ a pstar u ≠ 0)
    (p : ℝ → Idx n → ℝ) (hp : Tendsto p atTop (nhds pstar)) (r : ℝ → ℝ)
    (hd : ∀ t, HasDerivAt r (frozenVelocity (n + 1) m β θ a (Fin.snoc (p t) (r t)) (Fin.last n)) t) :
    ∃ A B : ℝ, ∀ᶠ t in atTop, A ≤ r t ∧ r t ≤ B := by
  obtain ⟨u, hu, hGu⟩ := hzero
  obtain ⟨⟨v, hv⟩, ⟨w, hw⟩⟩ := frozen_simple_zero_both_signs (frozenLastForce β θ a pstar) u
    (frozenLastCurvature β θ a pstar u) hu (hasDerivAt_frozenLastForce β θ a pstar u) hGu
  have hvneg : frozenVelocity (n + 1) m β θ a (Fin.snoc pstar v) (Fin.last n) < 0 := by
    rw [frozenVelocity_last_eq_force]
    exact div_neg_of_neg_of_pos (neg_neg_of_pos hv) (frozenLastPartition_pos β θ a ha pstar v)
  have hwpos : 0 < frozenVelocity (n + 1) m β θ a (Fin.snoc pstar w) (Fin.last n) := by
    rw [frozenVelocity_last_eq_force]
    exact div_pos (neg_pos.mpr hw) (frozenLastPartition_pos β θ a ha pstar w)
  obtain ⟨Tv, hTv⟩ := eventually_atTop.mp ((frozenLastVelocity_tendsto_fixed_angle β θ a ha p pstar hp v).eventually_lt_const hvneg)
  obtain ⟨Tw, hTw⟩ := eventually_atTop.mp ((frozenLastVelocity_tendsto_fixed_angle β θ a ha p pstar hp w).eventually_const_lt hwpos)
  exact frozenScalar_bounded_of_periodic_barriers r
    (fun t x => frozenVelocity (n + 1) m β θ a (Fin.snoc (p t) x) (Fin.last n)) hd
    (2 * Real.pi) (by positivity) (fun t => frozenLastVelocity_periodic β θ a (p t)) w v (max Tw Tv)
    (fun t ht => hTw t ((le_max_left _ _).trans ht)) (fun t ht => hTv t ((le_max_right _ _).trans ht))

/-- Every actual original last angle converges to an actual original critical angle when its actual prefix converges and the actual limiting force has nonempty simple zeros.
Source: arXiv:2411.04990v2, §5 and Appendix A, the scalar convergence argument for the actual smooth frozen system in `thm:fixed_centers`. -/
theorem frozen_last_tendsto_critical_of_simple_limit (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (pstar : Idx n → ℝ) (hzero : ∃ u, frozenLastForce β θ a pstar u = 0)
    (hsimple : ∀ u, frozenLastForce β θ a pstar u = 0 → frozenLastCurvature β θ a pstar u ≠ 0)
    (p : ℝ → Idx n → ℝ) (hp : Tendsto p atTop (nhds pstar)) (r : ℝ → ℝ)
    (hd : ∀ t, HasDerivAt r (frozenVelocity (n + 1) m β θ a (Fin.snoc (p t) (r t)) (Fin.last n)) t) :
    ∃ l : ℝ, Tendsto r atTop (nhds l) ∧ frozenLastForce β θ a pstar l = 0 := by
  obtain ⟨u, hu⟩ := hzero
  obtain ⟨A, B, hbound⟩ := frozen_last_bounded_of_simple_limit β θ a ha pstar ⟨u, hu, hsimple u hu⟩ p hp r hd
  have hC := frozen_simple_zeros_countable (frozenLastForce β θ a pstar) (frozenLastCurvature β θ a pstar)
    (hasDerivAt_frozenLastForce β θ a pstar) hsimple
  obtain ⟨l, hl⟩ : ∃ l : ℝ, Tendsto r atTop (nhds l) := by
    apply frozenScalar_tendsto_of_countable_levels r _ hd A B hbound {u | frozenLastForce β θ a pstar u = 0} hC
    intro v hv
    have hne : frozenVelocity (n + 1) m β θ a (Fin.snoc pstar v) (Fin.last n) ≠ 0 :=
      fun hz => hv ((frozenVelocity_last_eq_zero_iff β θ a ha pstar v).mp hz)
    have hlim := frozenLastVelocity_tendsto_fixed_angle β θ a ha p pstar hp v
    rcases lt_or_gt_of_ne hne with hn | hpos
    · obtain ⟨T, hT⟩ := eventually_atTop.mp (hlim.eventually_lt_const hn)
      exact ⟨T, Or.inr (fun t ht he => by rw [he]; exact hT t ht)⟩
    · obtain ⟨T, hT⟩ := eventually_atTop.mp (hlim.eventually_const_lt hpos)
      exact ⟨T, Or.inl (fun t ht he => by rw [he]; exact hT t ht)⟩
  have hlim := frozenLastVelocity_tendsto β θ a ha p pstar hp r l hl
  have hv := frozenScalar_derivative_limit_eq_zero r _ hd A B hbound _ hlim
  exact ⟨l, hl, (frozenVelocity_last_eq_zero_iff β θ a ha pstar l).mp hv⟩

/-- The actual Borel repeller detector has a countable original last-initialization fiber over each converging original prefix whose actual limiting force has nonempty simple zeros.
Source: arXiv:2411.04990v2, §5 and Appendix A, the conditional actual null-fiber step of `thm:fixed_centers`. -/
theorem frozenRepellerDetector_fiber_countable_of_simple_limit (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (p₀ pstar : Idx n → ℝ) (hp : Tendsto (frozenFlow β θ a ha p₀) atTop (nhds pstar))
    (hzero : ∃ u, frozenLastForce β θ a pstar u = 0)
    (hsimple : ∀ u, frozenLastForce β θ a pstar u = 0 → frozenLastCurvature β θ a pstar u ≠ 0) :
    {u : ℝ | Fin.snoc p₀ u ∈ frozenRepellerDetector β θ a ha}.Countable := by
  have hC := frozen_simple_zeros_countable (frozenLastForce β θ a pstar) (frozenLastCurvature β θ a pstar)
    (hasDerivAt_frozenLastForce β θ a pstar) hsimple
  apply (frozenFlow_repelling_fiber_countable β θ a ha p₀ pstar hp {l | frozenLastForce β θ a pstar l = 0} hC).mono
  intro u hu
  have hprefix : Tendsto (fun t j => frozenFlow β θ a ha (Fin.snoc p₀ u) t j.castSucc) atTop (nhds pstar) := by
    simpa! only [frozenFlow_castSucc, Fin.snoc_castSucc] using hp
  obtain ⟨l, hl, hF⟩ := frozen_last_tendsto_critical_of_simple_limit β θ a ha pstar hzero hsimple
    (fun t j => frozenFlow β θ a ha (Fin.snoc p₀ u) t j.castSucc) hprefix _
    (frozenODE_last_scalar β θ a _ (frozenFlow_spec β θ a ha (Fin.snoc p₀ u)).2)
  exact ⟨l, hF, (frozenRepellerDetector_iff_of_prefix_last_tendsto β θ a ha (Fin.snoc p₀ u) pstar l hprefix hl).mp hu, hl⟩

/-- Outside the actual Borel detector, every actual original last angle converges to a strictly attracting critical angle over a converging actual prefix with nonempty simple limiting force zeros.
Source: arXiv:2411.04990v2, §5 and Appendix A, the actual scalar attracting-limit conclusion used for `thm:fixed_centers`. -/
theorem frozenFlow_last_attracting_of_simple_limit (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (φ₀ : Idx (n + 1) → ℝ) (pstar : Idx n → ℝ)
    (hp : Tendsto (fun t j => frozenFlow β θ a ha φ₀ t j.castSucc) atTop (nhds pstar))
    (hzero : ∃ u, frozenLastForce β θ a pstar u = 0)
    (hsimple : ∀ u, frozenLastForce β θ a pstar u = 0 → frozenLastCurvature β θ a pstar u ≠ 0)
    (hnot : φ₀ ∉ frozenRepellerDetector β θ a ha) :
    ∃ l : ℝ, Tendsto (fun t => frozenFlow β θ a ha φ₀ t (Fin.last n)) atTop (nhds l) ∧
      frozenLastForce β θ a pstar l = 0 ∧ 0 < frozenLastCurvature β θ a pstar l := by
  obtain ⟨l, hl, hF⟩ := frozen_last_tendsto_critical_of_simple_limit β θ a ha pstar hzero hsimple
    (fun t j => frozenFlow β θ a ha φ₀ t j.castSucc) hp _
    (frozenODE_last_scalar β θ a _ (frozenFlow_spec β θ a ha φ₀).2)
  have hv := (frozenVelocity_last_eq_zero_iff β θ a ha pstar l).mpr hF
  have hG : 0 ≤ frozenLastCurvature β θ a pstar l := by
    apply le_of_not_gt
    intro hneg
    exact hnot ((frozenRepellerDetector_iff_of_prefix_last_tendsto β θ a ha φ₀ pstar l hp hl).mpr
      ((frozenLastVelocityDeriv_pos_iff β θ a ha pstar l hv).mpr hneg))
  exact ⟨l, hl, hF, lt_of_le_of_ne hG (hsimple l hF).symm⟩

/-- The original one-center consensus prefix supplies genuine nonempty simple limiting force zeros and an actual converging original prefix trajectory.
Source: arXiv:2411.04990v2, §5 and Appendix A, the actual one-center witness for the original scalar convergence and repeller-fiber arguments. -/
example : ∃ l : ℝ, Tendsto (fun _ : ℝ => (0 : ℝ)) atTop (nhds l) ∧
    frozenLastForce 1 (fun _ : Idx 1 => 0) (fun _ => 1) (fun _ : Idx 1 => 0) l = 0 := by
  have hp : ∀ j : Idx 1, circDist ((fun _ : Idx 1 => (0 : ℝ)) j - 0) = 0 := fun _ => by rw [sub_self, circDist_zero]
  obtain ⟨hF, hG⟩ := frozenOneCenter_center_critical 1 0 1 zero_lt_one (fun _ : Idx 1 => 0) hp
  have hv := (frozenVelocity_last_eq_zero_iff 1 (fun _ : Idx 1 => 0) (fun _ => 1) (fun _ => zero_le_one)
    (fun _ : Idx 1 => 0) 0).mpr hF
  exact frozen_last_tendsto_critical_of_simple_limit 1 (fun _ : Idx 1 => 0) (fun _ => 1) (fun _ => zero_le_one)
    (fun _ : Idx 1 => 0) ⟨0, hF⟩ (frozenOneCenter_critical_curvature_ne 1 0 1 zero_lt_one _ hp)
    (fun _ _ => 0) tendsto_const_nhds (fun _ => 0) (fun t => by simpa only [hv] using hasDerivAt_const t (0 : ℝ))

/-- Actual original one-center canonical flows jointly realize the converging-prefix, simple-zero, and detector-avoidance premises, with a countable exceptional fiber.
Source: arXiv:2411.04990v2, §5 and Appendix A, a simultaneous original flow witness for the conditional scalar conclusions in `thm:fixed_centers`. -/
example : {u : ℝ | Fin.snoc (fun _ : Idx 1 => 0) u ∈ frozenRepellerDetector 1 (fun _ : Idx 1 => 0)
    (fun _ => 1) (fun _ => zero_le_one)}.Countable ∧
    ∃ l : ℝ, Tendsto (fun t => frozenFlow 1 (fun _ : Idx 1 => 0) (fun _ => 1) (fun _ => zero_le_one)
      (fun _ : Idx 2 => 0) t (Fin.last 1)) atTop (nhds l) ∧
      frozenLastForce 1 (fun _ : Idx 1 => 0) (fun _ => 1) (fun _ : Idx 1 => 0) l = 0 ∧
      0 < frozenLastCurvature 1 (fun _ : Idx 1 => 0) (fun _ => 1) (fun _ : Idx 1 => 0) l := by
  have hpre : ∀ j : Idx 1, circDist ((fun _ : Idx 1 => (0 : ℝ)) j - 0) = 0 := fun _ => by rw [sub_self, circDist_zero]
  obtain ⟨hF, hG⟩ := frozenOneCenter_center_critical 1 0 1 zero_lt_one (fun _ : Idx 1 => 0) hpre
  have hpv : frozenVelocity 1 1 1 (fun _ => 0) (fun _ => 1) (fun _ => 0) = 0 := by
    funext k
    simp [frozenVelocity]
  have hXv : frozenVelocity 2 1 1 (fun _ => 0) (fun _ => 1) (fun _ => 0) = 0 := by
    funext k
    simp [frozenVelocity]
  have hpf : frozenFlow 1 (fun _ : Idx 1 => 0) (fun _ => 1) (fun _ => zero_le_one) (fun _ : Idx 1 => 0) = fun _ _ => 0 :=
    frozenFlow_eq_of_ODE 1 _ _ (fun _ => zero_le_one) (fun _ _ => 0)
      (fun t => by simpa only [hpv] using hasDerivAt_const t (fun _ : Idx 1 => (0 : ℝ)))
  have hp : Tendsto (frozenFlow 1 (fun _ : Idx 1 => 0) (fun _ => 1) (fun _ => zero_le_one) (fun _ : Idx 1 => 0))
      atTop (nhds (fun _ => 0)) := by rw [hpf]; exact tendsto_const_nhds
  have hpfull : Tendsto (fun (t : ℝ) (j : Idx 1) => frozenFlow 1 (fun _ : Idx 1 => 0) (fun _ => 1)
      (fun _ => zero_le_one) (fun _ : Idx 2 => 0) t j.castSucc) atTop (nhds (fun _ => 0)) := by
    simpa! only [frozenFlow_castSucc] using hp
  have hnot : (fun _ : Idx 2 => (0 : ℝ)) ∉ frozenRepellerDetector 1 (fun _ : Idx 1 => 0) (fun _ => 1) (fun _ => zero_le_one) := by
    rw [frozenRepellerDetector_iff_of_equilibrium 1 _ _ (fun _ => zero_le_one) _ hXv]
    have hv := (frozenVelocity_last_eq_zero_iff 1 (fun _ : Idx 1 => 0) (fun _ => 1) (fun _ => zero_le_one) (fun _ : Idx 1 => 0) 0).mpr hF
    exact not_lt.mpr ((frozenLastVelocityDeriv_neg_iff 1 _ _ (fun _ => zero_le_one) _ _ hv).mpr hG).le
  exact ⟨frozenRepellerDetector_fiber_countable_of_simple_limit 1 _ _ (fun _ => zero_le_one) _ _ hp ⟨0, hF⟩
      (frozenOneCenter_critical_curvature_ne 1 0 1 zero_lt_one _ hpre),
    frozenFlow_last_attracting_of_simple_limit 1 _ _ (fun _ => zero_le_one) _ _ hpfull ⟨0, hF⟩
      (frozenOneCenter_critical_curvature_ne 1 0 1 zero_lt_one _ hpre) hnot⟩

end Transformer.Causal
