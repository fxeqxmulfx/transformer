/-
# A Borel detector for actual frozen repelling limits

ArXiv:2411.04990v2, §5 and Appendix A, the probability-one
avoidance step in `thm:fixed_centers`.
The actual normalized last-angle derivative is continuous
in the full original angular state. At each natural time,
its composition with the actual frozen flow is continuous
in the original initialization. Closed positive-threshold
tests at those times therefore give a Borel event.

The event says that some reciprocal natural threshold is a
lower bound for every sufficiently late natural-time sample.
Whenever the actual full trajectory converges, continuity
shows that this event is equivalent to strict positivity of
the actual derivative at its limit. No limit is selected as
a function of the initialization, and no measurability of an
existential convergence assertion is presumed. The definition
contains only the actual original field and its actual flow.
-/

import Transformer.Causal.Section5_FlowContinuity
import Transformer.Causal.Section5_RowDerivative

open Real Set Filter MeasureTheory

namespace Transformer.Causal

variable {n m : ℕ}

/-- For a genuinely convergent real sequence, an eventual positive reciprocal-natural lower bound is equivalent to a strictly positive limit.
Source: arXiv:2411.04990v2, Appendix A, the countable observation step used to detect actual repelling limits in `thm:fixed_centers`. -/
theorem frozen_sequence_threshold_iff (r : ℕ → ℝ) (l : ℝ) (hr : Tendsto r atTop (nhds l)) :
    (∃ q N : ℕ, ∀ k : ℕ, N ≤ k → 1 / ((q : ℝ) + 1) ≤ r k) ↔ 0 < l := by
  constructor
  · rintro ⟨q, N, hN⟩
    have he : ∀ᶠ k in atTop, 1 / ((q : ℝ) + 1) ≤ r k := eventually_atTop.mpr ⟨N, hN⟩
    have hp : 0 < 1 / ((q : ℝ) + 1) := by positivity
    exact hp.trans_le (ge_of_tendsto hr he)
  · intro hl
    obtain ⟨q, hq⟩ := exists_nat_one_div_lt hl
    obtain ⟨N, hN⟩ := eventually_atTop.mp (hr.eventually_const_lt hq)
    exact ⟨q, N, fun k hk => (hN k hk).le⟩

/-- The actual original repeller detector uses only countably many positive-threshold observations of the exact original last derivative along the actual original flow.
Source: arXiv:2411.04990v2, §5 and Appendix A, a Borel implementation of repelling-limit avoidance in `thm:fixed_centers`. -/
noncomputable def frozenRepellerDetector (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j) :
    Set (Idx (n + 1) → ℝ) :=
  ⋃ q : ℕ, ⋃ N : ℕ, ⋂ k : ℕ, ⋂ (_ : N ≤ k),
    {φ₀ | 1 / ((q : ℝ) + 1) ≤ frozenLastVelocityDeriv β θ a
      (fun j => frozenFlow β θ a ha φ₀ (k : ℝ) j.castSucc)
      (frozenFlow β θ a ha φ₀ (k : ℝ) (Fin.last n))}

/-- Membership in the original detector is exactly an eventual reciprocal-natural lower bound on natural-time samples of the actual original last derivative.
Source: arXiv:2411.04990v2, §5 and Appendix A, the explicit countable observation event for `thm:fixed_centers`. -/
theorem mem_frozenRepellerDetector_iff (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (φ₀ : Idx (n + 1) → ℝ) : φ₀ ∈ frozenRepellerDetector β θ a ha ↔
      ∃ q N : ℕ, ∀ k : ℕ, N ≤ k → 1 / ((q : ℝ) + 1) ≤ frozenLastVelocityDeriv β θ a
        (fun j => frozenFlow β θ a ha φ₀ (k : ℝ) j.castSucc)
        (frozenFlow β θ a ha φ₀ (k : ℝ) (Fin.last n)) := by
  simp only [frozenRepellerDetector, mem_iUnion, mem_iInter, mem_ofPred_eq]

/-- Each actual natural-time observation of the full original last derivative is continuous in the original initial angular tuple.
Source: arXiv:2411.04990v2, §5 and Appendix A, the smooth original observation used in the Borel argument for `thm:fixed_centers`. -/
theorem continuous_frozenLastVelocityDeriv_sample (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j) (k : ℕ) :
    Continuous (fun φ₀ : Idx (n + 1) → ℝ => frozenLastVelocityDeriv β θ a
      (fun j => frozenFlow β θ a ha φ₀ (k : ℝ) j.castSucc)
      (frozenFlow β θ a ha φ₀ (k : ℝ) (Fin.last n))) := by
  have h := (continuous_frozenLastVelocityDeriv_full (n := n) β θ a ha).comp
    (continuous_frozenFlow β θ a ha (k : ℝ) (Nat.cast_nonneg k))
  simpa! only [Function.comp_def] using h

/-- The actual original repeller detector is Borel in the full space of original initial angular tuples.
Source: arXiv:2411.04990v2, §5 and Appendix A, the measurable exceptional event required before applying original-product Fubini in `thm:fixed_centers`. -/
theorem measurableSet_frozenRepellerDetector (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j) :
    MeasurableSet (frozenRepellerDetector (n := n) β θ a ha) := by
  apply MeasurableSet.iUnion
  intro q
  apply MeasurableSet.iUnion
  intro N
  apply MeasurableSet.iInter
  intro k
  apply MeasurableSet.iInter
  intro hNk
  exact (isClosed_le continuous_const (continuous_frozenLastVelocityDeriv_sample β θ a ha k)).measurableSet

/-- Along every genuinely converging original frozen trajectory, the Borel detector holds exactly when the actual last-angle derivative at its limit is positive.
Source: arXiv:2411.04990v2, §5 and Appendix A, the actual repelling-limit interpretation of the measurable event for `thm:fixed_centers`. -/
theorem frozenRepellerDetector_iff_of_tendsto (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (φ₀ φstar : Idx (n + 1) → ℝ) (hφ : Tendsto (frozenFlow β θ a ha φ₀) atTop (nhds φstar)) :
    φ₀ ∈ frozenRepellerDetector β θ a ha ↔
      0 < frozenLastVelocityDeriv β θ a (fun j => φstar j.castSucc) (φstar (Fin.last n)) := by
  have hD := (continuous_frozenLastVelocityDeriv_full β θ a ha).tendsto φstar
  have hs := (hD.comp hφ).comp (tendsto_natCast_atTop_atTop (R := ℝ))
  rw [mem_frozenRepellerDetector_iff]
  exact frozen_sequence_threshold_iff _ _ hs

/-- At an actual original equilibrium, the Borel detector holds exactly when the actual normalized last-angle derivative is strictly positive.
Source: arXiv:2411.04990v2, §5 and Appendix A, the original stationary repelling case in `thm:fixed_centers`. -/
theorem frozenRepellerDetector_iff_of_equilibrium (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (φstar : Idx (n + 1) → ℝ) (hcrit : frozenVelocity (n + 1) m β θ a φstar = 0) :
    φstar ∈ frozenRepellerDetector β θ a ha ↔
      0 < frozenLastVelocityDeriv β θ a (fun j => φstar j.castSucc) (φstar (Fin.last n)) := by
  have heq : frozenFlow β θ a ha φstar = fun _ => φstar :=
    frozenFlow_eq_of_ODE β θ a ha (fun _ => φstar) (fun t => by simpa only [hcrit] using hasDerivAt_const t φstar)
  apply frozenRepellerDetector_iff_of_tendsto
  rw [heq]
  exact tendsto_const_nhds

/-- Convergence of the actual original prefix and last angle separately gives the same exact repelling-limit interpretation of the Borel detector.
Source: arXiv:2411.04990v2, §5 and Appendix A, the actual causal induction form of the measurable event used for `thm:fixed_centers`. -/
theorem frozenRepellerDetector_iff_of_prefix_last_tendsto (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (φ₀ : Idx (n + 1) → ℝ) (pstar : Idx n → ℝ) (l : ℝ)
    (hp : Tendsto (fun t j => frozenFlow β θ a ha φ₀ t j.castSucc) atTop (nhds pstar))
    (hl : Tendsto (fun t => frozenFlow β θ a ha φ₀ t (Fin.last n)) atTop (nhds l)) :
    φ₀ ∈ frozenRepellerDetector β θ a ha ↔ 0 < frozenLastVelocityDeriv β θ a pstar l := by
  have hfull : Tendsto (frozenFlow β θ a ha φ₀) atTop (nhds (Fin.snoc pstar l)) := by
    apply tendsto_pi_nhds.mpr
    intro j
    refine Fin.lastCases ?_ (fun k => ?_) j
    · simpa only [Fin.snoc_last] using hl
    · simpa only [Fin.snoc_castSucc] using (tendsto_pi_nhds.mp hp) k
  simpa only [Fin.snoc_castSucc, Fin.snoc_last] using
    frozenRepellerDetector_iff_of_tendsto β θ a ha φ₀ (Fin.snoc pstar l) hfull

/-- A genuine positive constant sequence jointly realizes the convergence and threshold premises.
Source: arXiv:2411.04990v2, Appendix A, the countable positive-limit detection used for `thm:fixed_centers`. -/
example : Tendsto (fun _ : ℕ => (1 : ℝ)) atTop (nhds 1) ∧
    ∃ q N : ℕ, ∀ k : ℕ, N ≤ k → 1 / ((q : ℝ) + 1) ≤ (1 : ℝ) := by
  exact ⟨tendsto_const_nhds, (frozen_sequence_threshold_iff _ 1 tendsto_const_nhds).mpr zero_lt_one⟩

/-- The genuine original antipodal equilibrium belongs to the Borel detector, so the measurable exceptional event need not be empty.
Source: arXiv:2411.04990v2, §5 and Appendix A, the actual one-center repelling equilibrium of `thm:fixed_centers`. -/
example : (fun _ : Idx 1 => Real.pi) ∈ frozenRepellerDetector 1 (fun _ : Idx 1 => 0)
    (fun _ => 1) (fun _ => zero_le_one) := by
  have hv : frozenVelocity 1 1 1 (fun _ => 0) (fun _ => 1) (fun _ => Real.pi) = 0 := by
    funext k
    simp [frozenVelocity]
  rw [frozenRepellerDetector_iff_of_equilibrium 1 _ _ (fun _ => zero_le_one) _ hv]
  apply (frozenLastVelocityDeriv_pos_iff 1 (fun _ => 0) (fun _ => 1) (fun _ => zero_le_one)
    (fun _ : Idx 0 => Real.pi) Real.pi ?_).mpr
  · norm_num [frozenLastCurvature, Fin.sum_univ_zero, Fin.sum_univ_one, g_pot]
    positivity
  · rw [frozenVelocity_last_eq_force]
    simp [frozenLastForce, h_pot]

/-- The genuine original attracting center is excluded from the Borel detector, with an actual stationary and converging original trajectory.
Source: arXiv:2411.04990v2, §5 and Appendix A, the original one-center attracting equilibrium in `thm:fixed_centers`. -/
example : (fun _ : Idx 1 => (0 : ℝ)) ∉ frozenRepellerDetector 1 (fun _ : Idx 1 => 0)
    (fun _ => 1) (fun _ => zero_le_one) := by
  have hv : frozenVelocity 1 1 1 (fun _ => 0) (fun _ => 1) (fun _ => 0) = 0 := by
    funext k
    simp [frozenVelocity]
  rw [frozenRepellerDetector_iff_of_equilibrium 1 _ _ (fun _ => zero_le_one) _ hv]
  apply not_lt.mpr
  norm_num [frozenLastVelocityDeriv, frozenLastCurvature, frozenLastForce, frozenLastPartition,
    Fin.sum_univ_zero, Fin.sum_univ_one, g_pot, h_pot]

end Transformer.Causal
