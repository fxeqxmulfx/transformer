module

public import Transformer.Causal.Section5_FrozenFlow
public import Mathlib.Analysis.ODE.Gronwall
public import Mathlib.Analysis.Calculus.MeanValue

/-!
# Continuous dependence of the actual frozen angular flow

ArXiv:2411.04990v2, §5 and Appendix A, the smooth frozen
specialization of `thm:fixed_centers`.
The actual original angular velocity has norm at most one.
Consequently an actual trajectory moves by at most elapsed
time in the original maximum norm. Initial tuples in a fixed
ball stay in one larger compact ball over any finite forward
time interval. The original smooth field is Lipschitz there.

Grönwall's inequality gives an exponential comparison for
those actual original trajectories. Thus at each nonnegative
time the original flow map is Lipschitz on every initial ball,
continuous on the whole lifted state space, and measurable.
The constants are chosen from proved compact-ball bounds.
No compactness of the full lifted state space is presumed.
These are the actual flow maps needed for the later Borel
repelling-limit detector and the original Fubini argument.
-/

@[expose] public section

open Real Set Filter MeasureTheory
open scoped NNReal

namespace Transformer.Causal

variable {n m : ℕ}

/-- An actual original frozen trajectory moves by at most its elapsed time in the original maximum angular norm.
Source: arXiv:2411.04990v2, §5, the bounded original angular velocity in `thm:fixed_centers`. -/
theorem norm_frozenODE_sub_le_time (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (φ : ℝ → Idx n → ℝ) (hφ : ∀ t, HasDerivAt φ (frozenVelocity n m β θ a (φ t)) t)
    (s t : ℝ) (hst : s ≤ t) : ‖φ t - φ s‖ ≤ t - s := by
  have h := norm_image_sub_le_of_norm_deriv_le_segment' (a := s) (b := t)
    (fun u hu => (hφ u).hasDerivWithinAt) (fun u hu => norm_frozenVelocity_le_one β θ a ha (φ u)) t ⟨hst, le_rfl⟩
  simpa only [one_mul] using h

/-- The actual forward angular norm is bounded by the actual initial norm plus elapsed time.
Source: arXiv:2411.04990v2, §5, the original lifted trajectories in `thm:fixed_centers`. -/
theorem norm_frozenODE_le (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (φ : ℝ → Idx n → ℝ) (hφ : ∀ t, HasDerivAt φ (frozenVelocity n m β θ a (φ t)) t)
    (t : ℝ) (ht : 0 ≤ t) : ‖φ t‖ ≤ ‖φ 0‖ + t := by
  have h := norm_frozenODE_sub_le_time β θ a ha φ hφ 0 t ht
  have hn := norm_le_norm_sub_add (φ t) (φ 0)
  linarith

/-- The actual canonical original angular flow satisfies the same proved initial-norm plus time estimate.
Source: arXiv:2411.04990v2, §5, the actual global flow in `thm:fixed_centers`. -/
theorem norm_frozenFlow_le (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (φ₀ : Idx n → ℝ) (t : ℝ) (ht : 0 ≤ t) : ‖frozenFlow β θ a ha φ₀ t‖ ≤ ‖φ₀‖ + t := by
  simpa only [frozenFlow_zero] using norm_frozenODE_le β θ a ha _ (frozenFlow_spec β θ a ha φ₀).2 t ht

/-- The actual original frozen field is Lipschitz on every original compact angular ball.
Source: arXiv:2411.04990v2, §5 and Appendix A, smooth dependence for the original dynamics in `thm:fixed_centers`. -/
theorem exists_lipschitzOnWith_frozenVelocity (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j) (R : ℝ) :
    ∃ L : ℝ≥0, LipschitzOnWith L (frozenVelocity n m β θ a) (Metric.closedBall 0 R) :=
  (contDiff_frozenVelocity β θ a ha).contDiffOn.exists_lipschitzOnWith (by simp)
    (convex_closedBall (0 : Idx n → ℝ) R) (isCompact_closedBall 0 R)

/-- A Lipschitz bound chosen only after proving it exists for the actual original field on the specified angular ball.
Source: arXiv:2411.04990v2, §5 and Appendix A, the compact-ball estimate for the original frozen flow of `thm:fixed_centers`. -/
noncomputable def frozenBallLipschitz (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j) (R : ℝ) : ℝ≥0 :=
  Classical.choose (exists_lipschitzOnWith_frozenVelocity (n := n) β θ a ha R)

/-- The chosen constant bounds the actual original field on the entire specified original angular ball.
Source: arXiv:2411.04990v2, §5 and Appendix A, the smooth original dependence used for `thm:fixed_centers`. -/
theorem lipschitzOnWith_frozenVelocity_ball (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j) (R : ℝ) :
    LipschitzOnWith (frozenBallLipschitz (n := n) β θ a ha R) (frozenVelocity n m β θ a) (Metric.closedBall 0 R) :=
  Classical.choose_spec (exists_lipschitzOnWith_frozenVelocity β θ a ha R)

/-- Two actual original frozen trajectories starting in the same original angular ball satisfy a proved finite-time exponential comparison.
Source: arXiv:2411.04990v2, §5 and Appendix A, continuous dependence for the actual smooth frozen system in `thm:fixed_centers`. -/
theorem dist_frozenODE_le_exp (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (φ ψ : ℝ → Idx n → ℝ)
    (hφ : ∀ t, HasDerivAt φ (frozenVelocity n m β θ a (φ t)) t)
    (hψ : ∀ t, HasDerivAt ψ (frozenVelocity n m β θ a (ψ t)) t)
    (R t : ℝ) (ht : 0 ≤ t) (hφ0 : ‖φ 0‖ ≤ R) (hψ0 : ‖ψ 0‖ ≤ R) :
    dist (φ t) (ψ t) ≤ dist (φ 0) (ψ 0) * Real.exp ((frozenBallLipschitz (n := n) β θ a ha (R + t) : ℝ) * t) := by
  have hφball : ∀ s ∈ Icc (0 : ℝ) t, φ s ∈ Metric.closedBall 0 (R + t) := by
    intro s hs
    rw [mem_closedBall_zero_iff]
    have h := norm_frozenODE_le β θ a ha φ hφ s hs.1
    linarith [hφ0, hs.2]
  have hψball : ∀ s ∈ Icc (0 : ℝ) t, ψ s ∈ Metric.closedBall 0 (R + t) := by
    intro s hs
    rw [mem_closedBall_zero_iff]
    have h := norm_frozenODE_le β θ a ha ψ hψ s hs.1
    linarith [hψ0, hs.2]
  have hd (s : ℝ) := (hφ s).sub (hψ s)
  have hc : Continuous (fun s => φ s - ψ s) := continuous_iff_continuousAt.mpr (fun s => (hd s).continuousAt)
  have hb : ∀ s ∈ Ico (0 : ℝ) t,
      ‖frozenVelocity n m β θ a (φ s) - frozenVelocity n m β θ a (ψ s)‖ ≤
        (frozenBallLipschitz (n := n) β θ a ha (R + t) : ℝ) * ‖φ s - ψ s‖ := by
    intro s hs
    exact lipschitzOnWith_iff_norm_sub_le.mp (lipschitzOnWith_frozenVelocity_ball β θ a ha (R + t))
      (hφball s ⟨hs.1, hs.2.le⟩) (hψball s ⟨hs.1, hs.2.le⟩)
  have hG := norm_le_gronwallBound_of_norm_deriv_right_le (a := 0) (b := t)
    (δ := ‖φ 0 - ψ 0‖) (K := (frozenBallLipschitz (n := n) β θ a ha (R + t) : ℝ)) (ε := 0)
    hc.continuousOn (fun s hs => (hd s).hasDerivWithinAt) le_rfl
    (fun s hs => by simpa only [add_zero] using hb s hs) t ⟨ht, le_rfl⟩
  simpa only [gronwallBound_ε0, sub_zero, ← dist_eq_norm] using hG

/-- At each forward time, the actual original flow is Lipschitz on every bounded ball of original initial angular tuples.
Source: arXiv:2411.04990v2, §5 and Appendix A, the original flow used in the conditional initialization argument of `thm:fixed_centers`. -/
theorem lipschitzOnWith_frozenFlow_ball (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j) (R t : ℝ) (ht : 0 ≤ t) :
    LipschitzOnWith (Real.toNNReal (Real.exp ((frozenBallLipschitz (n := n) β θ a ha (R + t) : ℝ) * t)))
      (fun φ₀ : Idx n → ℝ => frozenFlow β θ a ha φ₀ t) (Metric.closedBall 0 R) := by
  rw [lipschitzOnWith_iff_dist_le_mul]
  intro φ₀ hφ₀ ψ₀ hψ₀
  rw [Real.coe_toNNReal _ (Real.exp_pos _).le, mul_comm]
  simpa only [frozenFlow_zero] using dist_frozenODE_le_exp β θ a ha _ _
    (frozenFlow_spec β θ a ha φ₀).2 (frozenFlow_spec β θ a ha ψ₀).2 R t ht
    (by simpa only [frozenFlow_zero] using mem_closedBall_zero_iff.mp hφ₀)
    (by simpa only [frozenFlow_zero] using mem_closedBall_zero_iff.mp hψ₀)

/-- The actual original frozen angular flow is continuous in every original initial tuple at every forward time.
Source: arXiv:2411.04990v2, §5 and Appendix A, the measurable smooth-flow dependence needed for `thm:fixed_centers`. -/
theorem continuous_frozenFlow (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j) (t : ℝ) (ht : 0 ≤ t) :
    Continuous (fun φ₀ : Idx n → ℝ => frozenFlow β θ a ha φ₀ t) := by
  apply continuous_iff_continuousAt.mpr
  intro φ₀
  have hL := lipschitzOnWith_frozenFlow_ball (n := n) β θ a ha (‖φ₀‖ + 1) t ht
  have hx : φ₀ ∈ Metric.ball (0 : Idx n → ℝ) (‖φ₀‖ + 1) := by rw [Metric.mem_ball, dist_zero_right]; linarith
  exact (hL.continuousOn φ₀ (Metric.ball_subset_closedBall hx)).continuousAt (Metric.closedBall_mem_nhds_of_mem hx)

/-- The actual original frozen angular flow is measurable in its original initialization at every forward time.
Source: arXiv:2411.04990v2, §5 and Appendix A, the Borel events used in the original probability-one conclusion of `thm:fixed_centers`. -/
theorem measurable_frozenFlow (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j) (t : ℝ) (ht : 0 ≤ t) :
    Measurable (fun φ₀ : Idx n → ℝ => frozenFlow β θ a ha φ₀ t) := (continuous_frozenFlow β θ a ha t ht).measurable

/-- Every original angular coordinate of the actual frozen flow is measurable in the original initialization at each forward time.
Source: arXiv:2411.04990v2, §5 and Appendix A, the actual coordinate observations in the conditional argument of `thm:fixed_centers`. -/
theorem measurable_frozenFlow_coordinate (β : ℝ) (θ a : Idx m → ℝ) (ha : ∀ j, 0 ≤ a j)
    (t : ℝ) (ht : 0 ≤ t) (k : Idx n) :
    Measurable (fun φ₀ : Idx n → ℝ => frozenFlow β θ a ha φ₀ t k) :=
  (measurable_pi_apply k).comp (measurable_frozenFlow β θ a ha t ht)

/-- Actual nonempty original stationary trajectories jointly realize the velocity, finite-time, initial-ball, and measurable-flow premises.
Source: arXiv:2411.04990v2, §5, the original one-center equilibrium in `thm:fixed_centers`. -/
example : (∀ j : Idx 1, (0 : ℝ) ≤ (fun _ : Idx 1 => (1 : ℝ)) j) ∧ (0 : ℝ) ≤ 1 ∧
    (∀ t : ℝ, HasDerivAt (fun _ : ℝ => (fun _ : Idx 1 => (0 : ℝ)))
      (frozenVelocity 1 1 1 (fun _ => 0) (fun _ => 1) (fun _ => 0)) t) ∧
    ‖(fun _ : Idx 1 => (0 : ℝ))‖ ≤ 1 ∧
    Measurable (fun φ₀ : Idx 1 → ℝ => frozenFlow 1 (fun _ : Idx 1 => 0) (fun _ => 1) (fun _ => zero_le_one) φ₀ 1) := by
  have hv : frozenVelocity 1 1 1 (fun _ => 0) (fun _ => 1) (fun _ : Idx 1 => 0) = 0 := by
    funext k
    simp [frozenVelocity]
  refine ⟨fun _ => zero_le_one, zero_le_one, fun t => ?_, by simp, measurable_frozenFlow 1 _ _ (fun _ => zero_le_one) 1 zero_le_one⟩
  simpa only [hv] using hasDerivAt_const t (fun _ : Idx 1 => (0 : ℝ))

end Transformer.Causal
