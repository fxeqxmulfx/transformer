module

public import Transformer.Causal.Section4_SphereFlow
public import Transformer.Perspective.Section2_FlowContinuity
public import Mathlib.MeasureTheory.Constructions.Polish.Basic
public import Mathlib.Analysis.Calculus.Deriv.Prod

/-!
# Continuous dependence of the original causal flow on its initial tokens

ArXiv:2411.04990v2, §4, proof of `thm1`.
The actual causal ambient field is smooth, hence Lipschitz on every
compact ball. All original spherical trajectories lie in the unit
ball for the maximum token norm. Grönwall therefore gives one
exponential stability estimate uniformly over every initial tuple.

The deterministic original flow is Lipschitz and measurable in its
initial tuple at each nonnegative time. Convergence events sampled
at integer times are consequently Borel, including convergence to
a measurable target depending on the same original initialization.
The countable observation times give a measurable convergence test.
Convergence over all real times is handled by the later dynamical argument.
These are the measurable events needed for conditioning on a causal
prefix. No null-set or convergence assertion is part of their definition.
-/

@[expose] public section

open Real Set MeasureTheory Filter
open scoped NNReal

namespace Transformer.Causal

variable {d n : ℕ}

/-- The actual original causal field is Lipschitz on every compact ambient ball.
Source: arXiv:2411.04990v2, §4, smooth autonomous dependence in the proof of `thm1`. -/
theorem exists_lipschitzOnWith_csaAmbientField (β : ℝ) (Q K V : ParamMatrix d) (R : ℝ) :
    ∃ L : ℝ≥0, LipschitzOnWith L (csaAmbientField (n := n) β Q K V)
      (Metric.closedBall 0 R) := by
  exact (contDiff_csaAmbientField β Q K V).contDiffOn.exists_lipschitzOnWith
    (by simp) (convex_closedBall (0 : Idx n → EucSpace d) R) (isCompact_closedBall 0 R)

/-- A Lipschitz bound chosen from the proved compact-ball estimate for the original field.
Source: arXiv:2411.04990v2, §4, the original flow used in `thm1`.
This number records only a proved field bound and imposes no regularity or convergence premise. -/
noncomputable def csaUnitLipschitz (β : ℝ) (Q K V : ParamMatrix d) : ℝ≥0 :=
  Classical.choose (exists_lipschitzOnWith_csaAmbientField (n := n) β Q K V 1)

/-- The chosen bound applies to the actual original field on the whole maximum-norm unit ball.
Source: arXiv:2411.04990v2, §4, the compact-state estimate in the proof of `thm1`. -/
theorem lipschitzOnWith_csaAmbientField_unit (β : ℝ) (Q K V : ParamMatrix d) :
    LipschitzOnWith (csaUnitLipschitz (n := n) β Q K V) (csaAmbientField (n := n) β Q K V)
      (Metric.closedBall 0 1) :=
  Classical.choose_spec (exists_lipschitzOnWith_csaAmbientField β Q K V 1)

/-- Every original spherical token tuple belongs to the ambient maximum-norm unit ball.
Source: arXiv:2411.04990v2, §2, the state space of `eqn:csa`. -/
theorem coe_sphereTuple_mem_closedBall (X : SphereTuple d n) :
    (fun i => (X i : EucSpace d)) ∈ Metric.closedBall (0 : Idx n → EucSpace d) 1 := by
  rw [Metric.mem_closedBall, dist_zero_right]
  exact (pi_norm_le_iff_of_nonneg zero_le_one).mpr fun i =>
    (mem_sphere_zero_iff_norm.mp (X i).2).le

/-- The original causal trajectories depend exponentially continuously on their original initial tuples.
Source: arXiv:2411.04990v2, §4, the flow used to condition on initial data in `thm1`.
The constant is uniform in both actual trajectories and in forward time. -/
theorem dist_CSA_le_exp (β : ℝ) (Q K V : ParamMatrix d) (X Y : ℝ → SphereTuple d n)
    (hX : CSA d n β Q K V X) (hY : CSA d n β Q K V Y) (t : ℝ) (ht : 0 ≤ t) :
    dist (X t) (Y t) ≤ dist (X 0) (Y 0) *
      Real.exp ((csaUnitLipschitz (n := n) β Q K V : ℝ) * t) := by
  let x (s : ℝ) (i : Idx n) := (X s i : EucSpace d)
  let y (s : ℝ) (i : Idx n) := (Y s i : EucSpace d)
  have hx (s : ℝ) : HasDerivAt x (csaAmbientField β Q K V (x s)) s :=
    hasDerivAt_pi.mpr ((CSA_iff_csaAmbientField β Q K V X).mp hX s)
  have hy (s : ℝ) : HasDerivAt y (csaAmbientField β Q K V (y s)) s :=
    hasDerivAt_pi.mpr ((CSA_iff_csaAmbientField β Q K V Y).mp hY s)
  have hd (s : ℝ) : HasDerivAt (fun r => x r - y r)
      (csaAmbientField β Q K V (x s) - csaAmbientField β Q K V (y s)) s := (hx s).sub (hy s)
  have hc : Continuous (fun s => x s - y s) :=
    continuous_iff_continuousAt.mpr fun s => (hd s).continuousAt
  have hb (s : ℝ) : ‖csaAmbientField β Q K V (x s) - csaAmbientField β Q K V (y s)‖ ≤
      (csaUnitLipschitz (n := n) β Q K V : ℝ) * ‖x s - y s‖ :=
    lipschitzOnWith_iff_norm_sub_le.mp (lipschitzOnWith_csaAmbientField_unit β Q K V)
      (coe_sphereTuple_mem_closedBall (X s)) (coe_sphereTuple_mem_closedBall (Y s))
  have hG := norm_le_gronwallBound_of_norm_deriv_right_le (a := 0) (b := t)
    (δ := ‖x 0 - y 0‖) (K := (csaUnitLipschitz (n := n) β Q K V : ℝ)) (ε := 0)
    hc.continuousOn (fun s hs => (hd s).hasDerivWithinAt) le_rfl
    (fun s hs => by simpa only [add_zero] using hb s) t ⟨ht, le_rfl⟩
  simpa only [gronwallBound_ε0, sub_zero, ← dist_eq_norm, x, y,
    Perspective.dist_sphereTuple_coe] using hG

/-- At each forward time the actual original causal flow is Lipschitz in all initial tokens.
Source: arXiv:2411.04990v2, §4, the deterministic flow in the proof of `thm1`. -/
theorem lipschitzWith_csaFlow (β : ℝ) (Q K V : ParamMatrix d) (t : ℝ) (ht : 0 ≤ t) :
    LipschitzWith (Real.toNNReal (Real.exp ((csaUnitLipschitz (n := n) β Q K V : ℝ) * t)))
      (fun X₀ : SphereTuple d n => csaFlow β Q K V X₀ t) := by
  apply LipschitzWith.of_dist_le_mul
  intro X₀ Y₀
  rw [Real.coe_toNNReal _ (Real.exp_pos _).le, mul_comm]
  simpa only [csaFlow_zero] using
    dist_CSA_le_exp β Q K V _ _ (csaFlow_spec β Q K V X₀).2 (csaFlow_spec β Q K V Y₀).2 t ht

/-- The actual original causal flow is continuous in its initial tuple at each nonnegative time.
Source: arXiv:2411.04990v2, §4, conditioning on the causal flow in `thm1`. -/
theorem continuous_csaFlow (β : ℝ) (Q K V : ParamMatrix d) (t : ℝ) (ht : 0 ≤ t) :
    Continuous (fun X₀ : SphereTuple d n => csaFlow β Q K V X₀ t) :=
  (lipschitzWith_csaFlow β Q K V t ht).continuous

/-- The actual original causal flow at forward times is measurable under the original Borel state spaces.
Source: arXiv:2411.04990v2, §4, the initialization events in the proof of `thm1`. -/
theorem measurable_csaFlow (β : ℝ) (Q K V : ParamMatrix d) (t : ℝ) (ht : 0 ≤ t) :
    Measurable (fun X₀ : SphereTuple d n => csaFlow β Q K V X₀ t) :=
  (continuous_csaFlow β Q K V t ht).measurable

/-- Integer-time convergence of an original token to a measurable initialization-dependent target is a Borel event.
Source: arXiv:2411.04990v2, §4, the conditional convergence events in the proof of `thm1`. -/
theorem measurableSet_csaFlow_tendsto (β : ℝ) (Q K V : ParamMatrix d) (i : Idx n)
    (g : SphereTuple d n → SSphere d) (hg : Measurable g) :
    MeasurableSet {X₀ : SphereTuple d n |
      Tendsto (fun m : ℕ => csaFlow β Q K V X₀ (m : ℝ) i) atTop (nhds (g X₀))} := by
  exact measurableSet_tendsto_fun
    (fun m : ℕ => (measurable_pi_apply i).comp (measurable_csaFlow β Q K V m (by positivity))) hg

/-- Integer-time convergence of all original tokens to one original initial token is a Borel event.
Source: arXiv:2411.04990v2, §4, the initialization event in `thm1`. -/
theorem measurableSet_csaFlow_consensus_nat (β : ℝ) (Q K V : ParamMatrix d) (j : Idx n) :
    MeasurableSet {X₀ : SphereTuple d n | ∀ i : Idx n,
      Tendsto (fun m : ℕ => csaFlow β Q K V X₀ (m : ℝ) i) atTop (nhds (X₀ j))} := by
  rw [Set.ofPred_forall]
  exact MeasurableSet.iInter fun i =>
    measurableSet_csaFlow_tendsto β Q K V i (fun X₀ => X₀ j) (measurable_pi_apply j)

/-- Convergence of an original token to the antipode of an original initial token is a Borel event.
Source: arXiv:2411.04990v2, §4, the exceptional antipodal limit in the proof of `thm1`.
The event uses the constructed original flow and the original integer-time observations. -/
theorem measurableSet_csaFlow_antipode_nat (β : ℝ) (Q K V : ParamMatrix d) (i j : Idx n) :
    MeasurableSet {X₀ : SphereTuple d n |
      Tendsto (fun m : ℕ => csaFlow β Q K V X₀ (m : ℝ) i) atTop
        (nhds (Perspective.antipode d (X₀ j)))} := by
  have hc : Continuous (fun X₀ : SphereTuple d n => Perspective.antipode d (X₀ j)) := by
    apply continuous_induced_rng.mpr
    change Continuous (fun X₀ : SphereTuple d n => -(X₀ j : EucSpace d))
    fun_prop
  exact measurableSet_csaFlow_tendsto β Q K V i _ hc.measurable

/-- Nonzero stationary original trajectories jointly realize both dynamical stability premises,
the forward-time condition and a measurable initial-coordinate target for convergence events. -/
example (β : ℝ) (Q K : ParamMatrix 2) :
    CSA 2 2 β Q K (ContinuousLinearMap.id ℝ _) (fun _ _ => basePoint 1) ∧
    CSA 2 2 β Q K (ContinuousLinearMap.id ℝ _) (fun _ _ => basePoint 1) ∧
    (0 : ℝ) ≤ 1 ∧ Measurable (fun X : SphereTuple 2 2 => X 0) :=
  ⟨CSA_const_consensus β Q K (basePoint 1), CSA_const_consensus β Q K (basePoint 1),
    zero_le_one, measurable_pi_apply 0⟩

end Transformer.Causal
