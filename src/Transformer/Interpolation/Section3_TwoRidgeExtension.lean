/-
# Complete ambient trajectories for two original ReLU neurons

ArXiv:2411.04551v3, §1, `eq: vf` and `eq: neural.ode.sphere`,
and Appendix, the constant-control construction toward `lem: two.balls`.

The sum of the two complete ridge extensions equals the original two-neuron
field on the unit sphere. It remains locally Lipschitz across both ReLU
kinks and has an explicit global linear growth bound. Thus a global ambient
trajectory exists from every initial unit vector.

The squared-norm defect obeys a homogeneous scalar equation, which forces
the original sphere to be invariant. Local ODE uniqueness then applies to
any two actual spherical trajectories. No particle existence, sphere
invariance or uniqueness premise is left unproved.
-/

import Transformer.Interpolation.Section3_TwoRidgeFields

open Real MeasureTheory Set Filter
open scoped Topology NNReal BigOperators

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The sum of the complete ambient extensions of the original neurons.
Auxiliary for arXiv:2411.04551v3, §1, `eq: neural.ode.sphere`. -/
noncomputable def twoRidgeGlobalField (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ)
    (x : EucSpace d) : EucSpace d := ridgeGlobalField p u b x + ridgeGlobalField q v c x

/-- The extension preserves the exact original field on unit vectors.
Source: arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem twoRidgeGlobalField_eq_of_unit (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ)
    (x : EucSpace d) (hx : ‖x‖ = 1) : twoRidgeGlobalField p u b q v c x = twoRidgeField p u b q v c x := by
  rw [twoRidgeGlobalField, ridgeGlobalField_eq_of_unit p u b x hx, ridgeGlobalField_eq_of_unit q v c x hx]
  rfl

/-- The extension is locally Lipschitz, including at both activation kinks.
Auxiliary for arXiv:2411.04551v3, §1, the original particle equation. -/
theorem locallyLipschitz_twoRidgeGlobalField (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ) :
    LocallyLipschitz (twoRidgeGlobalField p u b q v c) :=
  (locallyLipschitz_ridgeGlobalField p u b).add (locallyLipschitz_ridgeGlobalField q v c)

/-- The actual ambient extension has explicit global linear growth.
Auxiliary for arXiv:2411.04551v3, §1, the original particle equation. -/
theorem norm_twoRidgeGlobalField_le (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ) (x : EucSpace d) :
    ‖twoRidgeGlobalField p u b q v c x‖ ≤
      (4 * (|b| + ‖u‖) * ‖p‖ + 4 * (|c| + ‖v‖) * ‖q‖) * ‖x‖ := by
  calc
    _ ≤ ‖ridgeGlobalField p u b x‖ + ‖ridgeGlobalField q v c x‖ := norm_add_le _ _
    _ ≤ 4 * (|b| + ‖u‖) * ‖p‖ * ‖x‖ + 4 * (|c| + ‖v‖) * ‖q‖ * ‖x‖ :=
      add_le_add (norm_ridgeGlobalField_le p u b x) (norm_ridgeGlobalField_le q v c x)
    _ = _ := by ring


/-- The original two-neuron field is Lipschitz on the whole sphere.
Source: arXiv:2411.04551v3, §1, the Lipschitz well-posedness argument. -/
theorem exists_lipschitzOnWith_twoRidgeField (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ) :
    ∃ K : ℝ≥0, LipschitzOnWith K (twoRidgeField p u b q v c) (Sphere d) := by
  obtain ⟨K, hK⟩ := GlobalFlow.exists_lipschitzOnWith_closedBall (locallyLipschitz_twoRidgeGlobalField p u b q v c) 1
  refine ⟨K, lipschitzOnWith_iff_norm_sub_le.mpr ?_⟩
  intro x hx y hy
  have hxu := mem_sphere_zero_iff_norm.mp hx
  have hyu := mem_sphere_zero_iff_norm.mp hy
  rw [← twoRidgeGlobalField_eq_of_unit p u b q v c x hxu,
    ← twoRidgeGlobalField_eq_of_unit p u b q v c y hyu]
  exact lipschitzOnWith_iff_norm_sub_le.mp hK (mem_closedBall_zero_iff.mpr hxu.le)
    (mem_closedBall_zero_iff.mpr hyu.le)

/-- Every actual ambient trajectory from a unit vector stays on the sphere.
Source: arXiv:2411.04551v3, §1, the tangent projection in `eq: vf`. -/
theorem norm_eq_one_of_twoRidgeGlobalField (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ)
    (Y : ℝ → EucSpace d) (hY : ∀ t : ℝ, HasDerivAt Y (twoRidgeGlobalField p u b q v c (Y t)) t)
    (h0 : ‖Y 0‖ = 1) (t : ℝ) : ‖Y t‖ = 1 := by
  have hc : Continuous Y := continuous_iff_continuousAt.mpr fun s => (hY s).continuousAt
  let a (s : ℝ) := -2 * ridgeExtensionScale (Y s) *
    (max (b - inner ℝ (Y s) u) 0 * inner ℝ (Y s) p +
      max (c - inner ℝ (Y s) v) 0 * inner ℝ (Y s) q)
  have ha : Continuous a := by
    have hs : Continuous (fun s => ridgeExtensionScale (Y s)) :=
      (continuous_const.mul hc.norm).div ((continuous_const.add (hc.norm.pow 2)).pow 2)
        (fun s => pow_ne_zero _ (by positivity))
    have hα : Continuous (fun s => max (b - inner ℝ (Y s) u) 0) :=
      (continuous_const.sub (hc.inner continuous_const)).max continuous_const
    have hβ : Continuous (fun s => max (c - inner ℝ (Y s) v) 0) :=
      (continuous_const.sub (hc.inner continuous_const)).max continuous_const
    exact (continuous_const.mul hs).mul ((hα.mul (hc.inner continuous_const)).add
      (hβ.mul (hc.inner continuous_const)))
  have hd (s : ℝ) : HasDerivAt (fun r => ‖Y r‖ ^ 2 - 1) (a s * (‖Y s‖ ^ 2 - 1)) s := by
    convert (hY s).norm_sq.sub_const 1 using 1
    simp only [twoRidgeGlobalField, inner_add_right, ridgeGlobalField,
      real_inner_smul_right, proj, inner_sub_right, real_inner_self_eq_norm_sq]
    dsimp only [a]
    ring
  have hz := Perspective.scalar_eq_zero_of_linearODE a _ ha hd (by rw [h0]; norm_num) t
  nlinarith [norm_nonneg (Y t)]

/-- Every spherical initial point supplies an actual global trajectory
of the original two-neuron field. Source: arXiv:2411.04551v3,
§1, `eq: neural.ode.sphere`. -/
theorem exists_twoRidgeTrajectory (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ) (x₀ : SSphere d) :
    ∃ Y : ℝ → SSphere d, Y 0 = x₀ ∧ ∀ t : ℝ,
      HasDerivAt (fun s => (Y s : EucSpace d)) (twoRidgeField p u b q v c (Y t)) t := by
  obtain ⟨Y, hY0, hY⟩ := GlobalFlow.exists_global (locallyLipschitz_twoRidgeGlobalField p u b q v c)
    (C := 4 * (|b| + ‖u‖) * ‖p‖ + 4 * (|c| + ‖v‖) * ‖q‖) (by positivity)
    (norm_twoRidgeGlobalField_le p u b q v c) (x₀ : EucSpace d)
  have hunit (t : ℝ) : ‖Y t‖ = 1 := norm_eq_one_of_twoRidgeGlobalField p u b q v c Y hY
    (by rw [hY0]; exact mem_sphere_zero_iff_norm.mp x₀.2) t
  let X (t : ℝ) : SSphere d := ⟨Y t, mem_sphere_zero_iff_norm.mpr (hunit t)⟩
  refine ⟨X, Subtype.ext hY0, fun t => ?_⟩
  simpa only [twoRidgeGlobalField_eq_of_unit p u b q v c (Y t) (hunit t)] using hY t

/-- Actual original spherical trajectories with the same initial datum
agree globally. Source: arXiv:2411.04551v3, §1, `eq: neural.ode.sphere`. -/
theorem twoRidgeTrajectory_unique (p u : EucSpace d) (b : ℝ) (q v : EucSpace d) (c : ℝ)
    (X Y : ℝ → SSphere d)
    (hX : ∀ t : ℝ, HasDerivAt (fun s => (X s : EucSpace d)) (twoRidgeField p u b q v c (X t)) t)
    (hY : ∀ t : ℝ, HasDerivAt (fun s => (Y s : EucSpace d)) (twoRidgeField p u b q v c (Y t)) t)
    (h0 : X 0 = Y 0) : X = Y := by
  have he := GlobalFlow.eq_of_hasDerivAt (locallyLipschitz_twoRidgeGlobalField p u b q v c)
    (f := fun s => (X s : EucSpace d)) (g := fun s => (Y s : EucSpace d))
    (fun t => by rw [twoRidgeGlobalField_eq_of_unit _ _ _ _ _ _ _ (mem_sphere_zero_iff_norm.mp (X t).2)]; exact hX t)
    (fun t => by rw [twoRidgeGlobalField_eq_of_unit _ _ _ _ _ _ _ (mem_sphere_zero_iff_norm.mp (Y t).2)]; exact hY t)
    (congrArg Subtype.val h0)
  funext t
  exact Subtype.ext (congrFun he t)

/-- Two genuinely activated nonzero neurons admit an actual stationary
unit trajectory at their common output axis. The ambient and original
particle equations and unit initial condition hold jointly. -/
example : ∃ Y : ℝ → SSphere 2, ‖(Y 0 : EucSpace 2)‖ = 1 ∧
    (∀ t, HasDerivAt (fun s => (Y s : EucSpace 2))
      (twoRidgeGlobalField (basePoint 1 : EucSpace 2) 0 1 (basePoint 1 : EucSpace 2) 0 1 (Y t)) t) ∧
    (∀ t, HasDerivAt (fun s => (Y s : EucSpace 2))
      (twoRidgeField (basePoint 1 : EucSpace 2) 0 1 (basePoint 1 : EucSpace 2) 0 1 (Y t)) t) ∧
    Y 0 = Y 0 := by
  let a : SSphere 2 := basePoint 1
  have ha : ‖(a : EucSpace 2)‖ = 1 := mem_sphere_zero_iff_norm.mp a.2
  have hf : twoRidgeField (a : EucSpace 2) 0 1 (a : EucSpace 2) 0 1 a = 0 := by
    simp [twoRidgeField, proj, ha]
  refine ⟨fun _ => a, ha, ?_, ?_, rfl⟩
  · intro t
    rw [twoRidgeGlobalField_eq_of_unit _ _ _ _ _ _ _ ha, hf]
    exact hasDerivAt_const t _
  · intro t
    rw [hf]
    exact hasDerivAt_const t _

end Transformer.Interpolation
