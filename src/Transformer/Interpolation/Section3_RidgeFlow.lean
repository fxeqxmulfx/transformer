/-
# The actual one-neuron flow exists globally and is unique

ArXiv:2411.04551v3, §1, `eq: vf` and `eq: neural.ode.sphere`,
and §3.2, `lem: perturbation`, Part 2.

The complete ambient extension from the preceding module gives a
trajectory through every spherical initial point. Its proved norm
invariant recovers the source's original projected ReLU field.
Local ODE uniqueness applies to any two such spherical trajectories.

The selected flow therefore solves the actual neuron equation for all
real times, depends continuously on time, obeys the autonomous flow
law, and has a two-sided inverse given by reversing time.
-/

import Transformer.Interpolation.Section3_RidgeExtension

open Real MeasureTheory Filter Set

namespace Transformer.Interpolation

/-- Every spherical initial point admits an actual global trajectory.
Source: arXiv:2411.04551v3, §1, `eq: neural.ode.sphere`, for the explicit
one-neuron parameters used in §3.2, `lem: perturbation`. -/
theorem exists_ridgeTrajectory {d : ℕ} (p u : EucSpace d) (b : ℝ) (x₀ : SSphere d) :
    ∃ Y : ℝ → SSphere d, Y 0 = x₀ ∧ ∀ t : ℝ,
      HasDerivAt (fun s => (Y s : EucSpace d))
        (max (b - inner ℝ (Y t : EucSpace d) u) 0 • proj d (Y t) p) t := by
  obtain ⟨Y, hY0, hY⟩ := GlobalFlow.exists_global (locallyLipschitz_ridgeGlobalField p u b)
    (C := 4 * (|b| + ‖u‖) * ‖p‖) (by positivity) (norm_ridgeGlobalField_le p u b)
    (x₀ : EucSpace d)
  have hunit (t : ℝ) : ‖Y t‖ = 1 := norm_eq_one_of_ridgeGlobalField p u b Y hY
    (by rw [hY0]; exact mem_sphere_zero_iff_norm.mp x₀.2) t
  let X (t : ℝ) : SSphere d := ⟨Y t, mem_sphere_zero_iff_norm.mpr (hunit t)⟩
  refine ⟨X, Subtype.ext hY0, fun t => ?_⟩
  simpa only [ridgeGlobalField_eq_of_unit p u b (Y t) (hunit t)] using hY t

/-- Two actual spherical trajectories with the same initial value agree.
The proved sphere invariant is used only for existence; uniqueness tests
the original equation against the ambient extension.
Source: arXiv:2411.04551v3, §1, `eq: neural.ode.sphere`. -/
theorem ridgeTrajectory_unique {d : ℕ} (p u : EucSpace d) (b : ℝ) (X Y : ℝ → SSphere d)
    (hX : ∀ t : ℝ, HasDerivAt (fun s => (X s : EucSpace d))
      (max (b - inner ℝ (X t : EucSpace d) u) 0 • proj d (X t) p) t)
    (hY : ∀ t : ℝ, HasDerivAt (fun s => (Y s : EucSpace d))
      (max (b - inner ℝ (Y t : EucSpace d) u) 0 • proj d (Y t) p) t)
    (h0 : X 0 = Y 0) : X = Y := by
  have heq := GlobalFlow.eq_of_hasDerivAt (locallyLipschitz_ridgeGlobalField p u b)
    (f := fun s => (X s : EucSpace d)) (g := fun s => (Y s : EucSpace d))
    (fun t => by
      rw [ridgeGlobalField_eq_of_unit _ _ _ _ (mem_sphere_zero_iff_norm.mp (X t).2)]
      exact hX t)
    (fun t => by
      rw [ridgeGlobalField_eq_of_unit _ _ _ _ (mem_sphere_zero_iff_norm.mp (Y t).2)]
      exact hY t)
    (congrArg Subtype.val h0)
  funext t
  exact Subtype.ext (congrFun heq t)

/-- The actual global one-neuron flow, chosen after proving existence.
Source: arXiv:2411.04551v3, §1, `eq: neural.ode.sphere`. -/
noncomputable def ridgeFlow {d : ℕ} (p u : EucSpace d) (b : ℝ) (x₀ : SSphere d) :
    ℝ → SSphere d := Classical.choose (exists_ridgeTrajectory p u b x₀)

/-- The selected actual flow has the prescribed initial datum.
Source: arXiv:2411.04551v3, §1, `eq: neural.ode.sphere`. -/
theorem ridgeFlow_zero {d : ℕ} (p u : EucSpace d) (b : ℝ) (x₀ : SSphere d) :
    ridgeFlow p u b x₀ 0 = x₀ := (Classical.choose_spec (exists_ridgeTrajectory p u b x₀)).1

/-- The selected flow follows the original projected ReLU field at every
real time. Source: arXiv:2411.04551v3, §1, `eq: neural.ode.sphere`. -/
theorem hasDerivAt_ridgeFlow {d : ℕ} (p u : EucSpace d) (b : ℝ) (x₀ : SSphere d) (t : ℝ) :
    HasDerivAt (fun s => (ridgeFlow p u b x₀ s : EucSpace d))
      (max (b - inner ℝ (ridgeFlow p u b x₀ t : EucSpace d) u) 0 •
        proj d (ridgeFlow p u b x₀ t) p) t :=
  (Classical.choose_spec (exists_ridgeTrajectory p u b x₀)).2 t

/-- The trajectory solves the original network field at the explicit
single-neuron matrices, for every spherical probability law.
Source: arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem hasDerivAt_ridgeFlow_fullVF {d : ℕ} (i : Fin d) (p u : EucSpace d) (b : ℝ)
    (x₀ : SSphere d) (μ : Perspective.ProbSphere d) (t : ℝ) :
    HasDerivAt (fun s => (ridgeFlow p u b x₀ s : EucSpace d))
      (fullVF d (fun _ => reluRidgeParams i p u b) μ t (ridgeFlow p u b x₀ t)) t := by
  rw [fullVF_reluRidgeParams]
  exact hasDerivAt_ridgeFlow p u b x₀ t

/-- The same selected trajectory also solves the complete ambient ODE.
Auxiliary for arXiv:2411.04551v3, §3.2, `lem: perturbation`. -/
theorem hasDerivAt_ridgeFlow_extension {d : ℕ} (p u : EucSpace d) (b : ℝ)
    (x₀ : SSphere d) (t : ℝ) :
    HasDerivAt (fun s => (ridgeFlow p u b x₀ s : EucSpace d))
      (ridgeGlobalField p u b (ridgeFlow p u b x₀ t)) t := by
  rw [ridgeGlobalField_eq_of_unit _ _ _ _ (mem_sphere_zero_iff_norm.mp (ridgeFlow p u b x₀ t).2)]
  exact hasDerivAt_ridgeFlow p u b x₀ t

/-- The actual one-neuron trajectory is continuous in time.
Source: arXiv:2411.04551v3, §1, `eq: neural.ode.sphere`. -/
theorem continuous_ridgeFlow_time {d : ℕ} (p u : EucSpace d) (b : ℝ) (x₀ : SSphere d) :
    Continuous (ridgeFlow p u b x₀) :=
  (continuous_iff_continuousAt.mpr fun t =>
    (hasDerivAt_ridgeFlow p u b x₀ t).continuousAt).subtype_mk _

/-- The actual autonomous neuron flow satisfies the flow law.
Source: arXiv:2411.04551v3, §1, `eq: neural.ode.sphere`. -/
theorem ridgeFlow_add {d : ℕ} (p u : EucSpace d) (b : ℝ) (x₀ : SSphere d) (s t : ℝ) :
    ridgeFlow p u b x₀ (s + t) = ridgeFlow p u b (ridgeFlow p u b x₀ s) t := by
  have h := ridgeTrajectory_unique p u b (fun t => ridgeFlow p u b x₀ (s + t))
    (ridgeFlow p u b (ridgeFlow p u b x₀ s))
    (fun r => by
      simpa only [Function.comp_def, one_smul] using (hasDerivAt_ridgeFlow p u b x₀ (s + r)).scomp r
        ((hasDerivAt_id r).const_add s))
    (hasDerivAt_ridgeFlow p u b (ridgeFlow p u b x₀ s))
    (by simp only [add_zero, ridgeFlow_zero])
  exact congrFun h t

/-- Time reversal gives a left inverse for every actual ridge flow map.
Source: arXiv:2411.04551v3, §1, `eq: neural.ode.sphere`. -/
theorem ridgeFlow_inverse {d : ℕ} (p u : EucSpace d) (b : ℝ) (x₀ : SSphere d) (t : ℝ) :
    ridgeFlow p u b (ridgeFlow p u b x₀ t) (-t) = x₀ := by
  rw [← ridgeFlow_add, add_neg_cancel, ridgeFlow_zero]

/-- The actual ridge flow map is bijective, with inverse the negative-time
map. Source: arXiv:2411.04551v3, §1, `eq: neural.ode.sphere`. -/
theorem bijective_ridgeFlow {d : ℕ} (p u : EucSpace d) (b t : ℝ) :
    Function.Bijective (fun x : SSphere d => ridgeFlow p u b x t) := by
  have hl := ridgeFlow_inverse p u b
  have hr (x : SSphere d) : ridgeFlow p u b (ridgeFlow p u b x (-t)) t = x := by
    simpa only [neg_neg] using hl x (-t)
  have hleft : Function.LeftInverse (fun x : SSphere d => ridgeFlow p u b x (-t))
      (fun x => ridgeFlow p u b x t) := fun x => hl x t
  exact ⟨hleft.injective, fun x => ⟨ridgeFlow p u b x (-t), hr x⟩⟩

/-- A nonzero stationary unit trajectory jointly satisfies both original
ODE hypotheses and the equal-initial-value premise of uniqueness. -/
example :
    (∀ t : ℝ, HasDerivAt (fun _ : ℝ => (basePoint 0 : EucSpace 1))
      (max (1 - inner ℝ (basePoint 0 : EucSpace 1) 0) 0 •
        proj 1 (basePoint 0) (basePoint 0)) t) ∧
    (fun _ : ℝ => basePoint 0) 0 = (fun _ : ℝ => basePoint 0) 0 := by
  refine ⟨fun t => ?_, rfl⟩
  have hunit : ‖(basePoint 0 : EucSpace 1)‖ = 1 :=
    mem_sphere_zero_iff_norm.mp (basePoint 0).2
  have hf : max (1 - inner ℝ (basePoint 0 : EucSpace 1) 0) 0 •
      proj 1 (basePoint 0) (basePoint 0) = 0 := by
    simp [proj, hunit]
  rw [hf]
  exact hasDerivAt_const t _

end Transformer.Interpolation
