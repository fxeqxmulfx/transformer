/-
# The exact projection clock of a complete original ReLU trajectory

ArXiv:2411.04551v3, §1, `eq: neural.ode.sphere`, and §4,
the local point transfers in `prop: interpolation.neural.ode`.
A unit forcing direction makes every original one-neuron trajectory
an exact constant-projection orbit, with a scalar clock given by the
integral of its actual activation. This clock is differentiable,
nondecreasing, and nonnegative at nonnegative physical times.

The time-changed projection orbit and the actual ReLU trajectory solve
the same tangent equation with the same scalar coefficient. The
coefficient is bounded by the original gate norms. The actual tangent
projection estimate then bounds their squared gap by a constant times
itself, proving equality from their common initial unit point.
No representation is assumed in the definition of the original flow.
-/

import Transformer.Interpolation.Section3_RidgeFlow
import Transformer.Interpolation.Section3_ProjectionDynamics
import Transformer.Interpolation.SeparationField
import Transformer.Perspective.DrivenFlow

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The intrinsic projection clock accumulated by the actual original
one-neuron trajectory. Source auxiliary: arXiv:2411.04551v3, §4,
point matching under `eq: neural.ode.sphere`. -/
noncomputable def ridgeProjectionClock (a : SSphere d) (u : EucSpace d)
    (b : ℝ) (x : SSphere d) (t : ℝ) : ℝ :=
  ∫ s in (0 : ℝ)..t, max (b - inner ℝ (ridgeFlow (a : EucSpace d) u b x s : EucSpace d) u) 0

/-- The actual clock starts at zero. Source auxiliary:
arXiv:2411.04551v3, §4, the original one-neuron transfer. -/
theorem ridgeProjectionClock_zero (a : SSphere d) (u : EucSpace d) (b : ℝ) (x : SSphere d) :
    ridgeProjectionClock a u b x 0 = 0 := by
  rw [ridgeProjectionClock, intervalIntegral.integral_same]

/-- The original activation is continuous along its actual trajectory.
Source auxiliary: arXiv:2411.04551v3, §1, the ReLU gate in `eq: vf`. -/
theorem continuous_ridgeFlow_activation (a : SSphere d) (u : EucSpace d) (b : ℝ) (x : SSphere d) :
    Continuous (fun s => max (b - inner ℝ (ridgeFlow (a : EucSpace d) u b x s : EucSpace d) u) 0) :=
  (continuous_const.sub (((continuous_subtype_val.comp
    (continuous_ridgeFlow_time (a : EucSpace d) u b x)).inner continuous_const))).max continuous_const

/-- The actual activation differentiates the accumulated clock.
Source auxiliary: arXiv:2411.04551v3, §4, `eq: neural.ode.sphere`. -/
theorem hasDerivAt_ridgeProjectionClock (a : SSphere d) (u : EucSpace d)
    (b : ℝ) (x : SSphere d) (t : ℝ) :
    HasDerivAt (ridgeProjectionClock a u b x)
      (max (b - inner ℝ (ridgeFlow (a : EucSpace d) u b x t : EucSpace d) u) 0) t :=
  ((continuous_ridgeFlow_activation a u b x).integral_hasStrictDerivAt 0 t).hasDerivAt

/-- The accumulated actual clock is continuous, so its intrinsic
levels can be reached by the intermediate value theorem. Source
auxiliary: arXiv:2411.04551v3, §4, the exact local transfer clock. -/
theorem continuous_ridgeProjectionClock (a : SSphere d) (u : EucSpace d)
    (b : ℝ) (x : SSphere d) : Continuous (ridgeProjectionClock a u b x) :=
  continuous_iff_continuousAt.mpr
    (fun t => (hasDerivAt_ridgeProjectionClock a u b x t).continuousAt)

/-- Nonnegative physical times have nonnegative intrinsic clocks.
Source auxiliary: arXiv:2411.04551v3, §4, the original forward transfer. -/
theorem ridgeProjectionClock_nonneg (a : SSphere d) (u : EucSpace d)
    (b : ℝ) (x : SSphere d) (t : ℝ) (ht : 0 ≤ t) : 0 ≤ ridgeProjectionClock a u b x t := by
  exact intervalIntegral.integral_nonneg ht (fun _ _ => le_max_right _ _)

/-- The original one-neuron projection clock is nondecreasing.
Source auxiliary: arXiv:2411.04551v3, §4, the positive gate clock. -/
theorem monotone_ridgeProjectionClock (a : SSphere d) (u : EucSpace d) (b : ℝ) (x : SSphere d) :
    Monotone (ridgeProjectionClock a u b x) := by
  apply monotone_of_deriv_nonneg (fun t => (hasDerivAt_ridgeProjectionClock a u b x t).differentiableAt)
  intro t
  rw [(hasDerivAt_ridgeProjectionClock a u b x t).deriv]
  exact le_max_right _ _

/-- The exact original gate is bounded by its bias and input norms
along every unit trajectory. Source auxiliary:
arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem ridgeFlow_activation_le (a : SSphere d) (u : EucSpace d) (b : ℝ) (x : SSphere d) (t : ℝ) :
    max (b - inner ℝ (ridgeFlow (a : EucSpace d) u b x t : EucSpace d) u) 0 ≤ |b| + ‖u‖ := by
  have hi := abs_real_inner_le_norm (ridgeFlow (a : EucSpace d) u b x t : EucSpace d) u
  rw [mem_sphere_zero_iff_norm.mp (ridgeFlow (a : EucSpace d) u b x t).2, one_mul] at hi
  apply max_le _ (by positivity)
  linarith [le_abs_self b, neg_abs_le (inner ℝ (ridgeFlow (a : EucSpace d) u b x t : EucSpace d) u)]

/-- The actual original one-neuron trajectory is its exact projection
orbit at the actual accumulated activation clock. Source auxiliary:
arXiv:2411.04551v3, §4, the local matching flow; no orbit or time-change
hypothesis is added to the original ODE. -/
theorem ridgeFlow_eq_projection_clock (a : SSphere d) (u : EucSpace d)
    (b : ℝ) (x : SSphere d) (t : ℝ) (ht : 0 ≤ t) :
    ridgeFlow (a : EucSpace d) u b x t = sphereProjectionFlow a (ridgeProjectionClock a u b x t) x := by
  let Y (s : ℝ) := ridgeFlow (a : EucSpace d) u b x s
  let c := ridgeProjectionClock a u b x
  let g (s : ℝ) := max (b - inner ℝ (Y s : EucSpace d) u) 0
  let Z (s : ℝ) := sphereProjectionFlow a (c s) x
  let v (s : ℝ) := g s • proj d (Y s : EucSpace d) (a : EucSpace d) -
    g s • proj d (Z s : EucSpace d) (a : EucSpace d)
  have hY (s : ℝ) : HasDerivAt (fun r => (Y r : EucSpace d))
      (g s • proj d (Y s) (a : EucSpace d)) s := hasDerivAt_ridgeFlow _ _ _ x s
  have hZ (s : ℝ) : HasDerivAt (fun r => (Z r : EucSpace d))
      (g s • proj d (Z s) (a : EucSpace d)) s := by
    simpa only [Function.comp_def] using
      (hasDerivAt_sphereProjectionFlow a x (c s)).scomp s (hasDerivAt_ridgeProjectionClock a u b x s)
  let e (s : ℝ) := ‖(Y s : EucSpace d) - (Z s : EucSpace d)‖ ^ 2
  let e' (s : ℝ) := 2 * inner ℝ ((Y s : EucSpace d) - (Z s : EucSpace d)) (v s)
  have hd (s : ℝ) : HasDerivAt e (e' s) s := ((hY s).sub (hZ s)).norm_sq
  have hv (s : ℝ) : ‖v s‖ ≤ 2 * (|b| + ‖u‖) * ‖(Y s : EucSpace d) - (Z s : EucSpace d)‖ := by
    have hy : ‖(Y s : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp (Y s).2
    have hz : ‖(Z s : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp (Z s).2
    have hp := Perspective.norm_proj_sub_proj_le_mul hy hz (a : EucSpace d)
    rw [mem_sphere_zero_iff_norm.mp a.2, mul_one] at hp
    have hg : 0 ≤ g s := le_max_right _ _
    have hgb : g s ≤ |b| + ‖u‖ := ridgeFlow_activation_le a u b x s
    dsimp only [v]
    rw [← smul_sub, norm_smul, Real.norm_of_nonneg hg]
    calc
      _ ≤ g s * (2 * ‖(Y s : EucSpace d) - (Z s : EucSpace d)‖) :=
        mul_le_mul_of_nonneg_left hp hg
      _ ≤ _ := by nlinarith [norm_nonneg ((Y s : EucSpace d) - (Z s : EucSpace d))]
  have hbnd (s : ℝ) : e' s ≤ (4 * (|b| + ‖u‖)) * e s := by
    have hi := real_inner_le_norm ((Y s : EucSpace d) - (Z s : EucSpace d)) (v s)
    have hh := mul_le_mul_of_nonneg_left (hv s)
      (norm_nonneg ((Y s : EucSpace d) - (Z s : EucSpace d)))
    dsimp only [e', e]
    nlinarith
  have hzero : e 0 ≤ 0 := by simp [e, Y, Z, c, ridgeFlow_zero, ridgeProjectionClock_zero]
  have he := nonpos_of_deriv_le_mul e e' (4 * (|b| + ‖u‖)) t
    (fun s _ => hd s) hzero (fun s _ _ => hbnd s) t ⟨ht, le_rfl⟩
  have hn : ‖(Y t : EucSpace d) - (Z t : EucSpace d)‖ = 0 := by
    dsimp only [e] at he
    nlinarith [norm_nonneg ((Y t : EucSpace d) - (Z t : EucSpace d))]
  exact Subtype.ext (sub_eq_zero.mp (norm_eq_zero.mp hn))

/-- A positive constant gate, positive physical time and distinct unit
input and forcing directions supply every clock hypothesis with a
nonstationary original trajectory. -/
example : ∃ a x : SSphere 2, a ≠ x ∧ (0 : ℝ) ≤ 1 ∧
    ridgeProjectionClock a 0 1 x 1 = 1 := by
  let a : SSphere 2 := ⟨PiLp.single 2 (0 : Fin 2) (1 : ℝ), by simp⟩
  let x : SSphere 2 := ⟨PiLp.single 2 (1 : Fin 2) (1 : ℝ), by simp⟩
  refine ⟨a, x, ?_, zero_le_one, ?_⟩
  · intro he
    have hh := congrArg (fun z : SSphere 2 => (z : EucSpace 2) (0 : Fin 2)) he
    norm_num [a, x] at hh
  · simp [ridgeProjectionClock]

end Transformer.Interpolation
