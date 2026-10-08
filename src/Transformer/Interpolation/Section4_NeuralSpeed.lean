/-
# Neural dynamics and the quadratic dependence on parameter norms

arXiv:2411.04551v3, §4, `eq: neural.ode.sphere`. The field is
`Proj_x W (Ux + b)_+`. On the unit sphere, a bound `K` on all three parameter
norms gives speed at most `2K²`, hence displacement at most `2K²T` over a
time interval of length `T`. The estimates hold for an essential bound on
the parameters, as required by the source's `L∞` norm.

The dynamics definitions below are unchanged from `NeuralODE`; the speed
estimate is used by `Section4_NeuralBoundFalse` to refute both of the §4
uniform parameter estimates proportional to `1/T`.
-/

import Transformer.Interpolation.Basic

open scoped BigOperators Topology
open MeasureTheory Real

namespace Transformer.Interpolation

variable (d M : ℕ)

/-- The right-hand side `Proj_x W(t) (U(t)x + b(t))_+`.
Source: arXiv:2411.04551v3, §4, `eq: neural.ode.sphere`. -/
noncomputable def neuralVF
    (W U : ℝ → ParamMatrix d) (b : ℝ → EucSpace d) (t : ℝ) (x : EucSpace d) :
    EucSpace d :=
  proj d x ((W t) (EuclideanSpace.equiv _ ℝ |>.symm
    (fun k => max ((EuclideanSpace.equiv _ ℝ ((U t) x + b t)) k) 0)))

/-- **Equation (eq: neural.ode.sphere).** The actual integrated particle
equation, with an integrable velocity. Its initial condition is imposed
separately. The integrated form accommodates switches of the parameters,
at which a trajectory need not have a two-sided derivative.
Source: arXiv:2411.04551v3, §4. -/
def neuralODESphere
    (W U : ℝ → ParamMatrix d) (b : ℝ → EucSpace d)
    (x : ℝ → Idx M → EucSpace d) : Prop :=
  ∀ t : ℝ, ∀ i : Idx M,
    IntervalIntegrable (fun s => neuralVF d W U b s (x s i)) volume 0 t ∧
      x t i = x 0 i + ∫ s in (0 : ℝ)..t, neuralVF d W U b s (x s i)

/-- Parameters `(V,B,W,U,b) = (0,0,W,U,b)`, through which
`PiecewiseConstant` applies to the neural controls.
Source: arXiv:2411.04551v3, §4, `eq: neural.ode.sphere`. -/
def neuralParams (W U : ℝ → ParamMatrix d) (b : ℝ → EucSpace d) : TimeParams d :=
  fun t => { V := 0, B := 0, W := W t, U := U t, b := b t }

/-- Coordinatewise ReLU decreases the Euclidean norm: its squared
coordinates are bounded by the original squared coordinates.
Source: arXiv:2411.04551v3, §4, the activation in `eq: neural.ode.sphere`. -/
theorem norm_coordinate_relu_le {d : ℕ} (v : EucSpace d) :
    ‖(EuclideanSpace.equiv (Fin d) ℝ).symm (fun i => max (v i) 0)‖ ≤ ‖v‖ := by
  have hs : ‖(EuclideanSpace.equiv (Fin d) ℝ).symm (fun i => max (v i) 0)‖ ^ 2 ≤
      ‖v‖ ^ 2 := by
    rw [EuclideanSpace.real_norm_sq_eq, EuclideanSpace.real_norm_sq_eq]
    apply Finset.sum_le_sum
    intro i _
    change (max (v i) 0) ^ 2 ≤ (v i) ^ 2
    by_cases h : 0 ≤ v i
    · rw [max_eq_left h]
    · rw [max_eq_right (le_of_not_ge h)]
      simpa using sq_nonneg (v i)
  nlinarith [norm_nonneg v,
    norm_nonneg ((EuclideanSpace.equiv (Fin d) ℝ).symm (fun i => max (v i) 0))]

/-- Unit vectors have neural speed at most `2K²` when the maximum of the
three parameter norms is at most `K`. Projection and ReLU are contractions;
the operator bounds give `‖W‖ (‖U‖ + ‖b‖)`.
Source: arXiv:2411.04551v3, §4, `eq: neural.ode.sphere` and the norm bounds
in `prop: interpolation.neural.ode` and `lem: induction.neural.ode`. -/
theorem neuralVF_norm_le {d : ℕ} (W U : ℝ → ParamMatrix d) (b : ℝ → EucSpace d)
    (t K : ℝ) (x : EucSpace d) (hx : ‖x‖ = 1)
    (hmax : max (max ‖W t‖ ‖U t‖) ‖b t‖ ≤ K) :
    ‖neuralVF d W U b t x‖ ≤ 2 * K ^ 2 := by
  have hW : ‖W t‖ ≤ K := (le_max_left _ _).trans ((le_max_left _ _).trans hmax)
  have hU : ‖U t‖ ≤ K := (le_max_right _ _).trans ((le_max_left _ _).trans hmax)
  have hb : ‖b t‖ ≤ K := (le_max_right _ _).trans hmax
  have hK : 0 ≤ K := (norm_nonneg _).trans hW
  have hUx : ‖(U t) x‖ ≤ K := by
    exact ((U t).le_opNorm x).trans (by simpa [hx] using hU)
  have hr : ‖(EuclideanSpace.equiv (Fin d) ℝ).symm
      (fun i => max ((EuclideanSpace.equiv (Fin d) ℝ ((U t) x + b t)) i) 0)‖ ≤ K + K :=
    (norm_coordinate_relu_le ((U t) x + b t)).trans
      ((norm_add_le _ _).trans (add_le_add hUx hb))
  calc ‖neuralVF d W U b t x‖ ≤ ‖(W t) ((EuclideanSpace.equiv (Fin d) ℝ).symm
        (fun i => max ((EuclideanSpace.equiv (Fin d) ℝ ((U t) x + b t)) i) 0))‖ :=
      norm_proj_le hx _
    _ ≤ ‖W t‖ * ‖(EuclideanSpace.equiv (Fin d) ℝ).symm
        (fun i => max ((EuclideanSpace.equiv (Fin d) ℝ ((U t) x + b t)) i) 0)‖ :=
      (W t).le_opNorm _
    _ ≤ K * (K + K) := mul_le_mul hW hr (norm_nonneg _) hK
    _ = 2 * K ^ 2 := by ring

/-- Integrating the actual neural field along unit vectors gives
displacement at most `2K²T`. Only the terminal integral equation is needed,
so the estimate applies to every sphere-valued solution in the source.
The parameter bound is almost everywhere on the interval; values at
switching times therefore do not affect the estimate.
Source: arXiv:2411.04551v3, §4, `eq: neural.ode.sphere`. -/
theorem neural_displacement_le {d : ℕ} (W U : ℝ → ParamMatrix d)
    (b : ℝ → EucSpace d) (x : ℝ → EucSpace d) (T K : ℝ) (hT : 0 ≤ T)
    (hmax : ∀ᵐ s : ℝ, s ∈ Set.Icc 0 T → max (max ‖W s‖ ‖U s‖) ‖b s‖ ≤ K)
    (hunit : ∀ s ∈ Set.Icc 0 T, ‖x s‖ = 1)
    (heq : x T = x 0 + ∫ s in (0 : ℝ)..T, neuralVF d W U b s (x s)) :
    ‖x T - x 0‖ ≤ 2 * K ^ 2 * T := by
  have hbound := intervalIntegral.norm_integral_le_of_norm_le_const_ae
    (f := fun s => neuralVF d W U b s (x s)) (C := 2 * K ^ 2) (a := 0) (b := T) (by
      filter_upwards [hmax] with s hs
      intro hwin
      rw [Set.uIoc_of_le hT] at hwin
      have ht : s ∈ Set.Icc 0 T := ⟨hwin.1.le, hwin.2⟩
      exact neuralVF_norm_le W U b s K (x s) (hunit s ht) (hs ht))
  rw [heq, add_sub_cancel_left]
  simpa [abs_of_nonneg hT] using hbound

/-- All speed and displacement hypotheses have a witness: zero controls
and a constant unit curve on the nonzero time interval `[0,1]`. -/
example : ∃ (W U : ℝ → ParamMatrix 1) (b x : ℝ → EucSpace 1) (T K : ℝ),
    0 ≤ T ∧
    (∀ s ∈ Set.Icc 0 T, max (max ‖W s‖ ‖U s‖) ‖b s‖ ≤ K) ∧
    (∀ s ∈ Set.Icc 0 T, ‖x s‖ = 1) ∧
    x T = x 0 + ∫ s in (0 : ℝ)..T, neuralVF 1 W U b s (x s) := by
  refine ⟨fun _ => 0, fun _ => 0, fun _ => 0,
    fun _ => (basePoint 0 : EucSpace 1), 1, 0, by norm_num, fun _ _ => by simp,
    fun _ _ => mem_sphere_zero_iff_norm.mp (basePoint 0).2, ?_⟩
  simp [neuralVF, proj]

/-- The same estimate for the actual integrated particle system, with its
sphere-valued trajectories made explicit as in the source.
It holds for each chosen particle index.
Source: arXiv:2411.04551v3, §4, `eq: neural.ode.sphere`. -/
theorem neuralODESphere_displacement_le (W U : ℝ → ParamMatrix d)
    (b : ℝ → EucSpace d) (x : ℝ → Idx M → SSphere d) (i : Idx M)
    (T K : ℝ) (hT : 0 ≤ T)
    (hmax : ∀ᵐ s : ℝ, s ∈ Set.Icc 0 T → max (max ‖W s‖ ‖U s‖) ‖b s‖ ≤ K)
    (hx : neuralODESphere d M W U b (fun t j => (x t j : EucSpace d))) :
    ‖(x T i : EucSpace d) - (x 0 i : EucSpace d)‖ ≤ 2 * K ^ 2 * T := by
  apply neural_displacement_le W U b (fun s => (x s i : EucSpace d)) T K hT hmax
    (fun s _ => mem_sphere_zero_iff_norm.mp (x s i).2) (hx T i).2

/-- The zero controls and constant unit curves used above solve the actual
integrated system as well, witnessing the additional system hypothesis. -/
example : neuralODESphere 1 1 (fun _ => 0) (fun _ => 0) (fun _ => 0)
    (fun _ _ => (basePoint 0 : EucSpace 1)) := by
  intro t i
  simp [neuralVF, proj, intervalIntegrable_const]

end Transformer.Interpolation
