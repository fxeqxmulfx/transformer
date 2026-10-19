module

public import Transformer.Interpolation.Section3_ProjectionLaw
public import Transformer.Interpolation.Section3_ScaledProjectionTransport

/-!
# Actual rank-one controls for a constant tangent projection

ArXiv:2411.04551v3, §3.1, proof of `lem: first.quadrant`.
The source chooses `W 1 = v`, with `V=B=U=0` and `b=1`.
The rank-one matrix below reads one coordinate and multiplies by `v`.
It has operator norm at most `‖v‖` and gives the actual original field
`Proj_x v` for every original probability law. The constructed weak
solution and the transport formula therefore apply to these concrete
controls with no remaining vector-field assumption.
Any available coordinate suffices; the dimension hypothesis supplies
that coordinate when the final two-phase construction is assembled.
-/

@[expose] public section

open Real MeasureTheory

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The all-ones bias of the source's two constant-field phases.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
noncomputable def onesBias (d : ℕ) : EucSpace d :=
  (EuclideanSpace.equiv (Fin d) ℝ).symm (fun _ => 1)

/-- Its actual coordinates are all positive ones.
Source: arXiv:2411.04551v3, §3.1, `b=1` in `lem: first.quadrant`. -/
@[simp] theorem onesBias_apply (i : Fin d) : onesBias d i = 1 := rfl

/-- The source's constant tangent-field parameters, with an explicit
rank-one choice of `W`. Source: arXiv:2411.04551v3, §3.1,
`lem: first.quadrant`, the choices `W₁1=-ω`, `W₂1=α`. -/
noncomputable def projectionControl (i : Fin d) (v : EucSpace d) : Params d where
  V := 0
  B := 0
  W := (EuclideanSpace.proj i).smulRight v
  U := 0
  b := onesBias d

/-- The required zero components and bias are exact.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem projectionControl_components (i : Fin d) (v : EucSpace d) :
    (projectionControl i v).V = 0 ∧ (projectionControl i v).B = 0 ∧
      (projectionControl i v).U = 0 ∧ (projectionControl i v).b = onesBias d :=
  ⟨rfl, rfl, rfl, rfl⟩

/-- The chosen matrix has exactly the action used in the source proof.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem projectionControl_W_ones (i : Fin d) (v : EucSpace d) :
    (projectionControl i v).W (onesBias d) = v := by
  change onesBias d i • v = v
  rw [onesBias_apply, one_smul]

/-- Its actual operator norm is bounded before choosing a time horizon.
Source: arXiv:2411.04551v3, §3.1, the control norm in
`lem: first.quadrant`. -/
theorem norm_projectionControl_W_le (i : Fin d) (v : EucSpace d) :
    ‖(projectionControl i v).W‖ ≤ ‖v‖ := by
  apply ContinuousLinearMap.opNorm_le_bound _ (norm_nonneg v)
  intro x
  change ‖x i • v‖ ≤ ‖v‖ * ‖x‖
  rw [norm_smul]
  calc
    ‖x i‖ * ‖v‖ ≤ ‖x‖ * ‖v‖ :=
      mul_le_mul_of_nonneg_right (PiLp.norm_apply_le x i) (norm_nonneg v)
    _ = _ := mul_comm _ _

/-- The actual original field is the constant tangent projection,
independently of the original law.
Source: arXiv:2411.04551v3, §1, `eq: vf`, and §3.1,
`lem: first.quadrant`. -/
theorem fullVF_projectionControl (i : Fin d) (v : EucSpace d)
    (ν : Perspective.ProbSphere d) (t : ℝ) (x : EucSpace d) :
    fullVF d (fun _ => projectionControl i v) ν t x = proj d x v := by
  have hrelu : (EuclideanSpace.equiv (Fin d) ℝ).symm
      (fun j => max ((EuclideanSpace.equiv (Fin d) ℝ (onesBias d)) j) 0) = onesBias d := by
    apply (EuclideanSpace.equiv (Fin d) ℝ).injective
    ext j
    simp [onesBias]
  simp only [fullVF, projectionControl, zero_apply, zero_add]
  rw [hrelu]
  change proj d x ((projectionControl i v).W (onesBias d)) = _
  rw [projectionControl_W_ones]

/-- The same field identity applies inside a phase of the original
piecewise constant parameter curve.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem fullVF_projectionControl_at (θ : TimeParams d) (ν : Perspective.ProbSphere d)
    (t : ℝ) (x : EucSpace d) (i : Fin d) (v : EucSpace d)
    (hθ : θ t = projectionControl i v) : fullVF d θ ν t x = proj d x v := by
  have heq : fullVF d θ ν t x = fullVF d (fun _ => projectionControl i v) ν t x := by
    simp only [fullVF, hθ]
  rw [heq, fullVF_projectionControl]

/-- The control norm at a rescaled unit direction is at most its speed.
Source: arXiv:2411.04551v3, §3.1, time rescaling in `lem: first.quadrant`. -/
theorem norm_projectionControl_axis_le (i : Fin d) (w : SSphere d) (κ : ℝ) :
    ‖(projectionControl i (κ • (w : EucSpace d))).W‖ ≤ |κ| := by
  have hw : ‖(w : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp w.2
  simpa only [norm_smul, Real.norm_eq_abs, hw, mul_one] using
    norm_projectionControl_W_le i (κ • (w : EucSpace d))

/-- A rescaled original constant-field phase has a constructed weak
solution with its actual original initial law.
Source: arXiv:2411.04551v3, §1 and §3.1, `lem: first.quadrant`. -/
theorem cauchyPB_projectionControl (i : Fin d) (w : SSphere d)
    (ν : Perspective.ProbSphere d) (κ : ℝ) :
    (fun s => projectionLaw w ν (κ * s)) 0 = ν ∧
      cauchyPB d (fun _ => projectionControl i (κ • (w : EucSpace d)))
        (fun s => projectionLaw w ν (κ * s)) := by
  refine ⟨by simp, ?_⟩
  apply cauchyPB_projectionLaw_scale
  intro s x
  rw [fullVF_projectionControl, projection_field_smul]

/-- Every original weak solution of those actual parameters has the same
transported endpoint. Source: arXiv:2411.04551v3, §1 and §3.1,
`lem: first.quadrant`. -/
theorem projectionControl_endpoint (i : Fin d) (w : SSphere d)
    (ν : Perspective.ProbSphere d) (κ T : ℝ) (hT : 0 < T)
    (μ : ℝ → Perspective.ProbSphere d) (hinit : μ 0 = ν)
    (hμ : cauchyPB d (fun _ => projectionControl i (κ • (w : EucSpace d))) μ) :
    (μ T : Measure (SSphere d)) = (projectionLaw w ν (κ * T) : Measure (SSphere d)) := by
  rw [cauchyPB_scaled_projection_map_initial _ μ hμ w κ T hT, hinit, coe_projectionLaw]
  intro s hs x
  rw [fullVF_projectionControl, projection_field_smul]

/-- Each phase of intrinsic length `τ` can occupy half of a prescribed
positive horizon with the source's `1/T` control scaling.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem norm_projectionControl_half_time (i : Fin d) (w : SSphere d)
    (τ T : ℝ) (hτ : 0 ≤ τ) (hT : 0 < T) :
    ‖(projectionControl i ((2 * τ / T) • (w : EucSpace d))).W‖ ≤ 2 * τ / T := by
  have hs : 0 ≤ 2 * τ / T := by positivity
  simpa only [abs_of_nonneg hs] using norm_projectionControl_axis_le i w (2 * τ / T)

/-- Every restored phase input has nonzero actual data on the circle,
including a positive horizon and a genuine initial law. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 ∧
    (fun s => projectionLaw (basePoint 1) (Perspective.diracProb 2 (basePoint 1))
      ((2 : ℝ) * s)) 0 = Perspective.diracProb 2 (basePoint 1) ∧
    cauchyPB 2 (fun _ => projectionControl (0 : Fin 2) ((2 : ℝ) • (basePoint 1 : EucSpace 2)))
      (fun s => projectionLaw (basePoint 1) (Perspective.diracProb 2 (basePoint 1)) (2 * s)) := by
  have h := cauchyPB_projectionControl (0 : Fin 2) (basePoint 1 : SSphere 2)
    (Perspective.diracProb 2 (basePoint 1)) (2 : ℝ)
  exact ⟨one_pos, zero_le_one, h⟩

end Transformer.Interpolation
