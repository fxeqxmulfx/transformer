module

public import Transformer.Interpolation.Section1_WeakContinuity

/-!
# Exact time rescaling of the original weak equation

ArXiv:2411.04551v3, §1, `eq: cauchy.pb` and `eq: vf`, and
the time rescaling of the controls used in §3.

Replacing time by `κt` evaluates the original parameters at `κt` and
multiplies both output matrices `V,W` by `κ`. The attention score,
the activation input matrix, and the bias retain their original values
at the new time. The full original vector field scales by `κ`.

A change of variables in the integrated weak equation proves this
rescaling for every weak solution. For a nonzero factor, inverse
rescaling gives an equivalence of the original weak equations.
Constant controls remain constant, and zero attention scores remain
zero. Thus a small-time separation can fit any positive time budget.
-/

@[expose] public section

open Real MeasureTheory

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Actual network parameters implementing the change of time `t ↦ κt`.
Both output matrices scale; the source's input and score parameters
are evaluated at the new time. Source: arXiv:2411.04551v3, §1, `eq: vf`. -/
noncomputable def rescaleTimeParams (κ : ℝ) (θ : TimeParams d) : TimeParams d :=
  fun s =>
    { V := κ • (θ (κ * s)).V
      B := (θ (κ * s)).B
      W := κ • (θ (κ * s)).W
      U := (θ (κ * s)).U
      b := (θ (κ * s)).b }

/-- The original full network field scales by the time factor, for every
probability law and ambient point. The final spherical projection also
respects this scalar multiplication. Source: arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem fullVF_rescaleTimeParams (κ : ℝ) (θ : TimeParams d)
    (ν : Perspective.ProbSphere d) (s : ℝ) (x : EucSpace d) :
    fullVF d (rescaleTimeParams κ θ) ν s x = κ • fullVF d θ ν (κ * s) x := by
  simp only [fullVF, rescaleTimeParams, smul_apply, ← smul_add]
  simp only [proj, inner_smul_right, smul_sub, smul_smul]

/-- Every original weak solution transports to the exactly rescaled
parameters. The interval integrability requirement is transported too.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`, and time rescaling in §3. -/
theorem cauchyPB_rescaleTimeParams (κ : ℝ) (hκ : κ ≠ 0) (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d θ μ) :
    cauchyPB d (rescaleTimeParams κ θ) (fun s => μ (κ * s)) := by
  intro φ hφ t
  let g (s : ℝ) := ∫ x : SSphere d, inner ℝ (gradient φ (x : EucSpace d))
    (fullVF d θ (μ s) s x) ∂(μ s : Measure (SSphere d))
  have heq : (fun s => ∫ x : SSphere d, inner ℝ (gradient φ (x : EucSpace d))
      (fullVF d (rescaleTimeParams κ θ) (μ (κ * s)) s x)
        ∂(μ (κ * s) : Measure (SSphere d))) = (fun s => κ * g (κ * s)) := by
    funext s
    simp only [fullVF_rescaleTimeParams, real_inner_smul_right, integral_const_mul, g]
  rw [heq]
  have hi : IntervalIntegrable g volume 0 (κ * t) := (hμ φ hφ (κ * t)).1
  have hic : IntervalIntegrable (fun s => g (κ * s)) volume 0 t := by
    have h := hi.comp_mul_left (c := κ)
    simpa only [zero_div, mul_div_cancel_left₀ t hκ] using h
  refine ⟨hic.const_mul κ, ?_⟩
  simp only [mul_zero]
  rw [intervalIntegral.integral_const_mul]
  have he := intervalIntegral.smul_integral_comp_mul_left (f := g) (a := 0) (b := t) κ
  simp only [smul_eq_mul, mul_zero] at he
  rw [he]
  exact (hμ φ hφ (κ * t)).2

/-- Inverse rescaling recovers every component of the original controls.
Auxiliary for the prescribed time budgets of arXiv:2411.04551v3, §3. -/
theorem rescaleTimeParams_inverse (κ : ℝ) (hκ : κ ≠ 0) (θ : TimeParams d) :
    rescaleTimeParams κ⁻¹ (rescaleTimeParams κ θ) = θ := by
  funext s
  simp [rescaleTimeParams, smul_smul, hκ]

/-- Every weak solution of the rescaled parameters gives an original
weak solution by inverse rescaling. This applies to arbitrary weak
solutions, as required by arXiv:2411.04551v3, §3.2, `lem: perturbation`. -/
theorem cauchyPB_unscaleTimeParams (κ : ℝ) (hκ : κ ≠ 0) (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) (hμ : cauchyPB d (rescaleTimeParams κ θ) μ) :
    cauchyPB d θ (fun s => μ (κ⁻¹ * s)) := by
  simpa only [rescaleTimeParams_inverse κ hκ] using
    cauchyPB_rescaleTimeParams κ⁻¹ (inv_ne_zero hκ) (rescaleTimeParams κ θ) μ hμ

/-- The unit time factor preserves the actual parameter curve.
Source: arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem rescaleTimeParams_one (θ : TimeParams d) : rescaleTimeParams 1 θ = θ := by
  funext s
  simp [rescaleTimeParams]

/-- Successive time changes multiply their factors in the actual controls.
Source: arXiv:2411.04551v3, §1, `eq: vf`. -/
theorem rescaleTimeParams_mul (a b : ℝ) (θ : TimeParams d) :
    rescaleTimeParams a (rescaleTimeParams b θ) = rescaleTimeParams (b * a) θ := by
  funext s
  simp [rescaleTimeParams, smul_smul, mul_assoc, mul_comm a b]

/-- A constant original control stays constant under time rescaling.
In particular its switch count does not increase.
Source: arXiv:2411.04551v3, §3, the prescribed time budget. -/
theorem rescaleTimeParams_constant (κ : ℝ) (θ : Params d) :
    rescaleTimeParams κ (fun _ => θ) = fun _ =>
      { V := κ • θ.V, B := θ.B, W := κ • θ.W, U := θ.U, b := θ.b } := rfl

/-- The zero-score constraint survives exact time rescaling.
Source: arXiv:2411.04551v3, §3.2, `lem: perturbation`, Part 2. -/
theorem rescaleTimeParams_B_zero (κ : ℝ) (θ : TimeParams d) (hB : ∀ s : ℝ, (θ s).B = 0) :
    ∀ s : ℝ, (rescaleTimeParams κ θ s).B = 0 := fun s => hB (κ * s)

/-- The source's pure-perceptron constraint also survives time rescaling.
Source: arXiv:2411.04551v3, §3.2, the perceptron controls. -/
theorem rescaleTimeParams_V_zero (κ : ℝ) (θ : TimeParams d) (hV : ∀ s : ℝ, (θ s).V = 0) :
    ∀ s : ℝ, (rescaleTimeParams κ θ s).V = 0 := by
  intro s
  change κ • (θ (κ * s)).V = 0
  rw [hV]
  ext x
  simp only [smul_apply, zero_apply, smul_zero]

/-- Nonzero time rescaling is an equivalence of the original integrated
weak equations, including their integrability clauses.
Source: arXiv:2411.04551v3, §1, `eq: cauchy.pb`, and §3 time rescaling. -/
theorem cauchyPB_rescaleTimeParams_iff (κ : ℝ) (hκ : κ ≠ 0) (θ : TimeParams d)
    (μ : ℝ → Perspective.ProbSphere d) :
    cauchyPB d (rescaleTimeParams κ θ) (fun s => μ (κ * s)) ↔ cauchyPB d θ μ := by
  constructor
  · intro h
    have he := cauchyPB_unscaleTimeParams κ hκ θ (fun s => μ (κ * s)) h
    simpa only [mul_inv_cancel_left₀ hκ] using he
  · exact cauchyPB_rescaleTimeParams κ hκ θ μ

/-- A nonzero time factor, zero scores, and actual original and rescaled
weak solutions are jointly realized by a nonzero stationary spherical atom. -/
example : (2 : ℝ) ≠ 0 ∧
    (∀ s : ℝ, (pureAttentionParams 1 0 s).B = 0) ∧
    cauchyPB 1 (pureAttentionParams 1 0) (fun _ => Perspective.diracProb 1 (basePoint 0)) ∧
    cauchyPB 1 (rescaleTimeParams 2 (pureAttentionParams 1 0))
      (fun _ => Perspective.diracProb 1 (basePoint 0)) := by
  have h := cauchyPB_pureAttention_dirac 0 (basePoint 0)
  refine ⟨by norm_num, fun s => ?_, h, cauchyPB_rescaleTimeParams 2 (by norm_num) _ _ h⟩
  change (0 : ℝ) • ContinuousLinearMap.id ℝ (EucSpace 1) = 0
  ext x
  simp only [smul_apply, zero_smul, zero_apply]

example : ∀ s : ℝ,
    ((fun _ : ℝ => ({ V := 0, B := 0, W := 0, U := 0, b := 0 } : Params 1)) s).V = 0 :=
  fun _ => rfl

end Transformer.Interpolation
