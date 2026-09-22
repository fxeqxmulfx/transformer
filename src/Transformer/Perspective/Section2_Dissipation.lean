/-
# §3.2–§3.3 — Energy dissipation along `SA` and `USA`

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*, §3.2 `eq: dissipation.softmax`
and §3.3 `lem: dissipation`.

Both are `hasDerivAt_interactionEnergy` at a particular velocity.  Writing
`w(x) = ∫ e^{β⟨x,y⟩} y dμ(y)` and `Z = Z_{β,μ}(x)`, the `SA` field is
`Proj_x(Z⁻¹ w)` and `⟨w, Proj_x(Z⁻¹ w)⟩ = Z⁻¹ ‖Proj_x w‖² = Z ‖𝒳[μ](x)‖²`; the
`USA` field is `Proj_x w` and `⟨w, Proj_x w⟩ = ‖𝒳^{USA}[μ](x)‖²`.  What remains
is that both fields are continuous and bounded on the sphere.
-/

import Transformer.Perspective.Section2_EnergyDerivative
import Transformer.Perspective.Section2_GradientFlow
import Transformer.Perspective.AppendixA_Beta0

open Real MeasureTheory

namespace Transformer
namespace Perspective

variable {d : ℕ}

/-- `x ↦ ∫ e^{β⟨x,y⟩} y dν(y)` is continuous on the sphere. -/
theorem continuous_integral_expInner_smul (β : ℝ) (ν : ProbSphere d) :
    Continuous fun x : SSphere d => ∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d)
      (y : EucSpace d)) • (y : EucSpace d) ∂(ν : Measure (SSphere d)) := by
  have := continuous_parametric_integral_of_continuous (μ := (ν : Measure (SSphere d)))
    (f := fun x y : SSphere d =>
      exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) • (y : EucSpace d))
    ((continuous_const.mul continuous_inner_sphere).rexp.smul
      (continuous_subtype_val.comp continuous_snd)) isCompact_univ
  simpa only [Measure.restrict_univ] using this

/-- `x ↦ Proj_x f(x)` is continuous on the sphere when `f` is. -/
theorem continuous_proj_sphere {f : SSphere d → EucSpace d} (hf : Continuous f) :
    Continuous fun x : SSphere d => proj d x (f x) := by
  unfold proj; fun_prop

/-- `‖∫ e^{β⟨x,y⟩} y dν(y)‖ ≤ Z_{β,ν}(x)`, since `‖y‖ = 1`. -/
theorem norm_integral_expInner_smul_le (β : ℝ) (ν : ProbSphere d) (x : EucSpace d) :
    ‖∫ y, exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) • (y : EucSpace d)
      ∂(ν : Measure (SSphere d))‖ ≤ partitionMu d β ν x := by
  refine (norm_integral_le_integral_norm _).trans (le_of_eq ?_)
  rw [partitionMu]
  refine integral_congr_ae (ae_of_all _ fun y => ?_)
  have hy : ‖(y : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp y.2
  dsimp only
  rw [norm_smul, hy, mul_one, Real.norm_eq_abs, abs_of_pos (exp_pos _)]

/-- `𝒳[ν]` is continuous on the sphere. -/
theorem continuous_vectorField (β : ℝ) (hβ : 0 ≤ β) (ν : ProbSphere d) :
    Continuous fun x : SSphere d => vectorField d β ν x := by
  have hinv : Continuous fun x : SSphere d => (partitionMu d β ν x)⁻¹ :=
    (continuous_partitionMu d β hβ ν).inv₀ fun x => (partitionMu_pos d β ν x).ne'
  have h : Continuous fun x : SSphere d => (partitionMu d β ν x)⁻¹ •
      ∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d)) • (y : EucSpace d)
        ∂(ν : Measure (SSphere d)) :=
    hinv.smul (continuous_integral_expInner_smul β ν)
  exact continuous_proj_sphere h

/-- `‖𝒳[ν](x)‖ ≤ 1` on the sphere. -/
theorem norm_vectorField_le (β : ℝ) (ν : ProbSphere d) (x : SSphere d) :
    ‖vectorField d β ν x‖ ≤ 1 := by
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  have hZ := partitionMu_pos d β ν x
  refine (norm_proj_le hx _).trans ?_
  rw [norm_smul, Real.norm_eq_abs, abs_inv, abs_of_pos hZ, inv_mul_le_iff₀ hZ, mul_one]
  exact norm_integral_expInner_smul_le β ν x

/-- `⟨w, 𝒳[ν](x)⟩ = ‖𝒳[ν](x)‖² Z_{β,ν}(x)` for `w = ∫ e^{β⟨x,y⟩} y dν(y)`. -/
theorem inner_integral_vectorField (β : ℝ) (ν : ProbSphere d) (x : SSphere d) :
    inner (𝕜 := ℝ) (∫ y, exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d))
        • (y : EucSpace d) ∂(ν : Measure (SSphere d))) (vectorField d β ν x) =
      ‖vectorField d β ν x‖ ^ 2 * partitionMu d β ν x := by
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  have hZ := (partitionMu_pos d β ν x).ne'
  rw [vectorField, proj_smul, inner_smul_right, real_inner_comm, inner_proj_self d hx,
    norm_smul, mul_pow, Real.norm_eq_abs, sq_abs]
  field_simp

/-- **Equation (eq: dissipation.softmax).** Energy-dissipation identity:

  `d/dt 𝖤_β[μ(t)] = ∫ ‖𝒳[μ(t)](x)‖² Z_{β,μ(t)}(x) dμ(t,x)`.

In particular the interaction energy is non-decreasing along `SA`.

*`β > 0` is a hypothesis.*  It is the survey's standing assumption (§1,
`β > 0` a fixed number intrinsic to the model), and without it the statement
is about Lean's junk value: `𝖤_0 = (2·0)⁻¹ ∫∫ 1 = 0` is constant, while the
right-hand side is `∫ ‖Proj_x(∫ y dμ)‖² dμ`, which a moving curve keeps
nonzero.

Source: arXiv:2312.10794v5, §3.2, `eq: dissipation.softmax`. -/
theorem dissipation_softmax (β : ℝ) (hβ : 0 < β) (μ : ℝ → ProbSphere d)
    (hCE : continuityEquation d β μ) :
    ∀ t : ℝ,
      HasDerivAt (fun s => interactionEnergy d β (μ s))
        (∫ x, ‖vectorField d β (μ t) (x : EucSpace d)‖ ^ 2
              * partitionMu d β (μ t) (x : EucSpace d)
          ∂(μ t : Measure (SSphere d))) t := by
  intro t
  convert hasDerivAt_interactionEnergy β hβ.ne' μ _ hCE
    (fun s => continuous_vectorField β hβ.le (μ s)) 1
    (fun s x => norm_vectorField_le β (μ s) x) t using 1
  exact integral_congr_ae (ae_of_all _ fun x => (inner_integral_vectorField β (μ t) x).symm)

/-- The hypotheses of `dissipation_softmax` are satisfiable: `β = 1`, and the
constant curve at a Dirac mass solves the continuity equation. -/
example : (0 : ℝ) < 1 ∧ continuityEquation 1 1 (fun _ => diracProb 1 (basePoint 0)) :=
  ⟨one_pos, continuityEquation_const_diracProb 1 1 (basePoint 0)⟩

/-- `𝒳^{USA}[ν]` is continuous on the sphere. -/
theorem continuous_usaVectorField (β : ℝ) (ν : ProbSphere d) :
    Continuous fun x : SSphere d => usaVectorField d β ν x := by
  exact continuous_proj_sphere (continuous_integral_expInner_smul β ν)

/-- `‖𝒳^{USA}[ν](x)‖ ≤ e^β` on the sphere, for `β ≥ 0`. -/
theorem norm_usaVectorField_le (β : ℝ) (hβ : 0 ≤ β) (ν : ProbSphere d) (x : SSphere d) :
    ‖usaVectorField d β ν x‖ ≤ exp β := by
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  exact ((norm_proj_le hx _).trans (norm_integral_expInner_smul_le β ν x)).trans
    (partitionMu_le d β hβ ν x)

/-- **Lemma (lem: dissipation).**  Along `USA` the interaction energy
dissipates at rate

  `d/dt 𝖤_β[μ(t)] = ∫ ‖𝒳^{USA}[μ(t)](x)‖² dμ(t,x)`

— the same identity as `eq: dissipation.softmax` with the partition-function
weight `Z_{β,μ}` removed, which is precisely what the normalisation costs.

*`β > 0` is a hypothesis*, the survey's standing assumption: at `β = 0` the
energy is Lean's junk value `(2·0)⁻¹ ∫∫ 1 = 0`, as for `dissipation_softmax`.

Source: arXiv:2312.10794v5, §3.3, `lem: dissipation`. -/
theorem usa_dissipation (β : ℝ) (hβ : 0 < β) (μ : ℝ → ProbSphere d)
    (hCE : usaContinuityEquation d β μ) :
    ∀ t : ℝ,
      HasDerivAt (fun s => interactionEnergy d β (μ s))
        (∫ x, ‖usaVectorField d β (μ t) (x : EucSpace d)‖ ^ 2
          ∂(μ t : Measure (SSphere d))) t := by
  intro t
  convert hasDerivAt_interactionEnergy β hβ.ne' μ _ hCE
    (fun s => continuous_usaVectorField β (μ s)) (exp β)
    (fun s x => norm_usaVectorField_le β hβ.le (μ s) x) t using 1
  refine integral_congr_ae (ae_of_all _ fun x => ?_)
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  dsimp only
  rw [usaVectorField, real_inner_comm, inner_proj_self d hx]

/-- The hypotheses of `usa_dissipation` are satisfiable: `β = 1` and the
constant curve at a Dirac mass. -/
example : (0 : ℝ) < 1 ∧ usaContinuityEquation 1 1 (fun _ => diracProb 1 (basePoint 0)) :=
  ⟨one_pos, usaContinuityEquation_const_diracProb 1 1 (basePoint 0)⟩

end Perspective
end Transformer
