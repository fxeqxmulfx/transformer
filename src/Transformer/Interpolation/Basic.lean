/-
# Measure-to-measure interpolation — Basic definitions

Formalization of arXiv:2411.04551v3 — "Measure-to-measure interpolation
using Transformers" (Geshkovski, Karagodin, Polyanskiy, Rigollet).

This file collects:

* `eq: ips`                 — interacting-particle system on `𝕊^{d-1}`,
* `eq: vf`                  — full vector field with the attention `𝒜_𝐁`
                              and perceptron components,
* `eq: cauchy.pb`           — Cauchy problem for the continuity equation,
* `eq: average.vf`          — simplified vector field used in
                              `prop: separation`,
* parameter type `Θ = (𝐕, 𝐁, 𝐖, 𝐔, b)`,
* the flow map `Φ^t_θ : 𝒫(𝕊^{d-1}) → 𝒫(𝕊^{d-1})`,
* the hyperplane / spherical-cap shorthands `H_ε^γ` and `𝒮_q(ε)`,
* the geodesic convex hull `conv_g`,
* `eq_of_mem_support_dirac`, `antipode` and `basePoint_mem_positiveQuadrant`,
  the ingredients every satisfiability witness of §1–§5 is built from.
-/

import Transformer.Basic
import Transformer.Perspective.Section2_FlowMap
import Mathlib.MeasureTheory.Measure.MeasureSpaceDef
import Mathlib.MeasureTheory.Measure.Support
import Mathlib.Analysis.Calculus.Gradient.Basic

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Interpolation

variable (d : ℕ)

/-- The parameter space `Θ = M_{d×d}(ℝ)^4 × ℝ^d` of all admissible parameter
tuples `(𝐕, 𝐁, 𝐖, 𝐔, b)`. -/
structure Params (d : ℕ) where
  V : ParamMatrix d
  B : ParamMatrix d
  W : ParamMatrix d
  U : ParamMatrix d
  b : EucSpace d

/-- A time-dependent parameter is a curve in `Params d`. -/
abbrev TimeParams (d : ℕ) : Type := ℝ → Params d

/-- **Equation (eq: vf).** The full vector field driving the Transformer
particle system on `𝕊^{d-1}`:

  `𝐯[μ](t, x) = Proj_x (𝐕(t) · 𝒜_𝐁[μ](t, x) + 𝐖(t) · (𝐔(t) x + b(t))_+)`. -/
noncomputable def fullVF
    (θ : TimeParams d) (μ : Perspective.ProbSphere d)
    (t : ℝ) (x : EucSpace d) : EucSpace d :=
  let Bt := (θ t).B
  let Vt := (θ t).V
  let Wt := (θ t).W
  let Ut := (θ t).U
  let bt := (θ t).b
  let num : EucSpace d :=
    ∫ x', Real.exp (inner (𝕜 := ℝ) (Bt x) ((x' : EucSpace d)))
            • (x' : EucSpace d) ∂(μ : Measure (SSphere d))
  let den : ℝ :=
    ∫ ζ, Real.exp (inner (𝕜 := ℝ) (Bt x) ((ζ : EucSpace d)))
      ∂(μ : Measure (SSphere d))
  let attn : EucSpace d := den⁻¹ • num
  proj d x (Vt attn +
    Wt (EuclideanSpace.equiv _ ℝ |>.symm
        (fun i =>
          max ((EuclideanSpace.equiv _ ℝ (Ut x + bt)) i) 0)))

/-- **Equation (eq: average.vf).**  Simplified vector field with
`𝐁 ≡ 0` (so attention is just the barycenter `𝔼_μ[z]`):

  `𝐯[μ](t, x) = Proj_x (𝐕(t) · 𝔼_μ[z] + 𝐖(t) · (𝐔(t) x + b(t))_+)`. -/
noncomputable def averageVF
    (θ : TimeParams d) (μ : Perspective.ProbSphere d)
    (t : ℝ) (x : EucSpace d) : EucSpace d :=
  let Vt := (θ t).V
  let Wt := (θ t).W
  let Ut := (θ t).U
  let bt := (θ t).b
  let barycenter : EucSpace d :=
    ∫ x', (x' : EucSpace d) ∂(μ : Measure (SSphere d))
  proj d x (Vt barycenter +
    Wt (EuclideanSpace.equiv _ ℝ |>.symm
        (fun i =>
          max ((EuclideanSpace.equiv _ ℝ (Ut x + bt)) i) 0)))

/-- **Equation (eq: cauchy.pb).** Continuity equation on the sphere,

  `∂_t μ(t) + div(μ(t) · 𝐯[μ(t)]) = 0`,

in distributional form: for every `C¹` test function `φ` on the ambient space,

  `d/dt ∫ φ dμ(t) = ∫ ⟨∇φ(x), 𝐯[μ(t)](t,x)⟩ dμ(t)(x)`.

The initial condition `μ(0) = μ_0` is imposed separately by the statements that
use this, since it is what varies between them.  As in
`Perspective.continuityEquation`, the ambient gradient is the right pairing
because `fullVF` ends in `proj`. -/
def cauchyPB
    (θ : TimeParams d) (μ : ℝ → Perspective.ProbSphere d) : Prop :=
  ∀ φ : EucSpace d → ℝ, ContDiff ℝ 1 φ → ∀ t : ℝ,
    HasDerivAt (fun s => ∫ x, φ (x : EucSpace d) ∂(μ s : Measure (SSphere d)))
      (∫ x, inner (𝕜 := ℝ) (gradient φ (x : EucSpace d))
          (fullVF d θ (μ t) t (x : EucSpace d))
        ∂(μ t : Measure (SSphere d))) t

/-- **Flow map associated with parameters `θ`** — the solution operator of
`eq: cauchy.pb`.

It is a property of a candidate `Φ`, not a construction: `Φ^t_θ μ₀` is the
solution of the Cauchy problem started at `μ₀`, and producing one is exactly
the well-posedness the paper assumes.  Stating it this way keeps that
assumption visible wherever a flow map is used. -/
def IsFlowMap
    (θ : TimeParams d) (Φ : ℝ → Perspective.ProbSphere d → Perspective.ProbSphere d) :
    Prop :=
  ∀ μ₀ : Perspective.ProbSphere d, Φ 0 μ₀ = μ₀ ∧ cauchyPB d θ (fun t => Φ t μ₀)

/-- The hyperplane `H_ε^γ = { x ∈ 𝕊^{d-1} : |⟨x, γ⟩| ≤ ε }`. -/
def Hε (γ : SSphere d) (ε : ℝ) : Set (SSphere d) :=
  { x | |inner (𝕜 := ℝ) ((x : EucSpace d)) ((γ : EucSpace d))| ≤ ε }

/-- The positive quadrant of `𝕊^{d-1}`:

  `ℚ_1^{d-1} = 𝕊^{d-1} ∩ (ℝ_{>0})^d`. -/
def positiveQuadrant : Set (SSphere d) :=
  { x | ∀ i : Fin d, 0 < (EuclideanSpace.equiv _ ℝ ((x : EucSpace d))) i }

/-! ### Witnesses

The statements of §2–§3 hypothesize supports inside `ℚ_1^{d-1}`, barycenters,
and families of measures.  The cheapest measure satisfying such conditions is a
Dirac mass, and the cheapest point of `ℚ_1^{d-1}` is `basePoint 0 = +1 ∈ 𝕊^0`;
these two facts are what the satisfiability examples of those statements use. -/

/-- **A Dirac mass sits at its own point.**  Any other point has the open
complement of `{x}` as a null neighbourhood, hence lies outside the support.
So a support hypothesis on `δ_x` is a hypothesis on `x` alone. -/
theorem eq_of_mem_support_dirac {X : Type*} [MeasurableSpace X]
    [TopologicalSpace X] [T1Space X] [MeasurableSingletonClass X] {x y : X}
    (hy : y ∈ (Measure.dirac x).support) : y = x := by
  by_contra hne
  refine Measure.notMem_support_iff_exists.mpr ⟨{x}ᶜ, ?_, ?_⟩ hy
  · exact isOpen_compl_singleton.mem_nhds (by simpa using hne)
  · simp

/-- The antipode `-x` of a point of `𝕊^{d-1}`. -/
noncomputable def antipode (x : SSphere d) : SSphere d :=
  ⟨-(x : EucSpace d), by
    rw [mem_sphere_zero_iff_norm, norm_neg]
    exact mem_sphere_zero_iff_norm.mp x.2⟩

/-- A point of the sphere and its antipode are distinct — `-x = x` would force
`x = 0`, and `‖x‖ = 1`.  So a Dirac mass has neither full support nor all of
the sphere in its support's complement. -/
theorem antipode_ne (x : SSphere d) : antipode d x ≠ x := by
  intro h
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  have hv : -(x : EucSpace d) = (x : EucSpace d) := congrArg Subtype.val h
  have h2 : (2 : ℝ) • (x : EucSpace d) = 0 := by
    rw [two_smul]
    nth_rewrite 1 [← hv]
    exact neg_add_cancel _
  rcases smul_eq_zero.mp h2 with h0 | h0
  · exact absurd h0 (by norm_num)
  · rw [h0, norm_zero] at hx
    exact absurd hx (by norm_num)

/-- `basePoint 0`, the point `+1` of `𝕊^0 ⊂ ℝ^1`, lies in the positive
quadrant, so `ℚ_1^0` is inhabited.  In dimension `d ≥ 2` a point of
`ℚ_1^{d-1}` has to have *all* its coordinates positive, so no basis vector
will do and the witnesses of §2–§3 all take `d = 1`. -/
theorem basePoint_mem_positiveQuadrant : basePoint 0 ∈ positiveQuadrant 1 := by
  intro i
  fin_cases i
  simp [basePoint]

/-- The geodesic convex hull `conv_g A` of a set `A ⊆ 𝕊^{d-1}` (§1.5,
"Notation and basic definitions"): the radial projection onto the sphere of the Euclidean convex
hull of `A`, and the whole sphere when that hull contains `0`, i.e. when `A`
lies in no open hemisphere.  For `A` inside an open hemisphere it is the
smallest set containing `A` and every minimizing geodesic between two of its
points, which is the only case in which the source uses it.

Source: arXiv:2411.04551v3, §1.5. -/
def convG (A : Set (SSphere d)) : Set (SSphere d) :=
  {x | (0 : EucSpace d) ∈ convexHull ℝ ((↑) '' A) ∨
    ∃ v ∈ convexHull ℝ ((↑) '' A : Set (EucSpace d)), ∃ c : ℝ, 0 < c ∧
      (x : EucSpace d) = c • v}

/-- A set lies in its geodesic convex hull. -/
theorem subset_convG (A : Set (SSphere d)) : A ⊆ convG d A :=
  fun x hx => Or.inr ⟨x, subset_convexHull ℝ _ ⟨x, hx, rfl⟩, 1, one_pos, (one_smul ℝ _).symm⟩

end Interpolation
end Transformer
