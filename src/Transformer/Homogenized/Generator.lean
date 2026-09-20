/-
# Homogenized Transformers — the limiting dynamics

Formalization of `def: ito.formula`, `lem:toolkit_geo_riem`, `eq: first.sde`,
`eq:SDE_ito_clean`, `eq: deterministic` and `eq: deterministic.modified` of
arXiv:2604.01978v1, *Homogenized Transformers*.

**How the SDE is written here.**  `def: ito.formula` calls an adapted
`(𝕊^{d-1})^n`-valued process `X` a solution of

  `dX = B(X) dt + ∫_Θ G(X,θ) W(dθ,dt)`

when, for every smooth `g`, `g(X(t))` equals its initial value plus the time
integral of `(𝖫 g)(X(s))` plus a stochastic integral, where

  `𝖫 g = ⟨grad g, B⟩ + ½ ∫_Θ Hess g[G(·,θ), G(·,θ)] ρ*(dθ)`.

Mathlib has no Itô integral and no cylindrical Wiener process, so the pathwise
identity cannot be written.  What is written instead is what taking
expectations in `def: ito.formula` leaves — the stochastic integral is a
martingale, so it drops out — differentiated in `t`:

  `d/dt E g(X(t)) = E (𝖫 g)(X(t))`.

This is the form in which the approximation theorems of the paper are stated
and used: `thm:weak_error_clean` compares `E φ(X(t))` with `E φ(X^η(t))`.  It
is *weaker* than `def: ito.formula`: it pins the one-dimensional marginals of
the process and nothing beyond them.  Every statement below that carries it as
a hypothesis is therefore about any process with the right marginal evolution,
which is a stronger statement than one about a designated solution, and every
statement that would have to *produce* a solution is not stated here.

The Riemannian gradient and Hessian are expanded in ambient coordinates
exactly as `lem:toolkit_geo_riem` does, which is what makes `𝖫` writable at
all: for tangent `V`,

  `⟨grad φ(X), V⟩ = Dφ̃(X)[V]`,
  `Hess φ(X)[V,V] = D²φ̃(X)[V,V] - Σ_i ⟨x_i, ∇_{x_i} φ̃(X)⟩ ‖v_i‖²`,

and `⟨x_i, ∇_{x_i} φ̃(X)⟩` is the ambient derivative of `φ̃` in the direction
that carries `x_i` in slot `i` and `0` elsewhere.
-/

import Transformer.Homogenized.Basic
import Mathlib.Analysis.Calculus.IteratedDeriv.Defs
import Mathlib.Analysis.Calculus.ContDiff.Basic

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Homogenized

/-- The Riemannian Hessian of `φ` on `(𝕊^{d-1})^n` at `X`, evaluated at a
tangent vector `V` twice, written through an ambient `C²` extension:

  `Hess φ(X)[V,V] = D²φ(X)[V,V] - Σ_i ⟨x_i, ∇_{x_i} φ(X)⟩ ‖v_i‖²`.

Source: arXiv:2604.01978v1, `lem:toolkit_geo_riem`. -/
noncomputable def sphHess {d n : ℕ} (φ : (Idx n → EucSpace d) → ℝ)
    (X V : Idx n → EucSpace d) : ℝ :=
  iteratedFDeriv ℝ 2 φ X ![V, V]
    - ∑ i : Idx n, fderiv ℝ φ X (Pi.single i (X i)) * ‖V i‖ ^ 2

/-- The generator of the Itô SDE `eq: ito.sde.final` on `(𝕊^{d-1})^n` with
tangent drift `B` and diffusion kernel `G`:

  `𝖫 φ (X) = ⟨grad φ(X), B(X)⟩ + ½ ∫_Θ Hess φ(X)[G(θ,X), G(θ,X)] ρ*(dθ)`.

Source: arXiv:2604.01978v1, `def: ito.formula`. -/
noncomputable def sphGenerator {d n : ℕ} (ρ : Measure (HeadParam d))
    (B : (Idx n → EucSpace d) → Idx n → EucSpace d)
    (G : HeadParam d → (Idx n → EucSpace d) → Idx n → EucSpace d)
    (φ : (Idx n → EucSpace d) → ℝ) (X : Idx n → EucSpace d) : ℝ :=
  fderiv ℝ φ X (B X) + (1 / 2 : ℝ) * ∫ θ, sphHess φ X (G θ X) ∂ρ

/-- `X` is a solution of the Itô SDE `dX = B(X) dt + ∫_Θ G(θ,X) W(dθ,dt)` on
`(𝕊^{d-1})^n`, at the level of its one-dimensional marginals: it takes values
in `(𝕊^{d-1})^n`, and for every `C^∞` observable `φ` the expectation
`t ↦ E φ(X(t))` is differentiable on `[0,∞)` with derivative `E (𝖫 φ)(X(t))`.

See the module docstring for what this does and does not say.

Source: arXiv:2604.01978v1, `def: ito.formula`. -/
def IsItoSolution {d n : ℕ} (ρ : Measure (HeadParam d))
    (B : (Idx n → EucSpace d) → Idx n → EucSpace d)
    (G : HeadParam d → (Idx n → EucSpace d) → Idx n → EucSpace d)
    {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω)
    (X : ℝ → Ω → (Idx n → EucSpace d)) : Prop :=
  (∀ t : ℝ, ∀ ω : Ω, ∀ i : Idx n, ‖X t ω i‖ = 1) ∧
    ∀ φ : (Idx n → EucSpace d) → ℝ, ContDiff ℝ (⊤ : ℕ∞) φ →
      ∀ t ∈ Set.Ici (0 : ℝ),
        HasDerivWithinAt (fun r => ∫ ω, φ (X r ω) ∂P)
          (∫ ω, sphGenerator ρ B G φ (X t ω) ∂P) (Set.Ici 0) t

/-- The covariant derivative `(∇_{b(X)} b)(X)` of the drift along itself, the
Levi-Civita connection of `(𝕊^{d-1})^n` being the tangential part of the
ambient derivative:

  `(∇_U V)(X) = Proj_X (DV(X)[U])`.

Source: arXiv:2604.01978v1, `lem:toolkit_geo_riem` and `eq:SDE_ito_clean`. -/
noncomputable def covDerivB {d n : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    (x : Idx n → EucSpace d) (i : Idx n) : EucSpace d :=
  proj d (x i) (fderiv ℝ (fun y => bField β ρ y i) x (bField β ρ x))

/-- The diffusion kernel of the homogenized model: `√α G^σ = (√α/σ) G`.

Source: arXiv:2604.01978v1, `eq:G_alpha_def`, `eq: first.sde`. -/
noncomputable def noiseField {d n : ℕ} (β α s : ℝ) (ρ : Measure (HeadParam d))
    (θ : HeadParam d) (x : Idx n → EucSpace d) (i : Idx n) : EucSpace d :=
  (Real.sqrt α / s) • Gfield β ρ x θ i

/-- **Equation (eq: first.sde).**  The homogenized model: the Itô SDE on
`(𝕊^{d-1})^n` with drift `b(X)` and diffusion `(√α/σ) G(X,θ)`.

The source writes it in ambient coordinates, with the normal Itô correction
`-(α/2σ²) x_i ∫_Θ ‖Proj_{x_i} ξ_θ[μ](x_i)‖² ρ*(dθ)` that keeps `‖x_i‖ = 1`.
That correction is exactly what `def: ito.formula` builds into the intrinsic
formulation through the Riemannian Hessian, so the intrinsic drift is `b`.

Source: arXiv:2604.01978v1, `eq: first.sde`. -/
def IsFirstSde {d n : ℕ} (β α s : ℝ) (ρ : Measure (HeadParam d))
    {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω)
    (X : ℝ → Ω → (Idx n → EucSpace d)) : Prop :=
  IsItoSolution ρ (bField β ρ) (noiseField β α s ρ) P X

/-- **Equation (eq:SDE_ito_clean).**  The modified homogenized model, whose
Itô generator matches the one-step generator of `eq:update_tokens` to one order
more: the drift carries the corrector `-(η/2) ∇_{b(X)} b(X)`.

Source: arXiv:2604.01978v1, `eq:SDE_ito_clean`. -/
def IsModifiedSde {d n : ℕ} (η β α s : ℝ) (ρ : Measure (HeadParam d))
    {Ω : Type*} [MeasurableSpace Ω] (P : Measure Ω)
    (X : ℝ → Ω → (Idx n → EucSpace d)) : Prop :=
  IsItoSolution ρ (fun x i => bField β ρ x i - (η / 2) • covDerivB β ρ x i)
    (noiseField β α s ρ) P X

/-- **Equation (eq: deterministic).**  The ballistic limit: the flow of the
mean field `b` on `(𝕊^{d-1})^n`.

Source: arXiv:2604.01978v1, `eq: deterministic`. -/
def IsBallisticFlow {d n : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    (X : ℝ → Idx n → EucSpace d) : Prop :=
  (∀ t : ℝ, ∀ i : Idx n, ‖X t i‖ = 1) ∧
    ∀ t : ℝ, ∀ i : Idx n, HasDerivAt (fun r => X r i) (bField β ρ (X t) i) t

/-- **Equation (eq: deterministic.modified).**  The ballistic limit with the
`O(η)` corrector of `subsec:modified_regime`.

Source: arXiv:2604.01978v1, `eq: deterministic.modified`. -/
def IsModifiedFlow {d n : ℕ} (η β : ℝ) (ρ : Measure (HeadParam d))
    (X : ℝ → Idx n → EucSpace d) : Prop :=
  (∀ t : ℝ, ∀ i : Idx n, ‖X t i‖ = 1) ∧
    ∀ t : ℝ, ∀ i : Idx n,
      HasDerivAt (fun r => X r i)
        (bField β ρ (X t) i - (η / 2) • covDerivB β ρ (X t) i) t

/-! ### The trivial head solves everything

Under `ρ* = δ_0` every field of the model vanishes, so a constant
configuration of unit vectors solves each of the four limiting dynamics.  This
is the witness of satisfiability for the statements that carry them. -/

@[simp]
theorem sphHess_zero {d n : ℕ} (φ : (Idx n → EucSpace d) → ℝ) (X : Idx n → EucSpace d) :
    sphHess φ X 0 = 0 := by
  have h : (iteratedFDeriv ℝ 2 φ X) ![(0 : Idx n → EucSpace d), 0] = 0 :=
    ContinuousMultilinearMap.map_coord_zero _ (0 : Fin 2) (by simp)
  simp [sphHess, h]

theorem isItoSolution_dirac_zero {d n : ℕ} {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) (β α s : ℝ) (x : Idx n → EucSpace d) (hx : ∀ i : Idx n, ‖x i‖ = 1) :
    IsFirstSde β α s (Measure.dirac (0 : HeadParam d)) P (fun _ _ => x) := by
  constructor
  · intro _ _ i; exact hx i
  · intro φ _ t _
    have hB : bField β (Measure.dirac (0 : HeadParam d)) x = 0 := by
      funext i; simp
    have hgen : sphGenerator (Measure.dirac (0 : HeadParam d))
        (bField β (Measure.dirac (0 : HeadParam d)))
        (noiseField β α s (Measure.dirac (0 : HeadParam d))) φ x = 0 := by
      have hG : noiseField β α s (Measure.dirac (0 : HeadParam d)) (0 : HeadParam d) x = 0 := by
        funext i; simp [noiseField]
      simp [sphGenerator, hB, hG]
    simp only [hgen, integral_zero]
    exact hasDerivWithinAt_const _ _ _

/-- The hypothesis of `isItoSolution_dirac_zero` is satisfiable: the constant
tuple at `basePoint d`. -/
example (d : ℕ) : ‖((basePoint d : SSphere (d + 1)) : EucSpace (d + 1))‖ = 1 := by
  simp [basePoint, PiLp.norm_single]

theorem isBallisticFlow_dirac_zero {d n : ℕ} (β : ℝ) (x : Idx n → EucSpace d)
    (hx : ∀ i : Idx n, ‖x i‖ = 1) :
    IsBallisticFlow β (Measure.dirac (0 : HeadParam d)) (fun _ => x) := by
  refine ⟨fun _ i => hx i, fun t i => ?_⟩
  have : bField β (Measure.dirac (0 : HeadParam d)) x i = 0 := by simp
  rw [this]
  exact hasDerivAt_const _ _

/-- The hypothesis of `isBallisticFlow_dirac_zero` is satisfiable. -/
example (d : ℕ) : ‖((basePoint d : SSphere (d + 1)) : EucSpace (d + 1))‖ = 1 := by
  simp [basePoint, PiLp.norm_single]

end Homogenized
end Transformer
