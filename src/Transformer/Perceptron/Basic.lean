/-
# Perceptrons and attention's mean-field landscape — the coupled energy

Formalization of §2 of arXiv:2601.21366v2, *Perceptrons and localization of
attention's mean-field landscape* (Álvarez-López, Geshkovski, Ruiz-Balet):
the vocabulary its results are written in — the perceptron potential `v_ϑ`,
its drift `u_ϑ`, the coupled energy `E_{β,ϑ}`, the Wasserstein gradient of its
first variation, and the stationarity condition `eq: steady.state`.

The interaction half of the energy is `Perspective.interactionEnergy` of
arXiv:2312.10794v5 — it is the same `E_β`, and there is one of it in this
tree; `eq_of_mem_support_dirac` is the one from arXiv:2411.04551v3, which is
about measures and not about that paper.

**What the source says and what is carried here.**

* The source gives the perceptron exactly `d` neurons in dimension `d`,
  `ϑ = (a_j, ω_j)_{j ∈ ⟦1,d⟧} ∈ (ℝ^{d+1})^d`; that is kept, although nothing
  below uses the coincidence.

* The perceptron is written in its gradient form throughout: the source first
  allows output weights `𝛚_j ∈ ℝ^d` and then restricts to `𝛚_j = ω_j a_j`,
  which is exactly when the coupled dynamics is still a Wasserstein gradient
  flow, and every statement of the paper is made under that restriction.

* `φ` is the source's primitive, `φ' = 2σ`.  It is carried as the hypothesis
  `∀ s, HasDerivAt φ (2 * σ s) s` where it is needed, rather than constructed:
  a primitive exists for every continuous `σ`, and no statement depends on
  which one.

* The biases `b_j` are omitted, as the source omits them from `eq: WGF-main`
  onwards; `rem: ext` (i) says what changes when they are put back.

Source: arXiv:2601.21366v2, §2.
-/

import Transformer.Interpolation.Basic

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Perceptron

variable {d : ℕ}

/-! ### The perceptron block -/

/-- **`eq: primitive.field`.**  The perceptron potential

  `v_ϑ(x) = Σ_j ω_j φ(a_j · x)`,

whose spherical gradient is twice the perceptron drift.

Source: arXiv:2601.21366v2, `eq: primitive.field`. -/
noncomputable def potential (φ : ℝ → ℝ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (x : EucSpace d) : ℝ :=
  ∑ j : Idx d, ω j * φ (inner (𝕜 := ℝ) (a j) x)

/-- **The perceptron drift of `eq:WGF-main`**, in the gradient case
`𝛚_j = ω_j a_j`:

  `u_ϑ(x) = Proj_x Σ_j ω_j σ(a_j · x) a_j`.

Source: arXiv:2601.21366v2, §2.3. -/
noncomputable def drift (σ : ℝ → ℝ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (x : EucSpace d) : EucSpace d :=
  proj d x (∑ j : Idx d, (ω j * σ (inner (𝕜 := ℝ) (a j) x)) • a j)

/-- A perceptron with zero output weights has no drift. -/
@[simp] theorem drift_zero (σ : ℝ → ℝ) (a : Idx d → EucSpace d) (x : EucSpace d) :
    drift σ (0 : Idx d → ℝ) a x = 0 := by
  simp [drift, proj]

/-- **The ambient gradient of the perceptron potential**:

  `∇_{ℝ^d} v_ϑ(x) = 2 Σ_j ω_j σ(a_j · x) a_j`,

the computation behind the source's `(1/2)∇v_ϑ = u_ϑ`. -/
theorem hasGradientAt_potential (φ σ : ℝ → ℝ) (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (x : EucSpace d) :
    HasGradientAt (potential φ ω a)
      ((2 : ℝ) • ∑ j : Idx d, (ω j * σ (inner (𝕜 := ℝ) (a j) x)) • a j) x := by
  rw [hasGradientAt_iff_hasFDerivAt]
  have hterm : ∀ j : Idx d,
      HasFDerivAt (fun y : EucSpace d => ω j * φ (inner (𝕜 := ℝ) (a j) y))
        ((ω j * (2 * σ (inner (𝕜 := ℝ) (a j) x))) • (innerSL ℝ (a j))) x := by
    intro j
    have hlin : HasFDerivAt (fun y : EucSpace d => inner (𝕜 := ℝ) (a j) y)
        (innerSL ℝ (a j)) x := (innerSL ℝ (a j)).hasFDerivAt
    have hcomp := (hφ (inner (𝕜 := ℝ) (a j) x)).comp_hasFDerivAt x hlin
    simpa [mul_smul] using hcomp.const_mul (ω j)
  have hsum := HasFDerivAt.fun_sum (fun j (_ : j ∈ Finset.univ) => hterm j)
  refine hsum.congr_fderiv ?_
  ext y
  simp only [FunLike.coe_sum, Finset.sum_apply, smul_apply, innerSL_apply_apply, smul_eq_mul,
    InnerProductSpace.toDual_apply_apply, real_inner_smul_left, sum_inner]
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun j _ => by ring

/-- The hypotheses of `hasGradientAt_potential` are satisfiable: `σ = 0` has
the primitive `φ = 0`. -/
example : ∀ s : ℝ, HasDerivAt (fun _ : ℝ => (0 : ℝ)) (2 * (0 : ℝ → ℝ) s) s := by
  intro s
  simpa using hasDerivAt_const s (0 : ℝ)

/-- **`(1/2)∇v_ϑ = u_ϑ`**, the identity the source states just below
`eq: primitive.field`: the spherical gradient `Proj_x ∇_{ℝ^d} v_ϑ(x)` of the
potential is twice the perceptron drift.

Source: arXiv:2601.21366v2, `eq: primitive.field`. -/
theorem proj_gradient_potential (φ σ : ℝ → ℝ) (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s)
    (ω : Idx d → ℝ) (a : Idx d → EucSpace d) (x : EucSpace d) :
    proj d x (gradient (potential φ ω a) x) = (2 : ℝ) • drift σ ω a x := by
  rw [(hasGradientAt_potential φ σ hφ ω a x).gradient, drift, proj, proj]
  simp only [inner_smul_right, smul_sub, smul_smul]

/-- The hypotheses of `proj_gradient_potential` are satisfiable: `σ = 0` has
the primitive `φ = 0`. -/
example : ∀ s : ℝ, HasDerivAt (fun _ : ℝ => (0 : ℝ)) (2 * (0 : ℝ → ℝ) s) s := by
  intro s
  simpa using hasDerivAt_const s (0 : ℝ)

/-! ### The coupled energy and its critical points -/

/-- **The coupled energy**

  `E_{β,ϑ}[μ] = (1/(2β)) ∬ e^{β x·y} dμ(x) dμ(y) + (1/2) ∫ v_ϑ dμ`.

Source: arXiv:2601.21366v2, §2.3. -/
noncomputable def energy (β : ℝ) (φ : ℝ → ℝ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) : ℝ :=
  Perspective.interactionEnergy d β μ +
    (2 : ℝ)⁻¹ * ∫ x, potential φ ω a (x : EucSpace d) ∂(μ : Measure (SSphere d))

/-- **The Wasserstein gradient of the first variation**,

  `∇ δE_{β,ϑ}/δμ[μ](x) = ∫ e^{β x·y} Proj_x y dμ(y) + u_ϑ(x)`,

the field whose vanishing on `supp μ` is stationarity.

Source: arXiv:2601.21366v2, §2.3. -/
noncomputable def energyGrad (β : ℝ) (σ : ℝ → ℝ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) (x : EucSpace d) :
    EucSpace d :=
  (∫ y, Real.exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) • proj d x (y : EucSpace d)
      ∂(μ : Measure (SSphere d))) + drift σ ω a x

/-- **`eq: steady.state`.**  `μ` is a *stationary* measure for `E_{β,ϑ}`: the
Wasserstein gradient of the first variation vanishes on the whole support.

Source: arXiv:2601.21366v2, `eq: steady.state`. -/
def IsStationary (β : ℝ) (σ : ℝ → ℝ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) : Prop :=
  ∀ x ∈ (μ : Measure (SSphere d)).support,
    energyGrad β σ ω a μ (x : EucSpace d) = 0

/-- **A Dirac mass is stationary for pure attention.**  Its own point is the
only point of its support, and there the attention field is `e^β Proj_x x = 0`.
This is the inhabitant every satisfiability witness below is built from.

Source: arXiv:2601.21366v2, `eq: steady.state`. -/
theorem isStationary_diracProb (β : ℝ) (σ : ℝ → ℝ) (a : Idx d → EucSpace d)
    (x : SSphere d) : IsStationary β σ (0 : Idx d → ℝ) a (Perspective.diracProb d x) := by
  intro y hy
  have hyx : y = x := Interpolation.eq_of_mem_support_dirac hy
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  subst hyx
  rw [energyGrad, drift_zero, add_zero]
  show (∫ z, Real.exp (β * inner (𝕜 := ℝ) (y : EucSpace d) (z : EucSpace d)) •
    proj d (y : EucSpace d) (z : EucSpace d) ∂(Measure.dirac y)) = 0
  rw [integral_dirac]
  rw [show proj d (y : EucSpace d) (y : EucSpace d)
      = proj d (y : EucSpace d) ((1 : ℝ) • (y : EucSpace d)) by rw [one_smul],
    proj_smul_self hx, smul_zero]

end Perceptron
end Transformer
