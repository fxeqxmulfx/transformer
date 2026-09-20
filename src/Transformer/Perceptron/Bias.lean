/-
# Perceptrons and attention's mean-field landscape — biases

Formalization of `rem: ext` (i) of arXiv:2601.21366v2: the results extend to a
perceptron with biases, `a_j · x + b_j` inside the activation, provided at
least one of the hyperplanes `{x : a_j · x + b_j = 0}` meets `𝕊^{d-1}` in more
than one point.  The geometry of that proviso is `Hyperplane.lean`.

**What the source says and what is carried here.**

* The biased potential is `v_ϑ(x) = Σ_j ω_j φ(a_j·x + b_j)`, and its spherical
  gradient is twice the biased drift — `proj_gradient_biasedPotential`, the
  biased form of the identity `(1/2)∇v_ϑ = u_ϑ` of `eq: primitive.field`.  At
  `b = 0` every biased object is the unbiased one of `Basic.lean`; that is
  `biasedPotential_zero` and its companions, which is also what makes the
  witnesses below available.

* "The results above" are `thm: circle`, `thm: circle.gelu` and `thm: any.d`.
  Carried here are the two that name the ReLU — `thm: circle` and
  `thm: any.d` (i) — in their biased form.  The strict-SOPD statements are not:
  they are about second-order conditions on the *energy*, so their biased form
  needs the biased geodesic second derivative as well, and the source gives no
  more reason to believe it than it gives for these two.

* "at least one of the hyperplanes intersects `𝕊^{d-1}` in more than one point"
  is carried in the source's geometric form, `¬ Set.Subsingleton` of
  `sphereHyperplane`; it is `|b_j| < ‖a_j‖` by
  `not_subsingleton_sphereHyperplane_iff`, which is how the witnesses below
  discharge it.

Source: arXiv:2601.21366v2, `rem: ext` (i).
-/

import Transformer.Perceptron.Hyperplane
import Transformer.Perceptron.HigherDim

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Perceptron

variable {d : ℕ}

/-! ### The biased perceptron -/

/-- **`eq: primitive.field` with biases.**  The biased perceptron potential

  `v_ϑ(x) = Σ_j ω_j φ(a_j · x + b_j)`.

Source: arXiv:2601.21366v2, `rem: ext` (i). -/
noncomputable def biasedPotential (φ : ℝ → ℝ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (b : Idx d → ℝ) (x : EucSpace d) : ℝ :=
  ∑ j : Idx d, ω j * φ (inner (𝕜 := ℝ) (a j) x + b j)

/-- **The biased perceptron drift**

  `u_ϑ(x) = Proj_x Σ_j ω_j σ(a_j · x + b_j) a_j`.

Source: arXiv:2601.21366v2, `rem: ext` (i). -/
noncomputable def biasedDrift (σ : ℝ → ℝ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (b : Idx d → ℝ) (x : EucSpace d) : EucSpace d :=
  proj d x (∑ j : Idx d, (ω j * σ (inner (𝕜 := ℝ) (a j) x + b j)) • a j)

/-- **The Wasserstein gradient of the first variation, with biases.**

Source: arXiv:2601.21366v2, `rem: ext` (i). -/
noncomputable def biasedEnergyGrad (β : ℝ) (σ : ℝ → ℝ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (b : Idx d → ℝ) (μ : Perspective.ProbSphere d)
    (x : EucSpace d) : EucSpace d :=
  (∫ y, Real.exp (β * inner (𝕜 := ℝ) x (y : EucSpace d)) • proj d x (y : EucSpace d)
      ∂(μ : Measure (SSphere d))) + biasedDrift σ ω a b x

/-- **`eq: steady.state` with biases.**

Source: arXiv:2601.21366v2, `rem: ext` (i). -/
def IsBiasedStationary (β : ℝ) (σ : ℝ → ℝ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (b : Idx d → ℝ) (μ : Perspective.ProbSphere d) : Prop :=
  ∀ x ∈ (μ : Measure (SSphere d)).support,
    biasedEnergyGrad β σ ω a b μ (x : EucSpace d) = 0

/-- **`v_ϑ` is real-analytic on `𝕊^{d-1}`, with biases**: see
`IsAnalyticOnSphere`, of which this is the biased form.

Source: arXiv:2601.21366v2, `rem: ext` (i). -/
def IsBiasedAnalyticOnSphere (φ : ℝ → ℝ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (b : Idx d → ℝ) : Prop :=
  ∃ g : EucSpace d → ℝ, AnalyticOnNhd ℝ g {y : EucSpace d | ‖y‖ = 1} ∧
    ∀ y : EucSpace d, ‖y‖ = 1 → g y = biasedPotential φ ω a b y

/-! ### At zero bias the perceptron is the one of `Basic.lean` -/

@[simp] theorem biasedPotential_zero (φ : ℝ → ℝ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d) :
    biasedPotential φ ω a 0 = potential φ ω a := by
  funext x; simp [biasedPotential, potential]

@[simp] theorem biasedDrift_zero (σ : ℝ → ℝ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d) :
    biasedDrift σ ω a 0 = drift σ ω a := by
  funext x; simp [biasedDrift, drift]

@[simp] theorem biasedEnergyGrad_zero (β : ℝ) (σ : ℝ → ℝ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) :
    biasedEnergyGrad β σ ω a 0 μ = energyGrad β σ ω a μ := by
  funext x; simp [biasedEnergyGrad, energyGrad]

@[simp] theorem isBiasedStationary_zero_iff (β : ℝ) (σ : ℝ → ℝ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) (μ : Perspective.ProbSphere d) :
    IsBiasedStationary β σ ω a 0 μ ↔ IsStationary β σ ω a μ := by
  simp [IsBiasedStationary, IsStationary]

@[simp] theorem isBiasedAnalyticOnSphere_zero_iff (φ : ℝ → ℝ) (ω : Idx d → ℝ)
    (a : Idx d → EucSpace d) :
    IsBiasedAnalyticOnSphere φ ω a 0 ↔ IsAnalyticOnSphere φ ω a := by
  simp [IsBiasedAnalyticOnSphere, IsAnalyticOnSphere]

/-! ### The gradient of the biased potential -/

/-- **`(1/2)∇v_ϑ = u_ϑ` with biases.**  The ambient gradient of the biased
potential is `2 Σ_j ω_j σ(a_j·x + b_j) a_j`; the bias shifts the argument of
`σ` and nothing else, because `x ↦ a_j·x + b_j` has the same derivative as
`x ↦ a_j·x`.

Source: arXiv:2601.21366v2, `eq: primitive.field`, `rem: ext` (i). -/
theorem hasGradientAt_biasedPotential (φ σ : ℝ → ℝ)
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (b : Idx d → ℝ) (x : EucSpace d) :
    HasGradientAt (biasedPotential φ ω a b)
      ((2 : ℝ) • ∑ j : Idx d,
        (ω j * σ (inner (𝕜 := ℝ) (a j) x + b j)) • a j) x := by
  rw [hasGradientAt_iff_hasFDerivAt]
  have hterm : ∀ j : Idx d,
      HasFDerivAt (fun y : EucSpace d => ω j * φ (inner (𝕜 := ℝ) (a j) y + b j))
        ((ω j * (2 * σ (inner (𝕜 := ℝ) (a j) x + b j))) • (innerSL ℝ (a j))) x := by
    intro j
    have hlin : HasFDerivAt (fun y : EucSpace d => inner (𝕜 := ℝ) (a j) y + b j)
        (innerSL ℝ (a j)) x := ((innerSL ℝ (a j)).hasFDerivAt).add_const (b j)
    have hcomp := (hφ (inner (𝕜 := ℝ) (a j) x + b j)).comp_hasFDerivAt x hlin
    simpa [mul_smul] using hcomp.const_mul (ω j)
  have hsum := HasFDerivAt.fun_sum (fun j (_ : j ∈ Finset.univ) => hterm j)
  refine hsum.congr_fderiv ?_
  ext y
  simp only [FunLike.coe_sum, Finset.sum_apply, smul_apply, innerSL_apply_apply, smul_eq_mul,
    InnerProductSpace.toDual_apply_apply, real_inner_smul_left, sum_inner]
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun j _ => by ring

/-- The hypothesis of `hasGradientAt_biasedPotential` is satisfiable: `σ = 0`
has the primitive `φ = 0`. -/
example : ∀ s : ℝ, HasDerivAt (fun _ : ℝ => (0 : ℝ)) (2 * (0 : ℝ → ℝ) s) s := by
  intro s
  simpa using hasDerivAt_const s (0 : ℝ)

/-- **The biased spherical gradient is twice the biased drift.**

Source: arXiv:2601.21366v2, `eq: primitive.field`, `rem: ext` (i). -/
theorem proj_gradient_biasedPotential (φ σ : ℝ → ℝ)
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (b : Idx d → ℝ) (x : EucSpace d) :
    proj d x (gradient (biasedPotential φ ω a b) x) = (2 : ℝ) • biasedDrift σ ω a b x := by
  rw [(hasGradientAt_biasedPotential φ σ hφ ω a b x).gradient, biasedDrift, proj, proj]
  simp only [inner_smul_right, smul_sub, smul_smul]

/-- The hypothesis of `proj_gradient_biasedPotential` is satisfiable: `σ = 0`
has the primitive `φ = 0`. -/
example : ∀ s : ℝ, HasDerivAt (fun _ : ℝ => (0 : ℝ)) (2 * (0 : ℝ → ℝ) s) s := by
  intro s
  simpa using hasDerivAt_const s (0 : ℝ)

end Perceptron
end Transformer
