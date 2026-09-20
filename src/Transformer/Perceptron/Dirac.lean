/-
# Perceptrons and attention's mean-field landscape — a strict SOPD critical point

The statements of arXiv:2601.21366v2 are about *strictly SOPD* Wasserstein
critical points, and a hypothesis nothing satisfies is worth nothing.  This
file exhibits one, proved from the definitions: for the one-neuron perceptron

  `ω = e_{j₀}`,  `a_{j₀} = -x`,  `φ(s) = s`,  `σ ≡ 1/2`,

the Dirac mass `δ_x` is a strictly SOPD critical point of `E_{β,ϑ}` with
`κ = 1/2`, in every dimension.  The reason is visible in one line: along the
geodesic issued from `x` with velocity `ξ(x)`, the interaction energy is
constant — `e^β/(2β)` for any Dirac — and the potential is `-cos(t‖ξ(x)‖)`, by
`inner_sphereExp_smul`.

This is what the satisfiability witnesses of `thm: circle.gelu`, `thm: any.d`
and `thm: bound` are built from; `σ ≡ 1/2` is real-analytic and nowhere zero,
which is what those statements ask of it.

Source: arXiv:2601.21366v2, §2.2–2.3.
-/

import Transformer.Perceptron.Geodesic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Perceptron

variable {d : ℕ}

/-! ### The energy of a Dirac mass -/

/-- The interaction energy of a Dirac mass on the sphere is `e^β/(2β)`,
whatever the point: `⟪x,x⟫ = 1`. -/
theorem interactionEnergy_of_dirac (β : ℝ) (μ : Perspective.ProbSphere d) (z : SSphere d)
    (hμ : (μ : Measure (SSphere d)) = Measure.dirac z) :
    Perspective.interactionEnergy d β μ = (2 * β)⁻¹ * Real.exp β := by
  have hz : inner (𝕜 := ℝ) (z : EucSpace d) (z : EucSpace d) = (1 : ℝ) := by
    rw [real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp z.2]; norm_num
  rw [Perspective.interactionEnergy, hμ, integral_dirac, integral_dirac, hz]
  norm_num

/-- The coupled energy of a Dirac mass: the interaction part is constant, and
the potential part is the potential at the atom. -/
theorem energy_of_dirac (β : ℝ) (φ : ℝ → ℝ) (ω : Idx d → ℝ) (a : Idx d → EucSpace d)
    (μ : Perspective.ProbSphere d) (z : SSphere d)
    (hμ : (μ : Measure (SSphere d)) = Measure.dirac z) :
    energy β φ ω a μ
      = (2 * β)⁻¹ * Real.exp β + 2⁻¹ * potential φ ω a (z : EucSpace d) := by
  rw [energy, interactionEnergy_of_dirac β μ z hμ, hμ, integral_dirac]

/-! ### The one-neuron witness -/

/-- The witness perceptron's potential is `-⟪x, ·⟫`. -/
theorem potential_pin (j₀ : Idx d) (x : EucSpace d) (y : EucSpace d) :
    potential (fun s => s) (Pi.single j₀ (1 : ℝ)) (Pi.single j₀ (-x)) y
      = -inner (𝕜 := ℝ) x y := by
  rw [potential, Finset.sum_eq_single j₀]
  · simp [inner_neg_left]
  · intro j _ hj
    simp [Pi.single_eq_of_ne hj]
  · intro h
    exact absurd (Finset.mem_univ j₀) h

/-- The witness perceptron's drift is `Proj_·(-x/2)`; it vanishes at `x`. -/
theorem drift_pin (j₀ : Idx d) (x : EucSpace d) (y : EucSpace d) :
    drift (fun _ => (2 : ℝ)⁻¹) (Pi.single j₀ (1 : ℝ)) (Pi.single j₀ (-x)) y
      = proj d y ((-(2 : ℝ)⁻¹) • x) := by
  rw [drift, Finset.sum_eq_single j₀]
  · simp [neg_smul]
  · intro j _ hj
    simp [Pi.single_eq_of_ne hj]
  · intro h
    exact absurd (Finset.mem_univ j₀) h

/-- **`δ_x` is a critical point** of the witness perceptron: the attention
field vanishes at the only atom, and the drift is a multiple of `x` there. -/
theorem isStationary_pin (β : ℝ) (j₀ : Idx d) (x : SSphere d) :
    IsStationary β (fun _ => (2 : ℝ)⁻¹) (Pi.single j₀ (1 : ℝ))
      (Pi.single j₀ (-(x : EucSpace d))) (Perspective.diracProb d x) := by
  intro y hy
  have hyx : y = x := Interpolation.eq_of_mem_support_dirac hy
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  subst hyx
  rw [energyGrad, drift_pin, proj_smul_self hx]
  show (∫ z, Real.exp (β * inner (𝕜 := ℝ) (y : EucSpace d) (z : EucSpace d)) •
    proj d (y : EucSpace d) (z : EucSpace d) ∂(Measure.dirac y)) + 0 = 0
  rw [integral_dirac, add_zero,
    show proj d (y : EucSpace d) (y : EucSpace d)
      = proj d (y : EucSpace d) ((1 : ℝ) • (y : EucSpace d)) by rw [one_smul],
    proj_smul_self hx, smul_zero]

/-- **`δ_x` is a *strictly* SOPD critical point** of the witness perceptron,
with `κ = 1/2`: along every geodesic issued from `x`, the energy is

  `e^β/(2β) - (1/2) cos(t‖ξ(x)‖)`,

whose second derivative at `0` is `‖ξ(x)‖²/2 = (1/2)‖ξ‖²_{L²(δ_x)}`.

Source: arXiv:2601.21366v2, `eq:strict2order`. -/
theorem isStrictSOPD_pin (β : ℝ) (j₀ : Idx d) (x : SSphere d) :
    IsStrictSOPD β (fun s => s) (fun _ => (2 : ℝ)⁻¹) (Pi.single j₀ (1 : ℝ))
      (Pi.single j₀ (-(x : EucSpace d))) (Perspective.diracProb d x) := by
  refine ⟨isStationary_pin β j₀ x, 2⁻¹, by norm_num, fun ξ hξ ν hν H hH => ?_⟩
  set r : ℝ := ‖ξ x‖ with hr
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  -- the energy along an arbitrary geodesic issued from `x`
  have hg : (fun t => energy β (fun s => s) (Pi.single j₀ (1 : ℝ))
      (Pi.single j₀ (-(x : EucSpace d))) (ν t))
      = fun t => (2 * β)⁻¹ * Real.exp β + 2⁻¹ * -Real.cos (t * r) := by
    funext t
    obtain ⟨T, hTmeas, hTval, hTmap⟩ := hν t
    have hνt : (ν t : Measure (SSphere d)) = Measure.dirac (T x) := by
      rw [hTmap]; exact Measure.map_dirac' hTmeas x
    rw [energy_of_dirac β _ _ _ (ν t) (T x) hνt, potential_pin, hTval x,
      inner_sphereExp_smul hx (inner_isGradientField hξ x) t]
  -- its first derivative, everywhere
  have hd1 : ∀ t : ℝ, HasDerivAt (fun t => energy β (fun s => s) (Pi.single j₀ (1 : ℝ))
      (Pi.single j₀ (-(x : EucSpace d))) (ν t)) (2⁻¹ * (Real.sin (t * r) * r)) t := by
    intro t
    rw [hg]
    have hlin : HasDerivAt (fun t : ℝ => t * r) r t := by
      simpa using (hasDerivAt_id t).mul_const r
    have hc : HasDerivAt (fun t : ℝ => Real.cos (t * r)) (-Real.sin (t * r) * r) t := hlin.cos
    have := (hc.neg.const_mul (2 : ℝ)⁻¹).const_add ((2 * β)⁻¹ * Real.exp β)
    convert this using 1
    ring
  have hderiv : (deriv fun t => energy β (fun s => s) (Pi.single j₀ (1 : ℝ))
      (Pi.single j₀ (-(x : EucSpace d))) (ν t))
      = fun t => 2⁻¹ * (Real.sin (t * r) * r) := funext fun t => (hd1 t).deriv
  -- and its second derivative at `0`
  have hd2 : HasDerivAt (fun t => 2⁻¹ * (Real.sin (t * r) * r)) (2⁻¹ * (r * r)) 0 := by
    have hlin : HasDerivAt (fun t : ℝ => t * r) r 0 := by
      simpa using (hasDerivAt_id (0 : ℝ)).mul_const r
    have hs : HasDerivAt (fun t : ℝ => Real.sin (t * r)) (Real.cos (0 * r) * r) 0 := hlin.sin
    have := (hs.mul_const r).const_mul (2 : ℝ)⁻¹
    simpa using this
  rw [hderiv] at hH
  rw [hH.unique hd2]
  have hint : ∫ y, ‖ξ y‖ ^ 2 ∂((Perspective.diracProb d x : Perspective.ProbSphere d) :
      Measure (SSphere d)) = r ^ 2 := by
    show ∫ y, ‖ξ y‖ ^ 2 ∂(Measure.dirac x) = r ^ 2
    rw [integral_dirac, hr]
  rw [hint]
  ring_nf
  exact le_refl _

end Perceptron
end Transformer
