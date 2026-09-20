/-
# Homogenized Transformers — the generator of one token

The vocabulary in which §4 of arXiv:2604.01978v1, *Homogenized Transformers*,
writes its two solution concepts: the Riemannian Hessian of a function of one
token, the generator `𝖫_μ` of `def:weak_spde`, the integrand of the stochastic
integral that both displays carry, and the notion of martingale on a finite
horizon that replaces those integrals in `McKeanVlasov.lean`.

Also the trivial head `ρ* = δ_0`, at which every one of these fields vanishes:
it is the witness that the solution concepts built on them are satisfiable.
-/

import Transformer.Homogenized.MeanField
import Mathlib.Probability.Martingale.Basic

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Homogenized

/-! ### Martingales on a finite horizon -/

/-- `M` is a martingale on `[0,T]` for the filtration `ℱ` and the law `P`.

Mathlib's `MeasureTheory.Martingale` asks for the martingale property at every
pair of times of the index type — all of `ℝ` here, negative times included —
which a process given on `[0,T]` cannot supply; and it leaves integrability
out, whereas `condExp` of a non-integrable function is `0`, so without it the
condition below would not say what a martingale is.  Both are spelled out. -/
def IsMartingaleOn {Ω : Type*} {m : MeasurableSpace Ω} (T : ℝ) (ℱ : Filtration ℝ m)
    (P : Measure Ω) (M : ℝ → Ω → ℝ) : Prop :=
  (∀ t ∈ Set.Icc (0 : ℝ) T, StronglyMeasurable[ℱ t] (M t)) ∧
    (∀ t ∈ Set.Icc (0 : ℝ) T, Integrable (M t) P) ∧
      ∀ s ∈ Set.Icc (0 : ℝ) T, ∀ t ∈ Set.Icc (0 : ℝ) T, s ≤ t →
        condExp (ℱ s) P (M t) =ᵐ[P] M s

theorem isMartingaleOn_const {Ω : Type*} {m : MeasurableSpace Ω} (T : ℝ)
    (ℱ : Filtration ℝ m) (P : Measure Ω) [IsFiniteMeasure P] (c : ℝ) :
    IsMartingaleOn T ℱ P (fun _ _ => c) := by
  refine ⟨fun _ _ => stronglyMeasurable_const, fun _ _ => integrable_const c, ?_⟩
  intro s _ _ _ _
  rw [condExp_const (ℱ.le s)]

/-! ### The generator -/

/-- The Riemannian Hessian on `𝕊^{d-1}` of a function of one token, at `x` and
evaluated at a tangent vector `v` twice:

  `Hess φ(x)[v,v] = D²φ(x)[v,v] - ⟨x, ∇φ(x)⟩ ‖v‖²`.

This is `sphHess` at `n = 1`, written on `EucSpace d` because the mean-field
limit carries one token and a measure, not a tuple.

Source: arXiv:2604.01978v1, `lem:toolkit_geo_riem`. -/
noncomputable def sphHess₁ {d : ℕ} (φ : EucSpace d → ℝ) (x v : EucSpace d) : ℝ :=
  iteratedFDeriv ℝ 2 φ x ![v, v] - fderiv ℝ φ x x * ‖v‖ ^ 2

@[simp]
theorem sphHess₁_zero {d : ℕ} (φ : EucSpace d → ℝ) (x : EucSpace d) :
    sphHess₁ φ x 0 = 0 := by
  have h : (iteratedFDeriv ℝ 2 φ x) ![(0 : EucSpace d), 0] = 0 :=
    ContinuousMultilinearMap.map_coord_zero _ (0 : Fin 2) (by simp)
  simp [sphHess₁, h]

/-- **The generator of `def:weak_spde`.**  At a measure `μ` on `𝕊^{d-1}`,

  `𝖫_μ φ(x) = ½ ∫_Θ Hess φ(x)[G_μ(x,θ), G_μ(x,θ)] ρ*(dθ)`.

There is no drift: at a general measure the mean field is subtracted off in
`G_μ = Proj_x ξ_θ[μ]`, so the whole first-order part of the limiting dynamics
sits in the noise.

Source: arXiv:2604.01978v1, `def:weak_spde`. -/
noncomputable def mvGenerator {d : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    (μ : Measure (EucSpace d)) (φ : EucSpace d → ℝ) (x : EucSpace d) : ℝ :=
  (1 / 2 : ℝ) * ∫ θ, sphHess₁ φ x (GfieldOf β ρ μ x θ) ∂ρ

/-- The integrand of the stochastic integral of `def:weak_spde`, at one head:

  `⟨∇φ · G_μ(·,θ), μ⟩ = ∫ Dφ(x)[G_μ(x,θ)] μ(dx)`.

The source's `∇` is the spherical gradient; `G_μ(x,θ)` is tangent at `x`, so
the pairing is the ambient differential applied to it, as in `sphGenerator`.

Source: arXiv:2604.01978v1, `def:weak_spde`. -/
noncomputable def spdeNoise {d : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    (μ : Measure (EucSpace d)) (φ : EucSpace d → ℝ) (θ : HeadParam d) : ℝ :=
  ∫ x, fderiv ℝ φ x (GfieldOf β ρ μ x θ) ∂μ

/-! ### The trivial head

Under `ρ* = δ_0` every head is the zero head, the attention field vanishes, and
so do the fluctuation, its projection, the generator and the noise integrand —
as `attnField_zero` and its companions do in `Basic.lean` for the empirical
measure. -/

@[simp]
theorem attnFieldOf_zero {d : ℕ} (β : ℝ) (μ : Measure (EucSpace d)) (x : EucSpace d) :
    attnFieldOf β (0 : HeadParam d) μ x = 0 := by
  simp [attnFieldOf, valueMap]

@[simp]
theorem meanFieldOf_dirac_zero {d : ℕ} (β : ℝ) (μ : Measure (EucSpace d)) (x : EucSpace d) :
    meanFieldOf β (Measure.dirac (0 : HeadParam d)) μ x = 0 := by
  simp [meanFieldOf]

@[simp]
theorem fluctOf_dirac_zero {d : ℕ} (β : ℝ) (μ : Measure (EucSpace d)) (x : EucSpace d) :
    fluctOf β (Measure.dirac (0 : HeadParam d)) μ (0 : HeadParam d) x = 0 := by
  simp [fluctOf]

@[simp]
theorem GfieldOf_dirac_zero {d : ℕ} (β : ℝ) (μ : Measure (EucSpace d)) (x : EucSpace d) :
    GfieldOf β (Measure.dirac (0 : HeadParam d)) μ x (0 : HeadParam d) = 0 := by
  simp [GfieldOf, proj]

@[simp]
theorem mvGenerator_dirac_zero {d : ℕ} (β : ℝ) (μ : Measure (EucSpace d))
    (φ : EucSpace d → ℝ) (x : EucSpace d) :
    mvGenerator β (Measure.dirac (0 : HeadParam d)) μ φ x = 0 := by
  simp [mvGenerator]

@[simp]
theorem spdeNoise_dirac_zero {d : ℕ} (β : ℝ) (μ : Measure (EucSpace d))
    (φ : EucSpace d → ℝ) :
    spdeNoise β (Measure.dirac (0 : HeadParam d)) μ φ (0 : HeadParam d) = 0 := by
  simp [spdeNoise]

end Homogenized
end Transformer
