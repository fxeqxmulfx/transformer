/-
# Homogenized Transformers — two systems under one noise

The object `prop: poc` of arXiv:2604.01978v1, *Homogenized Transformers*, §4
is stated about: the `n`-token system

  `dx_i^n = ∫_Θ G_{μ^n(t)}(x_i^n(t);θ) W(dθ,dt)`,  `μ^n = μ_{X^n}`,

and `n` conditionally i.i.d. copies of the solution of
`eq:non_linear_SDE_common`, **driven by the same** cylindrical `W`.

Sharing the noise is the whole content of `prop: poc`: under independent
noises the two families decorrelate and no pathwise bound between them holds.
Mathlib has no stochastic integral, so "the same `W`" is written here as
`CoupledPair.lean` writes it for two tokens — through the joint martingale
problem, whose generator `pocGenerator` sends a single `θ` into all `2n`
coordinates at once.  The `n` particles are driven by their own empirical
measure `μ^n = μ_{X^n}` and the `n` copies by the conditional law `μ(t)`,
which is what makes the two blocks different equations under one noise.

The `n`-particle equation is not a separate condition: it is the joint
martingale problem read on test functions of the first block alone.

Source: arXiv:2604.01978v1, `prop: poc`.
-/

import Transformer.Homogenized.McKeanVlasov

open scoped BigOperators NNReal ENNReal
open Real MeasureTheory

namespace Transformer
namespace Homogenized

/-! ### The generator of the coupled `2n`-token system -/

/-- The Riemannian Hessian of a function of the **`2n` tokens** of `prop: poc`,
in ambient coordinates: for `φ` on `(ℝ^d)^n × (ℝ^d)^n`, a tangent vector `U`
along the first block and `V` along the second,

  `Hess φ(X,X̄)[(U,V),(U,V)] = D²φ(X,X̄)[(U,V),(U,V)]
      - Σ_i ⟨x_i, ∇_{x_i}φ⟩‖u_i‖² - Σ_i ⟨x̄_i, ∇_{x̄_i}φ⟩‖v_i‖²`.

This is `sphHess` on the product of two `n`-token configurations:
`lem:toolkit_geo_riem` makes the correction componentwise, and here there are
`2n` components.

Source: arXiv:2604.01978v1, `lem:toolkit_geo_riem`. -/
noncomputable def pocHess {d n : ℕ}
    (φ : (Idx n → EucSpace d) × (Idx n → EucSpace d) → ℝ)
    (p : (Idx n → EucSpace d) × (Idx n → EucSpace d))
    (U V : Idx n → EucSpace d) : ℝ :=
  iteratedFDeriv ℝ 2 φ p ![(U, V), (U, V)]
    - ∑ i : Idx n, fderiv ℝ φ p (Pi.single i (p.1 i), 0) * ‖U i‖ ^ 2
    - ∑ i : Idx n, fderiv ℝ φ p (0, Pi.single i (p.2 i)) * ‖V i‖ ^ 2

@[simp] theorem pocHess_zero {d n : ℕ}
    (φ : (Idx n → EucSpace d) × (Idx n → EucSpace d) → ℝ)
    (p : (Idx n → EucSpace d) × (Idx n → EucSpace d)) : pocHess φ p 0 0 = 0 := by
  have h : (iteratedFDeriv ℝ 2 φ p)
      ![((0 : Idx n → EucSpace d), (0 : Idx n → EucSpace d)), (0, 0)] = 0 :=
    ContinuousMultilinearMap.map_coord_zero _ (0 : Fin 2) (by simp)
  simp [pocHess, h]

/-- The generator of the `2n` tokens of `prop: poc` under **one** common noise:

  `𝖫^{n,μ}φ(X,X̄) = ½ ∫_Θ Hess φ(X,X̄)[(G_{μ_X}(x_i,θ))_i, (G_μ(x̄_i,θ))_i]^{⊗2} ρ*(dθ)`.

The first block is driven by its own empirical measure and the second by the
conditional law `μ`, both by the same `θ`; at independent noises the second
block's `θ` would be integrated separately and the mixed term would factor.

Source: arXiv:2604.01978v1, `prop: poc`. -/
noncomputable def pocGenerator {d n : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    (ν : Measure (EucSpace d))
    (φ : (Idx n → EucSpace d) × (Idx n → EucSpace d) → ℝ)
    (p : (Idx n → EucSpace d) × (Idx n → EucSpace d)) : ℝ :=
  (1 / 2 : ℝ) * ∫ θ, pocHess φ p
    (fun i => GfieldOf β ρ (empMeasure p.1) (p.1 i) θ)
    (fun i => GfieldOf β ρ ν (p.2 i) θ) ∂ρ

/-- The compensated process
`φ(X(t),X̄(t)) - φ(X(0),X̄(0)) - ∫₀^t 𝖫^{n,μ(s)}φ ds` of the joint martingale
problem. -/
noncomputable def pocMart {d n : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    {Ω : Type*} (X Xbar : ℝ → Ω → (Idx n → EucSpace d))
    (μ : ℝ → Ω → Measure (EucSpace d))
    (φ : (Idx n → EucSpace d) × (Idx n → EucSpace d) → ℝ) (t : ℝ) (ω : Ω) : ℝ :=
  φ (X t ω, Xbar t ω) - φ (X 0 ω, Xbar 0 ω)
    - ∫ s in (0 : ℝ)..t, pocGenerator β ρ (μ s ω) φ (X s ω, Xbar s ω)

/-! ### The coupled system of `prop: poc` -/

/-- **The two systems of `prop: poc` on one space, under one noise.**  `X` is
the `n`-token system `dx_i^n = ∫_Θ G_{μ^n(t)}(x_i^n(t);θ) W(dθ,dt)` with
`μ^n = μ_{X^n}`, and `X̄` are `n` copies of the solution of
`eq:non_linear_SDE_common`, conditionally i.i.d. given the noise filtration and
driven by the same `W`.

The `n`-particle equation is the `mart` field read on test functions of the
first block alone, so it is not stated twice; `mart` is where sharing the noise
is recorded.

Source: arXiv:2604.01978v1, `prop: poc`. -/
structure IsPoCSystem {d n : ℕ} (β T : ℝ) (ρ : Measure (HeadParam d))
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) (ℱ 𝒢 : Filtration ℝ m)
    (μ₀ : Measure (EucSpace d)) (X Xbar : ℝ → Ω → (Idx n → EucSpace d))
    (μ : ℝ → Ω → Measure (EucSpace d)) : Prop where
  /-- The `n` tokens of the particle system live on `𝕊^{d-1}`. -/
  sphere : ∀ t ω i, ‖X t ω i‖ = 1
  /-- The particle system is `(ℱ_t)`-adapted. -/
  adapted : ∀ t ∈ Set.Icc (0 : ℝ) T, StronglyMeasurable[ℱ t] (X t)
  /-- Each copy solves `eq:non_linear_SDE_common` against the same conditional
  law `μ`, which is what makes the `x̄_i` identically distributed given the
  noise. -/
  copies : ∀ i : Idx n,
    IsMcKeanVlasovSolution β T ρ P ℱ 𝒢 (fun ω => Xbar 0 ω i) μ₀
      (fun t ω => Xbar t ω i) μ
  /-- The copies are conditionally independent given the noise filtration: the
  conditional law of `(x̄_1(t),…,x̄_n(t))` given `ℱ_t^W` is the product
  `μ(t)^{⊗n}`. -/
  condIid : ∀ t ∈ Set.Icc (0 : ℝ) T, ∀ A : Idx n → Set (EucSpace d),
    (∀ i, MeasurableSet (A i)) →
    condExp (𝒢 t) P
        (fun ω => ∏ i : Idx n, Set.indicator (A i) (1 : EucSpace d → ℝ) (Xbar t ω i))
      =ᵐ[P] fun ω => ∏ i : Idx n, (μ t ω (A i)).toReal
  /-- All `2n` tokens are driven by one noise: the joint martingale problem. -/
  mart : ∀ φ : (Idx n → EucSpace d) × (Idx n → EucSpace d) → ℝ,
    ContDiff ℝ (⊤ : ℕ∞) φ → IsMartingaleOn T ℱ P (pocMart β ρ X Xbar μ φ)

/-- **`IsPoCSystem` is satisfiable.**  At the trivial head law `ρ* = δ_0` the
field `G` vanishes, so all `2n` tokens frozen at one unit vector, with
`μ(t) = δ_e`, form a coupled system: every compensated process is identically
`0`, and constants are conditionally independent. -/
theorem isPoCSystem_dirac_zero {d n : ℕ} (β T : ℝ) {Ω : Type*} {m : MeasurableSpace Ω}
    (P : Measure Ω) [IsProbabilityMeasure P] (ℱ : Filtration ℝ m) (e : EucSpace d)
    (he : ‖e‖ = 1) :
    IsPoCSystem β T (Measure.dirac (0 : HeadParam d)) P ℱ ℱ (Measure.dirac e)
      (fun _ _ (_ : Idx n) => e) (fun _ _ (_ : Idx n) => e)
      (fun _ _ => Measure.dirac e) := by
  refine ⟨fun _ _ _ => he, fun _ _ => stronglyMeasurable_const,
    fun _ => isMcKeanVlasovSolution_dirac_zero β T P ℱ e he, fun t _ A _ => ?_,
    fun φ _ => ?_⟩
  · have hc : (fun _ : Ω => ∏ i : Idx n, Set.indicator (A i) (1 : EucSpace d → ℝ) e)
        = fun _ : Ω => ∏ i : Idx n, ((Measure.dirac e) (A i)).toReal := by
      funext ω
      refine Finset.prod_congr rfl fun i _ => ?_
      rw [Measure.dirac_apply]
      by_cases h : e ∈ A i <;> simp [h]
    show condExp (ℱ t) P (fun _ : Ω => ∏ i : Idx n,
        Set.indicator (A i) (1 : EucSpace d → ℝ) e)
      =ᵐ[P] fun _ : Ω => ∏ i : Idx n, ((Measure.dirac e) (A i)).toReal
    rw [hc, condExp_const (ℱ.le t)]
  · have hzero : pocMart β (Measure.dirac (0 : HeadParam d))
        (fun (_ : ℝ) (_ : Ω) (_ : Idx n) => e) (fun (_ : ℝ) (_ : Ω) (_ : Idx n) => e)
        (fun (_ : ℝ) (_ : Ω) => Measure.dirac e) φ = fun _ _ => (0 : ℝ) := by
      funext t ω
      simp [pocMart, pocGenerator]
      exact Or.inr (pocHess_zero φ _)
    rw [hzero]
    exact isMartingaleOn_const T ℱ P 0

end Homogenized
end Transformer
