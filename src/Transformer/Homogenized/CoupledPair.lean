/-
# Homogenized Transformers — two tokens under the same common noise

The object `thm:large_beta_meta` of arXiv:2604.01978v1, *Homogenized
Transformers*, §3 is stated about: two solutions `x⁽¹⁾`, `x⁽²⁾` of the
conditional McKean-Vlasov equation, with independent initial data drawn from
`μ(0)` and **driven by the same** cylindrical Wiener process.

Sharing the noise is the whole content of the theorem — it is what puts the
cross-covariance `K[μ]` of `eq:cross_variation_kernel` into the drift of the
overlap `R(t) = ⟨x⁽¹⁾(t), x⁽²⁾(t)⟩`, and without it the two tokens would be
independent and `R` would not move at all at leading order.  Since Mathlib has
no stochastic integral, "the same `W`" cannot be said directly; it is said
instead through the joint martingale problem, whose generator `mvGenerator₂`
carries the mixed second derivative

  `D²φ(x,y)[(G_μ(x,θ), G_μ(y,θ)), (G_μ(x,θ), G_μ(y,θ))]`

integrated in `θ` against `ρ*`.  That mixed term *is* `K[μ](x,y)`, so a pair
satisfying `IsCoupledPair` has exactly the cross-variation the source's
coupling argument uses, and a pair driven by independent noises does not
satisfy it.
-/

import Transformer.Homogenized.McKeanVlasov
import Mathlib.Probability.Independence.Basic

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Homogenized

/-! ### The two-token generator -/

/-- The Riemannian Hessian of a function of **two** tokens, in ambient
coordinates: for `φ : ℝ^d × ℝ^d → ℝ` and tangent vectors `u` at `x` and `v` at
`y`,

  `Hess φ(x,y)[(u,v),(u,v)] = D²φ(x,y)[(u,v),(u,v)] - ⟨∇_xφ, x⟩‖u‖² - ⟨∇_yφ, y⟩‖v‖²`.

This is `sphHess₁` on the product `(𝕊^{d-1})²`: `lem:toolkit_geo_riem` says the
correction is componentwise, and here the two components are `x` and `y`.

Source: arXiv:2604.01978v1, `lem:toolkit_geo_riem`. -/
noncomputable def sphHess₂ {d : ℕ} (φ : EucSpace d × EucSpace d → ℝ)
    (p : EucSpace d × EucSpace d) (u v : EucSpace d) : ℝ :=
  iteratedFDeriv ℝ 2 φ p ![(u, v), (u, v)]
    - fderiv ℝ φ p (p.1, 0) * ‖u‖ ^ 2 - fderiv ℝ φ p (0, p.2) * ‖v‖ ^ 2

@[simp] theorem sphHess₂_zero {d : ℕ} (φ : EucSpace d × EucSpace d → ℝ)
    (p : EucSpace d × EucSpace d) : sphHess₂ φ p 0 0 = 0 := by
  have h : (iteratedFDeriv ℝ 2 φ p) ![((0 : EucSpace d), (0 : EucSpace d)), (0, 0)] = 0 :=
    ContinuousMultilinearMap.map_coord_zero _ (0 : Fin 2) (by simp)
  simp [sphHess₂, h]

/-- The generator of the **pair** `(x⁽¹⁾, x⁽²⁾)` driven by one common noise:

  `𝖫₂^μ φ(x,y) = ½ ∫_Θ Hess φ(x,y)[(G_μ(x,θ), G_μ(y,θ))^⊗²] ρ*(dθ)`.

Both coordinates are hit by the *same* `θ`, which is what "the same `W`" means
after the Itô isometry; at independent noises the `θ` of the second coordinate
would be integrated separately and the mixed term would factor.

Source: arXiv:2604.01978v1, `thm:large_beta_meta` and its proof. -/
noncomputable def mvGenerator₂ {d : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    (μ : Measure (EucSpace d)) (φ : EucSpace d × EucSpace d → ℝ)
    (p : EucSpace d × EucSpace d) : ℝ :=
  (1 / 2 : ℝ) * ∫ θ, sphHess₂ φ p (GfieldOf β ρ μ p.1 θ) (GfieldOf β ρ μ p.2 θ) ∂ρ

/-- The compensated process `φ(x⁽¹⁾(t),x⁽²⁾(t)) - φ(x⁽¹⁾(0),x⁽²⁾(0)) - ∫₀^t 𝖫₂^{μ(s)}φ ds`
of the two-token martingale problem. -/
noncomputable def mvMart₂ {d : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    {Ω : Type*} (x₁ x₂ : ℝ → Ω → EucSpace d) (μ : ℝ → Ω → Measure (EucSpace d))
    (φ : EucSpace d × EucSpace d → ℝ) (t : ℝ) (ω : Ω) : ℝ :=
  φ (x₁ t ω, x₂ t ω) - φ (x₁ 0 ω, x₂ 0 ω)
    - ∫ s in (0 : ℝ)..t, mvGenerator₂ β ρ (μ s ω) φ (x₁ s ω, x₂ s ω)

/-- The **overlap** `R(t) = ⟨x⁽¹⁾(t), x⁽²⁾(t)⟩`, the quantity
`thm:large_beta_meta` tracks.

Source: arXiv:2604.01978v1, proof of `thm:large_beta_meta`. -/
noncomputable def overlap {d : ℕ} {Ω : Type*} (x₁ x₂ : ℝ → Ω → EucSpace d)
    (t : ℝ) (ω : Ω) : ℝ :=
  inner (𝕜 := ℝ) (x₁ t ω) (x₂ t ω)

/-! ### Two tokens under one noise -/

/-- **Two solutions of `eq:non_linear_SDE_common` driven by the same common
noise**, with initial data drawn independently from `μ(0)`.

Each coordinate solves the conditional McKean-Vlasov equation against the
*same* conditional-law process `μ`, the initial states are independent, and the
pair solves the joint martingale problem of `mvGenerator₂`, which is where the
sharing of the noise is recorded.

**What the source says and what is written here.**  `thm:large_beta_meta` cites
`eq:non_linear_SDE_cylindrical`, written with the full field `B_θ` and without
the `√α` scaling; its own proof works with `eq:non_linear_SDE_common`, whose
field is the projected fluctuation `G_μ(·,θ)` and which carries the Stratonovich
correction `-½(∫‖G‖²ρ*)x dt` explicitly.  The latter is the equation
formalized, here and in `McKeanVlasov.lean`: it is the one §4 defines, the one
`thm:PoC_wellposedness` is about, and the one the proof of
`thm:large_beta_meta` computes with.

Source: arXiv:2604.01978v1, `thm:large_beta_meta`. -/
structure IsCoupledPair {d : ℕ} (β T : ℝ) (ρ : Measure (HeadParam d))
    {Ω : Type*} {m : MeasurableSpace Ω} (P : Measure Ω) (ℱ 𝒢 : Filtration ℝ m)
    (μ₀ : Measure (EucSpace d)) (x₁ x₂ : ℝ → Ω → EucSpace d)
    (μ : ℝ → Ω → Measure (EucSpace d)) : Prop where
  /-- The first token solves the conditional McKean-Vlasov equation. -/
  fst : IsMcKeanVlasovSolution β T ρ P ℱ 𝒢 (x₁ 0) μ₀ x₁ μ
  /-- The second token solves it against the same conditional law `μ`. -/
  snd : IsMcKeanVlasovSolution β T ρ P ℱ 𝒢 (x₂ 0) μ₀ x₂ μ
  /-- The initial states are drawn independently. -/
  indep : ProbabilityTheory.IndepFun (x₁ 0) (x₂ 0) P
  /-- The pair solves the joint martingale problem: one noise, not two. -/
  mart : ∀ φ : EucSpace d × EucSpace d → ℝ, ContDiff ℝ (⊤ : ℕ∞) φ →
    IsMartingaleOn T ℱ P (mvMart₂ β ρ x₁ x₂ μ φ)

/-- **`IsCoupledPair` is satisfiable.**  At the trivial head law `ρ* = δ_0` the
field `G` vanishes, both generators are `0`, and the pair of tokens frozen at
one unit vector is a coupled pair: the compensated process is identically `0`,
and two constants are independent. -/
theorem isCoupledPair_dirac_zero {d : ℕ} (β T : ℝ) {Ω : Type*} {m : MeasurableSpace Ω}
    (P : Measure Ω) [IsProbabilityMeasure P] (ℱ : Filtration ℝ m) (e : EucSpace d)
    (he : ‖e‖ = 1) :
    IsCoupledPair β T (Measure.dirac (0 : HeadParam d)) P ℱ ℱ (Measure.dirac e)
      (fun _ _ => e) (fun _ _ => e) (fun _ _ => Measure.dirac e) := by
  refine ⟨isMcKeanVlasovSolution_dirac_zero β T P ℱ e he,
    isMcKeanVlasovSolution_dirac_zero β T P ℱ e he,
    ProbabilityTheory.indepFun_const_left _ _, ?_⟩
  intro φ _
  have hzero : mvMart₂ β (Measure.dirac (0 : HeadParam d)) (fun (_ : ℝ) (_ : Ω) => e)
      (fun (_ : ℝ) (_ : Ω) => e) (fun (_ : ℝ) (_ : Ω) => Measure.dirac e) φ
      = fun _ _ => 0 := by
    funext t ω
    simp [mvMart₂, mvGenerator₂]
  rw [hzero]
  exact isMartingaleOn_const T ℱ P 0

/-- The hypotheses of `isCoupledPair_dirac_zero` are satisfiable: the north
pole of `𝕊^d` is a unit vector of `ℝ^{d+1}`. -/
example (d : ℕ) : ‖((basePoint d : SSphere (d + 1)) : EucSpace (d + 1))‖ = 1 := by
  simp [basePoint, PiLp.norm_single]

end Homogenized
end Transformer
