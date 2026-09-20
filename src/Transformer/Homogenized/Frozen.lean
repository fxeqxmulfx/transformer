/-
# Homogenized Transformers — the frozen pair

Whenever the field `G_μ(·,θ)` of `eq:G_def` vanishes on the sphere,
`eq:non_linear_SDE_common` reads `dx = 0`: any sphere-valued random variable,
held fixed in time, solves it, with conditional law its own law under the
trivial noise filtration.

Two head laws make the field vanish, for different reasons.  At `ρ* = δ_0`
every head is the zero head and the attention field is `0`.  In dimension `1`
the sphere is two points, every tangent space is trivial, and `Proj_x` kills
everything — there the head law may be an honest Gaussian one, which is what
the satisfiability of `thm:large_beta_meta` needs (`OneDim.lean`).

This is what makes `def:nonlinear_SDE_common` and `thm:large_beta_meta` more
than assertions about an empty class: `isCoupledPair_uniformAmbient` exhibits,
in every dimension `d ≥ 1`, a coupled pair whose common conditional law is the
uniform measure `σ_d` — so it satisfies `ass:low-temperature` at the same time,
with density `1`, which the frozen witnesses of `McKeanVlasov.lean` (whose law
is a Dirac mass) do not.
-/

import Transformer.Homogenized.CoupledPair
import Transformer.Homogenized.UniformLaw

open scoped BigOperators ENNReal
open Real MeasureTheory

namespace Transformer
namespace Homogenized

/-! ### Head laws whose field vanishes -/

/-- The diffusion kernel `G_μ(x,θ) = Proj_x ξ_θ[μ](x)` of `eq:G_def` vanishes
at every measure and every point of the sphere, for `ρ*`-almost every head.

This is a property of the temperature and the head law together: `GfieldOf` is
what drives `eq:non_linear_SDE_common`, so where it vanishes the equation is
`dx = 0` and every constant solves it.  Almost every head is the right
quantifier and not every head: the dynamics only ever sees `G` under `∫ · ∂ρ*`,
and at `ρ* = δ_0` the field does *not* vanish at a head `θ ≠ 0`. -/
def HasVanishingField {d : ℕ} (β : ℝ) (ρ : Measure (HeadParam d)) : Prop :=
  ∀ (ν : Measure (EucSpace d)) (x : EucSpace d), ‖x‖ = 1 →
    ∀ᵐ θ ∂ρ, GfieldOf β ρ ν x θ = 0

/-- At the trivial head law the field vanishes: `δ_0`-almost every head is the
zero head, whose attention field is `0`. -/
theorem hasVanishingField_dirac_zero {d : ℕ} (β : ℝ) :
    HasVanishingField β (Measure.dirac (0 : HeadParam d)) := by
  intro ν x _
  rw [ae_dirac_eq]
  exact GfieldOf_dirac_zero β ν x

/-- Where the field vanishes, so does the generator of one token. -/
theorem mvGenerator_of_vanishing {d : ℕ} {β : ℝ} {ρ : Measure (HeadParam d)}
    (hG : HasVanishingField β ρ) (ν : Measure (EucSpace d)) (φ : EucSpace d → ℝ)
    {x : EucSpace d} (hx : ‖x‖ = 1) : mvGenerator β ρ ν φ x = 0 := by
  have h : ∫ θ, sphHess₁ φ x (GfieldOf β ρ ν x θ) ∂ρ = ∫ _θ, (0 : ℝ) ∂ρ := by
    refine integral_congr_ae ?_
    filter_upwards [hG ν x hx] with θ hθ
    rw [hθ, sphHess₁_zero]
  simp [mvGenerator, h]

/-- And so does the generator of the pair. -/
theorem mvGenerator₂_of_vanishing {d : ℕ} {β : ℝ} {ρ : Measure (HeadParam d)}
    (hG : HasVanishingField β ρ) (ν : Measure (EucSpace d))
    (φ : EucSpace d × EucSpace d → ℝ) {p : EucSpace d × EucSpace d}
    (h₁ : ‖p.1‖ = 1) (h₂ : ‖p.2‖ = 1) : mvGenerator₂ β ρ ν φ p = 0 := by
  have h : ∫ θ, sphHess₂ φ p (GfieldOf β ρ ν p.1 θ) (GfieldOf β ρ ν p.2 θ) ∂ρ
      = ∫ _θ, (0 : ℝ) ∂ρ := by
    refine integral_congr_ae ?_
    filter_upwards [hG ν p.1 h₁, hG ν p.2 h₂] with θ hθ₁ hθ₂
    rw [hθ₁, hθ₂, sphHess₂_zero]
  simp [mvGenerator₂, h]

/-! ### A token that does not move -/

/-- **A frozen token solves `eq:non_linear_SDE_common`** whenever the field
vanishes: with `x(t) = x_0` for all `t`, every compensated process is
identically `0`, and under the trivial noise filtration the conditional law of
`x(t)` is its law. -/
theorem isMcKeanVlasovSolution_frozen {d : ℕ} (β T : ℝ) (ρ : Measure (HeadParam d))
    (hG : HasVanishingField β ρ) {Ω : Type*} {m : MeasurableSpace Ω}
    (P : Measure Ω) [IsProbabilityMeasure P] (x₀ : Ω → EucSpace d)
    (hmeas : Measurable x₀) (hsphere : ∀ ω, ‖x₀ ω‖ = 1) :
    IsMcKeanVlasovSolution β T ρ P
      (Filtration.const ℝ m le_rfl) (Filtration.const ℝ ⊥ bot_le)
      x₀ (Measure.map x₀ P) (fun _ => x₀) (fun _ _ => Measure.map x₀ P) := by
  have hprob : IsProbabilityMeasure (Measure.map x₀ P) :=
    ⟨by rw [Measure.map_apply hmeas MeasurableSet.univ, Set.preimage_univ, measure_univ]⟩
  have hbr : ∀ (φ : EucSpace d → ℝ) (ω : Ω),
      ∫ θ, fderiv ℝ φ (x₀ ω) (GfieldOf β ρ (Measure.map x₀ P) (x₀ ω) θ) ^ 2 ∂ρ = 0 := by
    intro φ ω
    rw [show ∫ θ, fderiv ℝ φ (x₀ ω) (GfieldOf β ρ (Measure.map x₀ P) (x₀ ω) θ) ^ 2 ∂ρ
        = ∫ _θ, (0 : ℝ) ∂ρ from integral_congr_ae ?_, integral_zero]
    filter_upwards [hG _ _ (hsphere ω)] with θ hθ
    rw [hθ, map_zero]
    ring
  have hmart : ∀ φ : EucSpace d → ℝ,
      mvMart β ρ (fun (_ : ℝ) => x₀)
        (fun (_ : ℝ) (_ : Ω) => Measure.map x₀ P) φ = fun _ _ => (0 : ℝ) := by
    intro φ; funext t ω
    simp [mvMart, mvGenerator_of_vanishing hG _ φ (hsphere ω)]
  refine ⟨fun _ => bot_le, fun _ => hsphere, fun _ _ => ?_, fun _ _ => measurable_const,
    Filter.Eventually.of_forall fun _ => rfl, rfl, fun _ => rfl, fun _ _ => hprob,
    fun t _ A hA => ?_, fun φ _ => ?_, fun φ _ => ?_⟩
  · exact hmeas.stronglyMeasurable
  · have hval : ((Measure.map x₀ P) A).toReal
        = ∫ ω, Set.indicator A (1 : EucSpace d → ℝ) (x₀ ω) ∂P := by
      have hcomp : (fun ω => Set.indicator A (1 : EucSpace d → ℝ) (x₀ ω))
          = Set.indicator (x₀ ⁻¹' A) (1 : Ω → ℝ) := by
        funext ω
        exact (Set.indicator_comp_right (s := A) (g := (1 : EucSpace d → ℝ)) x₀).symm
      rw [hcomp, integral_indicator_one (hmeas hA), Measure.map_apply hmeas hA,
        measureReal_def]
    show (fun _ : Ω => ((Measure.map x₀ P) A).toReal)
      =ᵐ[P] condExp ((Filtration.const ℝ ⊥ bot_le : Filtration ℝ m) t) P
        (fun ω => Set.indicator A (1 : EucSpace d → ℝ) (x₀ ω))
    rw [Filtration.const_apply, condExp_bot, hval]
  · rw [hmart φ]; exact isMartingaleOn_const T _ P 0
  · simp only [hmart φ, hbr φ]
    simpa using isMartingaleOn_const T (Filtration.const ℝ m le_rfl) P 0

/-- **Two frozen tokens are a coupled pair** whenever the field vanishes: they
solve the same conditional McKean-Vlasov equation, their initial states are
independent when the variables are, and the joint compensated process is
identically `0`. -/
theorem isCoupledPair_frozen {d : ℕ} (β T : ℝ) (ρ : Measure (HeadParam d))
    (hG : HasVanishingField β ρ) {Ω : Type*} {m : MeasurableSpace Ω}
    (P : Measure Ω) [IsProbabilityMeasure P] (X Y : Ω → EucSpace d)
    (hX : Measurable X) (hY : Measurable Y)
    (hsX : ∀ ω, ‖X ω‖ = 1) (hsY : ∀ ω, ‖Y ω‖ = 1)
    (hlaw : Measure.map Y P = Measure.map X P)
    (hindep : ProbabilityTheory.IndepFun X Y P) :
    IsCoupledPair β T ρ P
      (Filtration.const ℝ m le_rfl) (Filtration.const ℝ ⊥ bot_le)
      (Measure.map X P) (fun _ => X) (fun _ => Y)
      (fun _ _ => Measure.map X P) := by
  refine ⟨isMcKeanVlasovSolution_frozen β T ρ hG P X hX hsX, ?_, hindep, ?_⟩
  · have := isMcKeanVlasovSolution_frozen β T ρ hG P Y hY hsY
    rwa [hlaw] at this
  · intro φ _
    have hzero : mvMart₂ β ρ (fun (_ : ℝ) => X)
        (fun (_ : ℝ) => Y) (fun (_ : ℝ) (_ : Ω) => Measure.map X P) φ = fun _ _ => 0 := by
      funext t ω
      simp [mvMart₂, mvGenerator₂_of_vanishing hG _ φ (p := (X ω, Y ω)) (hsX ω) (hsY ω)]
    rw [hzero]
    exact isMartingaleOn_const T _ P 0

/-! ### A pair whose law is the uniform measure -/

open Classical in
/-- The **clamp** onto the unit sphere: a point of the sphere is left alone,
and everything else is sent to a fixed unit vector `e`.

It is the identity `σ_d`-almost everywhere, so it does not change the law; what
it buys is that `‖x(t,ω)‖ = 1` holds at *every* `ω`, as `IsMcKeanVlasovSolution`
asks, and not only almost everywhere. -/
noncomputable def clampSphere {d : ℕ} (e y : EucSpace d) : EucSpace d :=
  if ‖y‖ = 1 then y else e

theorem measurable_clampSphere {d : ℕ} (e : EucSpace d) : Measurable (clampSphere e) := by
  classical
  exact Measurable.ite (measurableSet_eq_fun measurable_norm measurable_const)
    measurable_id measurable_const

@[simp]
theorem clampSphere_of_norm_eq_one {d : ℕ} (e : EucSpace d) {y : EucSpace d}
    (hy : ‖y‖ = 1) : clampSphere e y = y := by
  classical
  simp [clampSphere, hy]

theorem norm_clampSphere {d : ℕ} {e : EucSpace d} (he : ‖e‖ = 1) (y : EucSpace d) :
    ‖clampSphere e y‖ = 1 := by
  classical
  by_cases h : ‖y‖ = 1 <;> simp [clampSphere, h, he]

/-- The clamp does not change a measure carried by the sphere. -/
theorem map_clampSphere {d : ℕ} {σ : Measure (EucSpace d)} {e : EucSpace d}
    (hσ : σ {y : EucSpace d | ‖y‖ = 1}ᶜ = 0) : Measure.map (clampSphere e) σ = σ := by
  classical
  have hsub : {y : EucSpace d | ¬ clampSphere e y = id y} ⊆ {y : EucSpace d | ‖y‖ = 1}ᶜ := by
    intro y hy hy1
    exact hy (clampSphere_of_norm_eq_one e hy1)
  have hae : clampSphere e =ᵐ[σ] id := by
    rw [Filter.EventuallyEq, ae_iff]
    exact measure_mono_null hsub hσ
  rw [Measure.map_congr hae, Measure.map_id]

/-- **A coupled pair whose conditional law is the uniform measure**, in every
dimension `d ≥ 1` and at every head law whose field vanishes: two independent
uniform tokens, frozen.  Together with `isLowTemperature_uniformAmbient` this
witnesses the hypotheses of `thm:large_beta_meta`. -/
theorem isCoupledPair_uniformAmbient {d : ℕ} (hd : 0 < d) (β T : ℝ)
    (ρ : Measure (HeadParam d)) (hG : HasVanishingField β ρ)
    {e : EucSpace d} (he : ‖e‖ = 1) :
    IsCoupledPair β T ρ
      ((uniformAmbient d).prod (uniformAmbient d))
      (Filtration.const ℝ _ le_rfl) (Filtration.const ℝ ⊥ bot_le)
      (uniformAmbient d)
      (fun _ ω => clampSphere e ω.1) (fun _ ω => clampSphere e ω.2)
      (fun _ _ => uniformAmbient d) := by
  obtain ⟨hprob, hcarr, -⟩ := isUniformAmbient_uniformAmbient hd
  have hmapX : Measure.map (fun ω : EucSpace d × EucSpace d => clampSphere e ω.1)
      ((uniformAmbient d).prod (uniformAmbient d)) = uniformAmbient d := by
    rw [← Function.comp_def, ← Measure.map_map (measurable_clampSphere e) measurable_fst,
      Measure.map_fst_prod, measure_univ, one_smul, map_clampSphere hcarr]
  have hmapY : Measure.map (fun ω : EucSpace d × EucSpace d => clampSphere e ω.2)
      ((uniformAmbient d).prod (uniformAmbient d)) = uniformAmbient d := by
    rw [← Function.comp_def, ← Measure.map_map (measurable_clampSphere e) measurable_snd,
      Measure.map_snd_prod, measure_univ, one_smul, map_clampSphere hcarr]
  have h := isCoupledPair_frozen (d := d) β T ρ hG
    ((uniformAmbient d).prod (uniformAmbient d))
    (fun ω => clampSphere e ω.1) (fun ω => clampSphere e ω.2)
    ((measurable_clampSphere e).comp measurable_fst)
    ((measurable_clampSphere e).comp measurable_snd)
    (fun _ => norm_clampSphere he _) (fun _ => norm_clampSphere he _)
    (by rw [hmapX, hmapY])
    (ProbabilityTheory.indepFun_prod (measurable_clampSphere e) (measurable_clampSphere e))
  rwa [hmapX] at h

/-- A unit vector exists in every dimension `d ≥ 1`. -/
theorem exists_norm_eq_one {d : ℕ} (hd : 0 < d) : ∃ e : EucSpace d, ‖e‖ = 1 :=
  ⟨EuclideanSpace.single ⟨0, hd⟩ (1 : ℝ), by simp [PiLp.norm_single]⟩

end Homogenized
end Transformer
