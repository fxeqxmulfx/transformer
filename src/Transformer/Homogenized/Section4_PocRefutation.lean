module

public import Transformer.Homogenized.Section4_WellPosedRefutation

/-!
# Homogenized Transformers — the formal weak solution is not the conditional law

`thm:PoC_wellposedness` of arXiv:2604.01978v1 makes the conditional law `μ` of
the McKean–Vlasov solution the unique weak solution of `eq: the.spde`. Its
uniqueness half — every weak solution starting at `μ_0` has the law of `μ` — is
false for the formal solution concepts, on the data of
`Section4_WellPosedRefutation`. Under the trivial head `ρ* = δ_0` the
compensated process of a weak solution has no noise to follow, and a law at
rest solves `def:weak_spde` for every filtration
(`isWeakSpdeSolution_dirac_zero_const`); the revealing `μ` jumps at `0+`
(`not_poc_wellposedness`).

As in `Section4_WellPosedRefutation`, the source is not refuted: its `𝒢` is the
filtration of a noise independent of `x_0`, and then `μ(t) = Law(x_0) = μ_0`.

Source: arXiv:2604.01978v1, §4.
-/

@[expose] public section

open scoped NNReal
open MeasureTheory

namespace Transformer
namespace Homogenized

/-- **A law at rest solves `def:weak_spde` under the trivial head.**  Under
`ρ* = δ_0` the compensated process `N^φ` of a constant `μ(t) = ν` and its
bracket are `0`, whatever the space and the filtration. -/
theorem isWeakSpdeSolution_dirac_zero_const {d : ℕ} (β T : ℝ) {Ω : Type*}
    {m : MeasurableSpace Ω} (P : Measure Ω) [IsProbabilityMeasure P] (𝒢 : Filtration ℝ m)
    (ν : Measure (EucSpace d)) [IsProbabilityMeasure ν]
    (hν : ν {y : EucSpace d | ‖y‖ = 1}ᶜ = 0) :
    IsWeakSpdeSolution β T (Measure.dirac (0 : HeadParam d)) P 𝒢 (fun _ _ => ν) := by
  have hmart : ∀ φ : EucSpace d → ℝ,
      spdeMart β (Measure.dirac (0 : HeadParam d)) (fun (_ : ℝ) (_ : Ω) => ν) φ
        = fun _ _ => (0 : ℝ) := by
    intro φ; funext t ω; simp [spdeMart_dirac_zero]
  refine ⟨fun _ _ => inferInstance, fun _ _ => hν, fun _ _ => measurable_const,
    fun φ _ => ?_, fun φ _ => ?_⟩
  · rw [hmart φ]; exact isMartingaleOn_const T 𝒢 P 0
  · simp only [hmart φ, spdeBracket_dirac_zero]
    simpa using isMartingaleOn_const T 𝒢 P 0

/-- The coin's law `½δ_{e_0} + ½δ_{-e_0}` is carried by `𝕊^0`. -/
theorem map_fairCoin_coinPoint_sphere :
    fairCoin.map coinPoint {y : EucSpace 1 | ‖y‖ = 1}ᶜ = 0 := by
  have hpre : coinPoint ⁻¹' {y : EucSpace 1 | ‖y‖ = 1}ᶜ = ∅ := by
    ext b; simp [norm_coinPoint]
  rw [Measure.map_apply measurable_from_top
    (isClosed_eq continuous_norm continuous_const).measurableSet.compl, hpre, measure_empty]

/-- **`thm:PoC_wellposedness` is false for the formal solution concepts.**

The claim refuted is `poc_wellposedness`, which stood in `WellPosed.lean`:
under `ass:high_order_short`, for every `μ_0 ∈ 𝒫(𝕊^{d-1})` the formal
`def:nonlinear_SDE_common` has a solution, and for every solution `(x, μ)` and
every weak solution `μ'` of the formal `def:weak_spde` with `μ'(0) = μ_0`, the
processes `μ` and `μ'` have the same finite-dimensional distributions on
`[0,T]` (`SameFdd`). The source: "there exists a unique solution `(x,μ)` …
Moreover, `μ` is the unique weak solution to `eq: the.spde`".

The data: `d = 1`, `ρ* = δ_0` (`σ_V = σ_A = 0`), `β = 0`, `T = 1`, the fair
coin `x_0 = ±e_0` with the revealing filtration, which gives the solution
`μ(t) = δ_{x_0}` for `t > 0` (`isMcKeanVlasovSolution_reveal`), and the weak
solution `μ' = μ_0` at rest for the same filtration
(`isWeakSpdeSolution_dirac_zero_const`). The functional `1{ν{e_0} = 1}` of the
law at time `1` has mean `½` under `μ` and `0` under `μ'`. The existence half
alone is `exists_mckeanVlasovSolution`.

Source: arXiv:2604.01978v1, `thm:PoC_wellposedness`, `def:weak_spde`,
`def:nonlinear_SDE_common`. -/
theorem not_poc_wellposedness : ¬ ∀ (d : ℕ) (β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)), HasHighOrderLaw d σV σA ρ → ∀ T : ℝ, 0 < T →
    ∀ μ₀ : Measure (EucSpace d), IsProbabilityMeasure μ₀ →
    μ₀ {y : EucSpace d | ‖y‖ = 1}ᶜ = 0 →
    HasMcKeanVlasovSolution β T ρ μ₀ ∧
      ∀ (Ω : Type) (mΩ : MeasurableSpace Ω) (P : Measure Ω) (ℱ 𝒢 : Filtration ℝ mΩ)
        (x₀ : Ω → EucSpace d) (x : ℝ → Ω → EucSpace d) (μ : ℝ → Ω → Measure (EucSpace d)),
        IsProbabilityMeasure P → IsMcKeanVlasovSolution β T ρ P ℱ 𝒢 x₀ μ₀ x μ →
      ∀ (Ω' : Type) (mΩ' : MeasurableSpace Ω') (P' : Measure Ω') (𝒢' : Filtration ℝ mΩ')
        (μ' : ℝ → Ω' → Measure (EucSpace d)),
        IsProbabilityMeasure P' → IsWeakSpdeSolution β T ρ P' 𝒢' μ' →
        (∀ ω, μ' 0 ω = μ₀) → SameFdd T P P' μ μ' := by
  intro H
  have hx : Measurable coinPoint := measurable_from_top
  let S : Set (Measure (EucSpace 1)) := {ν | ν {coinPoint true} = 1}
  have hS : MeasurableSet S :=
    Measure.measurable_coe (measurableSet_singleton _) (measurableSet_singleton 1)
  have h := (H 1 0 0 0 _ (hasHighOrderLaw_dirac_zero 1) 1 one_pos (fairCoin.map coinPoint)
    inferInstance map_fairCoin_coinPoint_sphere).2 Bool _ fairCoin _ _ _ _ _ inferInstance
    (isMcKeanVlasovSolution_reveal 0 1 _ (hasVanishingField_dirac_zero 0) fairCoin coinPoint
      hx norm_coinPoint) Bool _ fairCoin (revealFiltration _) _ inferInstance
    (isWeakSpdeSolution_dirac_zero_const 0 1 fairCoin _ _ map_fairCoin_coinPoint_sphere)
    (fun _ => rfl) 1 (fun _ => 1) (fun _ => ⟨zero_le_one, le_rfl⟩) (fun z => S.indicator 1 (z 0))
    ((measurable_one.indicator hS).comp (measurable_pi_apply 0))
    ⟨1, fun z => by by_cases hz : z 0 ∈ S <;> simp [hz]⟩
  simp only [revealLaw_of_pos fairCoin coinPoint (by norm_num : ¬ (1 : ℝ) ≤ 0),
    integral_fairCoin] at h
  simp [S, map_fairCoin_coinPoint_true, coinPoint_false_ne] at h

/-- The hypotheses of `isWeakSpdeSolution_dirac_zero_const` hold: the fair coin
and the law of `coinPoint` under it are probability measures, and the latter is
carried by `𝕊^0`. -/
example : IsProbabilityMeasure fairCoin ∧ IsProbabilityMeasure (fairCoin.map coinPoint) ∧
    fairCoin.map coinPoint {y : EucSpace 1 | ‖y‖ = 1}ᶜ = 0 :=
  ⟨inferInstance, inferInstance, map_fairCoin_coinPoint_sphere⟩

end Homogenized
end Transformer
