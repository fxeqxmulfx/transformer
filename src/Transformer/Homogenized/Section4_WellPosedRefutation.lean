/-
# Homogenized Transformers — the formal well-posedness is false

The formal `def:nonlinear_SDE_common` of `McKeanVlasov.lean` admits any noise
filtration below `ℱ` (`Section4_RevealingFiltration`).  Under the trivial head
`ρ* = δ_0`, which satisfies `ass:high_order_short`, take a fair coin
`x_0 = ±e_0 ∈ 𝕊^0` frozen in time.  With the trivial filtration it solves the
equation with `μ(t) = μ_0 = ½δ_{e_0} + ½δ_{-e_0}`
(`isMcKeanVlasovSolution_frozen`); with `revealFiltration` it solves it with
`μ(t) = δ_{x_0}` for `t > 0` (`isMcKeanVlasovSolution_reveal`).  This refutes,
for the formal solution concepts,

* uniqueness in law, `prop:mckean_vlasov_common_noise`:
  `not_mckeanVlasovSolution_unique`;
* `prop: prop.3`: the revealing `μ` has a quadratic variation that
  `def:weak_spde` at `ρ* = δ_0` forbids:
  `not_isWeakSpdeSolution_of_isMcKeanVlasovSolution`.

Neither refutes the source, whose `𝒢` is the filtration of a noise
independent of `x_0`: no such `𝒢` can learn `x_0`.

Source: arXiv:2604.01978v1, §4.
-/

import Transformer.Homogenized.Section4_RevealingFiltration
import Transformer.Homogenized.WellPosed
import Transformer.Homogenized.RademacherLaw

open scoped ENNReal NNReal
open MeasureTheory

namespace Transformer
namespace Homogenized

variable {Ω : Type*}

/-- Under the trivial head the compensated process of `def:weak_spde` is the
plain increment `⟨μ(t),φ⟩ - ⟨μ(0),φ⟩`. -/
theorem spdeMart_dirac_zero {d : ℕ} (β : ℝ) (μ : ℝ → Ω → Measure (EucSpace d))
    (φ : EucSpace d → ℝ) (t : ℝ) (ω : Ω) :
    spdeMart β (Measure.dirac (0 : HeadParam d)) μ φ t ω
      = (∫ y, φ y ∂(μ t ω)) - ∫ y, φ y ∂(μ 0 ω) := by
  simp [spdeMart]

/-- Under the trivial head the bracket of `def:weak_spde` vanishes. -/
theorem spdeBracket_dirac_zero {d : ℕ} (β : ℝ) (μ : ℝ → Ω → Measure (EucSpace d))
    (φ : EucSpace d → ℝ) (t : ℝ) (ω : Ω) :
    spdeBracket β (Measure.dirac (0 : HeadParam d)) μ φ t ω = 0 := by
  simp [spdeBracket]

/-- `x_0 = ±e_0 ∈ 𝕊^0`, according to a coin. -/
noncomputable def coinPoint (b : Bool) : EucSpace 1 :=
  if b then EuclideanSpace.single 0 1 else -EuclideanSpace.single 0 1

theorem norm_coinPoint (b : Bool) : ‖coinPoint b‖ = 1 := by
  cases b <;> simp [coinPoint, PiLp.norm_single]

theorem coinPoint_false_ne : coinPoint false ≠ coinPoint true := by
  intro h
  have h0 := congrArg (fun y : EucSpace 1 => y 0) h
  norm_num [coinPoint] at h0

theorem map_fairCoin_coinPoint_true : fairCoin.map coinPoint {coinPoint true} = 2⁻¹ := by
  have hpre : coinPoint ⁻¹' {coinPoint true} = {true} := by
    ext b; cases b <;> simp [coinPoint_false_ne]
  rw [Measure.map_apply measurable_from_top (measurableSet_singleton _), hpre]
  simp [fairCoin]

/-- **`prop:mckean_vlasov_common_noise`, uniqueness, is false for the formal
solution concept.**

The claim refuted is `mckeanVlasovSolution_unique`, which stood in
`WellPosed.lean`: under `ass:high_order_short`, two solutions of the formal
`def:nonlinear_SDE_common` with one initial law `μ_0` have the same
finite-dimensional distributions of `(x, μ)` on `[0,T]` (`SameFdd`).  The
source claims pathwise uniqueness for solutions driven by one Wiener process
`W`, with `𝒢 = (ℱ_t^W)` and `x_0` independent of `W`.

The data: `d = 1`, `ρ* = δ_0` (`σ_V = σ_A = 0`), `β = 0`, `T = 1`, the coin of
the module docstring.  The functional `F(x, ν) = 1{ν{e_0} = 1}` of
`(x(1), μ(1))` has mean `0` under the trivial filtration and `½` under the
revealing one.

Source: arXiv:2604.01978v1, `prop:mckean_vlasov_common_noise`,
`def:nonlinear_SDE_common`. -/
theorem not_mckeanVlasovSolution_unique : ¬ ∀ (d : ℕ) (β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)), HasHighOrderLaw d σV σA ρ → ∀ T : ℝ, 0 < T →
    ∀ (μ₀ : Measure (EucSpace d)) (Ω : Type) (m : MeasurableSpace Ω) (P : Measure Ω),
      IsProbabilityMeasure P → ∀ (ℱ 𝒢 : Filtration ℝ m) (x₀ : Ω → EucSpace d)
      (x : ℝ → Ω → EucSpace d) (μ : ℝ → Ω → Measure (EucSpace d)),
      IsMcKeanVlasovSolution β T ρ P ℱ 𝒢 x₀ μ₀ x μ →
    ∀ (Ω' : Type) (m' : MeasurableSpace Ω') (P' : Measure Ω'), IsProbabilityMeasure P' →
      ∀ (ℱ' 𝒢' : Filtration ℝ m') (x₀' : Ω' → EucSpace d) (x' : ℝ → Ω' → EucSpace d)
      (μ' : ℝ → Ω' → Measure (EucSpace d)),
      IsMcKeanVlasovSolution β T ρ P' ℱ' 𝒢' x₀' μ₀ x' μ' →
      SameFdd T P P' (fun t ω => (x t ω, μ t ω)) (fun t ω => (x' t ω, μ' t ω)) := by
  intro H
  have hx : Measurable coinPoint := measurable_from_top
  have hG := hasVanishingField_dirac_zero (d := 1) 0
  let S : Set (Measure (EucSpace 1)) := {ν | ν {coinPoint true} = 1}
  have hS : MeasurableSet S :=
    Measure.measurable_coe (measurableSet_singleton _) (measurableSet_singleton 1)
  have h := H 1 0 0 0 _ (hasHighOrderLaw_dirac_zero 1) 1 one_pos _ Bool _ fairCoin
    inferInstance _ _ _ _ _ (isMcKeanVlasovSolution_frozen 0 1 _ hG fairCoin coinPoint hx
      norm_coinPoint) Bool _ fairCoin inferInstance _ _ _ _ _
    (isMcKeanVlasovSolution_reveal 0 1 _ hG fairCoin coinPoint hx norm_coinPoint)
    1 (fun _ => 1) (fun _ => ⟨zero_le_one, le_rfl⟩) (fun z => S.indicator 1 (z 0).2)
    ((measurable_one.indicator hS).comp (measurable_snd.comp (measurable_pi_apply 0)))
    ⟨1, fun z => by by_cases hz : (z 0).2 ∈ S <;> simp [hz]⟩
  simp only [revealLaw_of_pos fairCoin coinPoint (by norm_num : ¬ (1 : ℝ) ≤ 0),
    integral_fairCoin] at h
  simp [S, map_fairCoin_coinPoint_true, coinPoint_false_ne] at h

/-- **`prop: prop.3` is false for the formal solution concepts.**

The claim refuted is `isWeakSpdeSolution_of_isMcKeanVlasovSolution`, which
stood in `WellPosed.lean`: if `(x, μ)` solves the formal
`def:nonlinear_SDE_common` with noise filtration `𝒢`, then `μ` is a weak
solution of `eq: the.spde` for `𝒢` in the sense of the formal `def:weak_spde`.

The data are the revealing solution of `not_mckeanVlasovSolution_unique`.
Under `ρ* = δ_0` the noise integrand of `def:weak_spde` vanishes, so the
square of the compensated process `N^φ` must be a martingale; with
`φ(y) = y_0` it is `0` at time `0` and `x_0² = 1` at time `1`.

Source: arXiv:2604.01978v1, `prop: prop.3`, `def:weak_spde`,
`def:nonlinear_SDE_common`. -/
theorem not_isWeakSpdeSolution_of_isMcKeanVlasovSolution : ¬ ∀ (d : ℕ) (β T : ℝ)
    (ρ : Measure (HeadParam d)) (Ω : Type) (m : MeasurableSpace Ω) (P : Measure Ω)
    (ℱ 𝒢 : Filtration ℝ m) (x₀ : Ω → EucSpace d) (μ₀ : Measure (EucSpace d))
    (x : ℝ → Ω → EucSpace d) (μ : ℝ → Ω → Measure (EucSpace d)),
    IsMcKeanVlasovSolution β T ρ P ℱ 𝒢 x₀ μ₀ x μ → IsWeakSpdeSolution β T ρ P 𝒢 μ := by
  intro H
  have hx : Measurable coinPoint := measurable_from_top
  let φ : EucSpace 1 →L[ℝ] ℝ := EuclideanSpace.proj 0
  have h := ((H 1 0 1 _ Bool _ fairCoin _ _ coinPoint _ _ _
    (isMcKeanVlasovSolution_reveal 0 1 _ (hasVanishingField_dirac_zero 0) fairCoin coinPoint
      hx norm_coinPoint)).bracket φ φ.contDiff).2.2
    0 ⟨le_rfl, zero_le_one⟩ 1 ⟨zero_le_one, le_rfl⟩ zero_le_one
  rw [revealFiltration_apply, ite_eq_left le_rfl, condExp_bot] at h
  obtain ⟨b, hb⟩ := h.exists
  simp only [spdeMart_dirac_zero, spdeBracket_dirac_zero, sub_self,
    revealLaw_of_pos fairCoin coinPoint (by norm_num : ¬ (1 : ℝ) ≤ 0),
    revealLaw_of_nonpos fairCoin coinPoint (le_refl (0 : ℝ)), integral_dirac,
    integral_map hx.aemeasurable φ.continuous.aestronglyMeasurable, integral_fairCoin] at hb
  norm_num [φ, coinPoint] at hb

end Homogenized
end Transformer
