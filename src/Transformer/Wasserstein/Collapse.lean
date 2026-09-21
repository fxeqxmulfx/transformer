/-
# The 2-Wasserstein distance — collapsing mass onto a point

The upper bound companion of `Wasserstein/LowerBound.lean`.  Keep `ρ` on a set
`s` and move the mass of `sᶜ` onto one point `x₀`: the result is
`ρ|_s + ρ(sᶜ) δ_{x₀}`, the plan that moves it is the coupling
`(x ↦ (x,x))_# ρ|_s + (x ↦ (x,x₀))_# ρ|_{sᶜ}`, and it costs at most
`ρ(sᶜ) D²` when every point is within `D` of `x₀`.

This is the construction of arXiv:2411.04551v3, proof of
`lem: mass.concentration.Q1`, with the geodesic transport there replaced by a
jump onto a single point, which already gives `W_2 = O(√η)`.

Source: arXiv:2411.04551v3, `lem: mass.concentration.Q1`.
-/

import Transformer.Wasserstein.Basic

open scoped ENNReal
open Real MeasureTheory

namespace Transformer
namespace Wasserstein

variable {X : Type*} [MeasurableSpace X] [PseudoMetricSpace X]
  [OpensMeasurableSpace X] [SecondCountableTopology X]

/-- **Collapsing the mass outside `s` onto a point.**  If every point is within
`D` of `x₀`, then

  `W_2(ρ, ρ|_s + ρ(sᶜ) δ_{x₀})² ≤ ρ(sᶜ) D²`.

Source: arXiv:2411.04551v3, proof of `lem: mass.concentration.Q1`. -/
theorem W2_sq_le_collapse (ρ : Measure X) [IsFiniteMeasure ρ] {s : Set X}
    (hs : MeasurableSet s) (x₀ : X) {D : ℝ} (hD : ∀ x, dist x x₀ ≤ D) :
    W2 ρ (ρ.restrict s + ρ sᶜ • Measure.dirac x₀) ^ 2 ≤ ρ.real sᶜ * D ^ 2 := by
  have hdiag : Measurable (fun x : X => (x, x)) := by fun_prop
  have hpt : Measurable (fun x : X => (x, x₀)) := by fun_prop
  set γ := Measure.map (fun x : X => (x, x)) (ρ.restrict s) +
    Measure.map (fun x : X => (x, x₀)) (ρ.restrict sᶜ) with hγ
  have hcoup : IsCoupling ρ (ρ.restrict s + ρ sᶜ • Measure.dirac x₀) γ := by
    constructor
    · rw [hγ, Measure.map_add _ _ measurable_fst, Measure.map_map measurable_fst hdiag,
        Measure.map_map measurable_fst hpt]
      simp [Function.comp_def, Measure.restrict_add_restrict_compl hs]
    · rw [hγ, Measure.map_add _ _ measurable_snd, Measure.map_map measurable_snd hdiag,
        Measure.map_map measurable_snd hpt]
      simp [Function.comp_def, Measure.map_const]
  have hD0 : 0 ≤ D := dist_self x₀ ▸ hD x₀
  have hf : Measurable (fun p : X × X => dist p.1 p.2 ^ 2) := by fun_prop
  have hint : ∀ ν : Measure X, IsFiniteMeasure ν → ∀ g : X → X × X, Measurable g →
      (∀ x, dist (g x).1 (g x).2 ≤ D) →
      Integrable (fun p : X × X => dist p.1 p.2 ^ 2) (Measure.map g ν) := by
    intro ν _ g hg hgD
    refine (integrable_map_measure hf.aestronglyMeasurable hg.aemeasurable).2 ?_
    refine Integrable.of_bound (hf.comp hg).aestronglyMeasurable (D ^ 2)
      (Filter.Eventually.of_forall fun x => ?_)
    rw [Function.comp_apply, Real.norm_of_nonneg (by positivity)]
    exact pow_le_pow_left₀ dist_nonneg (hgD x) 2
  have hcost : ∫ p, dist p.1 p.2 ^ 2 ∂γ ≤ ρ.real sᶜ * D ^ 2 := by
    rw [hγ, integral_add_measure (hint _ inferInstance _ hdiag fun x => by simpa using hD0)
      (hint _ inferInstance _ hpt hD),
      integral_map hdiag.aemeasurable hf.aestronglyMeasurable,
      integral_map hpt.aemeasurable hf.aestronglyMeasurable]
    simp only [dist_self, ne_eq, OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow,
      integral_zero, zero_add]
    calc ∫ x, dist x x₀ ^ 2 ∂(ρ.restrict sᶜ) ≤ ∫ _x, D ^ 2 ∂(ρ.restrict sᶜ) :=
          integral_mono_of_nonneg (Filter.Eventually.of_forall fun _ => by positivity)
            (integrable_const _) (Filter.Eventually.of_forall fun x =>
              pow_le_pow_left₀ dist_nonneg (hD x) 2)
      _ = ρ.real sᶜ * D ^ 2 := by simp [measureReal_def]
  have hle := W2_le_of_coupling γ hcoup
  have hc0 : 0 ≤ ∫ p, dist p.1 p.2 ^ 2 ∂γ := integral_nonneg fun p => by positivity
  calc W2 ρ (ρ.restrict s + ρ sᶜ • Measure.dirac x₀) ^ 2
      ≤ Real.sqrt (∫ p, dist p.1 p.2 ^ 2 ∂γ) ^ 2 :=
        pow_le_pow_left₀ (W2_nonneg _ _) hle 2
    _ = ∫ p, dist p.1 p.2 ^ 2 ∂γ := Real.sq_sqrt hc0
    _ ≤ _ := hcost

/-- The hypothesis of `W2_sq_le_collapse` is satisfiable: every point of the
closed interval `[0, 1]`, as a subtype of `ℝ`, is within `1` of `0`. -/
example : ∀ x : Set.Icc (0 : ℝ) 1, dist x ⟨0, by simp⟩ ≤ 1 := by
  rintro ⟨x, hx0, hx1⟩
  rw [Subtype.dist_eq, Real.dist_eq, sub_zero, abs_of_nonneg hx0]
  exact hx1

end Wasserstein
end Transformer
