/-
# The expected uniform energy is strictly below consensus

This is the probabilistic step for arXiv:2410.06833v1, §4,
`sec: energy.levels`. Independently reflecting one of two distinct sampled
tokens preserves the product law. For each sample, the two reflected
pair deficits have a strictly positive sum: at least one of the inner
products is less than one. Their expectations agree, so the expected
pair deficit is positive. Integrating the pointwise energy bound then
puts the mean strictly below the consensus maximum.

Continuity on the compact configuration space supplies every integrability
hypothesis. The argument works even on the two-point sphere, and needs
no density formula or unproved concentration estimate.

The strict gap comes from an off-diagonal ordered pair. Independence supplies
the reflection invariance, and compactness controls the integrals.
-/

import Transformer.Metastability.Section4_EnergyGap
import Transformer.Metastability.Section4_UniformLaw
import Transformer.Perspective.WendelSignSymmetry

open Real MeasureTheory
open scoped BigOperators

namespace Transformer.Metastability
variable {d n : ℕ}

/-- An independent uniform sample of at least two points has mean energy
strictly below `1/(2β)` for every `β > 0`.

The sign-flip invariance is proved in `Perspective.WendelSignSymmetry`;
no almost-sure distinctness or atomlessness assumption is necessary.
Source: arXiv:2410.06833v1, §4, `sec: energy.levels`, and the energy of §1,
`eq: interaction.energy`. -/
theorem mean_energy_lt_max (hn : 2 ≤ n) (β : ℝ) (hβ : 0 < β)
    (ν : Measure (SSphere d)) (hν : IsUniformOn d ν) :
    (∫ X, Eβ d n β X ∂iidSphere d n ν) < 1 / (2 * β) := by
  let : IsProbabilityMeasure ν := hν.1
  let : IsProbabilityMeasure (iidSphere d n ν) :=
    inferInstanceAs (IsProbabilityMeasure (Measure.pi (fun _ : Idx n => ν)))
  let i : Idx n := ⟨0, by omega⟩
  let j : Idx n := ⟨1, by omega⟩
  let D : SphereTuple d n → ℝ := fun X => Real.exp β - Real.exp
    (β * inner (𝕜 := ℝ) (X i : EucSpace d) (X j : EucSpace d))
  let flip := Perspective.flipSigns d n (fun k => decide (k = i))
  have hcD : Continuous D := by dsimp [D]; fun_prop
  have hiD : Integrable D (iidSphere d n ν) :=
    hcD.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
  have hflip := Perspective.measurePreserving_flipSigns d n ν hν.2
    (fun k => decide (k = i))
  have hcomp : (∫ X, D (flip X) ∂iidSphere d n ν) = ∫ X, D X ∂iidSphere d n ν := by
    have hm : AEStronglyMeasurable D ((iidSphere d n ν).map flip) :=
      hcD.measurable.aestronglyMeasurable
    have h := integral_map hflip.measurable.aemeasurable hm
    dsimp [iidSphere] at h
    rw [hflip.map_eq] at h
    exact h.symm
  have hneg (X : SphereTuple d n) : D (flip X) = Real.exp β - Real.exp
      (-β * inner (𝕜 := ℝ) (X i : EucSpace d) (X j : EucSpace d)) := by
    have hji : j ≠ i := by
      intro h
      have hh := congrArg Fin.val h
      dsimp [i, j] at hh
      omega
    simp only [D, flip, Perspective.flipSigns, decide_eq_true_eq, ite_true,
      ite_eq_right hji, Perspective.sphereMap, LinearIsometryEquiv.coe_neg, inner_neg_left]
    congr 2
    ring
  have hiF : Integrable (fun X => D X + D (flip X)) (iidSphere d n ν) := by
    apply hiD.add
    have hc : Continuous (fun X => D (flip X)) := by simp_rw [hneg]; fun_prop
    exact hc.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
  have hpos (X : SphereTuple d n) : 0 < D X + D (flip X) := by
    have h1 : 0 ≤ D X := sub_nonneg.mpr (exp_pair_le_exp β hβ.le _ _)
    have h2 : 0 ≤ D (flip X) := sub_nonneg.mpr (exp_pair_le_exp β hβ.le _ _)
    by_cases hs : 0 ≤ inner (𝕜 := ℝ) (X i : EucSpace d) (X j : EucSpace d)
    · have he : Real.exp (-β * inner (𝕜 := ℝ) (X i : EucSpace d) (X j : EucSpace d)) <
          Real.exp β := Real.exp_lt_exp.mpr (by nlinarith)
      rw [hneg]
      linarith
    · have he : Real.exp (β * inner (𝕜 := ℝ) (X i : EucSpace d) (X j : EucSpace d)) <
          Real.exp β := Real.exp_lt_exp.mpr (by nlinarith)
      dsimp [D] at h1 ⊢
      linarith
  have hp : 0 < ∫ X, D X + D (flip X) ∂iidSphere d n ν := by
    apply (integral_pos_iff_support_of_nonneg (fun X => (hpos X).le) hiF).mpr
    have hs : Function.support (fun X => D X + D (flip X)) = Set.univ := by
      ext X
      simp only [Function.mem_support, Set.mem_univ, iff_true]
      exact (hpos X).ne'
    rw [hs, measure_univ]
    exact zero_lt_one
  have hiDC : Integrable (fun X => D (flip X)) (iidSphere d n ν) := by
    have hc : Continuous (fun X => D (flip X)) := by simp_rw [hneg]; fun_prop
    exact hc.integrable_of_hasCompactSupport (HasCompactSupport.of_compactSpace _)
  rw [integral_add hiD hiDC, hcomp] at hp
  have hE : Integrable (Eβ d n β) (iidSphere d n ν) :=
    integrable_energy β (iidSphere d n ν)
  have hn0 : (0 : ℝ) < n := by exact_mod_cast (show 0 < n by omega)
  have hD0 : 0 < 2 * β * Real.exp β * (n : ℝ) ^ 2 := by positivity
  have hle := integral_mono (hE.add (hiD.div_const _)) (integrable_const (1 / (2 * β)))
    (fun X => energy_add_pair_deficit_le β hβ (by omega) X i j)
  change (∫ X, Eβ d n β X + D X / (2 * β * Real.exp β * (n : ℝ) ^ 2)
    ∂iidSphere d n ν) ≤ ∫ X : SphereTuple d n, 1 / (2 * β) ∂iidSphere d n ν at hle
  rw [integral_add hE (hiD.div_const _), integral_div, integral_const, probReal_univ,
    one_smul] at hle
  have : 0 < (∫ X, D X ∂iidSphere d n ν) / (2 * β * Real.exp β * (n : ℝ) ^ 2) := by
    apply div_pos _ hD0
    linarith
  linarith

/-- The full hypotheses and strict conclusion hold for two uniform points
on the circle at `β = 1`; uniformity is exhibited, not postulated. -/
example : 2 ≤ 2 ∧ (0 : ℝ) < 1 ∧ IsUniformOn 2 (MeanField.uniformLaw 2) ∧
    (∫ X, Eβ 2 2 1 X ∂iidSphere 2 2 (MeanField.uniformLaw 2)) < 1 / 2 := by
  refine ⟨le_rfl, by norm_num, isUniformOn_uniformLaw (by norm_num), ?_⟩
  simpa only [mul_one] using mean_energy_lt_max (by norm_num) 1 (by norm_num) _
    (isUniformOn_uniformLaw (by norm_num))

/-- The uniform energy gap is positive, with the same explicit uniform law. -/
example : 0 < 1 / (2 * (10000 : ℝ)) -
    ∫ X, Eβ 2 2 10000 X ∂iidSphere 2 2 (MeanField.uniformLaw 2) := by
  exact sub_pos.mpr (mean_energy_lt_max (by norm_num) 10000 (by norm_num) _
    (isUniformOn_uniformLaw (by norm_num)))

/-- At `β > 1`, the positive gap between consensus and the uniform mean
is also strictly less than one. These are exactly the range conditions
needed to place the nonempty energy window inside `(0,1)`.
Source: arXiv:2410.06833v1, §4, `sec: energy.levels`. -/
theorem uniform_energy_gap_mem_Ioo (hn : 2 ≤ n) (β : ℝ) (hβ : 1 < β)
    (ν : Measure (SSphere d)) (hν : IsUniformOn d ν) :
    1 / (2 * β) - (∫ X, Eβ d n β X ∂iidSphere d n ν) ∈ Set.Ioo (0 : ℝ) 1 := by
  have hβ0 : 0 < β := by linarith
  refine ⟨sub_pos.mpr (mean_energy_lt_max hn β hβ0 ν hν), ?_⟩
  have hm : 0 ≤ ∫ X, Eβ d n β X ∂iidSphere d n ν :=
    integral_nonneg (Eβ_nonneg d n hβ0)
  have ht : 1 / (2 * β) < 1 := (div_lt_iff₀ (by positivity)).mpr (by linarith)
  linarith

/-- Two tokens, `β = 2`, and the constructed circle law satisfy every
hypothesis of the uniform-gap range estimate. -/
example : 2 ≤ 2 ∧ (1 : ℝ) < 2 ∧ IsUniformOn 2 (MeanField.uniformLaw 2) ∧
    1 / (2 * (2 : ℝ)) - (∫ X, Eβ 2 2 2 X ∂iidSphere 2 2 (MeanField.uniformLaw 2)) ∈
      Set.Ioo (0 : ℝ) 1 := by
  exact ⟨le_rfl, by norm_num, isUniformOn_uniformLaw (by norm_num),
    uniform_energy_gap_mem_Ioo (by norm_num) 2 (by norm_num) _
      (isUniformOn_uniformLaw (by norm_num))⟩

end Transformer.Metastability
