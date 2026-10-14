/-
# The original barycenter correlation is Lipschitz in its Gram matrix

ArXiv:2604.01978v1, §7, eq:SIGMA_R and the overlap comparison after
eq:Lispchitz_bound_softmax. The empirical barycenters are expanded
at their original head sample. Their inner product is exactly the
proved Gaussian Gram observable, and pushing the original head law
back to its independently Gaussian matrices gives its actual integral.

The Gaussian shape, zero coordinate means and full covariance entries
are then supplied by the original entry-law theorems. Applying the
proved coupling and coefficient comparison yields the full source
scaling for arbitrary unit configurations, including singular Grams.
The original initialization A = WW′ᵀ is retained throughout.
-/

import Transformer.Homogenized.Section7_GaussianGramCoupling
import Transformer.Homogenized.SimplexBary
import Transformer.Homogenized.AttnAverage
import Mathlib.MeasureTheory.Function.SpecialFunctions.Inner

open MeasureTheory ProbabilityTheory Set
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- Expanding the actual empirical barycenters gives exactly the original
Gram observable at the original query/key features.
Source: arXiv:2604.01978v1, §7, the overlap display following eq:SIGMA_R,
with transposes retained from the original initialization. -/
theorem inner_softBary_eq_gaussianGramObservable {d n : ℕ} (β : ℝ) (X : Idx n → EucSpace d)
    (z : GaussIdx d → ℝ) (i j : Idx n) :
    inner (𝕜 := ℝ) (softBary β (gaussBlock 1 z * (gaussBlock 2 z).transpose) (empMeasure X) (X i))
      (softBary β (gaussBlock 1 z * (gaussBlock 2 z).transpose) (empMeasure X) (X j)) =
    gaussianGramObservable β (fun k l => inner (𝕜 := ℝ) (X k) (X l)) i j (gaussianHeadFeatureMap X z) := by
  simp only [softBary_empMeasure, sum_inner, inner_sum, real_inner_smul_left, real_inner_smul_right,
    attnProb_eq_softmax_gaussianFeatures, gaussianGramObservable]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro k hk
  apply Finset.sum_congr rfl
  intro l hl
  ring

/-- The original head pushforward gives the actual Gaussian-feature
observable integral, without any assumed expectation identity.
Source: arXiv:2604.01978v1, §7, the expectation following eq:SIGMA_R. -/
theorem baryCorr_eq_integral_gaussianGramObservable {Ω : Type*} [MeasurableSpace Ω] {d n : ℕ}
    (P : Measure Ω) (Z : Ω → GaussIdx d → ℝ) (hZ : Measurable Z)
    (ρ : Measure (HeadParam d))
    (hρ : P.map (fun ω => ((gaussBlock 0 (Z ω),
      gaussBlock 1 (Z ω) * (gaussBlock 2 (Z ω)).transpose) : HeadParam d)) = ρ)
    (β : ℝ) (X : Idx n → EucSpace d) (i j : Idx n) :
    baryCorr β ρ (empMeasure X) (X i) (X j) =
      ∫ ω, gaussianGramObservable β (fun k l => inner (𝕜 := ℝ) (X k) (X l)) i j
        (gaussianHeadFeatureMap X (Z ω)) ∂P := by
  have hm (k : Idx n) : Measurable (fun θ : HeadParam d => softBary β θ.2 (empMeasure X) (X k)) := by
    simp only [softBary_empMeasure]
    exact (measurable_sum_attnProb_smul β X k).comp measurable_snd
  have hhead : Measurable (fun ω => ((gaussBlock 0 (Z ω),
      gaussBlock 1 (Z ω) * (gaussBlock 2 (Z ω)).transpose) : HeadParam d)) :=
    (measurable_gaussHead d).comp hZ
  rw [baryCorr, ← hρ]
  calc
    _ = ∫ ω, inner (𝕜 := ℝ)
        (softBary β (gaussBlock 1 (Z ω) * (gaussBlock 2 (Z ω)).transpose) (empMeasure X) (X i))
        (softBary β (gaussBlock 1 (Z ω) * (gaussBlock 2 (Z ω)).transpose) (empMeasure X) (X j)) ∂P :=
      integral_map hhead.aemeasurable ((hm i).inner (𝕜 := ℝ) (hm j)).aestronglyMeasurable
    _ = _ := integral_congr_ae (ae_of_all P fun ω => inner_softBary_eq_gaussianGramObservable β X (Z ω) i j)

/-- The actual original barycenter correlation is Lipschitz in the Gram
matrix, with constant 1 + 20 d σA⁴ β² n.
Source: arXiv:2604.01978v1, §7, the estimate after eq:Lispchitz_bound_softmax.
All laws, means and second moments are proved from the original head
entries. The argument applies to arbitrary pairs of unit configurations,
strengthening the source's comparison with a simplex configuration. -/
theorem abs_baryCorr_sub_le_gram {d n : ℕ} (hn : 0 < n) (β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : IsGaussianHeadLaw d σV σA ρ)
    (X Y : Idx n → EucSpace d) (hX : ∀ a, ‖X a‖ = 1) (hY : ∀ a, ‖Y a‖ = 1)
    (i j : Idx n) (ε : ℝ)
    (hR : ∀ a b, |inner (𝕜 := ℝ) (X a) (X b) - inner (𝕜 := ℝ) (Y a) (Y b)| ≤ ε) :
    |baryCorr β ρ (empMeasure X) (X i) (X j) - baryCorr β ρ (empMeasure Y) (Y i) (Y j)| ≤
      (1 + 20 * (d : ℝ) * (σA : ℝ) ^ 4 * β ^ 2 * (n : ℝ)) * ε := by
  obtain ⟨Ω, mΩ, P, Vr, Wr, Wr', hP, hV, hW, hW', hind, hmV, hmW, hmW', hρ⟩ := hρ
  have : IsProbabilityMeasure P := hP
  let Z (ω : Ω) (p : GaussIdx d) := headEntries Vr Wr Wr' p ω
  have hZ : Measurable Z := Measurable.of_eval (measurable_headEntries hV hW hW')
  have hhead : P.map (fun ω => ((gaussBlock 0 (Z ω),
      gaussBlock 1 (Z ω) * (gaussBlock 2 (Z ω)).transpose) : HeadParam d)) = ρ := by
    convert hρ using 1
    congr 1
  have hmap (p : GaussIdx d) :
      P.map (headEntries Vr Wr Wr' p) = gaussianReal 0 ((if p.1 = 0 then σV else σA) ^ 2) := by
    unfold headEntries
    obtain ⟨a, k, l⟩ := p
    fin_cases a
    · simpa using hmV k l
    · simpa using hmW k l
    · simpa using hmW' k l
  rw [baryCorr_eq_integral_gaussianGramObservable P Z hZ ρ hhead β X i j, baryCorr_eq_integral_gaussianGramObservable P Z hZ ρ hhead β Y i j]
  apply abs_integral_gaussianGramObservable_change_le P
    (fun ω => gaussianHeadFeatureMap X (Z ω)) (fun ω => gaussianHeadFeatureMap Y (Z ω))
    (hasGaussianLaw_original_head_features P Vr Wr Wr' hV hW hW' hind σV σA hmV hmW hmW' X)
    (hasGaussianLaw_original_head_features P Vr Wr Wr' hV hW hW' hind σV σA hmV hmW hmW' Y)
    (integral_gaussianHeadFeature (headEntries Vr Wr Wr') σV σA
      (measurable_headEntries hV hW hW') hmap X)
    (integral_gaussianHeadFeature (headEntries Vr Wr Wr') σV σA
      (measurable_headEntries hV hW hW') hmap Y) σA X Y hX hY
    (integral_original_gaussianHeadFeature_product Vr Wr Wr' σV σA hV hW hW' hind hmV hmW hmW' X)
    (integral_original_gaussianHeadFeature_product Vr Wr Wr' σV σA hV hW hW' hind hmV hmW hmW' Y)
    hn β _ _ (fun k l => by simpa [hX k, hX l] using abs_real_inner_le_norm (X k) (X l))
    (fun k l => by simpa [hY k, hY l] using abs_real_inner_le_norm (Y k) (Y l)) i j ε hR hR

/-- Correlations of empirical barycenters of unit tokens have absolute
value at most one under any probability head law.
Source: arXiv:2604.01978v1, §7, the bounded correlation used after
eq:Lispchitz_bound_softmax and in eq:Dij_explicit_clean. -/
theorem abs_baryCorr_empMeasure_le_one {d n : ℕ} (β : ℝ) (ρ : Measure (HeadParam d)) [IsProbabilityMeasure ρ]
    (X : Idx n → EucSpace d) (hX : ∀ a, ‖X a‖ = 1) (i j : Idx n) :
    |baryCorr β ρ (empMeasure X) (X i) (X j)| ≤ 1 := by
  have h := norm_integral_le_of_norm_le_const (μ := ρ)
    (f := fun θ => inner (𝕜 := ℝ) (softBary β θ.2 (empMeasure X) (X i))
      (softBary β θ.2 (empMeasure X) (X j))) (C := 1) (ae_of_all ρ fun θ => by
        rw [Real.norm_eq_abs, softBary_empMeasure, softBary_empMeasure]
        apply (abs_real_inner_le_norm _ _).trans
        simpa only [one_mul] using mul_le_mul (norm_sum_attnProb_smul_le β θ.2 hX i)
          (norm_sum_attnProb_smul_le β θ.2 hX j) (norm_nonneg _) zero_le_one)
  simpa only [baryCorr, Real.norm_eq_abs, Measure.real, measure_univ, ENNReal.toReal_one, mul_one] using h
/-- The canonical original head with positive value and attention variances,
unit basis tokens and a singular constant-token reference realizes every
head-law, unit, measurability, pushforward and entrywise-difference premise. -/
example : ∃ X Y : Idx 2 → EucSpace 2,
    (0 : ℕ) < 2 ∧ (∀ a, ‖X a‖ = 1) ∧ (∀ a, ‖Y a‖ = 1) ∧
    (∀ a b, |inner (𝕜 := ℝ) (X a) (X b) - inner (𝕜 := ℝ) (Y a) (Y b)| ≤ 1) ∧
    IsGaussianHeadLaw 2 1 1 (gaussHeadLaw 2 1 1) ∧ IsProbabilityMeasure (gaussHeadLaw 2 1 1) ∧
    Measurable (id : (GaussIdx 2 → ℝ) → (GaussIdx 2 → ℝ)) ∧
    (gaussEnsemble 2 1 1).map (fun z => ((gaussBlock 0 z,
      gaussBlock 1 z * (gaussBlock 2 z).transpose) : HeadParam 2)) = gaussHeadLaw 2 1 1 := by
  let X : Idx 2 → EucSpace 2 := fun a => EuclideanSpace.single a 1
  let Y : Idx 2 → EucSpace 2 := fun _ => EuclideanSpace.single (0 : Fin 2) 1
  refine ⟨X, Y, by norm_num, fun a => by simp [X], fun a => by simp [Y], ?_,
    isGaussianHeadLaw_gaussHeadLaw 2 1 1, inferInstance, measurable_id, rfl⟩
  intro a b
  fin_cases a <;> fin_cases b <;> norm_num [X, Y, EuclideanSpace.inner_single_left, PiLp.single_apply]

example : |baryCorr 1 (gaussHeadLaw 2 1 1)
    (empMeasure (fun a : Idx 2 => (EuclideanSpace.single a 1 : EucSpace 2)))
    (EuclideanSpace.single (0 : Idx 2) 1) (EuclideanSpace.single (1 : Idx 2) 1)| ≤ 1 := by
  exact abs_baryCorr_empMeasure_le_one _ _ _ (fun a => by simp) 0 1

end Transformer.Homogenized
