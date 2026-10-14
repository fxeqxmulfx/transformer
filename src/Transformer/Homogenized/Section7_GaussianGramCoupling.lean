/-
# Independent coupling and changing the original Gram coefficients

ArXiv:2604.01978v1, §7, eq:interpolation_gaussian and the overlap
comparison following eq:Lispchitz_bound_softmax.
The two actual feature vectors may share their original head sample.
Their Gaussian image laws admit independent coordinate projections on
the product of those laws, with exactly the original expectations,
means and second moments. Thus interpolation requires no additional
independence hypothesis on the original features.

The direct coefficient change is bounded at fixed original scores.
Boundedness justifies subtraction of the two original integrals;
the proved Gaussian comparison handles the change of feature laws.
-/

import Transformer.Homogenized.Section7_GaussianGramExpectation

open MeasureTheory ProbabilityTheory Set
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- The actual expectation bound depends only on the two Gaussian
marginals; independent copies are constructed rather than assumed.
Source: arXiv:2604.01978v1, §7, the independent copies preceding
eq:interpolation_gaussian and the estimate eq:Lispchitz_bound_softmax. -/
theorem abs_integral_gaussianGramObservable_sub_le_of_marginals {Ω : Type*} [MeasurableSpace Ω] {d n : ℕ}
    (P : Measure Ω) (Z W : Ω → EuclideanSpace ℝ (GaussianFeatureIdx d n))
    (hZ : HasGaussianLaw Z P) (hW : HasGaussianLaw W P)
    (hmZ : ∀ p, ∫ ω, Z ω p ∂P = 0) (hmW : ∀ p, ∫ ω, W ω p ∂P = 0)
    (σA : ℝ≥0) (X Y : Idx n → EucSpace d) (hX : ∀ a, ‖X a‖ = 1) (hY : ∀ a, ‖Y a‖ = 1)
    (hZZ : ∀ p r, ∫ ω, Z ω p * Z ω r ∂P = gaussianFeatureCovariance σA X p r)
    (hWW : ∀ p r, ∫ ω, W ω p * W ω r ∂P = gaussianFeatureCovariance σA Y p r)
    (hn : 0 < n) (β : ℝ) (A : Matrix (Idx n) (Idx n) ℝ)
    (hA : ∀ k l, |A k l| ≤ 1) (i j : Idx n) (ε : ℝ)
    (hR : ∀ a b, |inner (𝕜 := ℝ) (X a) (X b) - inner (𝕜 := ℝ) (Y a) (Y b)| ≤ ε) :
    |(∫ ω, gaussianGramObservable β A i j (Z ω) ∂P) -
      ∫ ω, gaussianGramObservable β A i j (W ω) ∂P| ≤
        20 * (d : ℝ) * (σA : ℝ) ^ 4 * β ^ 2 * (n : ℝ) * ε := by
  let E := EuclideanSpace ℝ (GaussianFeatureIdx d n)
  let μ := P.map Z
  let ν := P.map W
  have : IsGaussian μ := hZ.isGaussian_map
  have : IsGaussian ν := hW.isGaussian_map
  have hlZ : HasLaw Z μ P := ⟨hZ.aemeasurable, rfl⟩
  have hlW : HasLaw W ν P := ⟨hW.aemeasurable, rfl⟩
  have hf : HasLaw (Prod.fst : E × E → E) μ (μ.prod ν) := measurePreserving_fst.hasLaw
  have hg : HasLaw (Prod.snd : E × E → E) ν (μ.prod ν) := measurePreserving_snd.hasLaw
  have hfi (f : E → ℝ) (hc : Continuous f) :
      (∫ z : E × E, f z.1 ∂(μ.prod ν)) = ∫ ω, f (Z ω) ∂P :=
    (hf.integral_comp hc.measurable.aestronglyMeasurable).trans
      (hlZ.integral_comp hc.measurable.aestronglyMeasurable).symm
  have hgi (f : E → ℝ) (hc : Continuous f) :
      (∫ z : E × E, f z.2 ∂(μ.prod ν)) = ∫ ω, f (W ω) ∂P :=
    (hg.integral_comp hc.measurable.aestronglyMeasurable).trans
      (hlW.integral_comp hc.measurable.aestronglyMeasurable).symm
  have hc (p : GaussianFeatureIdx d n) : Continuous (fun z : E => z p) :=
    (EuclideanSpace.proj p).continuous
  have hφ := (gaussianGramObservable_polynomial_growth (d := d) hn β A i j).1.continuous
  rw [← hfi _ hφ, ← hgi _ hφ]
  apply abs_integral_gaussianGramObservable_sub_le (μ.prod ν) Prod.fst Prod.snd
    hf.hasGaussianLaw hg.hasGaussianLaw (indepFun_prod measurable_id measurable_id)
    (fun p => (hfi _ (hc p)).trans (hmZ p)) (fun p => (hgi _ (hc p)).trans (hmW p)) σA X Y hX hY
    (fun p r => (hfi _ ((hc p).mul (hc r))).trans (hZZ p r))
    (fun p r => (hgi _ ((hc p).mul (hc r))).trans (hWW p r)) hn β A hA i j ε hR

/-- Bounded original coefficients make the actual observable integrable
under any finite measure and measurable feature sample.
Source: arXiv:2604.01978v1, §7, the normalized overlap sum after eq:SIGMA_R. -/
theorem integrable_gaussianGramObservable_comp {Ω : Type*} [MeasurableSpace Ω] {d n : ℕ}
    (P : Measure Ω) [IsFiniteMeasure P] (Z : Ω → EuclideanSpace ℝ (GaussianFeatureIdx d n))
    (hZ : AEMeasurable Z P) (hn : 0 < n) (β : ℝ)
    (A : Matrix (Idx n) (Idx n) ℝ) (hA : ∀ k l, |A k l| ≤ 1) (i j : Idx n) :
    Integrable (fun ω => gaussianGramObservable β A i j (Z ω)) P := by
  have hφ := (gaussianGramObservable_polynomial_growth (d := d) hn β A i j).1.continuous
  apply Integrable.of_bound (hφ.measurable.comp_aemeasurable hZ).aestronglyMeasurable 1
  apply ae_of_all
  intro ω
  rw [Real.norm_eq_abs]
  exact abs_sum_matrix_softmax_pair_le hn hn _ _ A 1 hA

/-- Changing both the original Gram coefficients and Gaussian feature
marginals has the source's full overlap Lipschitz scaling.
Source: arXiv:2604.01978v1, §7, the two-term estimate following
eq:Lispchitz_bound_softmax, with normalization before expectation. -/
theorem abs_integral_gaussianGramObservable_change_le {Ω : Type*} [MeasurableSpace Ω] {d n : ℕ}
    (P : Measure Ω) [IsProbabilityMeasure P]
    (Z W : Ω → EuclideanSpace ℝ (GaussianFeatureIdx d n))
    (hZ : HasGaussianLaw Z P) (hW : HasGaussianLaw W P)
    (hmZ : ∀ p, ∫ ω, Z ω p ∂P = 0) (hmW : ∀ p, ∫ ω, W ω p ∂P = 0)
    (σA : ℝ≥0) (X Y : Idx n → EucSpace d) (hX : ∀ a, ‖X a‖ = 1) (hY : ∀ a, ‖Y a‖ = 1)
    (hZZ : ∀ p r, ∫ ω, Z ω p * Z ω r ∂P = gaussianFeatureCovariance σA X p r)
    (hWW : ∀ p r, ∫ ω, W ω p * W ω r ∂P = gaussianFeatureCovariance σA Y p r)
    (hn : 0 < n) (β : ℝ) (A B : Matrix (Idx n) (Idx n) ℝ)
    (hA : ∀ k l, |A k l| ≤ 1) (hB : ∀ k l, |B k l| ≤ 1) (i j : Idx n) (ε : ℝ)
    (hR : ∀ a b, |inner (𝕜 := ℝ) (X a) (X b) - inner (𝕜 := ℝ) (Y a) (Y b)| ≤ ε)
    (hAB : ∀ k l, |A k l - B k l| ≤ ε) :
    |(∫ ω, gaussianGramObservable β A i j (Z ω) ∂P) -
      ∫ ω, gaussianGramObservable β B i j (W ω) ∂P| ≤
        (1 + 20 * (d : ℝ) * (σA : ℝ) ^ 4 * β ^ 2 * (n : ℝ)) * ε := by
  have hiA := integrable_gaussianGramObservable_comp P Z hZ.aemeasurable hn β A hA i j
  have hiB := integrable_gaussianGramObservable_comp P Z hZ.aemeasurable hn β B hB i j
  have hd := norm_integral_le_of_norm_le_const (μ := P)
    (f := fun ω => gaussianGramObservable β A i j (Z ω) - gaussianGramObservable β B i j (Z ω))
    (ae_of_all P fun ω => by
      rw [Real.norm_eq_abs]
      exact abs_sub_gram_softmax_pair_le hn hn _ _ A B ε hAB)
  rw [Real.norm_eq_abs, integral_sub hiA hiB] at hd
  simp only [Measure.real, measure_univ, ENNReal.toReal_one, mul_one] at hd
  have hg := abs_integral_gaussianGramObservable_sub_le_of_marginals P Z W hZ hW hmZ hmW σA X Y hX hY hZZ hWW hn β B hB i j ε hR
  exact (abs_sub_le _ _ _).trans ((add_le_add hd hg).trans_eq (by ring))
/-- Unit basis tokens, zero Gaussian features and bounded identity/zero
coefficients satisfy every law, moment and entrywise bound premise.
The coefficient change has a nonzero expectation difference. -/
example : ∃ X : Idx 2 → EucSpace 2,
    (0 : ℕ) < 2 ∧ (∀ a, ‖X a‖ = 1) ∧
    (∀ k l : Idx 2, |(1 : Matrix (Idx 2) (Idx 2) ℝ) k l| ≤ 1) ∧
    (∀ k l : Idx 2, |(0 : Matrix (Idx 2) (Idx 2) ℝ) k l| ≤ 1) ∧
    (∀ k l : Idx 2, |(1 : Matrix (Idx 2) (Idx 2) ℝ) k l -
      (0 : Matrix (Idx 2) (Idx 2) ℝ) k l| ≤ 1) ∧
    (∀ a b, |inner (𝕜 := ℝ) (X a) (X b) - inner (𝕜 := ℝ) (X a) (X b)| ≤ 1) ∧
    AEMeasurable (fun _ : Unit => (0 : EuclideanSpace ℝ (GaussianFeatureIdx 2 2))) (Measure.dirac ()) ∧
    HasGaussianLaw (fun _ : Unit => (0 : EuclideanSpace ℝ (GaussianFeatureIdx 2 2))) (Measure.dirac ()) ∧
    (∀ p : GaussianFeatureIdx 2 2, ∫ _ω : Unit, (0 : EuclideanSpace ℝ (GaussianFeatureIdx 2 2)) p
      ∂Measure.dirac () = 0) ∧
    (∀ p r : GaussianFeatureIdx 2 2, ∫ _ω : Unit,
      (0 : EuclideanSpace ℝ (GaussianFeatureIdx 2 2)) p * (0 : EuclideanSpace ℝ (GaussianFeatureIdx 2 2)) r
      ∂Measure.dirac () = gaussianFeatureCovariance 0 X p r) := by
  refine ⟨fun a => EuclideanSpace.single a 1, by norm_num, fun a => by simp,
    ?_, fun k l => by simp, ?_, fun a b => by simp, measurable_const.aemeasurable,
    ?_, fun p => by simp, fun p r => by simp [gaussianFeatureCovariance]⟩
  · intro k l
    simp only [Matrix.one_apply]
    split_ifs <;> norm_num
  · intro k l
    simp only [Matrix.one_apply, Matrix.zero_apply, sub_zero]
    split_ifs <;> norm_num
  · refine ⟨measurable_const.aemeasurable, ?_⟩
    rw [Measure.map_const]
    simp
    infer_instance

example : (∫ _ω : Unit,
      gaussianGramObservable (d := 2) 1 (1 : Matrix (Idx 2) (Idx 2) ℝ) 0 1 0 ∂Measure.dirac ()) -
    (∫ _ω : Unit,
      gaussianGramObservable (d := 2) 1 (0 : Matrix (Idx 2) (Idx 2) ℝ) 0 1 0 ∂Measure.dirac ()) = (1 / 2 : ℝ) := by
  norm_num [gaussianGramObservable, gaussianFeatureScores, Perspective.softmaxWeight, Fin.sum_univ_two,
    Matrix.one_apply]

end Transformer.Homogenized
