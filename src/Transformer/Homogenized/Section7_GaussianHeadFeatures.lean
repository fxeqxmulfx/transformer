module

public import Transformer.Homogenized.Section7_GaussianCoordinateLaw
public import Transformer.Homogenized.HeadConj
public import Transformer.Homogenized.Simplex
public import Transformer.Perspective.Softmax

/-!
# The actual Gaussian query and key features of the original head

ArXiv:2604.01978v1, §7, `eq:SIGMA_R` and the scores preceding it,
under assumption (G), `eq: tformers.at.initialization`.
The original initialization uses `A = WW′ᵀ`, while the Gram proof switches
to `A = W′ᵀW`. To retain the original pointwise score identity, this module
uses `h_i = W′ᵀx_i` and `g_j = Wᵀx_j`. These are the original features
used to derive the source's Gram covariance from its independently Gaussian
entries. The transpose convention is the only change to the features.

The actual feature vector is a continuous linear image of the original
entry vector. Its Gaussian law follows from the proved original entry law,
including every singular image. The exact original head score is the
inner product of these actual feature vectors; the original attention
probability is therefore the original softmax of their bilinear scores.
No covariance or score identity is assumed in the feature definition.
-/

@[expose] public section

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- Two original Gaussian feature blocks, one coordinate per original
token and head feature. Source: arXiv:2604.01978v1, §7, `eq:SIGMA_R`. -/
abbrev GaussianFeatureIdx (d n : ℕ) := Fin 2 × Idx n × Fin d

/-- The actual linear feature image of the original entry sample.
Block zero is `W′ᵀx_i`, and block one is `Wᵀx_i`, following initialization
`A = WW′ᵀ`. Source: arXiv:2604.01978v1, §7, the features in `eq:SIGMA_R`,
with the transpose convention corrected to the original initialization. -/
noncomputable def gaussianHeadFeatureMap {d n : ℕ} (X : Idx n → EucSpace d) :
    (GaussIdx d → ℝ) →L[ℝ] EuclideanSpace ℝ (GaussianFeatureIdx d n) :=
  (EuclideanSpace.equiv (GaussianFeatureIdx d n) ℝ).symm.toContinuousLinearMap.comp
    (ContinuousLinearMap.pi fun p => ∑ r : Fin d, X p.2.1 r •
      (ContinuousLinearMap.proj ((if p.1 = 0 then (2 : Fin 3) else 1), r, p.2.2) :
        (GaussIdx d → ℝ) →L[ℝ] ℝ))

/-- The actual feature coordinate is its prescribed original linear
combination of entries. Source: arXiv:2604.01978v1, §7, the
original `h_i` and `g_i` used in `eq:SIGMA_R`. -/
theorem gaussianHeadFeatureMap_apply {d n : ℕ} (X : Idx n → EucSpace d)
    (z : GaussIdx d → ℝ) (p : GaussianFeatureIdx d n) :
    gaussianHeadFeatureMap X z p = ∑ r : Fin d,
      X p.2.1 r * z ((if p.1 = 0 then (2 : Fin 3) else 1), r, p.2.2) := by
  simp [gaussianHeadFeatureMap]

/-- The linear feature image has its genuine original Gaussian law.
Source: arXiv:2604.01978v1, §7, the Gaussian shape in `eq:SIGMA_R`.
The original raw entry law is proved in `Section7_GaussianCoordinateLaw`. -/
theorem hasGaussianLaw_gaussianHeadFeatures {Ω : Type*} [MeasurableSpace Ω] {d n : ℕ}
    (P : Measure Ω) (Z : Ω → GaussIdx d → ℝ) (hZ : HasGaussianLaw Z P) (X : Idx n → EucSpace d) :
    HasGaussianLaw (fun ω => gaussianHeadFeatureMap X (Z ω)) P :=
  hZ.map (gaussianHeadFeatureMap X)

/-- The source's original independently Gaussian matrices give the
actual Gaussian feature vector on their original probability space.
Source: arXiv:2604.01978v1, §7, the original Gaussian features
preceding `eq:SIGMA_R`, with the initialization convention retained. -/
theorem hasGaussianLaw_original_head_features {Ω : Type*} [MeasurableSpace Ω] {d n : ℕ}
    (P : Measure Ω) [IsProbabilityMeasure P] (Vr Wr Wr' : Ω → Matrix (Fin d) (Fin d) ℝ)
    (hV : Measurable Vr) (hW : Measurable Wr) (hW' : Measurable Wr')
    (hind : iIndepFun (headEntries Vr Wr Wr') P) (σV σA : ℝ≥0)
    (hmV : ∀ i j, P.map (fun ω => Vr ω i j) = gaussianReal 0 (σV ^ 2))
    (hmW : ∀ i j, P.map (fun ω => Wr ω i j) = gaussianReal 0 (σA ^ 2))
    (hmW' : ∀ i j, P.map (fun ω => Wr' ω i j) = gaussianReal 0 (σA ^ 2))
    (X : Idx n → EucSpace d) :
    HasGaussianLaw (fun ω => gaussianHeadFeatureMap X (fun p => headEntries Vr Wr Wr' p ω)) P :=
  hasGaussianLaw_gaussianHeadFeatures P _
    (hasGaussianLaw_head_entry_vector P Vr Wr Wr' hV hW hW' hind σV σA hmV hmW hmW') X

/-- The actual feature coordinates are the original transposed
matrix images, rather than a different score representation.
Source: arXiv:2604.01978v1, §7, the feature definitions corrected
to initialization `A = WW′ᵀ` of `eq: tformers.at.initialization`. -/
theorem gaussianHeadFeatureMap_blocks {d n : ℕ} (X : Idx n → EucSpace d)
    (z : GaussIdx d → ℝ) (a : Fin 2) (i : Idx n) (q : Fin d) :
    gaussianHeadFeatureMap X z (a, i, q) =
      if a = 0 then Matrix.toEuclideanLin (gaussBlock 2 z).transpose (X i) q
      else Matrix.toEuclideanLin (gaussBlock 1 z).transpose (X i) q := by
  rw [gaussianHeadFeatureMap_apply]
  by_cases ha : a = 0
  · simp only [ha, ite_true]
    rw [Matrix.toLpLin_apply]
    change (∑ r : Fin d, X i r * z (2, r, q)) = ∑ r : Fin d, z (2, r, q) * X i r
    exact Finset.sum_congr rfl fun r hr => mul_comm _ _
  · simp only [ha, ite_false]
    rw [Matrix.toLpLin_apply]
    change (∑ r : Fin d, X i r * z (1, r, q)) = ∑ r : Fin d, z (1, r, q) * X i r
    exact Finset.sum_congr rfl fun r hr => mul_comm _ _

/-- The original attention score is exactly the original bilinear
feature score. Source: arXiv:2604.01978v1, §7, the score identity
preceding `eq:SIGMA_R`, with transposes fixed to the original initialization. -/
theorem inner_head_eq_gaussianFeature_score {d n : ℕ} (X : Idx n → EucSpace d)
    (z : GaussIdx d → ℝ) (i j : Idx n) :
    inner (𝕜 := ℝ) (Matrix.toEuclideanLin (gaussBlock 1 z * (gaussBlock 2 z).transpose) (X i)) (X j) =
      ∑ q : Fin d, gaussianHeadFeatureMap X z (0, i, q) * gaussianHeadFeatureMap X z (1, j, q) := by
  rw [inner_toEuclideanLin_mul_transpose, EuclideanSpace.inner_eq_star_dotProduct]
  simp only [dotProduct, star_trivial]
  apply Finset.sum_congr rfl
  intro q hq
  rw [gaussianHeadFeatureMap_blocks, gaussianHeadFeatureMap_blocks]
  simp only [ite_true, show (1 : Fin 2) ≠ 0 by decide, ite_false]
  ring

/-- The original scores as actual functions of the original Gaussian
feature vector. Source: arXiv:2604.01978v1, §7, the `βh_iᵀg_j`
scores preceding `eq:SIGMA_R`. -/
noncomputable def gaussianFeatureScores {d n : ℕ} (β : ℝ)
    (z : EuclideanSpace ℝ (GaussianFeatureIdx d n)) (i : Idx n) : Idx n → ℝ :=
  fun j => β * ∑ q : Fin d, z (0, i, q) * z (1, j, q)

/-- The actual original head probability is the original softmax
of its actual bilinear Gaussian features. Source: arXiv:2604.01978v1,
§7, the attention formula following `eq:SIGMA_R`, with the original
initialization convention retained pointwise. -/
theorem attnProb_eq_softmax_gaussianFeatures {d n : ℕ} (β : ℝ) (X : Idx n → EucSpace d)
    (z : GaussIdx d → ℝ) (i j : Idx n) :
    attnProb β (gaussBlock 1 z * (gaussBlock 2 z).transpose) X i j =
      Perspective.softmaxWeight (gaussianFeatureScores β (gaussianHeadFeatureMap X z) i) j := by
  unfold attnProb attnWeight qkMap Perspective.softmaxWeight gaussianFeatureScores
  simp only [inner_head_eq_gaussianFeature_score]

/-- A genuine nondegenerate original entry law and orthonormal
tokens witness the feature-law hypothesis; their features read the
original query and key entries exactly. -/
example : ∃ X : Idx 2 → EucSpace 2,
    HasGaussianLaw (id : (GaussIdx 2 → ℝ) → (GaussIdx 2 → ℝ)) (gaussEnsemble 2 1 1) ∧
    (∀ i, ‖X i‖ = 1) ∧
    (∀ z i q, gaussianHeadFeatureMap X z (0, i, q) = z (2, i, q)) ∧
    (∀ z i q, gaussianHeadFeatureMap X z (1, i, q) = z (1, i, q)) := by
  let X : Idx 2 → EucSpace 2 := fun i => EuclideanSpace.single i 1
  refine ⟨X, ?_, fun i => by simp [X], ?_, ?_⟩
  · simpa only [gaussEnsemble, entryVar, one_pow, ite_self] using
      hasGaussianLaw_pi_coordinate_identity (fun _ : GaussIdx 2 => (1 : ℝ≥0))
  · intro z i q
    rw [gaussianHeadFeatureMap_apply]
    simp [X, PiLp.single_apply]
  · intro z i q
    rw [gaussianHeadFeatureMap_apply]
    simp [X, PiLp.single_apply]

/-- The actual source ensemble supplies every original measurability,
independence and marginal-law premise at the standard value variance,
with a nondegenerate query/key variance and a unit-token configuration. -/
example : IsGaussianHeadLaw 2 (stdSigmaV 2) 1 (gaussHeadLaw 2 (stdSigmaV 2) 1) ∧
    (∀ i : Idx 2, ‖(EuclideanSpace.single i (1 : ℝ) : EucSpace 2)‖ = 1) := by
  exact ⟨isGaussianHeadLaw_gaussHeadLaw 2 _ _, fun i => by simp⟩

end Transformer.Homogenized
