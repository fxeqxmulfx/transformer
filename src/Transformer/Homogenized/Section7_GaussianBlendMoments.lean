module

public import Transformer.Homogenized.Section7_GaussianCoordinateMoments
public import Transformer.Homogenized.Section7_GaussianHeadFeatures

/-!
# Actual Gaussian interpolation moments

ArXiv:2604.01978v1, §7, eq:interpolation_gaussian and the Gaussian
moment calculation preceding eq:Lispchitz_bound_softmax.

The source uses dimension times variance for the feature energies.
Here these energies are computed from the actual interpolation on
its original probability space. Scalar independence eliminates the
cross moment; square integrability justifies every integral expansion.
No nonsingularity is required, and both interpolation endpoints are
included. These statements will integrate the pointwise Hessian trace
bound after the random softmax weights have been summed pointwise.
-/

@[expose] public section

open MeasureTheory ProbabilityTheory Set
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- Independent centered square-integrable scalars have the interpolated
second moment. Source: arXiv:2604.01978v1, §7, the moment step preceding
eq:Lispchitz_bound_softmax; Gaussian shape is unnecessary for this identity. -/
theorem independent_centered_blend_second_moment {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) (X Y : Ω → ℝ) (hX : MemLp X 2 P) (hY : MemLp Y 2 P)
    (hind : IndepFun X Y P) (hmX : ∫ ω, X ω ∂P = 0) (hmY : ∫ ω, Y ω ∂P = 0)
    (vX vY : ℝ) (hXX : ∫ ω, X ω ^ 2 ∂P = vX) (hYY : ∫ ω, Y ω ^ 2 ∂P = vY)
    (t : ℝ) (ht : t ∈ Icc (0 : ℝ) 1) :
    ∫ ω, (Real.sqrt t * X ω + Real.sqrt (1 - t) * Y ω) ^ 2 ∂P = t * vX + (1 - t) * vY := by
  have hiX : Integrable (fun ω => X ω ^ 2) P := by
    convert hX.integrable_mul hX using 1
    funext ω
    exact pow_two _
  have hiY : Integrable (fun ω => Y ω ^ 2) P := by
    convert hY.integrable_mul hY using 1
    funext ω
    exact pow_two _
  have hiXY : Integrable (fun ω => X ω * Y ω) P := hX.integrable_mul hY
  have hXY : ∫ ω, X ω * Y ω ∂P = 0 := by
    rw [hind.integral_fun_mul_eq_mul_integral hX.aestronglyMeasurable hY.aestronglyMeasurable,
      hmX, hmY, mul_zero]
  have he : ∀ ω, (Real.sqrt t * X ω + Real.sqrt (1 - t) * Y ω) ^ 2 =
      (Real.sqrt t) ^ 2 * X ω ^ 2 + (Real.sqrt (1 - t)) ^ 2 * Y ω ^ 2 +
        (2 * Real.sqrt t * Real.sqrt (1 - t)) * (X ω * Y ω) := fun ω => by ring
  simp_rw [he]
  have hiSum : Integrable (fun ω => (Real.sqrt t) ^ 2 * X ω ^ 2 + (Real.sqrt (1 - t)) ^ 2 * Y ω ^ 2) P := by
    exact (hiX.const_mul ((Real.sqrt t) ^ 2)).add
      (hiY.const_mul ((Real.sqrt (1 - t)) ^ 2))
  rw [integral_add hiSum (hiXY.const_mul _),
    integral_add (hiX.const_mul _) (hiY.const_mul _)]
  simp only [integral_const_mul, hXX, hYY, hXY, mul_zero, add_zero,
    Real.sq_sqrt ht.1, Real.sq_sqrt (sub_nonneg.mpr ht.2)]

/-- Each actual interpolated Gaussian coordinate is square-integrable.
Source: arXiv:2604.01978v1, §7, eq:interpolation_gaussian. -/
theorem gaussian_interp_coordinate_memLp {ι Ω : Type*} [Fintype ι] [MeasurableSpace Ω]
    (P : Measure Ω) (X Y : Ω → EuclideanSpace ℝ ι) (hX : HasGaussianLaw X P) (hY : HasGaussianLaw Y P)
    (t : ℝ) (p : ι) :
    MemLp (fun ω => (Real.sqrt t • X ω + Real.sqrt (1 - t) • Y ω) p) 2 P := by
  have hx : MemLp (fun ω => X ω p) 2 P := (hX.map (EuclideanSpace.proj p)).memLp (by norm_num)
  have hy : MemLp (fun ω => Y ω p) 2 P := (hY.map (EuclideanSpace.proj p)).memLp (by norm_num)
  exact (hx.const_mul (Real.sqrt t)).add (hy.const_mul (Real.sqrt (1 - t)))

/-- Equal centered coordinate variances are preserved throughout the closed
interpolation interval, including singular laws. Source: arXiv:2604.01978v1,
§7, the moment calculation before eq:Lispchitz_bound_softmax. -/
theorem gaussian_interp_coordinate_second_moment {ι Ω : Type*} [Fintype ι] [MeasurableSpace Ω]
    (P : Measure Ω) (X Y : Ω → EuclideanSpace ℝ ι) (hX : HasGaussianLaw X P) (hY : HasGaussianLaw Y P)
    (hind : IndepFun X Y P) (hmX : ∀ p, ∫ ω, X ω p ∂P = 0) (hmY : ∀ p, ∫ ω, Y ω p ∂P = 0)
    (σ2 : ℝ) (hXX : ∀ p, ∫ ω, X ω p ^ 2 ∂P = σ2) (hYY : ∀ p, ∫ ω, Y ω p ^ 2 ∂P = σ2)
    (t : ℝ) (ht : t ∈ Icc (0 : ℝ) 1) (p : ι) :
    ∫ ω, (Real.sqrt t • X ω + Real.sqrt (1 - t) • Y ω) p ^ 2 ∂P = σ2 := by
  have hx : MemLp (fun ω => X ω p) 2 P := (hX.map (EuclideanSpace.proj p)).memLp (by norm_num)
  have hy : MemLp (fun ω => Y ω p) 2 P := (hY.map (EuclideanSpace.proj p)).memLp (by norm_num)
  have hindp := hind.comp (EuclideanSpace.proj (𝕜 := ℝ) p).continuous.measurable
    (EuclideanSpace.proj (𝕜 := ℝ) p).continuous.measurable
  have h := independent_centered_blend_second_moment P _ _ hx hy hindp (hmX p) (hmY p)
    σ2 σ2 (hXX p) (hYY p) t ht
  have hcalc : t * σ2 + (1 - t) * σ2 = σ2 := by ring
  simpa using h.trans hcalc

/-- Actual squared interpolated coordinates are integrable, without requiring
independence. Source: arXiv:2604.01978v1, §7, eq:interpolation_gaussian. -/
theorem gaussian_interp_coordinate_sq_integrable {ι Ω : Type*} [Fintype ι] [MeasurableSpace Ω]
    (P : Measure Ω) (X Y : Ω → EuclideanSpace ℝ ι) (hX : HasGaussianLaw X P) (hY : HasGaussianLaw Y P)
    (t : ℝ) (p : ι) :
    Integrable (fun ω => (Real.sqrt t • X ω + Real.sqrt (1 - t) • Y ω) p ^ 2) P := by
  have h := gaussian_interp_coordinate_memLp P X Y hX hY t p
  convert h.integrable_mul h using 1
  funext ω
  exact pow_two _

/-- The expected squared feature energy is exactly dimension times variance.
Source: arXiv:2604.01978v1, §7, the moment step in eq:Lispchitz_bound_softmax.
The assertion concerns the actual interpolation, not an assumed Gaussian moment. -/
theorem gaussian_interp_feature_energy {Ω : Type*} [MeasurableSpace Ω] {d n : ℕ}
    (P : Measure Ω) (X Y : Ω → EuclideanSpace ℝ (GaussianFeatureIdx d n))
    (hX : HasGaussianLaw X P) (hY : HasGaussianLaw Y P) (hind : IndepFun X Y P)
    (hmX : ∀ p, ∫ ω, X ω p ∂P = 0) (hmY : ∀ p, ∫ ω, Y ω p ∂P = 0)
    (σ2 : ℝ) (hXX : ∀ p, ∫ ω, X ω p ^ 2 ∂P = σ2) (hYY : ∀ p, ∫ ω, Y ω p ^ 2 ∂P = σ2)
    (t : ℝ) (ht : t ∈ Icc (0 : ℝ) 1) (c : Fin 2) (i : Idx n) :
    ∫ ω, ∑ q : Fin d, (Real.sqrt t • X ω + Real.sqrt (1 - t) • Y ω) (c, i, q) ^ 2 ∂P =
      (d : ℝ) * σ2 := by
  rw [integral_finsetSum _ fun q hq => gaussian_interp_coordinate_sq_integrable P X Y hX hY t (c, i, q)]
  simp only [gaussian_interp_coordinate_second_moment P X Y hX hY hind hmX hmY σ2 hXX hYY t ht,
    Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

/-- Summing the actual feature energies over the original token indices gives
exactly the token count times dimension times variance. Source: arXiv:2604.01978v1,
§7, the moment step in eq:Lispchitz_bound_softmax. -/
theorem gaussian_interp_block_energy {Ω : Type*} [MeasurableSpace Ω] {d n : ℕ}
    (P : Measure Ω) (X Y : Ω → EuclideanSpace ℝ (GaussianFeatureIdx d n))
    (hX : HasGaussianLaw X P) (hY : HasGaussianLaw Y P) (hind : IndepFun X Y P)
    (hmX : ∀ p, ∫ ω, X ω p ∂P = 0) (hmY : ∀ p, ∫ ω, Y ω p ∂P = 0)
    (σ2 : ℝ) (hXX : ∀ p, ∫ ω, X ω p ^ 2 ∂P = σ2) (hYY : ∀ p, ∫ ω, Y ω p ^ 2 ∂P = σ2)
    (t : ℝ) (ht : t ∈ Icc (0 : ℝ) 1) (c : Fin 2) :
    ∫ ω, ∑ i : Idx n, ∑ q : Fin d,
      (Real.sqrt t • X ω + Real.sqrt (1 - t) • Y ω) (c, i, q) ^ 2 ∂P =
        (n : ℝ) * (d : ℝ) * σ2 := by
  rw [integral_finsetSum _ fun i hi => integrable_finsetSum _ fun q hq =>
    gaussian_interp_coordinate_sq_integrable P X Y hX hY t (c, i, q)]
  simp only [gaussian_interp_feature_energy P X Y hX hY hind hmX hmY σ2 hXX hYY t ht,
    Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_assoc]

/-- Independent standard Gaussian scalars realize the scalar moment premises. -/
example : ∫ ω : ℝ × ℝ, (Real.sqrt (1 / 2) * ω.1 + Real.sqrt (1 - 1 / 2) * ω.2) ^ 2
    ∂((gaussianReal 0 1).prod (gaussianReal 0 1)) = 1 := by
  let P := (gaussianReal 0 1).prod (gaussianReal 0 1)
  have hf : P.map (Prod.fst : ℝ × ℝ → ℝ) = gaussianReal 0 1 := measurePreserving_fst.map_eq
  have hg : P.map (Prod.snd : ℝ × ℝ → ℝ) = gaussianReal 0 1 := measurePreserving_snd.map_eq
  have hF : HasGaussianLaw (Prod.fst : ℝ × ℝ → ℝ) P := by
    refine ⟨measurable_fst.aemeasurable, ?_⟩
    rw [hf]
    infer_instance
  have hG : HasGaussianLaw (Prod.snd : ℝ × ℝ → ℝ) P := by
    refine ⟨measurable_snd.aemeasurable, ?_⟩
    rw [hg]
    infer_instance
  simpa using independent_centered_blend_second_moment P Prod.fst Prod.snd
    (hF.memLp (by norm_num)) (hG.memLp (by norm_num)) (indepFun_prod measurable_id measurable_id)
    (integral_eq_zero_of_map_eq_gaussianReal measurable_fst hf)
    (integral_eq_zero_of_map_eq_gaussianReal measurable_snd hg) 1 1
    (integral_sq_of_map_eq_gaussianReal measurable_fst hf)
    (integral_sq_of_map_eq_gaussianReal measurable_snd hg) (1 / 2) (by norm_num)

/-- Two independent rank-one Gaussian vectors with positive variance in every
coordinate satisfy all interpolation moment assumptions. Their singular laws
also demonstrate that no covariance invertibility premise is required. -/
example : ∃ X Y : (ℝ × ℝ) → EuclideanSpace ℝ (GaussianFeatureIdx 2 2),
    HasGaussianLaw X ((gaussianReal 0 1).prod (gaussianReal 0 1)) ∧
    HasGaussianLaw Y ((gaussianReal 0 1).prod (gaussianReal 0 1)) ∧
    IndepFun X Y ((gaussianReal 0 1).prod (gaussianReal 0 1)) ∧
    (∀ p, ∫ ω, X ω p ∂((gaussianReal 0 1).prod (gaussianReal 0 1)) = 0) ∧
    (∀ p, ∫ ω, Y ω p ∂((gaussianReal 0 1).prod (gaussianReal 0 1)) = 0) ∧
    (∀ p, ∫ ω, X ω p ^ 2 ∂((gaussianReal 0 1).prod (gaussianReal 0 1)) = 1) ∧
    (∀ p, ∫ ω, Y ω p ^ 2 ∂((gaussianReal 0 1).prod (gaussianReal 0 1)) = 1) ∧
    ((1 / 2 : ℝ) ∈ Icc (0 : ℝ) 1) := by
  let μ := gaussianReal 0 1
  let P := μ.prod μ
  let v : EuclideanSpace ℝ (GaussianFeatureIdx 2 2) := WithLp.toLp 2 (fun _ => (1 : ℝ))
  let L : ℝ →L[ℝ] EuclideanSpace ℝ (GaussianFeatureIdx 2 2) :=
    (ContinuousLinearMap.id ℝ ℝ).smulRight v
  let X (ω : ℝ × ℝ) := L ω.1
  let Y (ω : ℝ × ℝ) := L ω.2
  have hfm : P.map (Prod.fst : ℝ × ℝ → ℝ) = gaussianReal 0 1 :=
    (measurePreserving_fst (μ := μ) (ν := μ)).map_eq
  have hsm : P.map (Prod.snd : ℝ × ℝ → ℝ) = gaussianReal 0 1 :=
    (measurePreserving_snd (μ := μ) (ν := μ)).map_eq
  have hf : HasGaussianLaw (Prod.fst : ℝ × ℝ → ℝ) P := by
    refine ⟨measurable_fst.aemeasurable, ?_⟩
    rw [hfm]
    infer_instance
  have hg : HasGaussianLaw (Prod.snd : ℝ × ℝ → ℝ) P := by
    refine ⟨measurable_snd.aemeasurable, ?_⟩
    rw [hsm]
    infer_instance
  have hx (ω : ℝ × ℝ) (p : GaussianFeatureIdx 2 2) : X ω p = ω.1 := by simp [X, L, v]
  have hy (ω : ℝ × ℝ) (p : GaussianFeatureIdx 2 2) : Y ω p = ω.2 := by simp [Y, L, v]
  refine ⟨X, Y, hf.map L, hg.map L,
    indepFun_prod L.continuous.measurable L.continuous.measurable, ?_, ?_, ?_, ?_, by norm_num⟩
  · intro p
    simp only [hx]
    exact integral_eq_zero_of_map_eq_gaussianReal measurable_fst hfm
  · intro p
    simp only [hy]
    exact integral_eq_zero_of_map_eq_gaussianReal measurable_snd hsm
  · intro p
    simp only [hx]
    exact integral_sq_of_map_eq_gaussianReal measurable_fst hfm
  · intro p
    simp only [hy]
    exact integral_sq_of_map_eq_gaussianReal measurable_snd hsm

end Transformer.Homogenized
