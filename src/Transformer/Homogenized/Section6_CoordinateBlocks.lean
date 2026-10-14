/-
# Actual independent coordinate blocks

Source: arXiv:2604.01978v1, §6.3, `lem:Hanson_wright`.
The off-diagonal argument separates a subset of the original coordinates
from its complement. Their actual projected vectors are independent,
retain coordinate independence and the original subgaussian proxy, and
have the genuine product joint law. Coordinates outside either block
are zero. This includes empty blocks and keeps the original centering
and random vector rather than postulating a conditional MGF.
-/

import Transformer.Homogenized.Section6_SubgaussianGaussianComparison

open Real MeasureTheory ProbabilityTheory
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- An actual subgaussian MGF remains bounded after enlarging its
variance proxy. Source: arXiv:2604.01978v1, §6.3, the common
coordinate proxy in `lem:Hanson_wright`. -/
theorem hasSubgaussianMGF_enlarge_proxy {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) (X : Ω → ℝ) (c d : ℝ≥0) (hX : HasSubgaussianMGF X c P)
    (hcd : c ≤ d) : HasSubgaussianMGF X d P := by
  refine ⟨hX.integrable_exp_mul, fun t => (hX.mgf_le t).trans ?_⟩
  apply Real.exp_le_exp.mpr
  gcongr

/-- The original coordinate projection has its exact coordinate action.
Source: arXiv:2604.01978v1, §6.3, the block mask for `lem:Hanson_wright`. -/
theorem coordinateMask_toEuclideanCLM_apply {n : ℕ} (s : Finset (Fin n)) (z : EucSpace n) (i : Fin n) :
    (Matrix.toEuclideanCLM (𝕜 := ℝ) (coordinateMask s) z) i = if i ∈ s then z i else 0 := by
  rw [Matrix.ofLp_toEuclideanCLM, coordinateMask, Matrix.mulVec_diagonal]
  split_ifs <;> simp

/-- Every actual masked coordinate retains the original common
subgaussian proxy, including the zero coordinates outside the block.
Source: arXiv:2604.01978v1, §6.3, `lem:Hanson_wright`. -/
theorem hasSubgaussianMGF_coordinateMask {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) [IsProbabilityMeasure P] (Z : Ω → EucSpace n) (c : ℝ≥0)
    (hZ : ∀ i : Fin n, HasSubgaussianMGF (fun ω => Z ω i) c P) (s : Finset (Fin n)) :
    ∀ i : Fin n, HasSubgaussianMGF
      (fun ω => (Matrix.toEuclideanCLM (𝕜 := ℝ) (coordinateMask s) (Z ω)) i) c P := by
  intro i
  simp only [coordinateMask_toEuclideanCLM_apply]
  by_cases hi : i ∈ s
  · simpa only [ite_eq_left hi] using hZ i
  · simp only [ite_eq_right hi]
    exact hasSubgaussianMGF_enlarge_proxy P (fun _ => (0 : ℝ)) 0 c HasSubgaussianMGF.fun_zero (by positivity)

/-- Applying the actual mask preserves independence of all original
coordinates. Source: arXiv:2604.01978v1, §6.3, `lem:Hanson_wright`. -/
theorem iIndepFun_coordinateMask {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) (Z : Ω → EucSpace n)
    (hindep : iIndepFun (fun i : Fin n => fun ω => Z ω i) P) (s : Finset (Fin n)) :
    iIndepFun (fun i : Fin n => fun ω =>
      (Matrix.toEuclideanCLM (𝕜 := ℝ) (coordinateMask s) (Z ω)) i) P := by
  simpa only [Function.comp_def, coordinateMask_toEuclideanCLM_apply] using
    hindep.comp (fun (i : Fin n) (x : ℝ) => if i ∈ s then x else 0)
      (fun i => by split_ifs <;> fun_prop)

/-- The actual subset and complement coordinate projections are
independent vectors. Source: arXiv:2604.01978v1, §6.3,
the two original blocks in the decoupling proof of `lem:Hanson_wright`. -/
theorem indepFun_coordinateMasks_compl {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) (Z : Ω → EucSpace n)
    (hindep : iIndepFun (fun i : Fin n => fun ω => Z ω i) P)
    (hZ : ∀ i : Fin n, AEMeasurable (fun ω => Z ω i) P) (s : Finset (Fin n)) :
    IndepFun (fun ω => Matrix.toEuclideanCLM (𝕜 := ℝ) (coordinateMask s) (Z ω))
      (fun ω => Matrix.toEuclideanCLM (𝕜 := ℝ) (coordinateMask sᶜ) (Z ω)) P := by
  let pad (t : Finset (Fin n)) (x : t → ℝ) : EucSpace n :=
    WithLp.toLp 2 (fun i => if hi : i ∈ t then x ⟨i, hi⟩ else 0)
  have hm : ∀ t : Finset (Fin n), Measurable (pad t) := by
    intro t
    apply (WithLp.measurable_toLp 2 (Fin n → ℝ)).comp
    apply measurable_pi_iff.mpr
    intro i
    dsimp only [pad]
    split_ifs <;> fun_prop
  have he : ∀ t : Finset (Fin n), (fun ω => pad t (fun i : t => Z ω i)) =
      (fun ω => Matrix.toEuclideanCLM (𝕜 := ℝ) (coordinateMask t) (Z ω)) := by
    intro t
    funext ω
    apply WithLp.ofLp_injective 2
    funext i
    simp only [pad, coordinateMask_toEuclideanCLM_apply]
    split_ifs <;> rfl
  have h := (iIndepFun.indepFun_finset₀ s sᶜ (disjoint_compl_right : Disjoint s sᶜ) hindep hZ).comp (hm s) (hm sᶜ)
  simpa only [Function.comp_def, he] using h

/-- The actual original projected pair has the product of its genuine
marginal laws. Source: arXiv:2604.01978v1, §6.3, the block expectation
factorization in the proof of `lem:Hanson_wright`. -/
theorem hasLaw_coordinateMask_pair {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) [IsProbabilityMeasure P] (Z : Ω → EucSpace n)
    (hindep : iIndepFun (fun i : Fin n => fun ω => Z ω i) P)
    (hZ : ∀ i : Fin n, AEMeasurable (fun ω => Z ω i) P) (s : Finset (Fin n)) :
    HasLaw (fun ω => (Matrix.toEuclideanCLM (𝕜 := ℝ) (coordinateMask s) (Z ω),
      Matrix.toEuclideanCLM (𝕜 := ℝ) (coordinateMask sᶜ) (Z ω)))
      ((P.map (fun ω => Matrix.toEuclideanCLM (𝕜 := ℝ) (coordinateMask s) (Z ω))).prod
        (P.map (fun ω => Matrix.toEuclideanCLM (𝕜 := ℝ) (coordinateMask sᶜ) (Z ω)))) P := by
  have hz := aemeasurable_eucSpace_of_coordinates P Z hZ
  have hleft : AEMeasurable (fun ω => Matrix.toEuclideanCLM (𝕜 := ℝ) (coordinateMask s) (Z ω)) P :=
    (Matrix.toEuclideanCLM (𝕜 := ℝ) (coordinateMask s)).continuous.measurable.comp_aemeasurable hz
  have hright : AEMeasurable (fun ω => Matrix.toEuclideanCLM (𝕜 := ℝ) (coordinateMask sᶜ) (Z ω)) P :=
    (Matrix.toEuclideanCLM (𝕜 := ℝ) (coordinateMask sᶜ)).continuous.measurable.comp_aemeasurable hz
  exact ⟨hleft.prodMk hright, (indepFun_coordinateMasks_compl P Z hindep hZ s).map_prod_eq_prod_map_map hleft hright⟩

/-- The original masked quadratic form equals the bilinear form of
the two actual coordinate blocks. Source: arXiv:2604.01978v1,
§6.3, the decoupling term in `lem:Hanson_wright`. -/
theorem cross_mask_quadratic_eq_inner {n : ℕ}
    (B : Matrix (Fin n) (Fin n) ℝ) (s : Finset (Fin n)) (z : EucSpace n) :
    hansonWrightQuadraticForm (coordinateMask s * B * coordinateMask sᶜ) z =
      inner ℝ (Matrix.toEuclideanCLM (𝕜 := ℝ) (coordinateMask s) z)
        (Matrix.toEuclideanCLM (𝕜 := ℝ) B (Matrix.toEuclideanCLM (𝕜 := ℝ) (coordinateMask sᶜ) z)) := by
  rw [Matrix.inner_toEuclideanCLM]
  simp only [hansonWrightQuadraticForm, cross_mask_apply, coordinateMask_toEuclideanCLM_apply,
    dotProduct, Matrix.mulVec, Finset.mul_sum, Finset.mem_compl]
  apply Finset.sum_congr rfl
  intro i _
  apply Finset.sum_congr rfl
  intro j _
  split_ifs <;> simp_all
  ring

/-- The original cross block has actual expectation zero, so the
decoupling term requires no artificial centering. Source:
arXiv:2604.01978v1, §6.3, the off-diagonal part of `lem:Hanson_wright`. -/
theorem integral_cross_mask_quadratic_eq_zero {Ω : Type*} [MeasurableSpace Ω] {n : ℕ}
    (P : Measure Ω) (Z : Ω → EucSpace n) (c : ℝ≥0)
    (hindep : iIndepFun (fun i : Fin n => fun ω => Z ω i) P)
    (hZ : ∀ i : Fin n, HasSubgaussianMGF (fun ω => Z ω i) c P)
    (B : Matrix (Fin n) (Fin n) ℝ) (s : Finset (Fin n)) :
    (∫ ω, hansonWrightQuadraticForm (coordinateMask s * B * coordinateMask sᶜ) (Z ω) ∂P) = 0 := by
  rw [integral_hansonWrightQuadraticForm P Z c hindep hZ]
  apply Finset.sum_eq_zero
  intro i _
  have hd : (coordinateMask s * B * coordinateMask sᶜ) i i = 0 := by rw [cross_mask_apply]; simp
  simp only [hd, zero_mul]

/-- Genuine independent standard coordinates witness the block
hypotheses; a singleton subset gives two nonempty original blocks.
The same law also witnesses proxy enlargement from one to two. -/
example : iIndepFun (fun i : Fin 2 => fun z : Fin 2 → ℝ => z i)
      (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1)) ∧
    (∀ i : Fin 2, HasSubgaussianMGF (fun z : Fin 2 → ℝ => z i) 1
      (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1))) ∧
    (∀ i : Fin 2, AEMeasurable (fun z : Fin 2 → ℝ => z i)
      (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1))) ∧
    IsProbabilityMeasure (Measure.pi (fun _ : Fin 2 => gaussianReal 0 1)) ∧ (1 : ℝ≥0) ≤ 2 ∧
    (0 : Fin 2) ∈ ({0} : Finset (Fin 2)) ∧ (1 : Fin 2) ∈ ({0} : Finset (Fin 2))ᶜ := by
  refine ⟨?_, hasSubgaussianMGF_standardProduct_coordinate,
    fun i => (hasSubgaussianMGF_standardProduct_coordinate i).aemeasurable, inferInstance,
    by norm_num, by decide, by decide⟩
  simpa only [id_eq] using iIndepFun_pi (μ := fun _ : Fin 2 => gaussianReal 0 1)
    (X := fun _ => id) (fun i => measurable_id.aemeasurable)

end Transformer.Homogenized
