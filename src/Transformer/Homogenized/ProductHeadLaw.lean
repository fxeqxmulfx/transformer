/-
# Homogenized Transformers — an admissible Gaussian-product head law

The sparse source of `ProductHeadSource` realizes a particularly simple law:
`V₀₀ = ±1` is a fair coin, `A₀₁ = GH` for independent standard Gaussians,
and every other entry vanishes. We prove every independence, centering,
subGaussian and pushforward clause of the source assumption.

The law below is defined on the actual three random inputs. The enlarged
coordinate space is used only to certify full entry independence.
Source: arXiv:2604.01978v1, `ass:high_order_short`, `eq:Lipschitz_measure`.
-/

import Transformer.Homogenized.ProductHeadSource

open scoped BigOperators NNReal ENNReal
open Real MeasureTheory ProbabilityTheory

namespace Transformer.Homogenized

/-- The sole nonzero attention entry maps the second axis into the first. -/
def productAttentionMatrix (r : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  Matrix.of fun i j => if i = 0 ∧ j = 1 then r else 0

/-- A value coin and a genuine product of two independent Gaussian inputs. -/
noncomputable def productHead (ω : Bool × (ℝ × ℝ)) : HeadParam 2 :=
  (radMatrix ω.1, productAttentionMatrix (ω.2.1 * ω.2.2))

/-- The concrete head law used to test uniformity in the token measure. -/
noncomputable def productHeadLaw : Measure (HeadParam 2) :=
  Measure.map productHead (fairCoin.prod gaussianProductLaw)

@[fun_prop] theorem measurable_productAttentionMatrix : Measurable productAttentionMatrix := by
  apply Measurable.of_eval
  intro i
  apply Measurable.of_eval
  intro j
  by_cases h : i = 0 ∧ j = 1
  · simp only [productAttentionMatrix, Matrix.of_apply, ite_eq_left h]
    exact measurable_id
  · simp only [productAttentionMatrix, Matrix.of_apply, ite_eq_right h]
    exact measurable_const

@[fun_prop] theorem measurable_productHead : Measurable productHead :=
  (measurable_radMatrix.comp measurable_fst).prodMk
    (measurable_productAttentionMatrix.comp
      ((measurable_fst.comp measurable_snd).mul (measurable_snd.comp measurable_snd)))

/-- The attention map reads only the query's second coordinate and returns
it along the first axis, as the actual sparse matrices require. -/
theorem productAttentionMatrix_apply (r : ℝ) (x : EucSpace 2) :
    Matrix.toEuclideanLin (productAttentionMatrix r) x =
      (r * x 1) • (EuclideanSpace.single (0 : Fin 2) (1 : ℝ) : EucSpace 2) := by
  ext i
  fin_cases i <;> simp [Matrix.toLpLin_apply, Matrix.mulVec, dotProduct,
    productAttentionMatrix]

instance : IsProbabilityMeasure productHeadLaw := by
  unfold productHeadLaw
  exact (Measure.isProbabilityMeasure_map_iff measurable_productHead.aemeasurable).mpr
    inferInstance

/-- Read the two active Gaussian coordinates from the enlarged query space. -/
def productQueryPair (ω : ProductQueryIdx → ℝ) : ℝ × ℝ :=
  (ω (true, 0, 0), ω (false, 1, 0))

@[fun_prop] theorem measurable_productQueryPair : Measurable productQueryPair :=
  (measurable_pi_apply (true, 0, 0)).prodMk (measurable_pi_apply (false, 1, 0))

/-- The two active coordinates have the actual independent Gaussian law. -/
theorem map_productQueryPair :
    Measure.map productQueryPair productQueryLaw = gaussianProductLaw := by
  have hi : iIndepFun (fun p : ProductQueryIdx => fun ω : ProductQueryIdx → ℝ => ω p)
      productQueryLaw := iIndepFun_pi (X := fun _ => (id : ℝ → ℝ)) fun _ => aemeasurable_id
  have hp := hi.indepFun (i := (true, 0, 0)) (j := (false, 1, 0)) (by decide)
  have hmap := hp.map_prod_eq_prod_map_map (measurable_pi_apply _).aemeasurable
    (measurable_pi_apply _).aemeasurable
  have hm (p : ProductQueryIdx) : Measure.map (fun ω : ProductQueryIdx → ℝ => ω p)
      productQueryLaw = gaussianReal 0 1 :=
    (measurePreserving_eval (fun _ : ProductQueryIdx => gaussianReal 0 1) p).map_eq
  change Measure.map productQueryPair productQueryLaw =
    (Measure.map (fun ω : ProductQueryIdx → ℝ => ω (true, 0, 0)) productQueryLaw).prod
      (Measure.map (fun ω : ProductQueryIdx → ℝ => ω (false, 1, 0)) productQueryLaw) at hmap
  rw [hm, hm] at hmap
  exact hmap

/-- The source's two masked matrices multiply to the displayed attention
matrix. Source: arXiv:2604.01978v1, `ass:high_order_short`. -/
theorem productQueryBlocks_mul (ω : ProductQueryIdx → ℝ) :
    productQueryBlock true ω * (productQueryBlock false ω).transpose =
      productAttentionMatrix ((productQueryPair ω).1 * (productQueryPair ω).2) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [Matrix.mul_apply, productQueryBlock, productAttentionMatrix,
      productQueryPair]

/-- Adjoining the independent value coin gives the actual three-input law. -/
theorem map_productHeadSource_pair :
    Measure.map (fun ω : ProductHeadSource => (ω.1, productQueryPair ω.2))
      productHeadSourceLaw = fairCoin.prod gaussianProductLaw := by
  have h := Measure.map_prod_map fairCoin productQueryLaw measurable_id
    measurable_productQueryPair
  have he : Prod.map (id : Bool → Bool) productQueryPair =
      (fun ω : ProductHeadSource => (ω.1, productQueryPair ω.2)) := by
    funext ω
    rfl
  rw [Measure.map_id, map_productQueryPair, he] at h
  exact h.symm

/-- The enlarged independent source pushes forward to the explicitly defined
head law. Source: arXiv:2604.01978v1, `ass:high_order_short`. -/
theorem map_productHeadSource_head :
    Measure.map (fun ω : ProductHeadSource =>
      ((radMatrix ω.1, productQueryBlock true ω.2 *
        (productQueryBlock false ω.2).transpose) : HeadParam 2))
      productHeadSourceLaw = productHeadLaw := by
  have he : (fun ω : ProductHeadSource =>
      ((radMatrix ω.1, productQueryBlock true ω.2 *
        (productQueryBlock false ω.2).transpose) : HeadParam 2)) =
      productHead ∘ (fun ω : ProductHeadSource => (ω.1, productQueryPair ω.2)) := by
    funext ω
    exact Prod.ext rfl (productQueryBlocks_mul ω.2)
  have hm : Measurable (fun ω : ProductHeadSource => (ω.1, productQueryPair ω.2)) := by
    fun_prop
  rw [he, ← Measure.map_map measurable_productHead hm, map_productHeadSource_pair]
  rfl

/-- This nonzero, unbounded query/key law satisfies every clause of the paper's
high-order assumption, with `C = σ_V = σ_A = 1` and zero means.
Source: arXiv:2604.01978v1, `ass:high_order_short`. -/
theorem isHighOrderLaw_productHead :
    IsHighOrderLaw 2 1 1 1 productHeadLaw productHeadSourceLaw
      (fun ω => radMatrix ω.1) (fun ω => productQueryBlock true ω.2)
      (fun ω => productQueryBlock false ω.2) 0 0 := by
  refine ⟨inferInstance, measurable_radMatrix.comp measurable_fst,
    (measurable_productQueryBlock true).comp measurable_snd,
    (measurable_productQueryBlock false).comp measurable_snd, ?_,
    integral_productHeadSource_value, indep_productHeadSource_value_query,
    iIndepFun_productHeadSource_values, iIndepFun_productQueryBlocks,
    fun i j => ?_, fun i j => ?_, fun i j => ?_⟩
  · simpa only [zero_add] using map_productHeadSource_head
  · simpa using hasSubgaussianMGF_productHeadSource_value i j
  · simpa using hasSubgaussianMGF_productQueryBlock true i j
  · simpa using hasSubgaussianMGF_productQueryBlock false i j

/-- The existential version of the source assumption holds for the same
actual law. Source: arXiv:2604.01978v1, `ass:high_order_short`. -/
theorem hasHighOrderLaw_productHead : HasHighOrderLaw 2 1 1 productHeadLaw :=
  ⟨1, ProductHeadSource, inferInstance, productHeadSourceLaw,
    (fun ω => radMatrix ω.1), (fun ω => productQueryBlock true ω.2),
    (fun ω => productQueryBlock false ω.2), 0, 0, one_pos, isHighOrderLaw_productHead⟩

/-- The constructed law is an explicit witness of the high-order assumption
at positive value and query/key variance proxies. -/
example : HasHighOrderLaw 2 1 1 productHeadLaw := hasHighOrderLaw_productHead

end Transformer.Homogenized
