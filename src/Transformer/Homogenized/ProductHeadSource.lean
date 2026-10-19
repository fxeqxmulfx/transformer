module

public import Transformer.Homogenized.RademacherLaw
public import Transformer.Homogenized.GaussianProductTail
public import Mathlib.Analysis.Matrix.MeasurableSpace
public import Mathlib.MeasureTheory.Constructions.Pi

/-!
# Homogenized Transformers — independent sparse Gaussian query/key weights

A fair coin supplies the centered value entry `V₀₀ = ±1`. Independent
standard Gaussians supply `W₀₀` and `W'₁₀`; all other entries are zero.
Thus `A = W W'ᵀ` has the single entry `A₀₁ = W₀₀ W'₁₀`.
The unused coordinates of the product probability space allow the full
entry-independence condition to be inherited by deterministic coordinate
maps, including the zero entries.

This is a concrete source allowed by arXiv:2604.01978v1,
`ass:high_order_short`, for testing `eq:Lipschitz_measure`.
-/

@[expose] public section

open scoped BigOperators NNReal ENNReal
open Real MeasureTheory ProbabilityTheory

namespace Transformer.Homogenized

/-- The source coordinates corresponding to all query/key matrix entries. -/
abbrev ProductQueryIdx := Bool × Fin 2 × Fin 2

/-- Independent standard Gaussian coordinates before deterministic masking. -/
noncomputable def productQueryLaw : Measure (ProductQueryIdx → ℝ) :=
  Measure.pi fun _ => gaussianReal 0 1

instance : IsProbabilityMeasure productQueryLaw := by
  unfold productQueryLaw
  infer_instance

/-- The coin and the Gaussian coordinates are independent. -/
abbrev ProductHeadSource := Bool × (ProductQueryIdx → ℝ)

noncomputable def productHeadSourceLaw : Measure ProductHeadSource :=
  fairCoin.prod productQueryLaw

instance : IsProbabilityMeasure productHeadSourceLaw := by
  unfold productHeadSourceLaw
  infer_instance

/-- `W` has entry `(0,0)` and `W'` has entry `(1,0)`; the other entries vanish. -/
def productQueryBlock (b : Bool) (ω : ProductQueryIdx → ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  Matrix.of fun i j => if i = (if b then 0 else 1) ∧ j = 0 then ω (b, i, j) else 0

/-- Each masked query/key matrix is measurable in the source. -/
theorem measurable_productQueryBlock (b : Bool) : Measurable (productQueryBlock b) := by
  apply Measurable.of_eval
  intro i
  apply Measurable.of_eval
  intro j
  by_cases h : i = (if b then 0 else 1) ∧ j = 0
  · simpa only [productQueryBlock, Matrix.of_apply, ite_eq_left h] using measurable_pi_apply (b, i, j)
  · simpa only [productQueryBlock, Matrix.of_apply, ite_eq_right h] using
      (measurable_const : Measurable fun _ : ProductQueryIdx → ℝ => (0 : ℝ))

/-- Every raw query/key coordinate has the actual standard Gaussian law. -/
theorem map_productHeadSource_coordinate (p : ProductQueryIdx) :
    Measure.map (fun ω : ProductHeadSource => ω.2 p) productHeadSourceLaw = gaussianReal 0 1 :=
  ((measurePreserving_eval (fun _ : ProductQueryIdx => gaussianReal 0 1) p).comp
    (measurePreserving_snd (μ := fairCoin))).map_eq

/-- The raw query/key coordinates remain independent after adjoining the
independent value coin. Source: arXiv:2604.01978v1, `ass:high_order_short`. -/
theorem iIndepFun_productHeadSource_coordinates :
    iIndepFun (fun p : ProductQueryIdx => fun ω : ProductHeadSource => ω.2 p)
      productHeadSourceLaw := by
  apply (iIndepFun_iff_map_fun_eq_pi_map
    (fun p => ((measurable_pi_apply p).comp measurable_snd).aemeasurable)).mpr
  change Measure.map (Prod.snd : ProductHeadSource → ProductQueryIdx → ℝ)
    productHeadSourceLaw = Measure.pi (fun p => Measure.map
      (fun ω : ProductHeadSource => ω.2 p) productHeadSourceLaw)
  simp only [map_productHeadSource_coordinate]
  exact (measurePreserving_snd (μ := fairCoin) (ν := productQueryLaw)).map_eq

/-- Deterministic masks preserve independence of the combined `W,W'` entry
family. Source: arXiv:2604.01978v1, `ass:high_order_short`. -/
theorem iIndepFun_productQueryBlocks :
    iIndepFun (fun p : ProductQueryIdx => fun ω : ProductHeadSource =>
      if p.1 then productQueryBlock true ω.2 p.2.1 p.2.2
      else productQueryBlock false ω.2 p.2.1 p.2.2) productHeadSourceLaw := by
  let mask (p : ProductQueryIdx) (x : ℝ) : ℝ :=
    if p.2.1 = (if p.1 then 0 else 1) ∧ p.2.2 = 0 then x else 0
  have hm : ∀ p, Measurable (mask p) := by
    intro p
    dsimp only [mask]
    split_ifs <;> fun_prop
  have h := iIndepFun_productHeadSource_coordinates.comp mask hm
  convert h using 1
  funext p ω
  rcases p with ⟨b, i, j⟩
  cases b <;> rfl

/-- The standard Gaussian coordinate has variance proxy exactly `1`. -/
theorem hasSubgaussianMGF_gaussian_unit : HasSubgaussianMGF id 1 (gaussianReal 0 1) := by
  refine ⟨fun t => integrable_exp_mul_gaussianReal t, fun t => ?_⟩
  rw [mgf_id_gaussianReal]
  simp

/-- A zero coordinate also satisfies the common positive variance proxy. -/
theorem hasSubgaussianMGF_zero_unit {Ω : Type*} [MeasurableSpace Ω]
    (P : Measure Ω) [IsProbabilityMeasure P] : HasSubgaussianMGF (fun _ : Ω => 0) 1 P := by
  refine ⟨fun t => by simp, fun t => ?_⟩
  simpa [mgf] using (Real.one_le_exp (by positivity : 0 ≤ t ^ 2 / 2))

/-- The positive-proxy zero-coordinate hypothesis is satisfiable. -/
example : IsProbabilityMeasure (Measure.dirac ()) := inferInstance

/-- Every masked entry is subGaussian with the same variance proxy `1`.
Source: arXiv:2604.01978v1, `ass:high_order_short`. -/
theorem hasSubgaussianMGF_productQueryBlock (b : Bool) (i j : Fin 2) :
    HasSubgaussianMGF (fun ω : ProductHeadSource => productQueryBlock b ω.2 i j)
      1 productHeadSourceLaw := by
  by_cases h : i = (if b then 0 else 1) ∧ j = 0
  · simp only [productQueryBlock, Matrix.of_apply, ite_eq_left h]
    apply (HasSubgaussianMGF.id_map_iff
      (((measurable_pi_apply (b, i, j)).comp measurable_snd).aemeasurable)).mp
    rw [map_productHeadSource_coordinate]
    exact hasSubgaussianMGF_gaussian_unit
  · simp only [productQueryBlock, Matrix.of_apply, ite_eq_right h]
    exact hasSubgaussianMGF_zero_unit productHeadSourceLaw

/-- The value coin is independent of both entire query/key matrices.
Source: arXiv:2604.01978v1, `ass:high_order_short`. -/
theorem indep_productHeadSource_value_query :
    IndepFun (fun ω : ProductHeadSource => radMatrix ω.1)
      (fun ω => (productQueryBlock true ω.2, productQueryBlock false ω.2))
      productHeadSourceLaw :=
  indepFun_prod measurable_radMatrix
    ((measurable_productQueryBlock true).prodMk (measurable_productQueryBlock false))

/-- Only one value entry is random, so the full value-entry family is
independent. Source: arXiv:2604.01978v1, `ass:high_order_short`. -/
theorem iIndepFun_productHeadSource_values :
    iIndepFun (fun p : Fin 2 × Fin 2 => fun ω : ProductHeadSource => radMatrix ω.1 p.1 p.2)
      productHeadSourceLaw := by
  refine iIndepFun_of_const_of_ne _ (0, 0) ?_
  intro p hp
  refine ⟨0, funext fun ω => (radMatrix_apply ω.1 p.1 p.2).trans (ite_eq_right ?_)⟩
  rintro ⟨h1, h2⟩
  exact hp (Prod.ext h1 h2)

/-- The centered value entries have the same variance proxy after adjoining
the independent Gaussian coordinates. Source: arXiv:2604.01978v1,
`ass:high_order_short`. -/
theorem hasSubgaussianMGF_productHeadSource_value (i j : Fin 2) :
    HasSubgaussianMGF (fun ω : ProductHeadSource => radMatrix ω.1 i j)
      1 productHeadSourceLaw := by
  have h := isHighOrderLaw_rademacher.subgaussianV i j
  norm_num at h
  have hm : Measure.map (Prod.fst : ProductHeadSource → Bool) productHeadSourceLaw = fairCoin :=
    (measurePreserving_fst (ν := productQueryLaw)).map_eq
  rw [← hm] at h
  exact HasSubgaussianMGF.of_map measurable_fst.aemeasurable h

/-- The value entries are centered in the full source law. Source:
arXiv:2604.01978v1, `ass:high_order_short`. -/
theorem integral_productHeadSource_value (i j : Fin 2) :
    ∫ ω : ProductHeadSource, radMatrix ω.1 i j ∂productHeadSourceLaw = 0 := by
  have hm : Measure.map (Prod.fst : ProductHeadSource → Bool) productHeadSourceLaw = fairCoin :=
    (measurePreserving_fst (ν := productQueryLaw)).map_eq
  rw [← integral_map measurable_fst.aemeasurable
    ((measurable_of_countable (fun b => radMatrix b i j)).aestronglyMeasurable), hm]
  exact isHighOrderLaw_rademacher.centered i j

end Transformer.Homogenized
