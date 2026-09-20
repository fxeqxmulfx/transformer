/-
# Homogenized Transformers — a nondegenerate weight law

`ass:high_order_short` of arXiv:2604.01978v1, *Homogenized Transformers*, is
satisfiable by more than the point mass: `isHighOrderLaw_dirac_zero` witnesses
that the assumption is not contradictory, but under `ρ* = δ_0` the fluctuation
`ξ_θ[μ]` vanishes identically and every statement about the diffusion kernel
`G_μ` of `eq:G_def` holds vacuously.

This file builds the smallest weight law that is not degenerate.  The source
space is a fair coin, the value matrix has one Rademacher entry `V₁₁ = ±1` and
is `0` elsewhere, and the query-key matrix is `A ≡ 0`.  The independence
hypotheses hold because one entry is random and all the others are constant;
the subGaussian hypotheses hold by Hoeffding's lemma, with `σ_V = 1`,
`σ_A = 0`, `C = 1`.

Source: arXiv:2604.01978v1, `ass:high_order_short`.
-/

import Transformer.Homogenized.RandomChain

open scoped BigOperators NNReal ENNReal
open Real MeasureTheory ProbabilityTheory

namespace Transformer
namespace Homogenized

/-! ### Independence when only one member of a family is random -/

/-- A family of functions of which all but one are constant is independent: a
constant function has only `∅` and the whole space as preimages, so for every
finite subfamily both the intersection of the preimages and the product of
their measures collapse onto the one random member.

This is `iIndepFun_of_unit` with the one-point space replaced by one degree of
freedom, and it is what makes the independence hypotheses of
`ass:high_order_short` satisfiable at a law with exactly one random entry. -/
theorem iIndepFun_of_const_of_ne {Ω ι : Type*} [MeasurableSpace Ω] {P : Measure Ω}
    [IsProbabilityMeasure P] {β : ι → Type*} [∀ i, MeasurableSpace (β i)]
    (f : ∀ i, Ω → β i) (i₀ : ι) (hf : ∀ i, i ≠ i₀ → ∃ c, f i = fun _ => c) :
    iIndepFun f P := by
  classical
  have key : ∀ i, i ≠ i₀ → ∀ s : Set (β i),
      f i ⁻¹' s = Set.univ ∨ f i ⁻¹' s = ∅ := by
    intro i hi s
    obtain ⟨c, hc⟩ := hf i hi
    rw [hc]
    by_cases h : c ∈ s
    · exact Or.inl (Set.eq_univ_of_forall fun _ => h)
    · exact Or.inr (Set.eq_empty_of_forall_notMem fun _ => h)
  rw [iIndepFun_iff_measure_inter_preimage_eq_mul]
  intro s g _
  by_cases hall : ∀ i ∈ s, i ≠ i₀ → f i ⁻¹' g i = Set.univ
  · by_cases hi₀ : i₀ ∈ s
    · have hinter : (⋂ i ∈ s, f i ⁻¹' g i) = f i₀ ⁻¹' g i₀ := by
        refine Set.Subset.antisymm (Set.iInter₂_subset i₀ hi₀) ?_
        refine Set.subset_iInter₂ fun i hi => ?_
        rcases eq_or_ne i i₀ with rfl | hne
        · exact subset_rfl
        · rw [hall i hi hne]; exact Set.subset_univ _
      rw [hinter, Finset.prod_eq_single i₀
        (fun b hb hne => by rw [hall b hb hne, measure_univ]) (fun h => absurd hi₀ h)]
    · have hne : ∀ i ∈ s, i ≠ i₀ := fun i hi h => hi₀ (h ▸ hi)
      have hinter : (⋂ i ∈ s, f i ⁻¹' g i) = Set.univ :=
        Set.eq_univ_of_forall fun ω => Set.mem_iInter₂.mpr fun i hi => by
          rw [hall i hi (hne i hi)]; trivial
      rw [hinter, measure_univ,
        Finset.prod_congr rfl (fun i hi => by rw [hall i hi (hne i hi), measure_univ]),
        Finset.prod_const_one]
  · push Not at hall
    obtain ⟨i, his, hi, hu⟩ := hall
    have hempty : f i ⁻¹' g i = ∅ := ((key i hi (g i)).resolve_left hu)
    rw [Finset.prod_eq_zero his (by rw [hempty, measure_empty])]
    exact measure_mono_null (Set.iInter₂_subset i his) (by rw [hempty, measure_empty])

/-! ### The fair coin -/

/-- The fair coin on `Bool`, the source space of the Rademacher weight law. -/
noncomputable def fairCoin : Measure Bool :=
  (2 : ℝ≥0∞)⁻¹ • (Measure.dirac true + Measure.dirac false)

instance : IsProbabilityMeasure fairCoin := by
  constructor
  rw [fairCoin, Measure.smul_apply, Measure.add_apply, measure_univ, measure_univ,
    smul_eq_mul]
  rw [show (1 : ℝ≥0∞) + 1 = 2 by norm_num, ENNReal.inv_mul_cancel (by norm_num) (by norm_num)]

/-- An integral against the fair coin is the average of the two values. -/
theorem integral_fairCoin {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [CompleteSpace E] (f : Bool → E) :
    ∫ ω, f ω ∂fairCoin = (2 : ℝ)⁻¹ • (f true + f false) := by
  rw [fairCoin, integral_smul_measure,
    integral_add_measure (integrable_dirac enorm_lt_top) (integrable_dirac enorm_lt_top),
    integral_dirac, integral_dirac]
  norm_num

/-! ### The law -/

/-- The value matrix of the Rademacher weight law: the entry `(0,0)` is the
sign of the coin, every other entry is `0`. -/
noncomputable def radMatrix (b : Bool) : Matrix (Fin 2) (Fin 2) ℝ :=
  Matrix.of fun i j => if i = 0 ∧ j = 0 then (if b then (1 : ℝ) else -1) else 0

theorem radMatrix_apply (b : Bool) (i j : Fin 2) :
    radMatrix b i j = if i = 0 ∧ j = 0 then (if b then (1 : ℝ) else -1) else 0 := rfl

theorem measurable_radMatrix : Measurable radMatrix := measurable_of_countable _

/-- The Rademacher weight law `ρ* = ½δ_{(V₊,0)} + ½δ_{(V₋,0)}`: the value
matrix carries one `±1` entry and the query-key matrix is `0`. -/
noncomputable def radLaw : Measure (HeadParam 2) :=
  (2 : ℝ≥0∞)⁻¹ • (Measure.dirac ((radMatrix true, 0) : HeadParam 2) +
    Measure.dirac ((radMatrix false, 0) : HeadParam 2))

/-- An integral against the Rademacher weight law is the average of the two
values. -/
theorem integral_radLaw {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [CompleteSpace E] (f : HeadParam 2 → E) :
    ∫ θ, f θ ∂radLaw
      = (2 : ℝ)⁻¹ • (f (radMatrix true, 0) + f (radMatrix false, 0)) := by
  rw [radLaw, integral_smul_measure,
    integral_add_measure (integrable_dirac enorm_lt_top) (integrable_dirac enorm_lt_top),
    integral_dirac, integral_dirac]
  norm_num

/-- **`ass:high_order_short` at a nondegenerate law.**  The Rademacher weight
law satisfies the assumption with `C = 1`, `σ_V = 1`, `σ_A = 0`: the value
matrix is centered with one entry of variance proxy `1`, the query-key matrix
is deterministically `0`, and every independence hypothesis holds because at
most one of the entries in question is random.

Source: arXiv:2604.01978v1, `ass:high_order_short`. -/
theorem isHighOrderLaw_rademacher :
    IsHighOrderLaw 2 1 1 0 radLaw fairCoin radMatrix 0 0 0 0 := by
  have hcentered : ∀ i j : Fin 2, ∫ ω, radMatrix ω i j ∂fairCoin = 0 := by
    intro i j
    rw [integral_fairCoin]
    simp only [radMatrix_apply, smul_eq_mul]
    split_ifs <;> simp_all
  refine ⟨inferInstance, measurable_radMatrix, measurable_const, measurable_const,
    ?_, hcentered, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · have hfun : (fun ω : Bool => (((0 : Matrix (Fin 2) (Fin 2) ℝ) + radMatrix ω,
        (0 : Matrix (Fin 2) (Fin 2) ℝ) + (0 : Bool → Matrix (Fin 2) (Fin 2) ℝ) ω *
          ((0 : Bool → Matrix (Fin 2) (Fin 2) ℝ) ω).transpose) : HeadParam 2))
        = fun ω : Bool => ((radMatrix ω, 0) : HeadParam 2) := by
      funext ω; simp
    have hmeas : Measurable fun ω : Bool => ((radMatrix ω, 0) : HeadParam 2) :=
      measurable_of_countable _
    rw [hfun, fairCoin, Measure.map_smul, Measure.map_add _ _ hmeas,
      Measure.map_dirac true, Measure.map_dirac false, radLaw]
    exact hmeas.aemeasurable
  · exact indepFun_const_right radMatrix (0, 0)
  · refine iIndepFun_of_const_of_ne _ (0, 0) ?_
    intro p hp
    refine ⟨0, funext fun ω => (radMatrix_apply ω p.1 p.2).trans (ite_eq_right ?_)⟩
    rintro ⟨h1, h2⟩
    exact hp (Prod.ext h1 h2)
  · exact iIndepFun_of_const_of_ne _ (true, 0, 0) fun p _ => ⟨0, funext fun ω => by simp⟩
  · intro i j
    have hb : ∀ᵐ ω ∂fairCoin, radMatrix ω i j ∈ Set.Icc (-1 : ℝ) 1 := by
      refine Filter.Eventually.of_forall fun ω => ?_
      rw [Set.mem_Icc, radMatrix_apply]
      cases ω <;> split_ifs <;> simp_all
    have hsg := hasSubgaussianMGF_of_mem_Icc_of_integral_eq_zero
      (measurable_of_countable _).aemeasurable hb (hcentered i j)
    have hcst : ((‖(1 : ℝ) - (-1)‖₊ / 2) ^ 2 : ℝ≥0) = 1 * 1 ^ 2 := by
      norm_num
    rwa [hcst] at hsg
  · intro i j
    simp
  · intro i j
    simp

/-- `ass:high_order_short` is satisfied by a law under which the fluctuation
`ξ_θ[μ]` does not vanish. -/
theorem hasHighOrderLaw_rademacher : HasHighOrderLaw 2 1 0 radLaw :=
  ⟨1, Bool, inferInstance, fairCoin, radMatrix, 0, 0, 0, 0, one_pos,
    isHighOrderLaw_rademacher⟩

end Homogenized
end Transformer
