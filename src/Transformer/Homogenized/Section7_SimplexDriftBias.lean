module

public import Transformer.Homogenized.Section7_OverlapDriftGramComparison
public import Transformer.Homogenized.SimplexDrift

/-!
# The finite-dimension bias of the simplex ODE

ArXiv:2604.01978v1, §7, eq:f_g_def_selfcontained,
lem:drift_on_simplex_selfcontained and eq:Phi_def_selfcontained.
The paper substitutes the simplex attention moments into its actual
overlap drift and asserts exact equality with b(γ). The retained
dimension coefficients instead leave the exact gap proved in
SimplexDrift. Here bounded actual barycenter correlations give
the uniform absolute bound 3/d on that gap.

All attention expectations used in the substitution are proved
measurable and integrable from their actual normalized weights.
The bias estimate allows arbitrary probability head laws with the
stated moments; Gaussian shape is unnecessary for this step.
The example computes a nonzero gap of -1/2.
-/

@[expose] public section

open MeasureTheory Set
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- Actual attention probabilities are measurable in the original head.
Source: arXiv:2604.01978v1, §7, eq:pi_simplex_def_selfcontained. -/
theorem measurable_attnProb_head {d n : ℕ} (β : ℝ) (X : Idx n → EucSpace d) (i k : Idx n) :
    Measurable (fun θ : HeadParam d => attnProb β θ.2 X i k) := by
  have hA : Measurable (fun A : Matrix (Fin d) (Fin d) ℝ => attnProb β A X i k) :=
    (continuous_attnWeight_qk β (X i) (X k)).measurable.div
      (Finset.measurable_sum _ fun l hl => (continuous_attnWeight_qk β (X i) (X l)).measurable)
  exact hA.comp measurable_snd

/-- The actual overlap of two normalized attention rows lies in [0,1].
Source: arXiv:2604.01978v1, §7, eq:f_g_def_selfcontained.
This isolates the finite estimate used in SimplexWellDefined. -/
theorem sum_attnProb_product_mem_Icc {d n : ℕ} (β : ℝ) (A : Matrix (Fin d) (Fin d) ℝ)
    (X : Idx n → EucSpace d) (i j : Idx n) :
    (∑ k : Idx n, attnProb β A X i k * attnProb β A X j k) ∈ Icc (0 : ℝ) 1 := by
  constructor
  · exact Finset.sum_nonneg fun k hk => mul_nonneg (attnProb_nonneg _ _ _ _ _) (attnProb_nonneg _ _ _ _ _)
  · exact (Finset.sum_le_sum fun k hk =>
      mul_le_of_le_one_right (attnProb_nonneg _ _ _ _ _) (attnProb_le_one _ _ _ _ _)).trans
        (sum_attnProb_le_one β A X i)

/-- Actual attention-row overlaps are integrable under any finite head law.
Source: arXiv:2604.01978v1, §7, the expectations in eq:f_g_def_selfcontained. -/
theorem integrable_sum_attnProb_product {d n : ℕ} (β : ℝ) (ρ : Measure (HeadParam d)) [IsFiniteMeasure ρ]
    (X : Idx n → EucSpace d) (i j : Idx n) :
    Integrable (fun θ => ∑ k : Idx n, attnProb β θ.2 X i k * attnProb β θ.2 X j k) ρ := by
  have hm : Measurable (fun θ : HeadParam d =>
      ∑ k : Idx n, attnProb β θ.2 X i k * attnProb β θ.2 X j k) :=
    Finset.measurable_sum _ fun k hk =>
      (measurable_attnProb_head β X i k).mul (measurable_attnProb_head β X j k)
  apply Integrable.of_bound hm.aestronglyMeasurable 1
  apply ae_of_all
  intro θ
  rw [Real.norm_eq_abs, abs_of_nonneg (sum_attnProb_product_mem_Icc β θ.2 X i j).1]
  exact (sum_attnProb_product_mem_Icc β θ.2 X i j).2

/-- The squared norm of an actual attention row is integrable.
Source: arXiv:2604.01978v1, §7, the definition of f in eq:f_g_def_selfcontained. -/
theorem integrable_sum_attnProb_sq {d n : ℕ} (β : ℝ) (ρ : Measure (HeadParam d)) [IsFiniteMeasure ρ]
    (X : Idx n → EucSpace d) (i : Idx n) :
    Integrable (fun θ => ∑ k : Idx n, attnProb β θ.2 X i k ^ 2) ρ := by
  simpa only [pow_two] using integrable_sum_attnProb_product β ρ X i i

/-- Actual attention-overlap expectations lie in [0,1] under every probability
head law. Source: arXiv:2604.01978v1, §7, eq:f_g_def_selfcontained. -/
theorem integral_sum_attnProb_product_mem_Icc {d n : ℕ} (β : ℝ)
    (ρ : Measure (HeadParam d)) [IsProbabilityMeasure ρ] (X : Idx n → EucSpace d) (i j : Idx n) :
    (∫ θ, ∑ k : Idx n, attnProb β θ.2 X i k * attnProb β θ.2 X j k ∂ρ) ∈ Icc (0 : ℝ) 1 :=
  integral_mem_Icc_of_mem_Icc fun θ => sum_attnProb_product_mem_Icc β θ.2 X i j

/-- The actual simplex drift differs from the source's scalar ODE drift by
at most 3/d. Source: arXiv:2604.01978v1, §7, lem:drift_on_simplex_selfcontained
and eq:Phi_def_selfcontained. The source asserts exact equality; the
nonzero exact gap is proved in overlapDrift_simplex_sub_simplexDrift.
This bound retains that finite-dimension bias, with no integrability
assumptions added to the original bounded attention expectations. -/
theorem abs_overlapDrift_simplex_sub_simplexDrift_le {d n : ℕ} (hd : 0 < d) (β : ℝ)
    (ρ : Measure (HeadParam d)) [IsProbabilityMeasure ρ] (f g : ℝ → ℝ)
    (hf : ∀ (c : ℝ) (x : Idx (n + 1) → EucSpace d), IsSimplexConfig c x → ∀ k : Idx (n + 1),
      (∫ θ, ∑ l : Idx (n + 1), attnProb β θ.2 x k l ^ 2 ∂ρ) = f c)
    (hg : ∀ (c : ℝ) (x : Idx (n + 1) → EucSpace d), IsSimplexConfig c x →
      ∀ k l : Idx (n + 1), k ≠ l →
      (∫ θ, ∑ m : Idx (n + 1), attnProb β θ.2 x k m * attnProb β θ.2 x l m ∂ρ) = g c)
    (γ : ℝ) (X : Idx (n + 1) → EucSpace d) (hX : IsSimplexConfig γ X)
    (i j : Idx (n + 1)) (hij : i ≠ j) :
    |overlapDrift β ρ X i j - simplexDrift f g γ| ≤ 3 / (d : ℝ) := by
  have hdR : (0 : ℝ) < d := by exact_mod_cast hd
  have hself : baryCorr β ρ (empMeasure X) (X i) (X i) = γ + (1 - γ) * f γ := by
    rw [baryCorr_simplex_self β ρ γ X hX i (integrable_sum_attnProb_sq β ρ X i), hf γ X hX i]
  have hpair : baryCorr β ρ (empMeasure X) (X i) (X j) = γ + (1 - γ) * g γ := by
    rw [baryCorr_simplex β ρ γ X hX i j (integrable_sum_attnProb_product β ρ X i j), hg γ X hX i j hij]
  have hP : |γ + (1 - γ) * f γ| ≤ 1 := by
    rw [← hself]
    exact abs_baryCorr_empMeasure_le_one β ρ X hX.1 i i
  have hS : |γ + (1 - γ) * g γ| ≤ 1 := by
    rw [← hpair]
    exact abs_baryCorr_empMeasure_le_one β ρ X hX.1 i j
  have hγ : |γ| ≤ 1 := by
    rw [← hX.2 i j hij]
    simpa [hX.1 i, hX.1 j] using abs_real_inner_le_norm (X i) (X j)
  have hγ2 := (sq_le_one_iff_abs_le_one γ).2 hγ
  have htwo : |2 - γ ^ 2| ≤ 2 := by
    rw [abs_of_nonneg (by linarith : 0 ≤ 2 - γ ^ 2)]
    linarith [sq_nonneg γ]
  have hnum : |γ * (γ + (1 - γ) * f γ) - (2 - γ ^ 2) * (γ + (1 - γ) * g γ)| ≤ 3 := by
    calc
      _ ≤ |γ * (γ + (1 - γ) * f γ)| + |(2 - γ ^ 2) * (γ + (1 - γ) * g γ)| := by
        simpa only [sub_zero, zero_sub, abs_neg] using
          abs_sub_le (γ * (γ + (1 - γ) * f γ)) 0 ((2 - γ ^ 2) * (γ + (1 - γ) * g γ))
      _ = |γ| * |γ + (1 - γ) * f γ| + |2 - γ ^ 2| * |γ + (1 - γ) * g γ| := by rw [abs_mul, abs_mul]
      _ ≤ 1 * 1 + 2 * 1 :=
        add_le_add (mul_le_mul hγ hP (abs_nonneg _) zero_le_one)
          (mul_le_mul htwo hS (abs_nonneg _) (by norm_num))
      _ = 3 := by norm_num
  rw [overlapDrift_simplex_sub_simplexDrift hdR.ne' β ρ f g hf hg γ X hX
    (integrable_sum_attnProb_sq β ρ X) (integrable_sum_attnProb_product β ρ X) i j hij,
    abs_div, abs_of_pos hdR]
  exact div_le_div_of_nonneg_right hnum hdR.le
/-- Positive-variance original Gaussian heads and orthonormal unit tokens
realize every moment, probability, simplex and distinct-index premise. -/
example : ∃ f g : ℝ → ℝ,
    (0 : ℕ) < 2 ∧ IsProbabilityMeasure (gaussHeadLaw 2 1 1) ∧
    IsSimplexConfig (0 : ℝ) (fun k : Idx 2 => (EuclideanSpace.single k 1 : EucSpace 2)) ∧
    (0 : Idx 2) ≠ 1 ∧
    (∀ (c : ℝ) (x : Idx 2 → EucSpace 2), IsSimplexConfig c x → ∀ k : Idx 2,
      (∫ θ, ∑ l : Idx 2, attnProb 1 θ.2 x k l ^ 2 ∂(gaussHeadLaw 2 1 1)) = f c) ∧
    (∀ (c : ℝ) (x : Idx 2 → EucSpace 2), IsSimplexConfig c x → ∀ k l : Idx 2, k ≠ l →
      (∫ θ, ∑ m : Idx 2, attnProb 1 θ.2 x k m * attnProb 1 θ.2 x l m ∂(gaussHeadLaw 2 1 1)) = g c) := by
  obtain ⟨f, g, -, -, hf, hg⟩ :=
    simplex_overlap_wellDefined (d := 2) (n := 2) 1 1 1 _ (isGaussianHeadLaw_gaussHeadLaw 2 1 1)
  refine ⟨f, g, by norm_num, inferInstance, ?_, by decide, hf, hg⟩
  constructor
  · intro k
    simp
  · intro k l hkl
    simp [EuclideanSpace.inner_single_left, hkl.symm]

/-- The bias is genuinely nonzero at the source's smallest simplex size. -/
example : let X : Idx 2 → EucSpace 2 := fun k => EuclideanSpace.single k 1
    overlapDrift 1 (Measure.dirac (0 : HeadParam 2)) X 0 1 -
      simplexDrift (fun _ => 1 / 2) (fun _ => 1 / 2) 0 = (-1 / 2 : ℝ) := by
  dsimp
  norm_num [overlapDrift, simplexDrift, EuclideanSpace.inner_single_left, PiLp.single_apply]

end Transformer.Homogenized
