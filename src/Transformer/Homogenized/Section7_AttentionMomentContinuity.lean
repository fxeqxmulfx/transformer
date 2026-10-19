module

public import Transformer.Homogenized.Section7_SimplexOdeDomain

/-!
# Continuity of the original attention moments and scalar drift

ArXiv:2604.01978v1, §7, `eq:pi_simplex_def_selfcontained`,
`eq:f_g_def_selfcontained`, and `eq:gamma_ODE_selfcontained`.
Normalized attention probabilities depend continuously on the
actual configuration. Their row products lie in [0,1], so
bounded dominated convergence gives continuity of their actual
expectations under every finite head measure.

The explicit reference simplex transfers this continuity to the
source's f and g on the whole feasible closed overlap interval.
This includes both endpoints, where the same reference remains
a unit simplex. At those configurations the original moment
identities still identify the actual row expectations with f
and g, so the boundary values enter the continuity argument.
The proved ODE domain and continuity of gamma then give continuity
of b(gamma(t)) on the nonnegative time axis. No moment regularity
or time-integrability assumption is added to the source's data.
This supplies the original drift integral needed to combine the
noise probability estimate with the deterministic comparison.
-/

@[expose] public section

open MeasureTheory Set
open scoped BigOperators NNReal
namespace Transformer.Homogenized

/-- Actual normalized attention is continuous in the whole token configuration.
Source: arXiv:2604.01978v1, §7, `eq:pi_simplex_def_selfcontained`. -/
theorem continuous_attnProb_configuration {d n : ℕ} (β : ℝ) (A : Matrix (Fin d) (Fin d) ℝ) (i k : Idx n) :
    Continuous (fun X : Idx n → EucSpace d => attnProb β A X i k) := by
  have hw (l : Idx n) : Continuous (fun X : Idx n → EucSpace d =>
      attnWeight β ((0, A) : HeadParam d) (X i) (X l)) := by
    have hAx : Continuous (fun X : Idx n → EucSpace d => Matrix.toEuclideanLin A (X i)) :=
      (Matrix.toEuclideanLin A).continuous_of_finiteDimensional.comp (continuous_apply i)
    exact Real.continuous_exp.comp (continuous_const.mul (hAx.inner (continuous_apply l)))
  change Continuous (fun X : Idx n → EucSpace d =>
    attnWeight β ((0, A) : HeadParam d) (X i) (X k) /
      ∑ l, attnWeight β ((0, A) : HeadParam d) (X i) (X l))
  exact (hw k).div (continuous_finsetSum _ fun l hl => hw l) (fun X =>
    (Finset.sum_pos (fun l hl => attnWeight_pos β (0, A) (X i) (X l)) ⟨i, Finset.mem_univ i⟩).ne')

/-- Actual row-correlation expectations are continuous under every finite head law.
Source: arXiv:2604.01978v1, §7, `eq:f_g_def_selfcontained`. -/
theorem continuous_integral_sum_attnProb_product {d n : ℕ} (β : ℝ) (ρ : Measure (HeadParam d)) [IsFiniteMeasure ρ]
    (i j : Idx n) :
    Continuous (fun X : Idx n → EucSpace d => ∫ θ, ∑ k,
      attnProb β θ.2 X i k * attnProb β θ.2 X j k ∂ρ) := by
  apply continuous_of_dominated (bound := fun _ => (1 : ℝ))
  · intro X
    exact (integrable_sum_attnProb_product β ρ X i j).aestronglyMeasurable
  · intro X
    apply ae_of_all
    intro θ
    rw [Real.norm_eq_abs, abs_of_nonneg (sum_attnProb_product_mem_Icc β θ.2 X i j).1]
    exact (sum_attnProb_product_mem_Icc β θ.2 X i j).2
  · exact integrable_const 1
  · apply ae_of_all
    intro θ
    exact continuous_finsetSum _ fun k hk =>
      (continuous_attnProb_configuration β θ.2 i k).mul (continuous_attnProb_configuration β θ.2 j k)

/-- Actual squared-row expectations have the same continuity.
Source: arXiv:2604.01978v1, §7, the self moment in `eq:f_g_def_selfcontained`. -/
theorem continuous_integral_sum_attnProb_sq {d n : ℕ} (β : ℝ) (ρ : Measure (HeadParam d)) [IsFiniteMeasure ρ]
    (i : Idx n) :
    Continuous (fun X : Idx n → EucSpace d => ∫ θ, ∑ k, attnProb β θ.2 X i k ^ 2 ∂ρ) := by
  simpa only [pow_two] using continuous_integral_sum_attnProb_product β ρ i i

/-- The source's actual moments and scalar drift are continuous throughout
the feasible closed interval. Source: arXiv:2604.01978v1, §7,
`eq:f_g_def_selfcontained` and `eq:Phi_def_selfcontained`. -/
theorem continuousOn_simplex_moments_of_initial {d n : ℕ} (hn : 2 ≤ n + 1) (β : ℝ)
    (ρ : Measure (HeadParam d)) [IsFiniteMeasure ρ] (f g : ℝ → ℝ)
    (hf : ∀ (c : ℝ) (Y : Idx (n + 1) → EucSpace d), IsSimplexConfig c Y → ∀ i,
      (∫ θ, ∑ k, attnProb β θ.2 Y i k ^ 2 ∂ρ) = f c)
    (hg : ∀ (c : ℝ) (Y : Idx (n + 1) → EucSpace d), IsSimplexConfig c Y → ∀ i j, i ≠ j →
      (∫ θ, ∑ k, attnProb β θ.2 Y i k * attnProb β θ.2 Y j k ∂ρ) = g c)
    (γ₀ : ℝ) (hγ₀ : γ₀ ∈ Ioo (-(1 / (((n + 1 : ℕ) : ℝ) - 1))) 1)
    (X₀ : Idx (n + 1) → EucSpace d) (hX₀ : IsSimplexConfig γ₀ X₀) :
    ContinuousOn f (Icc (-(1 / (((n + 1 : ℕ) : ℝ) - 1))) 1) ∧
    ContinuousOn g (Icc (-(1 / (((n + 1 : ℕ) : ℝ) - 1))) 1) ∧
    ContinuousOn (simplexDrift f g) (Icc (-(1 / (((n + 1 : ℕ) : ℝ) - 1))) 1) := by
  let i : Idx (n + 1) := ⟨0, by omega⟩
  let j : Idx (n + 1) := ⟨1, by omega⟩
  have hij : i ≠ j := by
    intro h
    have hval := congrArg Fin.val h
    norm_num [i, j] at hval
  have hY (c : ℝ) (hc : c ∈ Icc (-(1 / (((n + 1 : ℕ) : ℝ) - 1))) 1) :
      IsSimplexConfig c (simplexRecenter X₀ γ₀ c) := isSimplexConfig_simplexRecenter hn hγ₀ hX₀ hc
  have hfc : ContinuousOn f (Icc (-(1 / (((n + 1 : ℕ) : ℝ) - 1))) 1) :=
    ((continuous_integral_sum_attnProb_sq β ρ i).comp (continuous_simplexRecenter X₀ γ₀)).continuousOn.congr
      (fun c hc => (hf c _ (hY c hc) i).symm)
  have hgc : ContinuousOn g (Icc (-(1 / (((n + 1 : ℕ) : ℝ) - 1))) 1) :=
    ((continuous_integral_sum_attnProb_product β ρ i j).comp (continuous_simplexRecenter X₀ γ₀)).continuousOn.congr
      (fun c hc => (hg c _ (hY c hc) i j hij).symm)
  refine ⟨hfc, hgc, ?_⟩
  unfold simplexDrift
  exact (continuousOn_id.add ((continuousOn_const.sub continuousOn_id).mul hgc)).sub
    (continuousOn_id.mul (continuousOn_id.add ((continuousOn_const.sub continuousOn_id).mul hfc)))
/-- The original ODE drift is continuous in time, as derived from its actual moments.
Source: arXiv:2604.01978v1, §7, `eq:gamma_ODE_selfcontained`. -/
theorem continuousOn_simplex_ode_drift {d n : ℕ} (hn : 2 ≤ n + 1) (β : ℝ)
    (ρ : Measure (HeadParam d)) [IsProbabilityMeasure ρ] (f g : ℝ → ℝ)
    (hf : ∀ (c : ℝ) (Y : Idx (n + 1) → EucSpace d), IsSimplexConfig c Y → ∀ i,
      (∫ θ, ∑ k, attnProb β θ.2 Y i k ^ 2 ∂ρ) = f c)
    (hg : ∀ (c : ℝ) (Y : Idx (n + 1) → EucSpace d), IsSimplexConfig c Y → ∀ i j, i ≠ j →
      (∫ θ, ∑ k, attnProb β θ.2 Y i k * attnProb β θ.2 Y j k ∂ρ) = g c)
    (γ₀ : ℝ) (hγ₀ : γ₀ ∈ Ioo (-(1 / (((n + 1 : ℕ) : ℝ) - 1))) 1)
    (X₀ : Idx (n + 1) → EucSpace d) (hX₀ : IsSimplexConfig γ₀ X₀)
    (γ : ℝ → ℝ) (hinit : γ 0 = γ₀)
    (hd : ∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt γ (simplexDrift f g (γ t)) (Ici 0) t) :
    ContinuousOn (fun t => simplexDrift f g (γ t)) (Ici (0 : ℝ)) := by
  have hB := (continuousOn_simplex_moments_of_initial hn β ρ f g hf hg γ₀ hγ₀ X₀ hX₀).2.2
  have hγ := (simplex_ode_domain_and_monotone hn β ρ f g hf hg hγ₀ hX₀ γ hinit hd).1
  exact hB.comp' (fun t ht => (hd t ht).continuousWithinAt) (fun t ht =>
    ⟨(hγ t ht).1.le, (hγ t ht).2.le⟩)


/-- Actual uniform attention, orthonormal initial tokens and the nonconstant
curve (exp(t)-1)/(exp(t)+1) satisfy all moment, domain and ODE hypotheses. -/
example : ∃ (X₀ : Idx 2 → EucSpace 2) (f g γ : ℝ → ℝ),
    (2 : ℕ) ≤ 2 ∧ (0 : ℝ) ∈ Ioo (-(1 / ((2 : ℝ) - 1))) 1 ∧ IsSimplexConfig 0 X₀ ∧
    (∀ (c : ℝ) (Y : Idx 2 → EucSpace 2), IsSimplexConfig c Y → ∀ i,
      (∫ θ, ∑ k, attnProb 1 θ.2 Y i k ^ 2 ∂(Measure.dirac (0 : HeadParam 2))) = f c) ∧
    (∀ (c : ℝ) (Y : Idx 2 → EucSpace 2), IsSimplexConfig c Y → ∀ i j, i ≠ j →
      (∫ θ, ∑ k, attnProb 1 θ.2 Y i k * attnProb 1 θ.2 Y j k ∂(Measure.dirac (0 : HeadParam 2))) = g c) ∧
    γ 0 = 0 ∧
    (∀ t ∈ Ici (0 : ℝ), HasDerivWithinAt γ (simplexDrift f g (γ t)) (Ici 0) t) ∧
    0 < γ 1 := by
  let X₀ : Idx 2 → EucSpace 2 := fun k => EuclideanSpace.single k 1
  let f : ℝ → ℝ := fun _ => 1 / 2
  let g : ℝ → ℝ := fun _ => 1 / 2
  let γ : ℝ → ℝ := fun t => (Real.exp t - 1) / (Real.exp t + 1)
  refine ⟨X₀, f, g, γ, le_rfl, by norm_num, ?_, ?_, ?_, by simp [γ], ?_, ?_⟩
  · constructor
    · intro k; simp [X₀]
    · intro i j hij; simp [X₀, EuclideanSpace.inner_single_left, hij.symm]
  · intro c Y hY i
    norm_num [f, attnProb, attnWeight, qkMap, Fin.sum_univ_two]
  · intro c Y hY i j hij
    norm_num [g, attnProb, attnWeight, qkMap, Fin.sum_univ_two]
  · intro t ht
    exact (hasDerivAt_simplex_uniform_attention t).hasDerivWithinAt
  · exact div_pos (sub_pos.mpr (Real.one_lt_exp_iff.mpr (by norm_num))) (by positivity)

end Transformer.Homogenized
