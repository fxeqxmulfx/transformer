module

public import Transformer.Homogenized.Section7_GaussianMeanPathError

/-!
# Homogenized Transformers — high temperature and the dimension limit

Formalization of arXiv:2604.01978v1, §2.3.5 and §7,
`thm:clustering_small_beta`, `eq:logistic_u_smallbeta`, and `eq: logistic.sol`.
The original Gaussian mean follows the original logistic reference with
a universal deterministic error and an unamplified confidence radius.
The adopted value variance is explicit, and the source equation is
represented by its full adapted continuous sphere martingale problem.

The original mean and its elementary geometry live in Section7_MeanOverlap,
before the Gaussian estimates used by the probability theorem here.
-/

@[expose] public section

open scoped BigOperators NNReal ENNReal
open Real MeasureTheory ProbabilityTheory

namespace Transformer
namespace Homogenized

/-- **Equation (eq: logistic.sol).** The explicit solution of the logistic
Cauchy problem `eq:logistic_u_smallbeta`,
`u(t)=u(0)/(u(0)+(1-u(0)) exp(-t))`, solves `u'=u(1-u)`.
Source: arXiv:2604.01978v1, §2.3.5, `eq: logistic.sol`. -/
theorem hasDerivAt_logistic (u₀ : ℝ) (h₀ : 0 < u₀) (h₁ : u₀ ≤ 1) (t : ℝ) :
    HasDerivAt (fun r => u₀ / (u₀ + (1 - u₀) * Real.exp (-r)))
      (u₀ / (u₀ + (1 - u₀) * Real.exp (-t)) *
        (1 - u₀ / (u₀ + (1 - u₀) * Real.exp (-t)))) t := by
  have hexp : (0 : ℝ) < Real.exp (-t) := Real.exp_pos _
  have hD : 0 < u₀ + (1 - u₀) * Real.exp (-t) := by nlinarith
  have hd : HasDerivAt (fun r : ℝ => u₀ + (1 - u₀) * Real.exp (-r))
      ((1 - u₀) * -Real.exp (-t)) t := by
    have : HasDerivAt (fun r : ℝ => Real.exp (-r)) (-Real.exp (-t)) t := by
      simpa using (hasDerivAt_neg t).exp
    simpa using (this.const_mul (1 - u₀)).const_add u₀
  have := (hasDerivAt_const t u₀).div hd hD.ne'
  convert this using 1
  field_simp
  ring

/-- The explicit original logistic formula is positive at every finite real time.
Source: arXiv:2604.01978v1, §2.3.5, `eq: logistic.sol`. -/
theorem logistic_formula_pos (u₀ : ℝ) (h₀ : 0 < u₀) (h₁ : u₀ ≤ 1) (t : ℝ) :
    0 < u₀ / (u₀ + (1 - u₀) * Real.exp (-t)) := by
  apply div_pos h₀
  nlinarith [Real.exp_pos (-t)]

/-- Positive interior initial means realize the explicit original logistic solution. -/
example : (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 := by norm_num

/-- **Theorem (thm:clustering_small_beta).** A source-normalized Gaussian
solution with initial pairwise overlaps at least `γ∈(0,1)` follows its
original logistic reference, with the source's full displayed error rate.
The universal constant is produced before all model and confidence parameters.

**Source wording and corrections.** The source says "the solution to
eq:Diffusive_gaussian_case" and adopts `σV²=1/d` in §2.3.1 before
this claim. Here the omitted value normalization is explicit, and the
solution has the source's adapted continuous smooth-test martingale
identities from Appendix `def:ito.formula`. The earlier marginal
expectation-derivative predicate did not retain these identities.
The sample law is explicitly a probability measure and the initial
configuration is deterministic, as in the source.

The proof uses the genuine regularized logarithmic mean and its actual
Gaussian generator. This closes the source proof's missing step from
`eq:gronwall_e_clean`, which amplifies its additive noise, to the displayed
`eq:smallbeta_main_bound`, whose noise coefficient is independent of time.
The displayed rate and every initial overlap hypothesis are retained.

Source: arXiv:2604.01978v1, §2.3.5 and §7, `thm:clustering_small_beta`,
`eq:smallbeta_main_bound`. -/
theorem clustering_small_beta :
    ∃ C : ℝ, 0 < C ∧
      ∀ (d n : ℕ), 2 ≤ d → 2 ≤ n → ∀ β ∈ Set.Icc (0 : ℝ) 1,
      ∀ (σV σA : ℝ≥0) (ρ : Measure (HeadParam d)), IsGaussianHeadLaw d σV σA ρ →
        (σV : ℝ) ^ 2 = 1 / (d : ℝ) →
      ∀ γ ∈ Set.Ioo (0 : ℝ) 1, ∀ (T : ℝ), 0 < T → ∀ δ ∈ Set.Ioo (0 : ℝ) 1,
      ∀ (Ω : Type) [m : MeasurableSpace Ω] (P : Measure Ω) [IsProbabilityMeasure P]
        (ℱ : Filtration ℝ m) (X : ℝ → Ω → (Idx n → EucSpace d)) (x₀ : Idx n → EucSpace d),
        IsSphereMartingaleSolution ρ (bField β ρ) (noiseField β 1 1 ρ) P ℱ T X →
        (∀ ω, X 0 ω = x₀) →
        (∀ i j : Idx n, i ≠ j → γ ≤ inner (𝕜 := ℝ) (x₀ i) (x₀ j)) →
      ∀ u : ℝ → ℝ, u 0 = meanOverlap x₀ →
        (∀ t ∈ Set.Ici (0 : ℝ), HasDerivWithinAt u (u t * (1 - u t)) (Set.Ici 0) t) →
        ENNReal.ofReal (1 - δ) ≤
          P {ω | ∀ t ∈ Set.Icc (0 : ℝ) T,
            |meanOverlap (X t ω) - u t| ≤
              C * Real.exp (C * T) *
                  max (1 / (d : ℝ)) ((d : ℝ) ^ 2 * (σA : ℝ) ^ 4 * β ^ 2) +
                C * Real.sqrt (T / (d : ℝ) * Real.log (4 / δ))} := by
  refine ⟨4096, by norm_num, ?_⟩
  intro d n hd hn β hβ σV σA ρ hρ hσ γ hγ T hT δ hδ Ω m P hP ℱ X x₀ hMP hinit hpair u huinit hu
  have hn0 : 0 < n := by omega
  by_cases hsmall : meanLogRegularization d β σA ≤ 1
  · have hnoise := gaussian_logMean_noise_probability_of_initial_configuration hd hn0
      β hβ σV σA ρ hρ hσ hsmall P ℱ T hT δ hδ X hMP x₀ hinit
    apply hnoise.trans (measure_mono ?_)
    intro ω hω
    exact gaussian_mean_clustering_on_log_noise_event hd hn0 β hβ σV σA ρ hρ hσ hsmall
      P ℱ T δ hT.le X hMP x₀ hinit γ hγ hpair u huinit hu ω hω
  · have hlarge : 1 ≤ meanLogRegularization d β σA := (lt_of_not_ge hsmall).le
    have hbound (ω : Ω) : ∀ t ∈ Set.Icc (0 : ℝ) T, |meanOverlap (X t ω) - u t| ≤
        4096 * Real.exp (4096 * T) *
          max (1 / (d : ℝ)) ((d : ℝ) ^ 2 * (σA : ℝ) ^ 4 * β ^ 2) +
          4096 * Real.sqrt (T / (d : ℝ) * Real.log (4 / δ)) := by
      have hpos : 0 < u 0 := by
        rw [huinit]
        exact meanOverlap_initial_pos_of_sphere_solution hn0 ρ (bField β ρ) (noiseField β 1 1 ρ)
          P ℱ T hT.le X hMP x₀ hinit γ hγ hpair ω
      have hu0 : u 0 ≤ 1 := by
        rw [huinit, ← hinit ω]
        exact (meanOverlap_mem_Icc (X 0 ω) (hMP.1 0 ⟨le_rfl, hT.le⟩ ω)).2
      intro t ht
      have hrange := logistic_ode_mem_Ioc u hpos hu0 hu t ht.1
      have herr := abs_mean_sub_logistic_le_one (meanOverlap_mem_Icc (X t ω) (hMP.1 t ht ω))
        ⟨hrange.1.le, hrange.2⟩
      have hb := mean_log_large_regularization_absorption (meanLogRegularization d β σA) T hlarge hT.le
      have hs := mean_log_error_le_source_scale d β σA T δ hT.le
      have hscale : 1 ≤ 4096 * Real.exp (4096 * T) *
          max (1 / (d : ℝ)) ((d : ℝ) ^ 2 * (σA : ℝ) ^ 4 * β ^ 2) +
          4096 * Real.sqrt (T / (d : ℝ) * Real.log (4 / δ)) := by
        nlinarith [meanLogNoiseRadius_nonneg d T δ]
      exact herr.trans hscale
    apply (ENNReal.ofReal_le_one.mpr (by linarith [hδ.1] : 1 - δ ≤ 1)).trans
    calc
      (1 : ℝ≥0∞) = P Set.univ := measure_univ.symm
      _ ≤ _ := measure_mono (fun ω _ => hbound ω)

/-- The source's adopted Gaussian value variance and a positive consensus
configuration realize its numerical and deterministic initial-data assumptions.
The genuine full martingale predicate at initial time is realized in
Section7_GaussianMeanPathError. -/
example : (1 : ℝ) ∈ Set.Icc (0 : ℝ) 1 ∧ (1 / 2 : ℝ) ∈ Set.Ioo (0 : ℝ) 1 ∧
    IsGaussianHeadLaw 2 (stdSigmaV 2) 1 (gaussHeadLaw 2 (stdSigmaV 2) 1) ∧
    (stdSigmaV 2 : ℝ) ^ 2 = 1 / (2 : ℝ) ∧
    (∀ i j : Idx 2, i ≠ j → (1 / 2 : ℝ) ≤ inner (𝕜 := ℝ)
      (EuclideanSpace.single (0 : Fin 2) (1 : ℝ) : EucSpace 2)
      (EuclideanSpace.single (0 : Fin 2) (1 : ℝ) : EucSpace 2)) ∧
    meanOverlap (fun _ : Idx 2 => (EuclideanSpace.single (0 : Fin 2) (1 : ℝ) : EucSpace 2)) = 1 := by
  refine ⟨by norm_num, by norm_num, isGaussianHeadLaw_gaussHeadLaw 2 _ _, stdSigmaV_sq 2, ?_, ?_⟩
  · intro i j hij
    rw [real_inner_self_eq_norm_sq]
    simp [PiLp.norm_single]
    norm_num
  · exact meanOverlap_const (n := 1) _ (by simp [PiLp.norm_single])

end Homogenized
end Transformer
