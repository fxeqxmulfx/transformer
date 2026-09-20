/-
# Homogenized Transformers — high temperature and the dimension limit

Formalization of `thm:clustering_small_beta`, of the logistic Cauchy problem
`eq:logistic_u_smallbeta` and of its explicit solution `eq: logistic.sol` of
arXiv:2604.01978v1, *Homogenized Transformers*.

Without any simplex assumption, but at high temperature `β ∈ [0,1]` and in
large dimension, the empirical second moment `m(t)` of the solution of
`eq:Diffusive_gaussian_case` follows the logistic equation `u̇ = u(1-u)`, whose
solution rises to `1`: the tokens cluster.

Source: arXiv:2604.01978v1, §2.3.5, `thm:clustering_small_beta`.
-/

import Transformer.Homogenized.GaussianInit

open scoped BigOperators NNReal
open Real MeasureTheory ProbabilityTheory

namespace Transformer
namespace Homogenized

/-- `eq:def_m_smallbeta`: the empirical second moment
`m = n^{-2} Σ_i Σ_j ⟨x_i, x_j⟩`.

Source: arXiv:2604.01978v1, `eq:def_m_smallbeta`. -/
noncomputable def meanOverlap {d n : ℕ} (x : Idx n → EucSpace d) : ℝ :=
  ((n : ℝ) ^ 2)⁻¹ * ∑ i : Idx n, ∑ j : Idx n, inner (𝕜 := ℝ) (x i) (x j)

/-- A configuration of equal unit tokens has second moment `1`. -/
theorem meanOverlap_const {d n : ℕ} (v : EucSpace (d + 1)) (hv : ‖v‖ = 1) :
    meanOverlap (fun _ : Idx (n + 1) => v) = 1 := by
  have hvv : inner (𝕜 := ℝ) v v = 1 := by
    rw [real_inner_self_eq_norm_sq, hv, one_pow]
  have hn : ((n : ℝ) + 1) ≠ 0 := by positivity
  simp only [meanOverlap, hvv, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    nsmul_eq_mul, mul_one, Nat.cast_add, Nat.cast_one]
  field_simp

/-- **Equation (eq: logistic.sol).**  The explicit solution of the logistic
Cauchy problem `eq:logistic_u_smallbeta`,

  `u(t) = u(0) / (u(0) + (1 - u(0)) e^{-t})`,

solves `u̇ = u(1-u)`.  Since `u(t) → 1` as `t → ∞`, the second moment of
`thm:clustering_small_beta` rises to `1`.

Source: arXiv:2604.01978v1, `eq: logistic.sol`. -/
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

/-- The hypotheses of `hasDerivAt_logistic` are satisfiable. -/
example : (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) ≤ 1 := by norm_num

/-- **Theorem (thm:clustering_small_beta).**  Fix `d, n ≥ 2` and `β ∈ [0,1]`,
and start from a configuration with all pairwise overlaps at least
`γ ∈ (0,1)`.  Let `u` solve `u̇ = u(1-u)`, `u(0) = m(0)`.  Then there is a
universal `C > 0` such that for every `T > 0` and `δ ∈ (0,1)`, with probability
at least `1 - δ` the solution of `eq:Diffusive_gaussian_case` satisfies

  `sup_{t∈[0,T]} |m(t) - u(t)| ≤ C e^{CT} max(1/d, d²σ_A⁴β²) + C√(T/d log(4/δ))`.

**What the source says and what is carried here.**  "There exists a universal
constant `C`" is read as a constant produced before `d, n, β, σ_A, T, δ` are
chosen, which is what *universal* means; and the initial configuration is
deterministic, as in the source, so that `u(0) = m(0)` is a number.

Not proved here.

Source: arXiv:2604.01978v1, `thm:clustering_small_beta`,
`eq:smallbeta_main_bound`. -/
theorem clustering_small_beta :
    ∃ C : ℝ, 0 < C ∧
      ∀ (d n : ℕ), 2 ≤ d → 2 ≤ n → ∀ β ∈ Set.Icc (0 : ℝ) 1,
      ∀ (σV σA : ℝ≥0) (ρ : Measure (HeadParam d)), IsGaussianHeadLaw d σV σA ρ →
      ∀ γ ∈ Set.Ioo (0 : ℝ) 1, ∀ (T : ℝ), 0 < T → ∀ δ ∈ Set.Ioo (0 : ℝ) 1,
      ∀ (Ω : Type) [MeasurableSpace Ω] (P : Measure Ω)
        (X : ℝ → Ω → (Idx n → EucSpace d)) (x₀ : Idx n → EucSpace d),
        IsDiffusiveSde β ρ P X → (∀ ω, X 0 ω = x₀) →
        (∀ i j : Idx n, i ≠ j → γ ≤ inner (𝕜 := ℝ) (x₀ i) (x₀ j)) →
      ∀ u : ℝ → ℝ, u 0 = meanOverlap x₀ →
        (∀ t ∈ Set.Ici (0 : ℝ), HasDerivWithinAt u (u t * (1 - u t)) (Set.Ici 0) t) →
        ENNReal.ofReal (1 - δ) ≤
          P {ω | ∀ t ∈ Set.Icc (0 : ℝ) T,
            |meanOverlap (X t ω) - u t| ≤
              C * Real.exp (C * T) *
                  max (1 / (d : ℝ)) ((d : ℝ) ^ 2 * (σA : ℝ) ^ 4 * β ^ 2) +
                C * Real.sqrt (T / (d : ℝ) * Real.log (4 / δ))} := by
  sorry

/-- The hypotheses `clustering_small_beta` quantifies over are satisfiable:
`d = n = 2`, `β = 1 ∈ [0,1]`, `ρ* = δ_0`, `γ = 1/2 ∈ (0,1)`, `T = 1`,
`δ = 1/2`, and the constant configuration `x₀ ≡ v` at a unit vector, whose
pairwise overlaps are all `1 ≥ γ` and whose second moment is `1`. -/
example :
    (1 : ℝ) ∈ Set.Icc (0 : ℝ) 1 ∧ (1 / 2 : ℝ) ∈ Set.Ioo (0 : ℝ) 1 ∧
      IsGaussianHeadLaw 2 0 0 (Measure.dirac (0 : HeadParam 2)) ∧
      (∀ i j : Idx 2, i ≠ j →
        (1 / 2 : ℝ) ≤ inner (𝕜 := ℝ)
          ((EuclideanSpace.single (0 : Fin 2) (1 : ℝ) : EucSpace 2))
          ((EuclideanSpace.single (0 : Fin 2) (1 : ℝ) : EucSpace 2))) ∧
      meanOverlap (fun _ : Idx 2 => (EuclideanSpace.single (0 : Fin 2) (1 : ℝ) : EucSpace 2))
        = 1 := by
  refine ⟨⟨by norm_num, le_rfl⟩, ⟨by norm_num, by norm_num⟩,
    isGaussianHeadLaw_dirac_zero 2, ?_, ?_⟩
  · intro i j _
    rw [real_inner_self_eq_norm_sq]
    simp [PiLp.norm_single]
    norm_num
  · exact meanOverlap_const (n := 1) _ (by simp [PiLp.norm_single])

end Homogenized
end Transformer
