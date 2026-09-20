/-
# Homogenized Transformers — simplex data at low temperature

Formalization of `thm:clustering_random_init` of arXiv:2604.01978v1,
*Homogenized Transformers*.

On a simplex configuration — unit tokens with a common pairwise overlap `γ` —
the attention probabilities `π^A_{i→k}(γ)` have expectations `f(γ)`, `g(γ)`
that depend on neither the configuration nor the indices, and the Gram matrix
of the solution of `eq:Diffusive_gaussian_case` tracks the scalar ODE
`γ̇ = b(γ)` up to a martingale fluctuation of size `√(T/d log(n/δ))`.

Source: arXiv:2604.01978v1, §2.3.4, `thm:clustering_random_init`.
-/

import Transformer.Homogenized.GaussianInit

open scoped BigOperators NNReal
open Real MeasureTheory ProbabilityTheory

namespace Transformer
namespace Homogenized

/-- `eq:pi_simplex_def_selfcontained`: the attention probability

  `π^A_{i→k}(γ) = e^{β⟨A x_i, x_k⟩} / Σ_ℓ e^{β⟨A x_i, x_ℓ⟩}`,

written for the query-key matrix `A` alone, i.e. for the head `(0, A)`.

Source: arXiv:2604.01978v1, `eq:pi_simplex_def_selfcontained`. -/
noncomputable def attnProb {d n : ℕ} (β : ℝ) (A : Matrix (Fin d) (Fin d) ℝ)
    (x : Idx n → EucSpace d) (i k : Idx n) : ℝ :=
  attnWeight β ((0, A) : HeadParam d) (x i) (x k) /
    ∑ l : Idx n, attnWeight β ((0, A) : HeadParam d) (x i) (x l)

theorem attnProb_pos {d n : ℕ} (β : ℝ) (A : Matrix (Fin d) (Fin d) ℝ)
    (x : Idx (n + 1) → EucSpace d) (i k : Idx (n + 1)) : 0 < attnProb β A x i k :=
  div_pos (attnWeight_pos _ _ _ _)
    (Finset.sum_pos (fun _ _ => attnWeight_pos _ _ _ _) ⟨i, Finset.mem_univ i⟩)

/-- The attention probabilities of a token sum to one. -/
theorem sum_attnProb {d n : ℕ} (β : ℝ) (A : Matrix (Fin d) (Fin d) ℝ)
    (x : Idx (n + 1) → EucSpace d) (i : Idx (n + 1)) :
    ∑ k : Idx (n + 1), attnProb β A x i k = 1 := by
  simp only [attnProb, ← Finset.sum_div]
  exact div_self (ne_of_gt
    (Finset.sum_pos (fun _ _ => attnWeight_pos _ _ _ _) ⟨i, Finset.mem_univ i⟩))

/-- A simplex configuration: unit tokens with a common pairwise overlap `γ`. -/
def IsSimplexConfig {d n : ℕ} (γ : ℝ) (x : Idx n → EucSpace d) : Prop :=
  (∀ i : Idx n, ‖x i‖ = 1) ∧ ∀ i j : Idx n, i ≠ j → inner (𝕜 := ℝ) (x i) (x j) = γ

/-- A one-token configuration is a simplex configuration for every `γ`: the
condition on distinct pairs is empty. -/
theorem isSimplexConfig_of_subsingleton {d : ℕ} (γ : ℝ)
    (x : Idx 1 → EucSpace (d + 1)) (hx : ∀ i, ‖x i‖ = 1) : IsSimplexConfig γ x :=
  ⟨hx, fun i j hij => absurd (Subsingleton.elim i j) hij⟩

/-- `eq:Phi_def_selfcontained`: the drift of the simplex ODE,

  `b(γ) = γ + (1-γ) g(γ) - γ(γ + (1-γ) f(γ))`.

Source: arXiv:2604.01978v1, `eq:Phi_def_selfcontained`. -/
noncomputable def simplexDrift (f g : ℝ → ℝ) (γ : ℝ) : ℝ :=
  γ + (1 - γ) * g γ - γ * (γ + (1 - γ) * f γ)

/-- **Theorem (thm:clustering_random_init), first half.**  The two expectations
of `eq:f_g_def_selfcontained`,

  `f(γ) = 𝔼 Σ_k (π^A_{i→k}(γ))²`,  `g(γ) = 𝔼 Σ_k π^A_{i→k}(γ) π^A_{j→k}(γ)`,

are well defined: they depend on neither the simplex configuration realizing
the overlap `γ` nor on the choice of indices, and they take values in `[0,1]`.

The expectation is over `A ∼ ρ*`, which is read off the head `θ = (V, A)`.

Not proved here.

Source: arXiv:2604.01978v1, `eq:f_g_def_selfcontained`. -/
theorem simplex_overlap_wellDefined {d n : ℕ} (hd : 2 ≤ d) (hn : 2 ≤ n) (β : ℝ)
    (σV σA : ℝ≥0) (ρ : Measure (HeadParam d)) (hρ : IsGaussianHeadLaw d σV σA ρ) :
    ∃ f g : ℝ → ℝ,
      (∀ γ : ℝ, f γ ∈ Set.Icc (0 : ℝ) 1) ∧ (∀ γ : ℝ, g γ ∈ Set.Icc (0 : ℝ) 1) ∧
      (∀ (γ : ℝ) (x : Idx n → EucSpace d), IsSimplexConfig γ x → ∀ i : Idx n,
        (∫ θ, ∑ k : Idx n, attnProb β θ.2 x i k ^ 2 ∂ρ) = f γ) ∧
      (∀ (γ : ℝ) (x : Idx n → EucSpace d), IsSimplexConfig γ x → ∀ i j : Idx n, i ≠ j →
        (∫ θ, ∑ k : Idx n, attnProb β θ.2 x i k * attnProb β θ.2 x j k ∂ρ) = g γ) := by
  sorry

/-- The hypotheses of `simplex_overlap_wellDefined` are satisfiable. -/
example : 2 ≤ 2 ∧ 2 ≤ 2 ∧ IsGaussianHeadLaw 2 0 0 (Measure.dirac (0 : HeadParam 2)) :=
  ⟨le_rfl, le_rfl, isGaussianHeadLaw_dirac_zero 2⟩

/-- **Theorem (thm:clustering_random_init), second half.**  Fix `d, n ≥ 2` and
`β > 0`, start from a simplex configuration with overlap
`γ₀ ∈ (-1/(n-1), 1)`, and let `γ` solve `γ̇ = b(γ)`, `γ(0) = γ₀`.  Then, with
probability at least `1 - δ`, the solution of `eq:Diffusive_gaussian_case`
satisfies

  `sup_{t∈[0,T]} max_{i≠j} |⟨x_i(t), x_j(t)⟩ - γ(t)| ≤ e^{CT} √(8T/d log(2n²/δ))`,

with `C = O(1 + d σ_A⁴ β² n)`.

**What the source says and what is carried here.**  The two statements "there
exists `C > 0`" and "`C = O(1 + dσ_A⁴β²n)`" are read as one: a universal `K`,
produced before `d, n, β, σ_A` are chosen, with `C = K(1 + dσ_A⁴β²n)`.  That is
the content of the `O(·)`, and it is stronger than an unquantified `∃ C`.

The functions `f` and `g` are taken as explicit hypotheses, in the form
`simplex_overlap_wellDefined` produces them, rather than used from it: that
theorem is not proved here, and a result resting on it would not be proved
either.

The initial configuration is deterministic, as in the source, where `X(0)` is
given.

The ODE is the source's, `γ̇ = b(γ)`, although the identity the source derives
it from is false: the drift of an overlap on the simplex line is `b(γ)` only
as `d → ∞`, by `not_forall_overlapDrift_eq_simplexDrift` and
`overlapDrift_simplex_sub_simplexDrift`.  The `O(1/d)` discrepancy is left
here, inside the `e^{CT}√(8T/d log(2n²/δ))` the conclusion allows.

Not proved here.

Source: arXiv:2604.01978v1, `thm:clustering_random_init`. -/
theorem clustering_random_init :
    ∃ K : ℝ, 0 < K ∧
      ∀ (d n : ℕ), 2 ≤ d → 2 ≤ n → ∀ (β : ℝ), 0 < β →
      ∀ (σV σA : ℝ≥0) (ρ : Measure (HeadParam d)), IsGaussianHeadLaw d σV σA ρ →
      ∀ f g : ℝ → ℝ,
        (∀ (γ : ℝ) (x : Idx n → EucSpace d), IsSimplexConfig γ x → ∀ i : Idx n,
          (∫ θ, ∑ k : Idx n, attnProb β θ.2 x i k ^ 2 ∂ρ) = f γ) →
        (∀ (γ : ℝ) (x : Idx n → EucSpace d), IsSimplexConfig γ x → ∀ i j : Idx n, i ≠ j →
          (∫ θ, ∑ k : Idx n, attnProb β θ.2 x i k * attnProb β θ.2 x j k ∂ρ) = g γ) →
      ∀ (γ₀ : ℝ), γ₀ ∈ Set.Ioo (-(1 / ((n : ℝ) - 1))) 1 →
      ∀ γ : ℝ → ℝ, γ 0 = γ₀ →
        (∀ t ∈ Set.Ici (0 : ℝ),
          HasDerivWithinAt γ (simplexDrift f g (γ t)) (Set.Ici 0) t) →
      ∀ (T : ℝ), 0 < T → ∀ δ ∈ Set.Ioo (0 : ℝ) 1,
      ∀ (Ω : Type) [MeasurableSpace Ω] (P : Measure Ω)
        (X : ℝ → Ω → (Idx n → EucSpace d)) (x₀ : Idx n → EucSpace d),
        IsDiffusiveSde β ρ P X → (∀ ω, X 0 ω = x₀) → IsSimplexConfig γ₀ x₀ →
        ENNReal.ofReal (1 - δ) ≤
          P {ω | ∀ t ∈ Set.Icc (0 : ℝ) T, ∀ i j : Idx n, i ≠ j →
            |inner (𝕜 := ℝ) (X t ω i) (X t ω j) - γ t| ≤
              Real.exp (K * (1 + (d : ℝ) * (σA : ℝ) ^ 4 * β ^ 2 * (n : ℝ)) * T) *
                Real.sqrt (8 * T / (d : ℝ) * Real.log (2 * (n : ℝ) ^ 2 / δ))} := by
  sorry

/-- The hypotheses `clustering_random_init` quantifies over are satisfiable:
`d = n = 2`, `β = 1`, `ρ* = δ_0`, `γ₀ = 0 ∈ (-1, 1)`, `T = δ/2 = 1/2`, and the
degenerate solution of `eq:Diffusive_gaussian_case` started at an orthonormal
pair, which is a simplex configuration with overlap `0`. -/
example :
    (0 : ℝ) ∈ Set.Ioo (-(1 / ((2 : ℝ) - 1))) 1 ∧ (0 : ℝ) < 1 ∧
      (1 / 2 : ℝ) ∈ Set.Ioo (0 : ℝ) 1 ∧
      IsSimplexConfig (0 : ℝ) (fun i : Idx 2 => (EuclideanSpace.single i (1 : ℝ) : EucSpace 2)) := by
  refine ⟨⟨by norm_num, by norm_num⟩, one_pos, ⟨by norm_num, by norm_num⟩, ?_, ?_⟩
  · intro i; simp [PiLp.norm_single]
  · intro i j hij
    simp [EuclideanSpace.inner_single_left, hij.symm]

end Homogenized
end Transformer
