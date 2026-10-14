/-
# The actual simplex error and its drift bound

ArXiv:2604.01978v1, §7, the error maximum E(t) and
eq:Gronwall_E_selfcontained.
The finite supremum norm below ranges over the original distinct
token pairs. It equals the source's largest overlap error and
controls the full original Gram difference because its diagonal
entries cancel for unit tokens. No error bound is assumed in its
definition.

The proved Gram stability and finite dimension simplex bias give
the actual drift bound with coefficient 80 and additive term 3/d.
Continuous original token and scalar paths give a continuous error,
as required for the integral Grönwall argument.
-/

import Transformer.Homogenized.Section7_SimplexDriftBias

open MeasureTheory Set
open scoped BigOperators NNReal

namespace Transformer.Homogenized

/-- The original distinct token pairs in the error maximum.
Source: arXiv:2604.01978v1, §7, eq:Gronwall_E_selfcontained. -/
abbrev DistinctPairIdx (n : ℕ) := {p : Idx n × Idx n // p.1 ≠ p.2}

/-- The actual largest off-diagonal overlap error, as a finite supremum norm.
Source: arXiv:2604.01978v1, §7, the displayed definition of E(t). -/
noncomputable def simplexOverlapError {d n : ℕ} (X : Idx n → EucSpace d) (γ : ℝ) : ℝ :=
  ‖fun p : DistinctPairIdx n => inner (𝕜 := ℝ) (X p.1.1) (X p.1.2) - γ‖

/-- Every original distinct-pair error is bounded by the actual maximum.
Source: arXiv:2604.01978v1, §7, the definition of E(t). -/
theorem abs_overlap_sub_simplex_le_error {d n : ℕ} (X : Idx n → EucSpace d) (γ : ℝ) (i j : Idx n) (hij : i ≠ j) :
    |inner (𝕜 := ℝ) (X i) (X j) - γ| ≤ simplexOverlapError X γ := by
  simpa only [simplexOverlapError, Real.norm_eq_abs] using
    norm_le_pi_norm (fun p : DistinctPairIdx n => inner (𝕜 := ℝ) (X p.1.1) (X p.1.2) - γ) ⟨(i, j), hij⟩

/-- Simultaneous bounds on all original distinct pairs bound their actual maximum.
Source: arXiv:2604.01978v1, §7, the definition of E(t). -/
theorem simplexOverlapError_le {d n : ℕ} (X : Idx n → EucSpace d) (γ C : ℝ) (hC : 0 ≤ C)
    (herror : ∀ i j : Idx n, i ≠ j → |inner (𝕜 := ℝ) (X i) (X j) - γ| ≤ C) :
    simplexOverlapError X γ ≤ C := by
  apply (pi_norm_le_iff_of_nonneg hC).2
  intro p
  rw [Real.norm_eq_abs]
  exact herror p.1.1 p.1.2 p.2

/-- The initial error vanishes exactly on the stated simplex Gram matrix.
Source: arXiv:2604.01978v1, §7, the initial-error step before eq:Gronwall_E_selfcontained. -/
theorem simplexOverlapError_eq_zero_iff {d n : ℕ} (X : Idx n → EucSpace d) (hX : ∀ i, ‖X i‖ = 1) (γ : ℝ) :
    simplexOverlapError X γ = 0 ↔ IsSimplexConfig γ X := by
  constructor
  · intro hzero
    refine ⟨hX, fun i j hij => ?_⟩
    have h := abs_overlap_sub_simplex_le_error X γ i j hij
    rw [hzero] at h
    exact sub_eq_zero.mp (abs_nonpos_iff.mp h)
  · intro hsimplex
    apply le_antisymm
    · apply simplexOverlapError_le X γ 0 le_rfl
      intro i j hij
      rw [hsimplex.2 i j hij, sub_self, abs_zero]
    · exact norm_nonneg _

/-- Continuous original token and ODE paths give a continuous actual error.
Source: arXiv:2604.01978v1, §7, the Grönwall step in eq:Gronwall_E_selfcontained. -/
theorem continuousOn_simplexOverlapError {d n : ℕ} (X : ℝ → Idx n → EucSpace d) (γ : ℝ → ℝ) (S : Set ℝ)
    (hX : ContinuousOn X S) (hγ : ContinuousOn γ S) :
    ContinuousOn (fun t => simplexOverlapError (X t) (γ t)) S := by
  apply ContinuousOn.norm
  apply continuousOn_pi.2
  intro p
  exact ((continuous_apply p.1.1).comp_continuousOn hX).inner
    ((continuous_apply p.1.2).comp_continuousOn hX) |>.sub hγ

/-- The full original Gram difference is bounded by the off-diagonal error;
the common unit diagonals cancel. Source: arXiv:2604.01978v1, §7, lem:Gram_stability. -/
theorem abs_gram_sub_simplex_le_error {d n : ℕ} (X Y : Idx n → EucSpace d) (hX : ∀ i, ‖X i‖ = 1)
    (γ : ℝ) (hY : IsSimplexConfig γ Y) (i j : Idx n) :
    |inner (𝕜 := ℝ) (X i) (X j) - inner (𝕜 := ℝ) (Y i) (Y j)| ≤ simplexOverlapError X γ := by
  by_cases hij : i = j
  · subst j
    simp only [real_inner_self_eq_norm_sq, hX i, hY.1 i, one_pow, sub_self, abs_zero]
    exact norm_nonneg _
  · rw [hY.2 i j hij]
    exact abs_overlap_sub_simplex_le_error X γ i j hij

/-- The actual maximum and the entrywise full Gram bound are equivalent.
Source: arXiv:2604.01978v1, §7, the passage from lem:Gram_stability to E(t). -/
theorem simplexOverlapError_le_iff_gram {d n : ℕ} (X Y : Idx n → EucSpace d) (hX : ∀ i, ‖X i‖ = 1)
    (γ ε : ℝ) (hε : 0 ≤ ε) (hY : IsSimplexConfig γ Y) :
    simplexOverlapError X γ ≤ ε ↔
      ∀ i j, |inner (𝕜 := ℝ) (X i) (X j) - inner (𝕜 := ℝ) (Y i) (Y j)| ≤ ε := by
  constructor
  · exact fun h i j => (abs_gram_sub_simplex_le_error X Y hX γ hY i j).trans h
  · intro h
    apply simplexOverlapError_le X γ ε hε
    intro i j hij
    simpa only [hY.2 i j hij] using h i j

/-- The actual drift error is at most 80(1+dσA⁴β²n) E + 3/d.
Source: arXiv:2604.01978v1, §7, the estimate before eq:Gronwall_E_selfcontained,
with the nonzero finite dimension simplex bias retained. -/
theorem abs_overlapDrift_sub_simplexDrift_le_error {d n : ℕ} (hd : 2 ≤ d) (β : ℝ) (σV σA : ℝ≥0)
    (ρ : Measure (HeadParam d)) (hρ : IsGaussianHeadLaw d σV σA ρ) (f g : ℝ → ℝ)
    (hf : ∀ (c : ℝ) (x : Idx (n + 1) → EucSpace d), IsSimplexConfig c x → ∀ k : Idx (n + 1),
      (∫ θ, ∑ l : Idx (n + 1), attnProb β θ.2 x k l ^ 2 ∂ρ) = f c)
    (hg : ∀ (c : ℝ) (x : Idx (n + 1) → EucSpace d), IsSimplexConfig c x →
      ∀ k l : Idx (n + 1), k ≠ l →
      (∫ θ, ∑ m : Idx (n + 1), attnProb β θ.2 x k m * attnProb β θ.2 x l m ∂ρ) = g c)
    (X Y : Idx (n + 1) → EucSpace d) (hX : ∀ i, ‖X i‖ = 1)
    (γ : ℝ) (hY : IsSimplexConfig γ Y) (i j : Idx (n + 1)) (hij : i ≠ j) :
    |overlapDrift β ρ X i j - simplexDrift f g γ| ≤
      (80 * (1 + (d : ℝ) * (σA : ℝ) ^ 4 * β ^ 2 * ((n + 1 : ℕ) : ℝ))) *
        simplexOverlapError X γ + 3 / (d : ℝ) := by
  have : IsProbabilityMeasure ρ := isProbabilityMeasure_of_isGaussianHeadLaw hρ
  have hgram := abs_overlapDrift_sub_le_gram hd β σV σA ρ hρ X Y hX hY.1 i j
    (simplexOverlapError X γ) (abs_gram_sub_simplex_le_error X Y hX γ hY)
  have hsimplex := abs_overlapDrift_simplex_sub_simplexDrift_le
    (lt_of_lt_of_le (by norm_num : 0 < 2) hd) β ρ f g hf hg γ Y hY i j hij
  exact (abs_sub_le _ _ _).trans (add_le_add hgram hsimplex)

/-- Different unit Gram matrices with actual error one and a positive-variance
Gaussian head realize every norm, continuity, moment and drift premise. -/
example : ∃ (f g : ℝ → ℝ) (X Y : Idx 2 → EucSpace 2),
    (2 : ℕ) ≤ 2 ∧ IsGaussianHeadLaw 2 1 1 (gaussHeadLaw 2 1 1) ∧
    (∀ i, ‖X i‖ = 1) ∧ IsSimplexConfig (0 : ℝ) Y ∧ (0 : Idx 2) ≠ 1 ∧
    simplexOverlapError X 0 = 1 ∧
    (∀ i j, |inner (𝕜 := ℝ) (X i) (X j) - inner (𝕜 := ℝ) (Y i) (Y j)| ≤ 1) ∧
    ContinuousOn (fun _ : ℝ => X) (Icc (0 : ℝ) 1) ∧
    ContinuousOn (fun _ : ℝ => (0 : ℝ)) (Icc (0 : ℝ) 1) ∧
    (∀ (c : ℝ) (x : Idx 2 → EucSpace 2), IsSimplexConfig c x → ∀ k : Idx 2,
      (∫ θ, ∑ l : Idx 2, attnProb 1 θ.2 x k l ^ 2 ∂(gaussHeadLaw 2 1 1)) = f c) ∧
    (∀ (c : ℝ) (x : Idx 2 → EucSpace 2), IsSimplexConfig c x → ∀ k l : Idx 2, k ≠ l →
      (∫ θ, ∑ m : Idx 2, attnProb 1 θ.2 x k m * attnProb 1 θ.2 x l m ∂(gaussHeadLaw 2 1 1)) = g c) := by
  let X : Idx 2 → EucSpace 2 := fun _ => EuclideanSpace.single (0 : Fin 2) 1
  let Y : Idx 2 → EucSpace 2 := fun k => EuclideanSpace.single k 1
  obtain ⟨f, g, -, -, hf, hg⟩ :=
    simplex_overlap_wellDefined (d := 2) (n := 2) 1 1 1 _ (isGaussianHeadLaw_gaussHeadLaw 2 1 1)
  refine ⟨f, g, X, Y, le_rfl, isGaussianHeadLaw_gaussHeadLaw 2 1 1,
    fun i => by simp [X], ?_, by decide, ?_, ?_, continuousOn_const, continuousOn_const, hf, hg⟩
  · constructor
    · intro i
      simp [Y]
    · intro i j hij
      simp [Y, EuclideanSpace.inner_single_left, hij.symm]
  · apply le_antisymm
    · apply simplexOverlapError_le X 0 1 zero_le_one
      intro i j hij
      norm_num [X, EuclideanSpace.inner_single_left, PiLp.single_apply]
    · have h := abs_overlap_sub_simplex_le_error X 0 0 1 (by decide)
      simpa [X, EuclideanSpace.inner_single_left, PiLp.single_apply] using h
  · intro i j
    fin_cases i <;> fin_cases j <;> norm_num [X, Y, EuclideanSpace.inner_single_left, PiLp.single_apply]

end Transformer.Homogenized
