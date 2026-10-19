module

public import Transformer.Homogenized.GaussianInit

/-!
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

@[expose] public section

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

/-- Attention probabilities are nonnegative. -/
theorem attnProb_nonneg {d n : ℕ} (β : ℝ) (A : Matrix (Fin d) (Fin d) ℝ)
    (x : Idx n → EucSpace d) (i k : Idx n) : 0 ≤ attnProb β A x i k :=
  div_nonneg (attnWeight_pos _ _ _ _).le
    (Finset.sum_nonneg fun _ _ => (attnWeight_pos _ _ _ _).le)

/-- The attention probabilities of a token sum to at most one (to zero when there
are no tokens). -/
theorem sum_attnProb_le_one {d n : ℕ} (β : ℝ) (A : Matrix (Fin d) (Fin d) ℝ)
    (x : Idx n → EucSpace d) (i : Idx n) : ∑ k : Idx n, attnProb β A x i k ≤ 1 := by
  simp only [attnProb, ← Finset.sum_div]
  exact div_self_le_one _

/-- An attention probability is at most one. -/
theorem attnProb_le_one {d n : ℕ} (β : ℝ) (A : Matrix (Fin d) (Fin d) ℝ)
    (x : Idx n → EucSpace d) (i k : Idx n) : attnProb β A x i k ≤ 1 :=
  (Finset.single_le_sum (fun l _ => attnProb_nonneg β A x i l) (Finset.mem_univ k)).trans
    (sum_attnProb_le_one β A x i)

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

/- The probability theorem is proved downstream in Section7_ClusteringProbability. -/

/-- The strict-interior initial geometry of `thm:clustering_random_init`
is realized by an orthonormal pair with overlap zero, positive horizon,
and failure probability in (0,1). Source: arXiv:2604.01978v1, §2.3.4. -/
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
