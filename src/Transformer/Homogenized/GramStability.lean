/-
# Homogenized Transformers — the drift is Lipschitz near the simplex line

Formalization of `lem:Gram_stability` of arXiv:2604.01978v1, *Homogenized
Transformers*, together with a refutation of the identity it takes for granted.

The lemma bounds `max_{i≠j} |𝒟_ij(R) - b(γ)|` by `L ‖R - R(γ)‖_∞` with
`L = O(1 + dσ_A⁴β²n)`, and adds parenthetically that `𝒟_ij(R(γ)) = b(γ)` —
the claim of `lem:drift_on_simplex_selfcontained`.  That claim is false, and
`not_forall_overlapDrift_eq_simplexDrift` refutes it: in `d = n = 2`, at the
head law (G) with `σ_V² = 1/d` and `σ_A = 0`, and an orthonormal pair, the trace factor
`d - 2 + R_ij²` of `eq:Dij_explicit_clean` vanishes, so the drift of the
overlap is `0`, while `b(0) = g(0) = 1/2`.  With it the lemma's own statement
is false too, since at `R = R(γ)` the right-hand side is `0` and the left-hand
side is not.

What survives, and is what the Grönwall argument of `thm:clustering_random_init`
uses, is the Lipschitz bound with the drift at a simplex configuration in place
of `b(γ)`: that is `gram_stability`.

Source: arXiv:2604.01978v1, `lem:Gram_stability`.
-/

import Transformer.Homogenized.SimplexDrift

open scoped BigOperators NNReal
open Real MeasureTheory

namespace Transformer
namespace Homogenized

/-! ### The identity `𝒟_ij(R(γ)) = b(γ)` is false -/

/-- **Refutation of `𝒟_ij(R(γ)) = b(γ)`**, the conclusion of
`lem:drift_on_simplex_selfcontained` and the parenthetical of
`lem:Gram_stability`.

The witness is the source's own setting at its smallest admissible size:
`d = n = 2`, `β = 1`, the head law (G) at the standard scaling `σ_V² = 1/d`
of §2.5 — under which `eq:Dij_explicit_clean`, with its prefactor `1/d`, is
derived — and `σ_A = 0`, which the source leaves free; and an orthonormal
pair, a simplex configuration with overlap `γ = 0 ∈ (-1/(n-1), 1) = (-1,1)`.
At `σ_A = 0` the matrix `A = WW'ᵀ` vanishes almost surely, so `f = g ≡ 1/2`
and `b(0) = 1/2`, while `eq:Dij_explicit_clean` gives
`𝒟_{12} = (1/d)(d - 2 + 0)·(1/2) = 0`: in the plane the projections `𝐏_{x_1}`
and `𝐏_{x_2}` onto two orthogonal lines compose to `0`, so
`Tr(𝐏_{x_1}𝐏_{x_2}) = d - 2 + R² = 0` and the overlap does not move at all.

`σ_A = 0` is not what the refutation rests on: at `γ = 0` the gap of
`overlapDrift_simplex_sub_simplexDrift` is `-2g(0)/d`, and `g(0) > 0` for
every head law, since attention probabilities are positive.  It is only what
makes `f` and `g` computable in closed form here.

The general gap is `overlapDrift_simplex_sub_simplexDrift`, which is `O(1/d)`;
`b(γ)` is the drift's `d → ∞` form.

Source: arXiv:2604.01978v1, `lem:drift_on_simplex_selfcontained`,
`eq:Phi_def_selfcontained`. -/
theorem not_forall_overlapDrift_eq_simplexDrift :
    ¬ ∀ (d n : ℕ) (β : ℝ) (σV σA : ℝ≥0) (ρ : Measure (HeadParam d))
        (f g : ℝ → ℝ) (γ : ℝ) (X : Idx (n + 1) → EucSpace d) (i j : Idx (n + 1)),
      2 ≤ d → 2 ≤ n + 1 → 0 < β → (σV : ℝ) ^ 2 = 1 / d → IsProbabilityMeasure ρ →
      IsGaussianHeadLaw d σV σA ρ →
      γ ∈ Set.Ioo (-(1 / (n : ℝ))) 1 → IsSimplexConfig γ X → i ≠ j →
      (∀ (c : ℝ) (x : Idx (n + 1) → EucSpace d), IsSimplexConfig c x → ∀ k : Idx (n + 1),
        (∫ θ, ∑ l : Idx (n + 1), attnProb β θ.2 x k l ^ 2 ∂ρ) = f c) →
      (∀ (c : ℝ) (x : Idx (n + 1) → EucSpace d), IsSimplexConfig c x →
        ∀ k l : Idx (n + 1), k ≠ l →
        (∫ θ, ∑ m : Idx (n + 1), attnProb β θ.2 x k m * attnProb β θ.2 x l m ∂ρ) = g c) →
      (∀ k : Idx (n + 1),
        Integrable (fun θ => ∑ l : Idx (n + 1), attnProb β θ.2 X k l ^ 2) ρ) →
      (∀ k l : Idx (n + 1),
        Integrable (fun θ => ∑ m : Idx (n + 1),
          attnProb β θ.2 X k m * attnProb β θ.2 X l m) ρ) →
      overlapDrift β ρ X i j = simplexDrift f g γ := by
  intro hall
  set X : Idx 2 → EucSpace 2 := fun k => (EuclideanSpace.single k (1 : ℝ) : EucSpace 2) with hXdef
  have hsimp : IsSimplexConfig (0 : ℝ) X := by
    refine ⟨fun k => by simp [hXdef, PiLp.norm_single], fun k l hkl => ?_⟩
    simp [hXdef, EuclideanSpace.inner_single_left, hkl.symm]
  set ρ := gaussHeadLaw 2 (stdSigmaV 2) 0 with hρ
  have hf : ∀ (c : ℝ) (x : Idx 2 → EucSpace 2), IsSimplexConfig c x → ∀ k : Idx 2,
      (∫ θ, ∑ l : Idx 2, attnProb 1 θ.2 x k l ^ 2 ∂ρ) = (1 / 2 : ℝ) := by
    intro c x _ k
    rw [hρ, integral_snd_gaussHeadLaw_zero 2 (stdSigmaV 2)
      (fun A => ∑ l : Idx 2, attnProb 1 A x k l ^ 2)]
    simp [attnProb, attnWeight, qkMap]
    norm_num
  have hg : ∀ (c : ℝ) (x : Idx 2 → EucSpace 2), IsSimplexConfig c x → ∀ k l : Idx 2, k ≠ l →
      (∫ θ, ∑ m : Idx 2, attnProb 1 θ.2 x k m * attnProb 1 θ.2 x l m ∂ρ)
        = (1 / 2 : ℝ) := by
    intro c x _ k l _
    rw [hρ, integral_snd_gaussHeadLaw_zero 2 (stdSigmaV 2)
      (fun A => ∑ m : Idx 2, attnProb 1 A x k m * attnProb 1 A x l m)]
    simp [attnProb, attnWeight, qkMap]
  have hsq : ∀ k : Idx 2, Integrable
      (fun θ : HeadParam 2 => ∑ l : Idx 2, attnProb 1 θ.2 X k l ^ 2) ρ := fun k =>
    integrable_snd_gaussHeadLaw_zero 2 _ (fun A => ∑ l : Idx 2, attnProb 1 A X k l ^ 2)
  have hpr : ∀ k l : Idx 2, Integrable
      (fun θ : HeadParam 2 => ∑ m : Idx 2, attnProb 1 θ.2 X k m * attnProb 1 θ.2 X l m) ρ :=
    fun k l => integrable_snd_gaussHeadLaw_zero 2 _
      (fun A => ∑ m : Idx 2, attnProb 1 A X k m * attnProb 1 A X l m)
  have hne : (0 : Idx 2) ≠ 1 := by decide
  have hgap := overlapDrift_simplex_sub_simplexDrift (d := 2) (n := 1) (by norm_num) 1
    ρ (fun _ => (1 / 2 : ℝ)) (fun _ => (1 / 2 : ℝ))
    hf hg 0 X hsimp hsq hpr 0 1 hne
  have heq := hall 2 1 1 (stdSigmaV 2) 0 ρ
    (fun _ => (1 / 2 : ℝ)) (fun _ => (1 / 2 : ℝ)) 0 X 0 1 (by norm_num) (by norm_num) one_pos
    (by simp) inferInstance (isGaussianHeadLaw_gaussHeadLaw 2 _ _) (by norm_num) hsimp hne
    hf hg hsq hpr
  rw [heq, sub_self] at hgap
  norm_num at hgap

/-! ### The Lipschitz bound -/

/-- **Lemma (lem:Gram_stability).**  The drift of an overlap is Lipschitz in the
Gram matrix, uniformly along the simplex line: there is a universal `K` such
that whenever `X` is a configuration of unit tokens, `Y` is a simplex
configuration with overlap `γ ∈ (-1/(n-1), 1)`, and the two Gram matrices
differ by at most `ε` entrywise, then for every `i ≠ j`

  `|𝒟_ij(X) - 𝒟_ij(Y)| ≤ K(1 + dσ_A⁴β²n) ε`.

**What the source says and what is carried here.**

* The source writes the right-hand side of the comparison as `b(γ)` of
  `eq:Phi_def_selfcontained`, on the strength of the parenthetical
  `𝒟_ij(R(γ)) = b(γ)`.  That identity is false —
  `not_forall_overlapDrift_eq_simplexDrift` — and with `b(γ)` in place the
  lemma is false as well, since at `X = Y` the right-hand side is `0` and the
  left-hand side is the nonzero gap of
  `overlapDrift_simplex_sub_simplexDrift`.  What is stated here is the source's
  own intended reading, the drift at a configuration realizing `R(γ)`.
* `‖R - R(γ)‖_∞ = max_{k,l}|R_kl - R(γ)_kl|` is carried as a bound `ε` on every
  entry rather than as a supremum, which is the same statement and avoids
  naming a maximum.
* "there exists `L > 0`" and "`L = O(1 + dσ_A⁴β²n)`" are read as one, as in
  `clustering_random_init`: a universal `K` produced before `d, n, β, σ_A`.
* `R(γ)` is not asserted to be realizable; `Y` is a hypothesis, and in
  dimension `d ≥ n` it exists.

Not proved here.

Source: arXiv:2604.01978v1, `lem:Gram_stability`. -/
theorem gram_stability :
    ∃ K : ℝ, 0 < K ∧
      ∀ (d n : ℕ), 2 ≤ d → 1 ≤ n → ∀ (β : ℝ), 0 < β →
      ∀ (σV σA : ℝ≥0) (ρ : Measure (HeadParam d)), IsGaussianHeadLaw d σV σA ρ →
      ∀ (γ : ℝ), γ ∈ Set.Ioo (-(1 / (n : ℝ))) 1 →
      ∀ (X Y : Idx (n + 1) → EucSpace d), (∀ k : Idx (n + 1), ‖X k‖ = 1) →
        IsSimplexConfig γ Y →
      ∀ (ε : ℝ), (∀ k l : Idx (n + 1),
          |inner (𝕜 := ℝ) (X k) (X l) - inner (𝕜 := ℝ) (Y k) (Y l)| ≤ ε) →
      ∀ i j : Idx (n + 1), i ≠ j →
        |overlapDrift β ρ X i j - overlapDrift β ρ Y i j|
          ≤ K * (1 + (d : ℝ) * (σA : ℝ) ^ 4 * β ^ 2 * ((n : ℝ) + 1)) * ε := by
  sorry

/-- The hypotheses of `gram_stability` are satisfiable non-trivially: `d = 2`,
`n + 1 = 2`, `β = 1`, `ρ* = δ_0`, `γ = 0 ∈ (-1, 1)`, and `X = Y` the
orthonormal pair, at `ε = 0`. -/
example : (2 : ℕ) ≤ 2 ∧ (1 : ℕ) ≤ 1 ∧ (0 : ℝ) < 1 ∧
    IsGaussianHeadLaw 2 0 0 (Measure.dirac (0 : HeadParam 2)) ∧
    (0 : ℝ) ∈ Set.Ioo (-(1 / ((1 : ℕ) : ℝ))) 1 ∧
    (∀ k : Idx 2, ‖(EuclideanSpace.single k (1 : ℝ) : EucSpace 2)‖ = 1) ∧
    IsSimplexConfig (0 : ℝ) (fun k : Idx 2 => (EuclideanSpace.single k (1 : ℝ) : EucSpace 2)) ∧
    (∀ k l : Idx 2,
      |inner (𝕜 := ℝ) (EuclideanSpace.single k (1 : ℝ) : EucSpace 2)
          (EuclideanSpace.single l (1 : ℝ) : EucSpace 2) -
        inner (𝕜 := ℝ) (EuclideanSpace.single k (1 : ℝ) : EucSpace 2)
          (EuclideanSpace.single l (1 : ℝ) : EucSpace 2)| ≤ 0) := by
  refine ⟨le_rfl, le_rfl, one_pos, isGaussianHeadLaw_dirac_zero 2, ⟨by norm_num, by norm_num⟩,
    fun k => by simp [PiLp.norm_single], ⟨fun k => by simp [PiLp.norm_single], fun k l hkl => ?_⟩,
    fun k l => by simp⟩
  simp [EuclideanSpace.inner_single_left, hkl.symm]

end Homogenized
end Transformer
