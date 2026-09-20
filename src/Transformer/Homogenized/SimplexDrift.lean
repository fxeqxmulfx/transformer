/-
# Homogenized Transformers — the drift on the simplex line

Formalization of `lem:drift_on_simplex_selfcontained` of arXiv:2604.01978v1,
*Homogenized Transformers*.  It has two halves.

* The drift `𝒟_ij(X)` of `eq:Dij_explicit_clean` depends on `X` only through
  its Gram matrix.  The content is that `s_{μ_X}` does (`baryCorr_of_gram`, an
  orthogonal-invariance statement about the Gaussian law of `A = W W'ᵀ`); the
  rest is the substitution `overlapDrift_congr`.

* On a simplex configuration with overlap `γ`, the two statistics collapse to
  `γ + (1-γ)f(γ)` and `γ + (1-γ)g(γ)` — that is `SimplexBary.lean` — and the
  drift becomes a function of `γ` alone, `overlapDrift_simplex`.

**Where this deviates from the source.**  The source says that plugging the two
statistics into `eq:Dij_explicit_clean` "yields exactly" the `b(γ)` of
`eq:Phi_def_selfcontained`, i.e. `𝒟_ij(R(γ)) = b(γ)`.  It does not: with
`S = γ + (1-γ)g(γ)` and `P = γ + (1-γ)f(γ)` the substitution gives

  `𝒟_ij = (1/d)(d-2+γ²) S - ((d-1)/d) γ P`,   while   `b(γ) = S - γ P`,

and the two differ by `(γP - (2-γ²)S)/d`, which is not zero — at `γ = 0` it is
`-2g(0)/d`, and `g(0) > 0` because the attention probabilities are positive.
`b(γ)` is the `d → ∞` form of the drift.  The corrected identity is
`overlapDrift_simplex` and the exact gap is
`overlapDrift_simplex_sub_simplexDrift`; the discrepancy is `O(1/d)`, below the
`O(√(T/d))` fluctuation `thm:clustering_random_init` allows, so the theorem the
lemma feeds is unaffected.

Source: arXiv:2604.01978v1, `lem:drift_on_simplex_selfcontained`.
-/

import Transformer.Homogenized.OverlapDrift
import Transformer.Homogenized.SimplexBary

open scoped BigOperators NNReal
open Real MeasureTheory

namespace Transformer
namespace Homogenized

/-! ### The drift is a function of the Gram matrix -/

/-- **Lemma (lem:drift_on_simplex_selfcontained), first half.**  Two
configurations of unit tokens with the same Gram matrix give the same value to
`s_{μ_X}(x_i, x_j)`.

Two configurations with the same Gram matrix differ by an orthogonal `O`, and
`⟨A x_i, x_j⟩ = ⟨Oᵀ A O x̃_i, x̃_j⟩` has the same law as `⟨A x̃_i, x̃_j⟩` because
the law of `A = W W'ᵀ` is orthogonally invariant.  That invariance is what is
unproved here; `overlapDrift_congr` is the substitution it feeds.

Not proved here.

Source: arXiv:2604.01978v1, proof of `lem:drift_on_simplex_selfcontained`. -/
theorem baryCorr_of_gram {d n : ℕ} (β : ℝ) (σV σA : ℝ≥0) (ρ : Measure (HeadParam d))
    (hρ : IsGaussianHeadLaw d σV σA ρ) (X Y : Idx n → EucSpace d)
    (hX : ∀ k : Idx n, ‖X k‖ = 1) (hY : ∀ k : Idx n, ‖Y k‖ = 1)
    (hgram : ∀ k l : Idx n, inner (𝕜 := ℝ) (X k) (X l) = inner (𝕜 := ℝ) (Y k) (Y l))
    (i j : Idx n) :
    baryCorr β ρ (empMeasure X) (X i) (X j) = baryCorr β ρ (empMeasure Y) (Y i) (Y j) := by
  sorry

/-- The hypotheses of `baryCorr_of_gram` are satisfiable: the degenerate head
law `ρ* = δ_0` and one orthonormal pair compared with itself. -/
example :
    IsGaussianHeadLaw 2 0 0 (Measure.dirac (0 : HeadParam 2)) ∧
      ∀ k : Idx 2, ‖(EuclideanSpace.single k (1 : ℝ) : EucSpace 2)‖ = 1 := by
  refine ⟨isGaussianHeadLaw_dirac_zero 2, fun k => by simp [PiLp.norm_single]⟩

/-- **Lemma (lem:drift_on_simplex_selfcontained), the substitution.**  Once the
three values of `s_{μ_X}` and the overlap `R_ij` agree, the drifts agree: this
is the "and hence `𝒟_ij(X)`" of the statement.

The three equalities are taken as hypotheses rather than read off
`baryCorr_of_gram`, which is not proved here; that keeps the dependence
legible in the signature.

Source: arXiv:2604.01978v1, `lem:drift_on_simplex_selfcontained`. -/
theorem overlapDrift_congr {d n : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    (X Y : Idx n → EucSpace d) (i j : Idx n)
    (hR : inner (𝕜 := ℝ) (X i) (X j) = inner (𝕜 := ℝ) (Y i) (Y j))
    (hij : baryCorr β ρ (empMeasure X) (X i) (X j) = baryCorr β ρ (empMeasure Y) (Y i) (Y j))
    (hii : baryCorr β ρ (empMeasure X) (X i) (X i) = baryCorr β ρ (empMeasure Y) (Y i) (Y i))
    (hjj : baryCorr β ρ (empMeasure X) (X j) (X j) = baryCorr β ρ (empMeasure Y) (Y j) (Y j)) :
    overlapDrift β ρ X i j = overlapDrift β ρ Y i j := by
  rw [overlapDrift, overlapDrift, hR, hij, hii, hjj]

/-- The hypotheses of `overlapDrift_congr` are satisfiable, at `Y = X`. -/
example {d n : ℕ} (β : ℝ) (ρ : Measure (HeadParam d)) (X : Idx n → EucSpace d) (i j : Idx n) :
    inner (𝕜 := ℝ) (X i) (X j) = inner (𝕜 := ℝ) (X i) (X j) ∧
      baryCorr β ρ (empMeasure X) (X i) (X j) = baryCorr β ρ (empMeasure X) (X i) (X j) :=
  ⟨rfl, rfl⟩

/-! ### The drift on the simplex line -/

/-- **Lemma (lem:drift_on_simplex_selfcontained), second half, corrected.**  On
a simplex configuration with overlap `γ`, and for `i ≠ j`,

  `𝒟_ij(X) = (1/d)(d-2+γ²)(γ + (1-γ)g(γ)) - ((d-1)/d) γ (γ + (1-γ)f(γ))`.

**What the source says and what is changed.**  The source asserts that this
substitution "yields exactly" `b(γ) = γ + (1-γ)g(γ) - γ(γ + (1-γ)f(γ))` of
`eq:Phi_def_selfcontained`.  The coefficients `1/d` and `(d-1)/(2d)` of
`eq:Dij_explicit_clean` do not disappear, and what is stated here is the result
of the substitution as written; `overlapDrift_simplex_sub_simplexDrift` is the
exact difference from `b(γ)`, which is `O(1/d)`.

`f` and `g` are taken as explicit hypotheses in the form
`simplex_overlap_wellDefined` produces them, and the integrability that lets
the head law be integrated term by term is likewise explicit.

Source: arXiv:2604.01978v1, `lem:drift_on_simplex_selfcontained`,
`eq:Dij_explicit_clean`. -/
theorem overlapDrift_simplex {d n : ℕ} (hd : (d : ℝ) ≠ 0) (β : ℝ)
    (ρ : Measure (HeadParam d)) [IsProbabilityMeasure ρ] (f g : ℝ → ℝ)
    (hf : ∀ (c : ℝ) (x : Idx (n + 1) → EucSpace d), IsSimplexConfig c x → ∀ k : Idx (n + 1),
      (∫ θ, ∑ l : Idx (n + 1), attnProb β θ.2 x k l ^ 2 ∂ρ) = f c)
    (hg : ∀ (c : ℝ) (x : Idx (n + 1) → EucSpace d), IsSimplexConfig c x →
      ∀ k l : Idx (n + 1), k ≠ l →
      (∫ θ, ∑ m : Idx (n + 1), attnProb β θ.2 x k m * attnProb β θ.2 x l m ∂ρ) = g c)
    (γ : ℝ) (X : Idx (n + 1) → EucSpace d) (hX : IsSimplexConfig γ X)
    (hsq : ∀ k : Idx (n + 1),
      Integrable (fun θ => ∑ l : Idx (n + 1), attnProb β θ.2 X k l ^ 2) ρ)
    (hpr : ∀ k l : Idx (n + 1),
      Integrable (fun θ => ∑ m : Idx (n + 1), attnProb β θ.2 X k m * attnProb β θ.2 X l m) ρ)
    (i j : Idx (n + 1)) (hij : i ≠ j) :
    overlapDrift β ρ X i j
      = 1 / (d : ℝ) * ((d : ℝ) - 2 + γ ^ 2) * (γ + (1 - γ) * g γ)
        - ((d : ℝ) - 1) / (d : ℝ) * γ * (γ + (1 - γ) * f γ) := by
  rw [overlapDrift, hX.2 i j hij,
    baryCorr_simplex β ρ γ X hX i j (hpr i j), hg γ X hX i j hij,
    baryCorr_simplex_self β ρ γ X hX i (hsq i), hf γ X hX i,
    baryCorr_simplex_self β ρ γ X hX j (hsq j), hf γ X hX j]
  field_simp
  ring

/-- **The gap between the drift and `eq:Phi_def_selfcontained`.**  With
`S = γ + (1-γ)g(γ)` and `P = γ + (1-γ)f(γ)`,

  `𝒟_ij(X) - b(γ) = (γP - (2-γ²)S)/d`.

This is the exact size of the source's "yields exactly": `b(γ)` is the drift's
`d → ∞` form, and the error is `O(1/d)`.  It does not vanish — at `γ = 0` it is
`-2g(0)/d` and `g(0) > 0`, the attention probabilities being positive.

Source: arXiv:2604.01978v1, `lem:drift_on_simplex_selfcontained`,
`eq:Phi_def_selfcontained`. -/
theorem overlapDrift_simplex_sub_simplexDrift {d n : ℕ} (hd : (d : ℝ) ≠ 0) (β : ℝ)
    (ρ : Measure (HeadParam d)) [IsProbabilityMeasure ρ] (f g : ℝ → ℝ)
    (hf : ∀ (c : ℝ) (x : Idx (n + 1) → EucSpace d), IsSimplexConfig c x → ∀ k : Idx (n + 1),
      (∫ θ, ∑ l : Idx (n + 1), attnProb β θ.2 x k l ^ 2 ∂ρ) = f c)
    (hg : ∀ (c : ℝ) (x : Idx (n + 1) → EucSpace d), IsSimplexConfig c x →
      ∀ k l : Idx (n + 1), k ≠ l →
      (∫ θ, ∑ m : Idx (n + 1), attnProb β θ.2 x k m * attnProb β θ.2 x l m ∂ρ) = g c)
    (γ : ℝ) (X : Idx (n + 1) → EucSpace d) (hX : IsSimplexConfig γ X)
    (hsq : ∀ k : Idx (n + 1),
      Integrable (fun θ => ∑ l : Idx (n + 1), attnProb β θ.2 X k l ^ 2) ρ)
    (hpr : ∀ k l : Idx (n + 1),
      Integrable (fun θ => ∑ m : Idx (n + 1), attnProb β θ.2 X k m * attnProb β θ.2 X l m) ρ)
    (i j : Idx (n + 1)) (hij : i ≠ j) :
    overlapDrift β ρ X i j - simplexDrift f g γ
      = (γ * (γ + (1 - γ) * f γ) - (2 - γ ^ 2) * (γ + (1 - γ) * g γ)) / (d : ℝ) := by
  rw [overlapDrift_simplex hd β ρ f g hf hg γ X hX hsq hpr i j hij, simplexDrift]
  field_simp
  ring

/-- The hypotheses of `overlapDrift_simplex` are satisfiable: `d = n = 2`,
`ρ* = δ_0`, an orthonormal pair with overlap `γ = 0`, and the constants
`f = g = 1/2` that the uniform attention of the degenerate head law produces. -/
example (β : ℝ) :
    ((2 : ℕ) : ℝ) ≠ 0 ∧ IsProbabilityMeasure (Measure.dirac (0 : HeadParam 2)) ∧
      IsSimplexConfig (0 : ℝ) (fun k : Idx 2 => (EuclideanSpace.single k (1 : ℝ) : EucSpace 2)) ∧
      (∀ (c : ℝ) (x : Idx 2 → EucSpace 2), IsSimplexConfig c x → ∀ k : Idx 2,
        (∫ θ, ∑ l : Idx 2, attnProb β θ.2 x k l ^ 2 ∂(Measure.dirac (0 : HeadParam 2)))
          = (1 / 2 : ℝ)) ∧
      (∀ (c : ℝ) (x : Idx 2 → EucSpace 2), IsSimplexConfig c x → ∀ k l : Idx 2, k ≠ l →
        (∫ θ, ∑ m : Idx 2, attnProb β θ.2 x k m * attnProb β θ.2 x l m
            ∂(Measure.dirac (0 : HeadParam 2))) = (1 / 2 : ℝ)) ∧
      (∀ k l : Idx 2, Integrable (fun θ : HeadParam 2 => ∑ m : Idx 2,
          attnProb β θ.2 (fun k : Idx 2 => (EuclideanSpace.single k (1 : ℝ) : EucSpace 2)) k m *
            attnProb β θ.2 (fun k : Idx 2 => (EuclideanSpace.single k (1 : ℝ) : EucSpace 2)) l m)
        (Measure.dirac (0 : HeadParam 2))) := by
  refine ⟨by norm_num, inferInstance,
    ⟨fun k => by simp [PiLp.norm_single], fun k l hkl => ?_⟩, ?_, ?_,
    fun _ _ => integrable_dirac enorm_lt_top⟩
  · simp [EuclideanSpace.inner_single_left, hkl.symm]
  · intro c x _ k
    rw [integral_dirac]
    simp [attnProb, attnWeight, qkMap]
    norm_num
  · intro c x _ k l _
    rw [integral_dirac]
    simp [attnProb, attnWeight, qkMap]

end Homogenized
end Transformer
