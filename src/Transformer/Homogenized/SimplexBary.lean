/-
# Homogenized Transformers — the barycenter of a simplex configuration

The computation inside the proof of `lem:drift_on_simplex_selfcontained` of
arXiv:2604.01978v1, *Homogenized Transformers*: on a configuration of unit
tokens with a common pairwise overlap `γ`, the softmax barycenter of `x_i` is
the attention average

  `m_i = Σ_k π^A_{i→k}(γ) x_k`,

and the Gram structure collapses the double sum,

  `⟨m_i, m_j⟩ = γ + (1-γ) Σ_k π^A_{i→k}(γ) π^A_{j→k}(γ)`,

whose expectation over the head law is `s_{μ_X}(x_i, x_j)`.  At `i = j` this is
the source's `‖m_i‖² = γ + (1-γ) Σ_k (π^A_{i→k})²`; there is one identity, not
two.

`Simplex.lean` holds `π^A_{i→k}` and the simplex configurations,
`Barycenter.lean` holds `m_{β,A}[μ]` and `s_μ`, and this file is the bridge.

Source: arXiv:2604.01978v1, proof of `lem:drift_on_simplex_selfcontained`.
-/

import Transformer.Homogenized.Barycenter
import Transformer.Homogenized.Simplex

open scoped BigOperators NNReal
open Real MeasureTheory

namespace Transformer
namespace Homogenized

/-! ### The barycenter of an empirical measure is the attention average -/

/-- The attention probability of `eq:pi_simplex_def_selfcontained` is written in
the weight of `eq: mbetaA`: the head `(0, A)` has query-key matrix `A`. -/
theorem attnProb_eq_softWeight {d n : ℕ} (β : ℝ) (A : Matrix (Fin d) (Fin d) ℝ)
    (x : Idx n → EucSpace d) (i k : Idx n) :
    attnProb β A x i k
      = softWeight β A (x i) (x k) / ∑ l : Idx n, softWeight β A (x i) (x l) := rfl

/-- **`m_{β,A}[μ_X](x_i) = Σ_k π^A_{i→k} x_k`**, the first display of the proof
of `lem:drift_on_simplex_selfcontained`: at the empirical measure the `1/n`
cancels between the numerator and the normalizer, and what is left is the
attention average of the tokens.

Source: arXiv:2604.01978v1, proof of `lem:drift_on_simplex_selfcontained`. -/
theorem softBary_empMeasure {d n : ℕ} (β : ℝ) (A : Matrix (Fin d) (Fin d) ℝ)
    (x : Idx (n + 1) → EucSpace d) (i : Idx (n + 1)) :
    softBary β A (empMeasure x) (x i) = ∑ k : Idx (n + 1), attnProb β A x i k • x k := by
  have hW : (0 : ℝ) < ∑ l : Idx (n + 1), softWeight β A (x i) (x l) :=
    Finset.sum_pos (fun l _ => softWeight_pos _ _ _ _) ⟨i, Finset.mem_univ i⟩
  have hn : ((n + 1 : ℕ) : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Nat.succ_ne_zero n)
  have hrhs : ∑ k : Idx (n + 1), attnProb β A x i k • x k
      = (∑ l : Idx (n + 1), softWeight β A (x i) (x l))⁻¹ •
          ∑ k : Idx (n + 1), softWeight β A (x i) (x k) • x k := by
    rw [Finset.smul_sum]
    exact Finset.sum_congr rfl fun k _ => by
      rw [attnProb_eq_softWeight, div_eq_inv_mul, smul_smul]
  rw [hrhs, softBary, integral_empMeasure, integral_empMeasure, smul_eq_mul, mul_inv,
    smul_smul, inv_inv]
  congr 1
  field_simp

/-! ### The Gram structure collapses the double sum -/

/-- **`⟨m_i, m_j⟩ = γ + (1-γ) Σ_k π^A_{i→k} π^A_{j→k}`.**  This is both displays
of the proof of `lem:drift_on_simplex_selfcontained`: the source writes
`‖m_i‖²` and `⟨m_i, m_j⟩` separately, but the Gram matrix of a simplex
configuration is `γ + (1-γ)δ_{kl}` at every pair, so one computation covers
both — at `i = j` the right-hand side is `γ + (1-γ) Σ_k (π^A_{i→k})²`.

Source: arXiv:2604.01978v1, proof of `lem:drift_on_simplex_selfcontained`. -/
theorem inner_softBary_simplex {d n : ℕ} (β : ℝ) (A : Matrix (Fin d) (Fin d) ℝ)
    (γ : ℝ) (X : Idx (n + 1) → EucSpace d) (hX : IsSimplexConfig γ X) (i j : Idx (n + 1)) :
    inner (𝕜 := ℝ) (softBary β A (empMeasure X) (X i)) (softBary β A (empMeasure X) (X j))
      = γ + (1 - γ) * ∑ k : Idx (n + 1), attnProb β A X i k * attnProb β A X j k := by
  have hgram : ∀ k l : Idx (n + 1),
      inner (𝕜 := ℝ) (X k) (X l) = γ + (1 - γ) * (if k = l then (1 : ℝ) else 0) := by
    intro k l
    rcases eq_or_ne k l with hkl | hkl
    · subst hkl
      rw [real_inner_self_eq_norm_sq, hX.1 k, ite_eq_left rfl]
      ring
    · rw [hX.2 k l hkl, ite_eq_right hkl]
      ring
  have hstep : ∀ x : Idx (n + 1),
      ∑ y : Idx (n + 1), attnProb β A X j x *
          (attnProb β A X i y * (γ + (1 - γ) * (if y = x then (1 : ℝ) else 0)))
        = γ * attnProb β A X j x + (1 - γ) * (attnProb β A X i x * attnProb β A X j x) := by
    intro x
    have hy : ∀ y : Idx (n + 1),
        attnProb β A X j x * (attnProb β A X i y * (γ + (1 - γ) * (if y = x then (1 : ℝ) else 0)))
          = γ * attnProb β A X j x * attnProb β A X i y
            + (if y = x then (1 - γ) * (attnProb β A X i x * attnProb β A X j x) else 0) := by
      intro y
      rcases eq_or_ne y x with h | h
      · subst h; rw [ite_eq_left rfl, ite_eq_left rfl]; ring
      · rw [ite_eq_right h, ite_eq_right h]; ring
    simp only [hy]
    rw [Finset.sum_add_distrib, ← Finset.mul_sum, sum_attnProb,
      Finset.sum_ite_eq' Finset.univ x
        (fun _ => (1 - γ) * (attnProb β A X i x * attnProb β A X j x)),
      ite_eq_left (Finset.mem_univ x)]
    ring
  simp only [softBary_empMeasure, sum_inner, inner_sum, real_inner_smul_left,
    real_inner_smul_right, hgram, hstep]
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, sum_attnProb, ← Finset.mul_sum]
  ring

/-! ### `s_μ` on a simplex configuration -/

/-- **`s_{μ_X}(x_i, x_j) = γ + (1-γ) 𝔼 Σ_k π^A_{i→k} π^A_{j→k}`.**  The previous
identity integrated over the head law.

The source integrates without comment; here the head law is asked to be a
probability measure and the sum of products to be `ρ*`-integrable, which is
what turns the constant `γ` into itself and lets `(1-γ)` leave the integral.

Source: arXiv:2604.01978v1, proof of `lem:drift_on_simplex_selfcontained`. -/
theorem baryCorr_simplex {d n : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    [IsProbabilityMeasure ρ] (γ : ℝ) (X : Idx (n + 1) → EucSpace d)
    (hX : IsSimplexConfig γ X) (i j : Idx (n + 1))
    (hint : Integrable
      (fun θ => ∑ k : Idx (n + 1), attnProb β θ.2 X i k * attnProb β θ.2 X j k) ρ) :
    baryCorr β ρ (empMeasure X) (X i) (X j)
      = γ + (1 - γ) *
          ∫ θ, ∑ k : Idx (n + 1), attnProb β θ.2 X i k * attnProb β θ.2 X j k ∂ρ := by
  have hpt : ∀ θ : HeadParam d,
      inner (𝕜 := ℝ) (softBary β θ.2 (empMeasure X) (X i))
          (softBary β θ.2 (empMeasure X) (X j))
        = γ + (1 - γ) * ∑ k : Idx (n + 1), attnProb β θ.2 X i k * attnProb β θ.2 X j k :=
    fun θ => inner_softBary_simplex β θ.2 γ X hX i j
  rw [baryCorr]
  rw [integral_congr_ae (Filter.Eventually.of_forall hpt)]
  rw [integral_add (integrable_const γ) (hint.const_mul _)]
  rw [integral_const_mul, integral_const]
  simp

/-- **`s_{μ_X}(x_i) = γ + (1-γ) f(γ)`**, in the notation of
`eq:f_g_def_selfcontained`: the diagonal case, where the sum of products is the
sum of squares.

Source: arXiv:2604.01978v1, proof of `lem:drift_on_simplex_selfcontained`. -/
theorem baryCorr_simplex_self {d n : ℕ} (β : ℝ) (ρ : Measure (HeadParam d))
    [IsProbabilityMeasure ρ] (γ : ℝ) (X : Idx (n + 1) → EucSpace d)
    (hX : IsSimplexConfig γ X) (i : Idx (n + 1))
    (hint : Integrable (fun θ => ∑ k : Idx (n + 1), attnProb β θ.2 X i k ^ 2) ρ) :
    baryCorr β ρ (empMeasure X) (X i) (X i)
      = γ + (1 - γ) * ∫ θ, ∑ k : Idx (n + 1), attnProb β θ.2 X i k ^ 2 ∂ρ := by
  have hint' : Integrable
      (fun θ => ∑ k : Idx (n + 1), attnProb β θ.2 X i k * attnProb β θ.2 X i k) ρ := by
    simpa only [pow_two] using hint
  rw [baryCorr_simplex β ρ γ X hX i i hint']
  simp only [pow_two]

/-- The hypotheses of `baryCorr_simplex` are satisfiable: the degenerate head
law `ρ* = δ_0`, for which every integrand is integrable, and an orthonormal
pair, which is a simplex configuration with overlap `0`. -/
example (β : ℝ) (i j : Idx 2) :
    IsProbabilityMeasure (Measure.dirac (0 : HeadParam 2)) ∧
      IsSimplexConfig (0 : ℝ) (fun k : Idx 2 => (EuclideanSpace.single k (1 : ℝ) : EucSpace 2)) ∧
      Integrable (fun θ : HeadParam 2 => ∑ k : Idx 2,
        attnProb β θ.2 (fun k : Idx 2 => (EuclideanSpace.single k (1 : ℝ) : EucSpace 2)) i k *
          attnProb β θ.2 (fun k : Idx 2 => (EuclideanSpace.single k (1 : ℝ) : EucSpace 2)) j k)
        (Measure.dirac (0 : HeadParam 2)) := by
  refine ⟨inferInstance, ⟨fun k => by simp [PiLp.norm_single], fun k l hkl => ?_⟩,
    integrable_dirac enorm_lt_top⟩
  simp [EuclideanSpace.inner_single_left, hkl.symm]

end Homogenized
end Transformer
