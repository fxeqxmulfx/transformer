module

public import Transformer.Grokking.FlatSpectrum.CopiesInvisible
public import Transformer.Grokking.FlatSpectrum.Intermediate

/-!
# Theorem B for `k` copies of `x³ + x y`, with explicit constants

Source: `review/flat_spectrum.md`, *Functions of n variables*, item NV6. Theorem B bounds the
covariance of a readout of one pair's label with that pair's intermediate `ψ_r(x y)` by
`√Var · √(9/(2p) + 8/p^{3/2})` (`norm_cov_cube_add_mul_le`). The other `m = k − 1` pairs scale
the covariance by `p^{−m}` (`cov_comp_add_copies`), while the variance changes by at most a
factor `2`: one pair's label law `(1 − 1/p) U + δ₀/p` is at most twice the law
`(1 − p^{−k}) U + p^{−k} δ₀` of `k` pairs at every value
(`var_cube_add_mul_le_two_mul_var_copies`). Hence, for `k` pairs and `r ≠ 0`
(`norm_cov_copies_cube_add_mul_le`),

    |Cov(g(f), ψ_r(x₁ y₁))| ≤ p^{−(k−1)} √(Var g(f)) √(9/p + 16/p^{3/2}),

so a readout of unit variance correlates with one pair's intermediate by at most
`p^{−(k−1)} √(9/p + 16/p^{3/2})`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

variable {α β : Type*} [Fintype α] [Fintype β] {p : ℕ}

/-- The variance `Var A = E |A − E A|²` on a finite uniform space. -/
noncomputable def var (A : α → ℂ) : ℝ := meanSq fun x => A x - mean A

lemma meanSq_nonneg (A : α → ℂ) : 0 ≤ meanSq A := by
  unfold meanSq
  positivity

lemma var_nonneg (A : α → ℂ) : 0 ≤ var A := meanSq_nonneg _

/-- `E |A − c|² = Var A + |E A − c|²`. -/
lemma meanSq_sub_eq_var_add [Nonempty α] (A : α → ℂ) (c : ℂ) :
    meanSq (fun x => A x - c) = var A + ‖mean A - c‖ ^ 2 := by
  have hn : (Fintype.card α : ℝ) ≠ 0 := by exact_mod_cast Fintype.card_ne_zero
  have hsum : ∑ x, (A x - mean A) = 0 := by
    rw [sum_sub_distrib, sum_const, card_univ, nsmul_eq_mul, mean,
      mul_div_cancel₀ _ (by exact_mod_cast Fintype.card_ne_zero), sub_self]
  have key : ∑ x, ‖A x - c‖ ^ 2 =
      ∑ x, ‖A x - mean A‖ ^ 2 + Fintype.card α * ‖mean A - c‖ ^ 2 := by
    have (x : α) : ‖A x - c‖ ^ 2 = ‖A x - mean A‖ ^ 2 + ‖mean A - c‖ ^ 2 +
        2 * ((A x - mean A) * (starRingEnd ℂ) (mean A - c)).re := by
      rw [show A x - c = (A x - mean A) + (mean A - c) by ring]
      simp only [Complex.sq_norm, Complex.normSq_add]
    simp only [this, sum_add_distrib, sum_const, card_univ, nsmul_eq_mul, ← mul_sum,
      ← Complex.re_sum, ← sum_mul, hsum, zero_mul, Complex.zero_re, mul_zero, add_zero]
  rw [var, meanSq, meanSq, key, add_div, mul_div_cancel_left₀ _ hn]

/-- The mean minimizes the mean square distance: `Var A ≤ E |A − c|²`. -/
lemma var_le_meanSq_sub (A : α → ℂ) (c : ℂ) : var A ≤ meanSq fun x => A x - c := by
  cases isEmpty_or_nonempty α
  · simp [var, meanSq]
  · rw [meanSq_sub_eq_var_add]
    exact le_add_of_nonneg_right (by positivity)

lemma ofReal_meanSq (A : α → ℂ) : (meanSq A : ℂ) = mean fun x => ((‖A x‖ ^ 2 : ℝ) : ℂ) := by
  simp only [meanSq, mean, Complex.ofReal_div, Complex.ofReal_sum, Complex.ofReal_natCast]

/-- The mean square under the law `(1 − ε) U + ε δ₀`. -/
lemma meanSq_comp_of_mean_stdAddChar [NeZero p] [Nonempty β] {ε : ℝ} {w : β → ZMod p}
    (hw : ∀ r ≠ 0, mean (fun b => ZMod.stdAddChar (r * w b)) = ε) (φ : ZMod p → ℂ) :
    meanSq (fun b => φ (w b)) = (1 - ε) * meanSq φ + ε * ‖φ 0‖ ^ 2 := by
  apply Complex.ofReal_injective
  rw [ofReal_meanSq, mean_comp_of_mean_stdAddChar hw fun t => ((‖φ t‖ ^ 2 : ℝ) : ℂ),
    ← ofReal_meanSq]
  push_cast
  ring

/-- A constant remainder (`ε = 1`) is the point mass at `0`. -/
example (φ : ZMod 5 → ℂ) := meanSq_comp_of_mean_stdAddChar (β := Unit) (w := fun _ => 0)
  (ε := 1) (fun r _ => by simp [mean]) φ

/-- Characters of independent parts: `E ψ_r(u(a) + w(b)) = E ψ_r(u) · E ψ_r(w)`. -/
lemma mean_stdAddChar_add [NeZero p] (u : α → ZMod p) (w : β → ZMod p) (r : ZMod p) :
    mean (fun z : α × β => ZMod.stdAddChar (r * (u z.1 + w z.2))) =
      mean (fun a => ZMod.stdAddChar (r * u a)) * mean fun b => ZMod.stdAddChar (r * w b) := by
  simp only [mul_add, AddChar.map_add_eq_mul]
  exact mean_mul_prod (fun a => ZMod.stdAddChar (r * u a)) fun b => ZMod.stdAddChar (r * w b)

variable [Fact p.Prime]

/-- **Theorem B as a covariance**:
`|Cov(g(x³ + x y), ψ_r(x y))| ≤ √(Var g) √(9/(2p) + 8/p^{3/2})` for `r ≠ 0`. -/
theorem norm_cov_cube_add_mul_le {r : ZMod p} (hr : r ≠ 0) (g : ZMod p → ℂ) :
    ‖cov (fun ab : ZMod p × ZMod p => g (ab.1 ^ 3 + ab.1 * ab.2))
        (fun ab => ZMod.stdAddChar (r * (ab.1 * ab.2)))‖ ≤
      √(var fun ab : ZMod p × ZMod p => g (ab.1 ^ 3 + ab.1 * ab.2)) *
        √(9 / (2 * p) + 8 / (p * √p)) := by
  rw [cov_eq_mean_sub_mul, ← norm_star, ← mean_star]
  simp only [star_mul', star_star]
  exact norm_mean_star_mul_cube_add_mul_le hr fun t =>
    g t - mean fun ab : ZMod p × ZMod p => g (ab.1 ^ 3 + ab.1 * ab.2)

example := @norm_cov_cube_add_mul_le 7 ⟨Nat.prime_seven⟩ 1 (by decide) fun _ => 1

/-- One pair's label law is at most twice the law of `k = 1 + m` pairs at every value, so the
variance of a readout under one pair is at most twice its variance under `k` pairs. -/
theorem var_cube_add_mul_le_two_mul_var_copies {τ : Type*} [Fintype τ] [DecidableEq τ]
    (g : ZMod p → ℂ) :
    var (fun ab : ZMod p × ZMod p => g (ab.1 ^ 3 + ab.1 * ab.2)) ≤
      2 * var (fun z : (ZMod p × ZMod p) × (τ → ZMod p × ZMod p) =>
        g (z.1.1 ^ 3 + z.1.1 * z.1.2 + ∑ j, ((z.2 j).1 ^ 3 + (z.2 j).1 * (z.2 j).2))) := by
  set c := mean fun z : (ZMod p × ZMod p) × (τ → ZMod p × ZMod p) =>
    g (z.1.1 ^ 3 + z.1.1 * z.1.2 + ∑ j, ((z.2 j).1 ^ 3 + (z.2 j).1 * (z.2 j).2))
  set φ : ZMod p → ℂ := fun t => g t - c
  have hP : (fun x : ZMod p => x ^ 3) 0 = 0 := zero_pow three_ne_zero
  have hone : (1 : ℝ) ≤ p := by exact_mod_cast (Fact.out : p.Prime).one_lt.le
  have h1 (r : ZMod p) (hr : r ≠ 0) : mean (fun ab : ZMod p × ZMod p =>
      ZMod.stdAddChar (r * (ab.1 ^ 3 + ab.1 * ab.2))) = (((p : ℝ)⁻¹ : ℝ) : ℂ) := by
    push_cast
    exact mean_stdAddChar_poly_add_mul _ hP hr
  have hk (r : ZMod p) (hr : r ≠ 0) : mean (fun z : (ZMod p × ZMod p) × (τ → ZMod p × ZMod p) =>
      ZMod.stdAddChar (r * (z.1.1 ^ 3 + z.1.1 * z.1.2 +
        ∑ j, ((z.2 j).1 ^ 3 + (z.2 j).1 * (z.2 j).2)))) =
      ((((p : ℝ)⁻¹) ^ (1 + Fintype.card τ) : ℝ) : ℂ) := by
    rw [mean_stdAddChar_add (fun ab : ZMod p × ZMod p => ab.1 ^ 3 + ab.1 * ab.2)
      (fun z : τ → ZMod p × ZMod p => ∑ j, ((z j).1 ^ 3 + (z j).1 * (z j).2)) r,
      mean_stdAddChar_poly_add_mul (fun x => x ^ 3) hP hr,
      mean_stdAddChar_copies (fun x => x ^ 3) hP hr]
    push_cast
    ring
  have hq := meanSq_comp_of_mean_stdAddChar h1 φ
  have hf := meanSq_comp_of_mean_stdAddChar hk φ
  have hM : ‖φ 0‖ ^ 2 * (p : ℝ)⁻¹ ≤ meanSq φ := by
    rw [meanSq, ZMod.card, div_eq_mul_inv]
    gcongr
    exact single_le_sum (f := fun t => ‖φ t‖ ^ 2) (fun t _ => by positivity) (mem_univ 0)
  have hε : ((p : ℝ)⁻¹) ^ (1 + Fintype.card τ) ≤ (p : ℝ)⁻¹ :=
    pow_le_of_le_one (by positivity) (inv_le_one_of_one_le₀ hone) (by omega)
  have hε0 : 0 ≤ ((p : ℝ)⁻¹) ^ (1 + Fintype.card τ) := by positivity
  have hip : (p : ℝ)⁻¹ ≤ 1 := inv_le_one_of_one_le₀ hone
  have hz : 0 ≤ ‖φ 0‖ ^ 2 := by positivity
  calc _ ≤ meanSq (fun ab : ZMod p × ZMod p => φ (ab.1 ^ 3 + ab.1 * ab.2)) :=
        var_le_meanSq_sub _ c
    _ = (1 - (p : ℝ)⁻¹) * meanSq φ + (p : ℝ)⁻¹ * ‖φ 0‖ ^ 2 := hq
    _ ≤ 2 * ((1 - ((p : ℝ)⁻¹) ^ (1 + Fintype.card τ)) * meanSq φ +
        ((p : ℝ)⁻¹) ^ (1 + Fintype.card τ) * ‖φ 0‖ ^ 2) := by
      nlinarith [mul_nonneg (by linarith : 0 ≤ 1 + (p : ℝ)⁻¹ -
          2 * ((p : ℝ)⁻¹) ^ (1 + Fintype.card τ)) (by linarith : 0 ≤ meanSq φ - ‖φ 0‖ ^ 2 *
          (p : ℝ)⁻¹), mul_nonneg hz (sq_nonneg (p : ℝ)⁻¹),
        mul_nonneg (mul_nonneg hz hε0) (by linarith : 0 ≤ 1 - (p : ℝ)⁻¹)]
    _ = _ := by rw [← hf]; rfl

/-- **NV6, Theorem B for `k = 1 + m` copies of `x³ + x y`**: for `r ≠ 0`, every readout `g` of
the label of `k` pairs has `|Cov(g(f), ψ_r(x₁ y₁))| ≤ p^{−m} √(Var g(f)) √(9/p + 16/p^{3/2})`;
a readout of unit variance correlates with one pair's intermediate by at most
`p^{−m} √(9/p + 16/p^{3/2})`. -/
theorem norm_cov_copies_cube_add_mul_le {τ : Type*} [Fintype τ] [DecidableEq τ] {r : ZMod p}
    (hr : r ≠ 0) (g : ZMod p → ℂ) :
    ‖cov (fun z : (ZMod p × ZMod p) × (τ → ZMod p × ZMod p) =>
        g (z.1.1 ^ 3 + z.1.1 * z.1.2 + ∑ j, ((z.2 j).1 ^ 3 + (z.2 j).1 * (z.2 j).2)))
        (fun z => ZMod.stdAddChar (r * (z.1.1 * z.1.2)))‖ ≤
      (p : ℝ)⁻¹ ^ Fintype.card τ *
        √(var fun z : (ZMod p × ZMod p) × (τ → ZMod p × ZMod p) =>
          g (z.1.1 ^ 3 + z.1.1 * z.1.2 + ∑ j, ((z.2 j).1 ^ 3 + (z.2 j).1 * (z.2 j).2))) *
        √(9 / p + 16 / (p * √p)) := by
  have hv := var_nonneg fun z : (ZMod p × ZMod p) × (τ → ZMod p × ZMod p) =>
    g (z.1.1 ^ 3 + z.1.1 * z.1.2 + ∑ j, ((z.2 j).1 ^ 3 + (z.2 j).1 * (z.2 j).2))
  rw [cov_comp_add_copies (fun x => x ^ 3) (zero_pow three_ne_zero) g
      (fun ab : ZMod p × ZMod p => ab.1 ^ 3 + ab.1 * ab.2)
      (fun ab : ZMod p × ZMod p => ZMod.stdAddChar (r * (ab.1 * ab.2))),
    norm_mul, norm_pow, norm_inv, Complex.norm_natCast, mul_assoc]
  gcongr
  calc _ ≤ _ := norm_cov_cube_add_mul_le hr g
    _ ≤ √(2 * var fun z : (ZMod p × ZMod p) × (τ → ZMod p × ZMod p) =>
          g (z.1.1 ^ 3 + z.1.1 * z.1.2 + ∑ j, ((z.2 j).1 ^ 3 + (z.2 j).1 * (z.2 j).2))) *
        √(9 / (2 * p) + 8 / (p * √p)) := by
      gcongr
      exact var_cube_add_mul_le_two_mul_var_copies g
    _ = _ := by
      rw [← Real.sqrt_mul (mul_nonneg zero_le_two hv), ← Real.sqrt_mul hv]
      congr 1
      ring

example := @norm_cov_copies_cube_add_mul_le 7 ⟨Nat.prime_seven⟩ (Fin 2) _ _ 1 (by decide)

end Transformer.Grokking.FlatSpectrum
