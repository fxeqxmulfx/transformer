module

public import Transformer.Grokking.FlatSpectrum.ContractionDual

/-!
# The row-sum test for contractions

Source: `review/flat_spectrum.md`, *What a part of the table determines*, PT6. If every row of
`G = Zᴴ Z` has absolute sum at most `c²`, then `Z / c` is a contraction. This is Schur's test:
`‖Z v‖² = vᴴ G v ≤ Σ |G_pq| |v_p| |v_q| ≤ Σ |G_pq| (|v_p|² + |v_q|²) / 2`, and `G` is Hermitian, so
its columns have the same sums as its rows.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.TablePart

open Matrix Finset
open scoped ComplexOrder

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq n]

omit [DecidableEq n] in
/-- **Schur's test**: the row sums of `Zᴴ Z` bound `‖Z v‖²`. -/
lemma sum_norm_mulVec_sq_le_of_row_sum (Z : Matrix m n ℂ) {C : ℝ}
    (h : ∀ p, ∑ q, ‖(Zᴴ * Z) p q‖ ≤ C) (v : n → ℂ) :
    ∑ i, ‖(Z *ᵥ v) i‖ ^ 2 ≤ C * ∑ j, ‖v j‖ ^ 2 := by
  set G := Zᴴ * Z with hG
  have hherm (p q : n) : ‖G q p‖ = ‖G p q‖ := by
    have hH : Gᴴ = G := by rw [hG, conjTranspose_mul, conjTranspose_conjTranspose]
    have h2 : G q p = star (G p q) := by
      conv_lhs => rw [← hH]
      rfl
    rw [h2, norm_star]
  have e : ((∑ i, ‖(Z *ᵥ v) i‖ ^ 2 : ℝ) : ℂ) = star v ⬝ᵥ (G *ᵥ v) := by
    rw [hG, ← mulVec_mulVec, dotProduct_mulVec (star v) Zᴴ, ← star_mulVec,
      star_dotProduct_self_eq_sum_norm_sq]
  have h1 : ∑ i, ‖(Z *ᵥ v) i‖ ^ 2 ≤ ‖star v ⬝ᵥ (G *ᵥ v)‖ := by
    rw [← e, Complex.norm_real, Real.norm_eq_abs]
    exact le_abs_self _
  have h2 : ‖star v ⬝ᵥ (G *ᵥ v)‖ ≤ ∑ p, ∑ q, ‖G p q‖ * (‖v p‖ * ‖v q‖) := by
    simp only [dotProduct, mulVec, Pi.star_apply]
    refine (norm_sum_le _ _).trans (Finset.sum_le_sum fun p _ => ?_)
    calc ‖star (v p) * ∑ q, G p q * v q‖ = ‖v p‖ * ‖∑ q, G p q * v q‖ := by
          rw [norm_mul, norm_star]
      _ ≤ ‖v p‖ * ∑ q, ‖G p q‖ * ‖v q‖ := by
          refine mul_le_mul_of_nonneg_left ?_ (norm_nonneg _)
          exact (norm_sum_le _ _).trans (le_of_eq (Finset.sum_congr rfl fun q _ => norm_mul _ _))
      _ = ∑ q, ‖G p q‖ * (‖v p‖ * ‖v q‖) := by
          rw [Finset.mul_sum]
          exact Finset.sum_congr rfl fun q _ => by ring
  have h3 : ∑ p, ∑ q, ‖G p q‖ * (‖v p‖ * ‖v q‖) ≤
      ∑ p, ∑ q, (‖G p q‖ * ‖v p‖ ^ 2 + ‖G p q‖ * ‖v q‖ ^ 2) / 2 := by
    refine Finset.sum_le_sum fun p _ => Finset.sum_le_sum fun q _ => ?_
    nlinarith [mul_nonneg (norm_nonneg (G p q)) (sq_nonneg (‖v p‖ - ‖v q‖))]
  have h4 : ∑ p, ∑ q, ‖G p q‖ * ‖v p‖ ^ 2 ≤ C * ∑ j, ‖v j‖ ^ 2 := by
    rw [Finset.mul_sum]
    refine Finset.sum_le_sum fun p _ => ?_
    rw [← Finset.sum_mul]
    exact mul_le_mul_of_nonneg_right (h p) (by positivity)
  have h5 : ∑ p, ∑ q, ‖G p q‖ * ‖v q‖ ^ 2 ≤ C * ∑ j, ‖v j‖ ^ 2 := by
    rw [Finset.sum_comm, Finset.mul_sum]
    refine Finset.sum_le_sum fun q _ => ?_
    rw [← Finset.sum_mul]
    refine mul_le_mul_of_nonneg_right ?_ (by positivity)
    calc ∑ p, ‖G p q‖ = ∑ p, ‖G q p‖ := Finset.sum_congr rfl fun p _ => hherm q p
      _ ≤ C := h q
  have h6 (f g : n → n → ℝ) : ∑ p, ∑ q, (f p q + g p q) / 2 =
      (∑ p, ∑ q, f p q + ∑ p, ∑ q, g p q) / 2 := by
    simp only [add_div, Finset.sum_add_distrib, Finset.sum_div]
  have h7 := h6 (fun p q => ‖G p q‖ * ‖v p‖ ^ 2) (fun p q => ‖G p q‖ * ‖v q‖ ^ 2)
  beta_reduce at h7
  linarith

example (v : Fin 1 → ℂ) :
    ∑ i, ‖((1 : Matrix (Fin 1) (Fin 1) ℂ) *ᵥ v) i‖ ^ 2 ≤ 1 * ∑ j, ‖v j‖ ^ 2 :=
  sum_norm_mulVec_sq_le_of_row_sum 1 (fun p => by
    rw [Subsingleton.elim p 0]
    simp) v

/-- If `‖W v‖² ≤ c² ‖v‖²` for every `v`, then `W / c` is a contraction. -/
theorem isContraction_smul_of_sum_norm_sq_le {W : Matrix m n ℂ} {c : ℝ} (hc : 0 < c)
    (h : ∀ v, ∑ i, ‖(W *ᵥ v) i‖ ^ 2 ≤ c ^ 2 * ∑ j, ‖v j‖ ^ 2) :
    IsContraction (((c : ℝ) : ℂ)⁻¹ • W) := by
  rw [isContraction_iff_sum_norm_sq_le]
  intro v
  have e (i : m) : ‖((((c : ℝ) : ℂ)⁻¹ • W) *ᵥ v) i‖ ^ 2 = (c ^ 2)⁻¹ * ‖(W *ᵥ v) i‖ ^ 2 := by
    rw [smul_mulVec, Pi.smul_apply, smul_eq_mul, norm_mul, norm_inv, Complex.norm_real,
      Real.norm_of_nonneg hc.le, mul_pow, inv_pow]
  simp only [e]
  rw [← Finset.mul_sum, inv_mul_le_iff₀ (by positivity)]
  exact h v

example : IsContraction (((1 : ℝ) : ℂ)⁻¹ • (1 : Matrix (Fin 1) (Fin 1) ℂ)) :=
  isContraction_smul_of_sum_norm_sq_le one_pos fun v => by simp

/-- **The row-sum test**: if every row of `Zᴴ Z` has absolute sum at most `c²`, then `Z / c` is a
contraction. -/
theorem isContraction_of_row_sum {Z : Matrix m n ℂ} {c : ℝ} (hc : 0 < c)
    (h : ∀ p, ∑ q, ‖(Zᴴ * Z) p q‖ ≤ c ^ 2) : IsContraction (((c : ℝ) : ℂ)⁻¹ • Z) :=
  isContraction_smul_of_sum_norm_sq_le hc (sum_norm_mulVec_sq_le_of_row_sum Z h)

example : IsContraction (((1 : ℝ) : ℂ)⁻¹ • (1 : Matrix (Fin 1) (Fin 1) ℂ)) :=
  isContraction_of_row_sum one_pos fun p => by
    rw [Subsingleton.elim p 0]
    simp

end Transformer.Grokking.FlatSpectrum.TablePart
