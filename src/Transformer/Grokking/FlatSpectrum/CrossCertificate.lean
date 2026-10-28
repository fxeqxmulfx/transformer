module

public import Transformer.Grokking.FlatSpectrum.TwoDiagonals
public import Transformer.Grokking.FlatSpectrum.CyclePoincare

/-!
# The cross: rigid, and a lower bound for its completions

Source: `review/flat_spectrum.md`, *Exact recovery of a rank-one kernel* (added). Let `Ω` be
the cross of row `i₀` and column `j₀`, `2N − 1` cells. It is rigid:
`r i + s j = (r i + s j₀) + (r i₀ + s j) − (r i₀ + s j₀)` (`rigid_cross`), and its values
determine a rank-one matrix `a bᵀ` with `a i₀ b j₀ ≠ 0` (`vecMulVec_eq_of_cross`). For
`N ≥ 2` and `m = √(N − 1)`, the matrix `Z`, equal to `1/m` on the cross except at `(i₀, j₀)` and
zero elsewhere, maps `v` to `(Σ_{j ≠ j₀} v j)/m` in row `i₀` and to `v j₀ / m` in the other
`N − 1` rows. By Cauchy–Schwarz `|Z v|² ≤ Σ_{j ≠ j₀} |v j|² + |v j₀|²`: `Z` is a contraction
(`isContraction_crossCert`). It pairs with every `X` equal to `J` on the cross to
`2 (N − 1)/m = 2m`, so all such `X` have `‖X‖_S1 ≥ 2√(N − 1)`
(`two_mul_sqrt_le_nuclearNorm_of_cross`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset

variable {n : Type*} [Fintype n] [DecidableEq n]

omit [Fintype n] [DecidableEq n] in
/-- The cross is rigid: if `r i + s j = 0` on row `i₀` and column `j₀`, then everywhere. -/
lemma rigid_cross {i₀ j₀ : n} {r s : n → ℂ}
    (h : ∀ ij ∈ {ij : n × n | ij.1 = i₀ ∨ ij.2 = j₀}, r ij.1 + s ij.2 = 0) (i j : n) :
    r i + s j = 0 := by
  have h1 := h (i, j₀) (Or.inr rfl)
  have h2 := h (i₀, j) (Or.inl rfl)
  have h3 := h (i₀, j₀) (Or.inl rfl)
  linear_combination h1 + h2 - h3

example := rigid_cross (i₀ := (0 : Fin 2)) (j₀ := 0) (r := 0) (s := 0) (fun _ _ => by simp) 1 1

omit [Fintype n] [DecidableEq n] in
/-- **The cross determines a rank-one matrix**, `review/flat_spectrum.md`, *Exact recovery of a
rank-one kernel*: if `a' b'ᵀ = a bᵀ` on row `i₀` and column `j₀` and `a i₀ b j₀ ≠ 0`, then
`a' b'ᵀ = a bᵀ`. -/
theorem vecMulVec_eq_of_cross {a b a' b' : n → ℂ} {i₀ j₀ : n} (h₀ : a i₀ * b j₀ ≠ 0)
    (h : ∀ ij ∈ {ij : n × n | ij.1 = i₀ ∨ ij.2 = j₀},
      vecMulVec a' b' ij.1 ij.2 = vecMulVec a b ij.1 ij.2) :
    vecMulVec a' b' = vecMulVec a b := by
  ext i j
  have h1 := h (i, j₀) (Or.inr rfl)
  have h2 := h (i₀, j) (Or.inl rfl)
  have h3 := h (i₀, j₀) (Or.inl rfl)
  simp only [vecMulVec_apply] at h1 h2 h3 ⊢
  refine mul_right_cancel₀ h₀ ?_
  linear_combination (-(a' i * b' j)) * h3 + (a' i₀ * b' j) * h1 + (a i * b j₀) * h2

example := vecMulVec_eq_of_cross (a := fun _ : Fin 2 => 1) (b := fun _ => 1) (a' := fun _ => 1)
  (b' := fun _ => 1) (i₀ := 0) (j₀ := 0) (by simp) fun _ _ => rfl

/-- The certificate on the cross: `c` on row `i₀` and on column `j₀`, except `0` at
`(i₀, j₀)`, and `0` off the cross. -/
noncomputable def crossCert (i₀ j₀ : n) (c : ℝ) : Matrix n n ℂ :=
  of fun i j => if i = i₀ then (if j = j₀ then 0 else (c : ℂ)) else (if j = j₀ then (c : ℂ) else 0)

/-- The complement of one index has `N − 1` elements. -/
lemma card_compl_singleton_cast (hN : 1 ≤ Fintype.card n) (x : n) :
    ((({x}ᶜ : Finset n).card : ℕ) : ℝ) = Fintype.card n - 1 := by
  rw [Finset.card_compl, Finset.card_singleton, Nat.cast_sub hN, Nat.cast_one]

example := card_compl_singleton_cast (n := Fin 2) (by simp) 0

/-- **The cross certificate is a contraction**: for `N ≥ 2`, `crossCert i₀ j₀ (1/√(N − 1))`
shortens every vector. -/
lemma isContraction_crossCert {i₀ j₀ : n} (hN : 2 ≤ Fintype.card n) :
    IsContraction (crossCert i₀ j₀ (√((Fintype.card n : ℝ) - 1))⁻¹) := by
  set N1 : ℝ := (Fintype.card n : ℝ) - 1 with hN1def
  have hN1 : 0 < N1 := by
    have : (2 : ℝ) ≤ Fintype.card n := by exact_mod_cast hN
    linarith
  set c : ℝ := (√N1)⁻¹
  have hc : c ^ 2 * N1 = 1 := by rw [inv_pow, Real.sq_sqrt hN1.le, inv_mul_cancel₀ hN1.ne']
  have hcard := card_compl_singleton_cast (n := n) (by omega)
  refine isContraction_of_forall_sum_norm_sq_le fun v => ?_
  have h0 : (crossCert i₀ j₀ c *ᵥ v) i₀ = c * ∑ j ∈ {j₀}ᶜ, v j := by
    rw [mulVec, dotProduct, Fintype.sum_eq_add_sum_compl j₀, mul_sum]
    simp only [crossCert, of_apply, ↓reduceIte, zero_mul, zero_add]
    exact sum_congr rfl fun j hj => by
      rw [ite_eq_right (by simpa using hj)]
  have h1 (i : n) (hi : i ∈ ({i₀}ᶜ : Finset n)) : ‖(crossCert i₀ j₀ c *ᵥ v) i‖ ^ 2 =
      c ^ 2 * ‖v j₀‖ ^ 2 := by
    have hi' : i ≠ i₀ := by simpa using hi
    simp only [mulVec, dotProduct, crossCert, of_apply, hi', ↓reduceIte, ite_mul, zero_mul,
      sum_ite_eq', mem_univ, norm_mul, Complex.norm_real, Real.norm_eq_abs, mul_pow, sq_abs]
  have hcs : ‖∑ j ∈ ({j₀}ᶜ : Finset n), v j‖ ^ 2 ≤ N1 * ∑ j ∈ ({j₀}ᶜ : Finset n), ‖v j‖ ^ 2 := by
    calc ‖∑ j ∈ ({j₀}ᶜ : Finset n), v j‖ ^ 2
        ≤ (∑ j ∈ ({j₀}ᶜ : Finset n), ‖v j‖) ^ 2 :=
          pow_le_pow_left₀ (norm_nonneg _) (norm_sum_le _ _) 2
      _ ≤ #({j₀}ᶜ : Finset n) * ∑ j ∈ ({j₀}ᶜ : Finset n), ‖v j‖ ^ 2 := sq_sum_le_card_mul_sum_sq
      _ = N1 * ∑ j ∈ ({j₀}ᶜ : Finset n), ‖v j‖ ^ 2 := by rw [hcard]
  rw [Fintype.sum_eq_add_sum_compl i₀, Fintype.sum_eq_add_sum_compl j₀ (fun j => ‖v j‖ ^ 2), h0,
    sum_congr rfl h1, sum_const, nsmul_eq_mul, hcard, norm_mul, Complex.norm_real,
    Real.norm_eq_abs, mul_pow, sq_abs]
  have hS := sum_nonneg fun j (_ : j ∈ ({j₀}ᶜ : Finset n)) => sq_nonneg ‖v j‖
  nlinarith [mul_le_mul_of_nonneg_left hcs (sq_nonneg c)]

example := isContraction_crossCert (i₀ := (0 : Fin 2)) (j₀ := 0) (by simp)

/-- **A lower bound on the cross**, `review/flat_spectrum.md`, *Exact recovery of a rank-one
kernel*: for `N ≥ 2`, every `X` equal to `J` on row `i₀` and column `j₀` has
`‖X‖_S1 ≥ 2√(N − 1)`. -/
theorem two_mul_sqrt_le_nuclearNorm_of_cross {i₀ j₀ : n} (hN : 2 ≤ Fintype.card n)
    {X : Matrix n n ℂ}
    (hX : ∀ ij ∈ {ij : n × n | ij.1 = i₀ ∨ ij.2 = j₀}, X ij.1 ij.2 = onesMatrix n ij.1 ij.2) :
    2 * √((Fintype.card n : ℝ) - 1) ≤ nuclearNorm X := by
  set N1 : ℝ := (Fintype.card n : ℝ) - 1 with hN1def
  have hN1 : 0 < N1 := by
    have : (2 : ℝ) ≤ Fintype.card n := by exact_mod_cast hN
    linarith
  have hcard := card_compl_singleton_cast (n := n) (by omega)
  have hrow (j : n) : X i₀ j = 1 := hX (i₀, j) (Or.inl rfl)
  have hcol (i : n) : X i j₀ = 1 := hX (i, j₀) (Or.inr rfl)
  have hsum (i : n) : ∑ j, star (crossCert i₀ j₀ (√N1)⁻¹ i j) * X i j =
      if i = i₀ then ((N1 * (√N1)⁻¹ : ℝ) : ℂ) else (((√N1)⁻¹ : ℝ) : ℂ) := by
    split_ifs with hi
    · subst hi
      rw [Fintype.sum_eq_add_sum_compl j₀]
      simp only [crossCert, of_apply, ↓reduceIte, star_zero, zero_add, hrow, mul_one]
      rw [sum_congr rfl fun j hj => by rw [ite_eq_right (by simpa using hj)], sum_const,
        nsmul_eq_mul, Complex.star_def, Complex.conj_ofReal, ← Complex.ofReal_natCast, hcard,
        Complex.ofReal_mul]
    · simp only [crossCert, of_apply, hi, ↓reduceIte, apply_ite star, star_zero, ite_mul,
        zero_mul, sum_ite_eq', mem_univ, hcol, mul_one, Complex.star_def, Complex.conj_ofReal]
  have htr : ((crossCert i₀ j₀ (√N1)⁻¹)ᴴ * X).trace = ((2 * √N1 : ℝ) : ℂ) := by
    rw [trace_conjTranspose_mul, sum_congr rfl fun i _ => hsum i,
      Fintype.sum_eq_add_sum_compl i₀, ite_eq_left rfl,
      sum_congr rfl fun i hi => ite_eq_right (by simpa using hi), sum_const, nsmul_eq_mul,
      ← Complex.ofReal_natCast, hcard, ← Complex.ofReal_mul, ← Complex.ofReal_add]
    congr 1
    have hs : √N1 * √N1 = N1 := Real.mul_self_sqrt hN1.le
    have hs0 : √N1 ≠ 0 := (Real.sqrt_pos.mpr hN1).ne'
    field_simp
    linear_combination (-2) * hs
  have h := norm_trace_le_nuclearNorm (isContraction_crossCert (i₀ := i₀) (j₀ := j₀) hN) X
  rwa [htr, Complex.norm_real, Real.norm_of_nonneg (by positivity)] at h

example := two_mul_sqrt_le_nuclearNorm_of_cross (i₀ := (0 : Fin 2)) (j₀ := 0) (by simp)
  (X := onesMatrix (Fin 2)) fun _ _ => rfl

end Transformer.Grokking.FlatSpectrum
