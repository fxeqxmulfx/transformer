module

public import Transformer.Grokking.FlatSpectrum.Blocks

/-!
# A Gram matrix made of a projection and one `2 × 2` block

Source: `review/flat_spectrum.md`, Torch check: "31.5 at p=97, 60.0 at p=199 | These are the
centered values". Let `E` be a Hermitian projection and `W` an isometry, `Wᴴ W = 1`, with
`E W = 0`. If `Xᴴ X = c² E + W B Wᴴ`, then `B = (X W)ᴴ (X W)` has a positive semidefinite square
root `R`, and `P = c E + W R Wᴴ` is a positive semidefinite square root of `Xᴴ X`. So
`‖X‖_S1 = tr P = c tr E + tr R`, and `tr R = σ₁ + σ₂ = √(tr B + 2 √|det B|)` for a `2 × 2` block
(`nuclearNorm_fin_two`). Always `‖X‖²_S2 = tr (Xᴴ X) = c² tr E + tr (M Uᴴ U)` when
`Xᴴ X = c² E + U M Uᴴ`.

For `U` with orthogonal columns, `Uᴴ U = diag d` with `d > 0`, the isometry `W = U diag(d)^(-1/2)`
and `B = diag(d)^(1/2) M diag(d)^(1/2)` give `tr B = tr (M diag d)` and
`det B = det M · d₀ d₁`, so no square root of `d` enters the statement.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix
open scoped ComplexOrder

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq n]

/-- If `Xᴴ X = c² E + W B Wᴴ`, with `E` a Hermitian projection, `Wᴴ W = 1`, `E W = 0` and
`c ≥ 0`, then `‖X‖_S1 = c tr E + √(tr B + 2 √|det B|)`. -/
theorem nuclearNorm_of_gram_eq_proj_add {X : Matrix m n ℂ} {E : Matrix n n ℂ}
    {W : Matrix n (Fin 2) ℂ} {B : Matrix (Fin 2) (Fin 2) ℂ} {c : ℝ} (hc : 0 ≤ c)
    (hEh : Eᴴ = E) (hEE : E * E = E) (hW : Wᴴ * W = 1) (hEW : E * W = 0)
    (hX : Xᴴ * X = ((c ^ 2 : ℝ) : ℂ) • E + W * B * Wᴴ) :
    nuclearNorm X = c * E.trace.re + √(B.trace.re + 2 * √‖B.det‖) := by
  have hWE : Wᴴ * E = 0 := by
    rw [← hEh, ← conjTranspose_mul, hEW, conjTranspose_zero]
  obtain ⟨R, hR, hRR⟩ := exists_posSemidef_mul_self_eq (X * W)
  have hRB : R * R = B := by
    rw [hRR, conjTranspose_mul, Matrix.mul_assoc, ← Matrix.mul_assoc Xᴴ, hX, Matrix.add_mul,
      Matrix.smul_mul, hEW, smul_zero, zero_add]
    simp only [Matrix.mul_assoc]
    rw [hW, Matrix.mul_one, ← Matrix.mul_assoc, hW, Matrix.one_mul]
  have hRh : Rᴴ = R := hR.1
  have hEpsd : E.PosSemidef := by
    have := posSemidef_conjTranspose_mul_self E
    rwa [hEh, hEE] at this
  have h1 : E * (W * R * Wᴴ) = 0 := by
    rw [← Matrix.mul_assoc, ← Matrix.mul_assoc, hEW, Matrix.zero_mul, Matrix.zero_mul]
  have h2 : W * R * Wᴴ * E = 0 := by rw [Matrix.mul_assoc, hWE, Matrix.mul_zero]
  have h3 : W * R * Wᴴ * (W * R * Wᴴ) = W * B * Wᴴ := by
    simp only [Matrix.mul_assoc]
    rw [← Matrix.mul_assoc Wᴴ W, hW, Matrix.one_mul, ← Matrix.mul_assoc R R, hRB]
  have hP := nuclearNorm_eq_trace_of_mul_self (X := X)
    ((hEpsd.smul (Complex.zero_le_real.mpr hc)).add (hR.mul_mul_conjTranspose_same W)) (by
      rw [hX, show ((c ^ 2 : ℝ) : ℂ) = (c : ℂ) * c by push_cast; ring]
      simp only [Matrix.add_mul, Matrix.mul_add, Matrix.smul_mul, Matrix.mul_smul, hEE, h1, h2,
        h3, smul_smul, smul_zero, add_zero, zero_add])
  have hRn : (nuclearNorm R : ℂ) = R.trace := nuclearNorm_eq_trace_of_mul_self hR (by rw [hRh])
  have hdR : ‖R.det‖ = √‖B.det‖ := by
    rw [← hRB, det_mul, norm_mul, Real.sqrt_mul_self (norm_nonneg _)]
  rw [trace_add, trace_smul, smul_eq_mul, Matrix.trace_mul_comm (W * R), ← Matrix.mul_assoc, hW,
    Matrix.one_mul, ← hRn, nuclearNorm_fin_two, frobeniusSq_eq_re_trace, hRh, hRB, hdR] at hP
  have h := congrArg Complex.re hP
  rwa [Complex.ofReal_re, Complex.add_re, Complex.re_ofReal_mul, Complex.ofReal_re] at h

example := nuclearNorm_of_gram_eq_proj_add (X := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (E := 0)
  (W := 1) (B := 1) le_rfl (by simp) (by simp) (by simp) (by simp) (by simp)

/-- `nuclearNorm_of_gram_eq_proj_add` for orthogonal columns, `Uᴴ U = diag d` with `d > 0`: if
`Xᴴ X = c² E + U M Uᴴ`, then `‖X‖_S1 = c tr E + √(tr (M diag d) + 2 √(|det M| d₀ d₁))`. -/
theorem nuclearNorm_of_gram_eq_proj_add_diagonal {X : Matrix m n ℂ} {E : Matrix n n ℂ}
    {U : Matrix n (Fin 2) ℂ} {M : Matrix (Fin 2) (Fin 2) ℂ} {c : ℝ} {d : Fin 2 → ℝ}
    (hc : 0 ≤ c) (hd : ∀ i, 0 < d i) (hEh : Eᴴ = E) (hEE : E * E = E)
    (hU : Uᴴ * U = diagonal fun i => (d i : ℂ)) (hEU : E * U = 0)
    (hX : Xᴴ * X = ((c ^ 2 : ℝ) : ℂ) • E + U * M * Uᴴ) :
    nuclearNorm X = c * E.trace.re +
      √((M * diagonal fun i => (d i : ℂ)).trace.re + 2 * √(‖M.det‖ * (d 0 * d 1))) := by
  have hsq (i : Fin 2) : √(d i) * √(d i) = d i := Real.mul_self_sqrt (hd i).le
  have hne (i : Fin 2) : √(d i) ≠ 0 := (Real.sqrt_pos.mpr (hd i)).ne'
  have hS : ∀ S T : Matrix (Fin 2) (Fin 2) ℂ, S = (diagonal fun i => ((√(d i) : ℝ) : ℂ)) →
      T = (diagonal fun i => (((√(d i))⁻¹ : ℝ) : ℂ)) →
      S * T = 1 ∧ T * S = 1 ∧ Tᴴ = T ∧ S * S = (diagonal fun i => (d i : ℂ)) ∧
        T * (diagonal fun i => (d i : ℂ)) * T = 1 ∧ S.det = ((√(d 0) * √(d 1) : ℝ) : ℂ) := by
    rintro S T rfl rfl
    refine ⟨?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [diagonal_mul_diagonal, ← diagonal_one]
      exact congrArg diagonal (funext fun i => by
        rw [← Complex.ofReal_mul, mul_inv_cancel₀ (hne i), Complex.ofReal_one])
    · rw [diagonal_mul_diagonal, ← diagonal_one]
      exact congrArg diagonal (funext fun i => by
        rw [← Complex.ofReal_mul, inv_mul_cancel₀ (hne i), Complex.ofReal_one])
    · rw [diagonal_conjTranspose]
      exact congrArg diagonal (funext fun i => by simp)
    · rw [diagonal_mul_diagonal]
      exact congrArg diagonal (funext fun i => by rw [← Complex.ofReal_mul, hsq])
    · rw [diagonal_mul_diagonal, diagonal_mul_diagonal, ← diagonal_one]
      refine congrArg diagonal (funext fun i => ?_)
      have h : (√(d i))⁻¹ * d i * (√(d i))⁻¹ = 1 := by
        calc (√(d i))⁻¹ * d i * (√(d i))⁻¹ = (√(d i))⁻¹ * (√(d i) * √(d i)) * (√(d i))⁻¹ := by
              rw [hsq]
          _ = 1 := by have := hne i; field_simp
      exact_mod_cast h
    · rw [det_diagonal, Fin.prod_univ_two]
      push_cast
      rfl
  obtain ⟨hST, hTS, hTh, hSS, hTdT, hdetS⟩ := hS _ _ rfl rfl
  set S : Matrix (Fin 2) (Fin 2) ℂ := diagonal fun i => ((√(d i) : ℝ) : ℂ)
  set T : Matrix (Fin 2) (Fin 2) ℂ := diagonal fun i => (((√(d i))⁻¹ : ℝ) : ℂ)
  have hW : (U * T)ᴴ * (U * T) = 1 := by
    rw [conjTranspose_mul, hTh, Matrix.mul_assoc, ← Matrix.mul_assoc Uᴴ, hU,
      ← Matrix.mul_assoc, hTdT]
  have hEW : E * (U * T) = 0 := by rw [← Matrix.mul_assoc, hEU, Matrix.zero_mul]
  have hUM : U * M * Uᴴ = U * T * (S * M * S) * (U * T)ᴴ := by
    rw [conjTranspose_mul, hTh]
    simp only [Matrix.mul_assoc]
    rw [← Matrix.mul_assoc T S, hTS, Matrix.one_mul, ← Matrix.mul_assoc S T, hST,
      Matrix.one_mul]
  have htr : (S * M * S).trace = (M * diagonal fun i => (d i : ℂ)).trace := by
    rw [Matrix.trace_mul_comm (S * M) S, ← Matrix.mul_assoc, hSS, Matrix.trace_mul_comm]
  have hdet : ‖(S * M * S).det‖ = ‖M.det‖ * (d 0 * d 1) := by
    rw [det_mul, det_mul, hdetS, norm_mul, norm_mul,
      Complex.norm_of_nonneg (mul_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _))]
    linear_combination (‖M.det‖ * (√(d 1) * √(d 1))) * hsq 0 + (‖M.det‖ * d 0) * hsq 1
  rw [nuclearNorm_of_gram_eq_proj_add hc hEh hEE hW hEW (hX.trans (by rw [hUM])), htr, hdet]

example := nuclearNorm_of_gram_eq_proj_add_diagonal (X := (1 : Matrix (Fin 2) (Fin 2) ℂ))
  (E := 0) (U := 1) (M := 1) (d := fun _ => 1) le_rfl (fun _ => one_pos) (by simp) (by simp)
  (by simp) (by simp) (by simp)

omit [DecidableEq n] in
/-- `‖X‖²_S2 = c² tr E + tr (M Uᴴ U)` when `Xᴴ X = c² E + U M Uᴴ`. -/
theorem frobeniusSq_of_gram_eq_add {k : Type*} [Fintype k] {X : Matrix m n ℂ}
    {E : Matrix n n ℂ} {U : Matrix n k ℂ} {M : Matrix k k ℂ} {c : ℝ}
    (hX : Xᴴ * X = ((c ^ 2 : ℝ) : ℂ) • E + U * M * Uᴴ) :
    frobeniusSq X = c ^ 2 * E.trace.re + (M * (Uᴴ * U)).trace.re := by
  rw [frobeniusSq_eq_re_trace, hX, trace_add, trace_smul, smul_eq_mul,
    Matrix.trace_mul_comm (U * M), ← Matrix.mul_assoc, Matrix.trace_mul_comm (Uᴴ * U),
    Complex.add_re, Complex.re_ofReal_mul]

example := frobeniusSq_of_gram_eq_add (X := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (E := 0)
  (U := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (M := 1) (c := 0) (by simp)

/-- Under the hypotheses of `nuclearNorm_of_gram_eq_proj_add_diagonal`,
`r_eff X = (c tr E + √(tr (M diag d) + 2 √(|det M| d₀ d₁)))² / (c² tr E + tr (M diag d))`. -/
theorem effectiveRank_of_gram_eq_proj_add_diagonal {X : Matrix m n ℂ} {E : Matrix n n ℂ}
    {U : Matrix n (Fin 2) ℂ} {M : Matrix (Fin 2) (Fin 2) ℂ} {c : ℝ} {d : Fin 2 → ℝ}
    (hc : 0 ≤ c) (hd : ∀ i, 0 < d i) (hEh : Eᴴ = E) (hEE : E * E = E)
    (hU : Uᴴ * U = diagonal fun i => (d i : ℂ)) (hEU : E * U = 0)
    (hX : Xᴴ * X = ((c ^ 2 : ℝ) : ℂ) • E + U * M * Uᴴ) :
    effectiveRank X = (c * E.trace.re +
      √((M * diagonal fun i => (d i : ℂ)).trace.re + 2 * √(‖M.det‖ * (d 0 * d 1)))) ^ 2 /
      (c ^ 2 * E.trace.re + (M * diagonal fun i => (d i : ℂ)).trace.re) := by
  rw [effectiveRank, div_pow, Real.sq_sqrt (frobeniusSq_nonneg X),
    nuclearNorm_of_gram_eq_proj_add_diagonal hc hd hEh hEE hU hEU hX,
    frobeniusSq_of_gram_eq_add hX, hU]

example := effectiveRank_of_gram_eq_proj_add_diagonal (X := (1 : Matrix (Fin 2) (Fin 2) ℂ))
  (E := 0) (U := 1) (M := 1) (d := fun _ => 1) le_rfl (fun _ => one_pos) (by simp) (by simp)
  (by simp) (by simp) (by simp)

end Transformer.Grokking.FlatSpectrum
