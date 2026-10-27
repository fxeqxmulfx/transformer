module

public import Transformer.Grokking.FlatSpectrum.Roots

/-!
# Nuclear norms of an identity with one small block

Source: `review/flat_spectrum.md`, Corollary, proof sketch: the one-entry change "changes a
single 2×2 block". For an isometry `J` (`Jᴴ J = 1`), the matrix `c (1 - J Jᴴ) + J B Jᴴ`
acts as `c` off the range of `J` and as `B` on it, so its nuclear norm is
`c (dim - rank J) + ‖B‖_S1`. A `2 × 2` block has `σ₁ + σ₂ = √(σ₁² + σ₂² + 2 σ₁ σ₂)`
with `σ₁ σ₂ = |det B|`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix
open scoped ComplexOrder

variable {n k : Type*} [Fintype n] [DecidableEq n] [Fintype k] [DecidableEq k]

/-- Splitting off an identity block: if `Jᴴ J = 1` and `c ≥ 0`, then
`‖c (1 - J Jᴴ) + J B Jᴴ‖_S1 = c (card n - card k) + ‖B‖_S1`. -/
theorem nuclearNorm_block {J : Matrix n k ℂ} (hJ : Jᴴ * J = 1) {c : ℝ} (hc : 0 ≤ c)
    (B : Matrix k k ℂ) :
    nuclearNorm ((c : ℂ) • (1 - J * Jᴴ) + J * B * Jᴴ) =
      c * (Fintype.card n - Fintype.card k) + nuclearNorm B := by
  obtain ⟨P, hP, hPB⟩ := exists_posSemidef_mul_self_eq B
  set E : Matrix n n ℂ := 1 - J * Jᴴ with hEdef
  have hEh : Eᴴ = E := by
    rw [hEdef, conjTranspose_sub, conjTranspose_one, conjTranspose_mul,
      conjTranspose_conjTranspose]
  have hEJ : E * J = 0 := by
    rw [hEdef, Matrix.sub_mul, Matrix.one_mul, Matrix.mul_assoc, hJ, Matrix.mul_one, sub_self]
  have hJE : Jᴴ * E = 0 := by
    rw [hEdef, Matrix.mul_sub, Matrix.mul_one, ← Matrix.mul_assoc, hJ, Matrix.one_mul,
      sub_self]
  have hEE : E * E = E := by
    calc E * E = E - J * (Jᴴ * E) := by
          rw [← Matrix.mul_assoc]; conv_lhs => rw [hEdef]
          rw [Matrix.sub_mul, Matrix.one_mul]
      _ = E := by rw [hJE, Matrix.mul_zero, sub_zero]
  have hEpsd : E.PosSemidef := by
    have := posSemidef_conjTranspose_mul_self E
    rwa [hEh, hEE] at this
  have key : ∀ A₁ A₂ : Matrix k k ℂ,
      ((c : ℂ) • E + J * A₁ * Jᴴ) * ((c : ℂ) • E + J * A₂ * Jᴴ) =
        ((c : ℂ) * c) • E + J * (A₁ * A₂) * Jᴴ := by
    intro A₁ A₂
    have h1 : E * (J * A₂ * Jᴴ) = 0 := by
      rw [← Matrix.mul_assoc, ← Matrix.mul_assoc, hEJ, Matrix.zero_mul, Matrix.zero_mul]
    have h2 : J * A₁ * Jᴴ * E = 0 := by rw [Matrix.mul_assoc, hJE, Matrix.mul_zero]
    have h3 : J * A₁ * Jᴴ * (J * A₂ * Jᴴ) = J * (A₁ * A₂) * Jᴴ := by
      simp only [Matrix.mul_assoc]
      rw [← Matrix.mul_assoc Jᴴ J, hJ, Matrix.one_mul]
    rw [Matrix.add_mul, Matrix.mul_add, Matrix.mul_add, Matrix.smul_mul, Matrix.mul_smul,
      Matrix.smul_mul, Matrix.mul_smul, hEE, h1, h2, h3, smul_smul, smul_zero,
      add_zero, zero_add]
  have hY : ((c : ℂ) • E + J * B * Jᴴ)ᴴ = (c : ℂ) • E + J * Bᴴ * Jᴴ := by
    rw [conjTranspose_add, conjTranspose_smul, hEh, conjTranspose_mul, conjTranspose_mul,
      conjTranspose_conjTranspose, Complex.star_def, Complex.conj_ofReal, Matrix.mul_assoc]
  have h := nuclearNorm_eq_trace_of_mul_self (X := (c : ℂ) • E + J * B * Jᴴ)
    ((hEpsd.smul (Complex.zero_le_real.mpr hc)).add (hP.mul_mul_conjTranspose_same J))
    (by rw [hY, key, key, hPB])
  have htrE : E.trace = (Fintype.card n : ℂ) - Fintype.card k := by
    rw [hEdef, trace_sub, trace_one, Matrix.trace_mul_comm, hJ, trace_one]
  rw [trace_add, trace_smul, smul_eq_mul, htrE, Matrix.trace_mul_comm, ← Matrix.mul_assoc, hJ,
    Matrix.one_mul, ← nuclearNorm_eq_trace_of_mul_self hP hPB] at h
  exact_mod_cast h

example (B : Matrix (Fin 2) (Fin 2) ℂ) :=
  nuclearNorm_block (J := (of fun i j => if (i : ℕ) = j then 1 else 0 : Matrix (Fin 3) (Fin 2) ℂ))
    (by ext i j; fin_cases i <;> fin_cases j <;> simp [mul_apply, Fin.sum_univ_three])
    zero_le_one B

/-- Cayley–Hamilton for `2 × 2` matrices. -/
lemma mul_self_fin_two (A : Matrix (Fin 2) (Fin 2) ℂ) :
    A * A = A.trace • A - A.det • (1 : Matrix (Fin 2) (Fin 2) ℂ) := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [Matrix.mul_apply, Fin.sum_univ_two, Matrix.trace_fin_two, Matrix.det_fin_two] <;> ring

/-- A `2 × 2` matrix has `‖B‖_S1 = σ₁ + σ₂ = √(‖B‖_S2² + 2 |det B|)`,
`review/flat_spectrum.md`, Corollary (the 2×2 block). -/
theorem nuclearNorm_fin_two (B : Matrix (Fin 2) (Fin 2) ℂ) :
    nuclearNorm B = √(frobeniusSq B + 2 * ‖B.det‖) := by
  rcases eq_or_ne B 0 with rfl | hB
  · simp [nuclearNorm_zero, frobeniusSq]
  set G := Bᴴ * B with hGdef
  set δ := ‖B.det‖
  set τ := frobeniusSq B
  have hτ : 0 < τ := lt_of_le_of_ne (frobeniusSq_nonneg B) fun h =>
    hB (eq_zero_of_frobeniusSq_eq_zero h.symm)
  have hδ : 0 ≤ δ := norm_nonneg _
  set s := √(τ + 2 * δ)
  have hs2 : s ^ 2 = τ + 2 * δ := Real.sq_sqrt (by positivity)
  have hs : s ≠ 0 := (Real.sqrt_pos.mpr (by positivity)).ne'
  have htrG : G.trace = (τ : ℂ) := trace_conjTranspose_mul_self B
  have hdetG : G.det = ((δ ^ 2 : ℝ) : ℂ) := by
    rw [hGdef, det_mul, det_conjTranspose, Complex.star_def, Complex.conj_mul']
    push_cast
    rfl
  have hsq : (G + (δ : ℂ) • 1) * (G + (δ : ℂ) • 1) = ((τ + 2 * δ : ℝ) : ℂ) • G := by
    rw [Matrix.add_mul, Matrix.mul_add, Matrix.mul_add, mul_self_fin_two, htrG, hdetG]
    ext i j
    simp only [Matrix.add_apply, Matrix.sub_apply, Matrix.smul_apply, Matrix.smul_mul,
      Matrix.mul_smul, Matrix.one_mul, Matrix.mul_one, smul_eq_mul]
    push_cast
    ring
  set P : Matrix (Fin 2) (Fin 2) ℂ := ((s⁻¹ : ℝ) : ℂ) • (G + (δ : ℂ) • 1)
  have hP : P.PosSemidef :=
    ((posSemidef_conjTranspose_mul_self B).add
      (PosSemidef.one.smul (Complex.zero_le_real.mpr hδ))).smul
      (Complex.zero_le_real.mpr (inv_nonneg.mpr (Real.sqrt_nonneg _)))
  have hPP : P * P = Bᴴ * B := by
    rw [Matrix.smul_mul, Matrix.mul_smul, smul_smul, hsq, smul_smul, ← Complex.ofReal_mul,
      ← Complex.ofReal_mul, show s⁻¹ * s⁻¹ * (τ + 2 * δ) = 1 by rw [← hs2]; field_simp,
      Complex.ofReal_one, one_smul]
  have htr : P.trace = (s : ℂ) := by
    rw [trace_smul, trace_add, trace_smul, trace_one, htrG, Fintype.card_fin, smul_eq_mul,
      smul_eq_mul]
    rw [show ((s⁻¹ : ℝ) : ℂ) * ((τ : ℂ) + (δ : ℂ) * ((2 : ℕ) : ℂ)) =
        ((s⁻¹ * s ^ 2 : ℝ) : ℂ) by rw [hs2]; push_cast; ring]
    congr 1
    field_simp
  exact_mod_cast (nuclearNorm_eq_trace_of_mul_self hP hPP).trans htr

end Transformer.Grokking.FlatSpectrum
