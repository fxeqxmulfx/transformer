module

public import Transformer.Grokking.FlatSpectrum.Roots
public import Mathlib.NumberTheory.LegendreSymbol.AddCharacter
public import Mathlib.LinearAlgebra.Matrix.Rank

/-!
# Proposition 2: the spectra of the `a + b` and `a³ + a b` kernels

Source: `review/flat_spectrum.md`, Proposition 2, with `ψ x = exp (2πi x / p)`. The kernel
of `a + b` is `ψ ψᵀ`: rank one, nuclear norm `p`. The kernel of `a³ + a b` is
`D F` with `D = diag ψ(a³)` and `F = [ψ(a b)]`, `F F* = p I`; so `K K* = K* K = p I`,
every singular value is `√p` and `‖K‖_S1 = p^{3/2}`. Neither needs `p` prime: `ψ` is a
primitive character of every `ℤ/p`, so `∑_b ψ (b x) = p [x = 0]`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix
open scoped ComplexOrder

variable (p : ℕ) [NeZero p]

/-- Orthogonality of the characters `b ↦ ψ (b x)`: `∑_b ψ (b x) = p [x = 0]`. -/
lemma sum_stdAddChar_mul (x : ZMod p) :
    ∑ b : ZMod p, ZMod.stdAddChar (b * x) = if x = 0 then (p : ℂ) else 0 := by
  rw [AddChar.sum_mulShift x (ZMod.isPrimitive_stdAddChar p)]
  split_ifs <;> simp [ZMod.card]

variable {p} in
/-- `conj (ψ x) ψ y = ψ (y - x)`. -/
lemma star_stdAddChar_mul (x y : ZMod p) :
    star (ZMod.stdAddChar x) * ZMod.stdAddChar y = ZMod.stdAddChar (y - x) := by
  rw [Complex.star_def, ← AddChar.map_neg_eq_conj, ← AddChar.map_add_eq_mul, neg_add_eq_sub]

/-- The `a + b` kernel is the outer product `ψ ψᵀ`, Proposition 2. -/
lemma addKernel_eq_vecMulVec :
    addKernel p = vecMulVec ZMod.stdAddChar ZMod.stdAddChar := by
  ext a b
  simp [addKernel, taskKernel, vecMulVec_apply, AddChar.map_add_eq_mul]

/-- The `a + b` kernel has rank one, Proposition 2. -/
theorem rank_addKernel : (addKernel p).rank = 1 := by
  refine le_antisymm (addKernel_eq_vecMulVec p ▸ rank_vecMulVec_le _ _) ?_
  rw [Nat.one_le_iff_ne_zero, Ne, rank_eq_finrank_span_cols, Submodule.finrank_eq_zero,
    Submodule.span_eq_bot]
  intro h
  have := congrFun (h _ ⟨0, rfl⟩) 0
  simp [addKernel, taskKernel, col] at this

/-- `‖K_{a+b}‖_S1 = p`: `P = conj ψ ψᵀ` is a positive semidefinite root of `K* K = p P`,
and `tr P = p`, Proposition 2. -/
theorem nuclearNorm_addKernel : nuclearNorm (addKernel p) = p := by
  have hd : star ⇑(ZMod.stdAddChar (N := p)) ⬝ᵥ ⇑ZMod.stdAddChar = (p : ℂ) := by
    simp only [dotProduct, Pi.star_apply, star_stdAddChar_mul, sub_self,
      AddChar.map_zero_eq_one, Finset.sum_const, Finset.card_univ, ZMod.card, nsmul_eq_mul,
      mul_one]
  have hd' : ⇑(ZMod.stdAddChar (N := p)) ⬝ᵥ star ⇑ZMod.stdAddChar = (p : ℂ) := by
    rw [dotProduct_comm, hd]
  have hPP : vecMulVec (star ⇑(ZMod.stdAddChar (N := p))) (star (star ⇑ZMod.stdAddChar)) *
      vecMulVec (star ⇑(ZMod.stdAddChar (N := p))) (star (star ⇑ZMod.stdAddChar)) =
      (addKernel p)ᴴ * addKernel p := by
    rw [addKernel_eq_vecMulVec, conjTranspose_vecMulVec, star_star, vecMulVec_mul_vecMulVec,
      vecMulVec_mul_vecMulVec, hd, hd']
  have h := nuclearNorm_eq_trace_of_mul_self (posSemidef_vecMulVec_self_star _) hPP
  rw [trace_vecMulVec, star_star, hd] at h
  exact_mod_cast h

/-- The `a³ + a b` kernel factors as `diag ψ(a³) · [ψ(a b)]`, Proposition 2. -/
lemma cubicKernel_eq_diagonal_mul :
    cubicKernel p = diagonal (fun a => ZMod.stdAddChar (a ^ 3)) *
      Matrix.of fun a b : ZMod p => ZMod.stdAddChar (a * b) := by
  ext a b
  simp [cubicKernel, taskKernel, diagonal_mul, AddChar.map_add_eq_mul]

/-- `F F* = p I` for `F = [ψ(a b)]`, Proposition 2. -/
theorem stdAddChar_mul_conjTranspose :
    (Matrix.of fun a b : ZMod p => ZMod.stdAddChar (a * b)) *
      (Matrix.of fun a b : ZMod p => ZMod.stdAddChar (a * b))ᴴ = (p : ℂ) • 1 := by
  ext a c
  simp only [mul_apply, of_apply, conjTranspose_apply, Matrix.smul_apply, one_apply, smul_eq_mul]
  rw [Finset.sum_congr rfl fun b _ => (by
    rw [mul_comm, star_stdAddChar_mul, show a * b - c * b = b * (a - c) by ring] :
      ZMod.stdAddChar (a * b) * star (ZMod.stdAddChar (c * b)) = ZMod.stdAddChar (b * (a - c))),
    sum_stdAddChar_mul]
  simp [sub_eq_zero]

/-- `K* K = p I` for the `a³ + a b` kernel, Proposition 2. -/
theorem conjTranspose_mul_cubicKernel :
    (cubicKernel p)ᴴ * cubicKernel p = (p : ℂ) • 1 := by
  ext b d
  simp only [mul_apply, conjTranspose_apply, cubicKernel, taskKernel, of_apply,
    Matrix.smul_apply, one_apply, smul_eq_mul]
  rw [Finset.sum_congr rfl fun a _ => (by
    rw [star_stdAddChar_mul, show a ^ 3 + a * d - (a ^ 3 + a * b) = a * (d - b) by ring] :
      star (ZMod.stdAddChar (a ^ 3 + a * b)) * ZMod.stdAddChar (a ^ 3 + a * d) =
        ZMod.stdAddChar (a * (d - b))),
    sum_stdAddChar_mul]
  simp [sub_eq_zero, eq_comm]

/-- `K K* = p I` for the `a³ + a b` kernel, Proposition 2. -/
theorem cubicKernel_mul_conjTranspose :
    cubicKernel p * (cubicKernel p)ᴴ = (p : ℂ) • 1 := by
  ext a c
  simp only [mul_apply, conjTranspose_apply, cubicKernel, taskKernel, of_apply,
    Matrix.smul_apply, one_apply, smul_eq_mul]
  rw [Finset.sum_congr rfl fun b _ => (by
    rw [mul_comm, star_stdAddChar_mul,
      show a ^ 3 + a * b - (c ^ 3 + c * b) = (a ^ 3 - c ^ 3) + b * (a - c) by ring,
      AddChar.map_add_eq_mul] :
      ZMod.stdAddChar (a ^ 3 + a * b) * star (ZMod.stdAddChar (c ^ 3 + c * b)) =
        ZMod.stdAddChar (a ^ 3 - c ^ 3) * ZMod.stdAddChar (b * (a - c))),
    ← Finset.mul_sum, sum_stdAddChar_mul]
  by_cases h : a = c
  · subst h
    simp
  · simp [h, sub_eq_zero]

variable {p} in
/-- A Hermitian matrix equal to `c I` has every eigenvalue `c`. -/
lemma eigenvalues_eq_of_eq_smul_one {n : Type*} [Fintype n] [DecidableEq n]
    {A : Matrix n n ℂ} (hA : A.IsHermitian) {c : ℝ} (h : A = (c : ℂ) • 1) (i : n) :
    hA.eigenvalues i = c := by
  subst h
  have h1 := congrFun (congrFun (eigenvectorUnitary_conj hA) i) i
  rw [Matrix.mul_smul, Matrix.mul_one, Matrix.smul_mul, eigenvectorUnitary_conjTranspose_mul hA,
    diagonal_apply_eq, Matrix.smul_apply, one_apply_eq, smul_eq_mul, mul_one] at h1
  exact_mod_cast h1.symm

example (i : ZMod 2) := eigenvalues_eq_of_eq_smul_one
  (isHermitian_conjTranspose_mul_self (cubicKernel 2)) (c := (2 : ℕ))
  (by rw [conjTranspose_mul_cubicKernel, Complex.ofReal_natCast]) i

/-- Every singular value of the `a³ + a b` kernel is `√p`, Proposition 2. -/
theorem singularValues_cubicKernel (i : ZMod p) : singularValues (cubicKernel p) i = √p := by
  rw [singularValues, eigenvalues_eq_of_eq_smul_one _ (by
    rw [conjTranspose_mul_cubicKernel, Complex.ofReal_natCast])]

/-- `‖K_{a³+ab}‖_S1 = p^{3/2}`, Proposition 2. -/
theorem nuclearNorm_cubicKernel : nuclearNorm (cubicKernel p) = (p : ℝ) ^ (3 / 2 : ℝ) := by
  simp only [nuclearNorm, singularValues_cubicKernel, Finset.sum_const, Finset.card_univ,
    ZMod.card, nsmul_eq_mul]
  have hp : (0 : ℝ) < p := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne p)
  rw [show (3 / 2 : ℝ) = 1 + 1 / 2 by norm_num, Real.rpow_add hp, Real.rpow_one,
    Real.sqrt_eq_rpow]

/-- Unimodular `p × p` kernels have `‖K‖_S2² = p²`. -/
lemma frobeniusSq_taskKernel (f : ZMod p → ZMod p → ZMod p) :
    frobeniusSq (taskKernel f ZMod.stdAddChar) = (p : ℝ) ^ 2 := by
  simp [frobeniusSq, taskKernel, ZMod.card, sq]

/-- `r_eff (K_{a+b}) = 1`. -/
theorem effectiveRank_addKernel : effectiveRank (addKernel p) = 1 := by
  have hp : (p : ℝ) ≠ 0 := by exact_mod_cast NeZero.ne p
  rw [effectiveRank, nuclearNorm_addKernel, addKernel, frobeniusSq_taskKernel,
    Real.sqrt_sq (Nat.cast_nonneg p), div_self hp, one_pow]

/-- `r_eff (K_{a³+ab}) = p`: the flat spectrum has full effective rank. -/
theorem effectiveRank_cubicKernel : effectiveRank (cubicKernel p) = p := by
  have hp : (0 : ℝ) < p := by exact_mod_cast Nat.pos_of_ne_zero (NeZero.ne p)
  rw [effectiveRank, nuclearNorm_cubicKernel, cubicKernel, frobeniusSq_taskKernel,
    Real.sqrt_sq hp.le, show (3 / 2 : ℝ) = 1 + 1 / 2 by norm_num, Real.rpow_add hp,
    Real.rpow_one, ← Real.sqrt_eq_rpow, mul_div_cancel_left₀ _ hp.ne', Real.sq_sqrt hp.le]

end Transformer.Grokking.FlatSpectrum
