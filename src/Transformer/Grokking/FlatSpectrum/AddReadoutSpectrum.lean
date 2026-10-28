module

public import Transformer.Grokking.FlatSpectrum.OneHotSpectrum
public import Transformer.Grokking.FlatSpectrum.Unimodular
public import Mathlib.Analysis.Fourier.ZMod

/-!
# The spectrum of every readout of `a + b`

Source: `review/flat_spectrum.md`, Readout-free version: "For (a+b)², gradient search finds
2.0". A kernel `K(a, b) = g(a + b)` on `ℤ/N` is diagonalized by the Fourier matrix
`F = [ψ(k b)]`. Fourier inversion `g(s) = (1/N) ∑_k ψ(k s) ĝ(k)`, with `ĝ = ZMod.dft g`, gives
`K = F diag(ĝ) F / N`. As `F* F = F F* = N I`, `K* K = F* diag|ĝ|² F / N` has the characteristic
polynomial `∏_k (X − |ĝ(k)|²)`, so the singular values of `K` are the `|ĝ(k)|`:
`‖K‖_S1 = ∑_k |ĝ(k)|`, `‖K‖²_S2 = ∑_k |ĝ(k)|²` and `r_eff K = (∑_k |ĝ(k)|)² / ∑_k |ĝ(k)|²`.

The readout `g(s) = cos(2π r s/N) = (ψ(r s) + ψ(−r s))/2` has `ĝ = (N/2)([k = r] + [k = −r])`.
A constant readout `g = c ≠ 0` has `ĝ = N c [k = 0]` and `r_eff = 1`, so the note's `2.0` for
`(a + b)²` is a minimum over centered readouts only.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix Finset Polynomial

variable {N : ℕ} [NeZero N]

variable (N) in
/-- The Fourier matrix `F = [ψ(k b)]` of `ℤ/N`. -/
noncomputable def fourierMatrix : Matrix (ZMod N) (ZMod N) ℂ :=
  Matrix.of fun k b => ZMod.stdAddChar (k * b)

/-- `F F* = N I`. -/
lemma fourierMatrix_mul_conjTranspose :
    fourierMatrix N * (fourierMatrix N)ᴴ = (N : ℂ) • 1 :=
  stdAddChar_mul_conjTranspose N

/-- `F* F = N I`. -/
lemma conjTranspose_fourierMatrix_mul :
    (fourierMatrix N)ᴴ * fourierMatrix N = (N : ℂ) • 1 := by
  ext b c
  simp only [fourierMatrix, mul_apply, conjTranspose_apply, of_apply, Matrix.smul_apply,
    one_apply, smul_eq_mul]
  rw [sum_congr rfl fun k _ => (by
    rw [star_stdAddChar_mul, show k * c - k * b = k * (c - b) by ring] :
      star (ZMod.stdAddChar (k * b)) * ZMod.stdAddChar (k * c) = ZMod.stdAddChar (k * (c - b))),
    sum_stdAddChar_mul]
  rcases eq_or_ne b c with rfl | h
  · simp
  · simp [sub_eq_zero, h, Ne.symm h]

/-- Fourier inversion: `K = F diag(ĝ) F / N` for `K(a, b) = g(a + b)`. -/
theorem taskKernel_add_eq_fourier (g : ZMod N → ℂ) :
    taskKernel (· + ·) g =
      (1 / N : ℂ) • (fourierMatrix N * diagonal (ZMod.dft g) * fourierMatrix N) := by
  ext a b
  have h := congrFun (ZMod.dft.symm_apply_apply g) (a + b)
  rw [ZMod.invDFT_apply, smul_eq_mul] at h
  rw [Matrix.smul_apply, smul_eq_mul, mul_apply]
  simp only [taskKernel, of_apply, mul_diagonal, fourierMatrix]
  rw [← h, one_div]
  congr 1
  refine sum_congr rfl fun k _ => ?_
  rw [show k * (a + b) = a * k + k * b by ring, AddChar.map_add_eq_mul]
  ring

/-- `K* K = F* diag|ĝ|² F / N`. -/
theorem conjTranspose_mul_taskKernel_add_eq_fourier (g : ZMod N → ℂ) :
    (taskKernel (· + ·) g)ᴴ * taskKernel (· + ·) g = (1 / N : ℂ) • ((fourierMatrix N)ᴴ *
      diagonal (fun k => ((‖ZMod.dft g k‖ ^ 2 : ℝ) : ℂ)) * fourierMatrix N) := by
  have hD : (diagonal (ZMod.dft g))ᴴ * diagonal (ZMod.dft g) =
      diagonal fun k => ((‖ZMod.dft g k‖ ^ 2 : ℝ) : ℂ) := by
    rw [diagonal_conjTranspose, diagonal_mul_diagonal]
    congr 1
    ext k
    rw [Pi.star_apply, Complex.star_def, Complex.conj_mul', Complex.ofReal_pow]
  have hF (F D : Matrix (ZMod N) (ZMod N) ℂ) (hF : Fᴴ * F = (N : ℂ) • 1) :
      Fᴴ * (Dᴴ * Fᴴ) * (F * D * F) = (N : ℂ) • (Fᴴ * (Dᴴ * D) * F) := by
    simp only [Matrix.mul_assoc]
    rw [← Matrix.mul_assoc Fᴴ F, hF, Matrix.smul_mul, Matrix.one_mul, Matrix.mul_smul,
      Matrix.mul_smul]
  have hc : star (1 / (N : ℂ)) * (1 / N) * N = 1 / N := by
    have hN : (N : ℂ) ≠ 0 := Nat.cast_ne_zero.mpr (NeZero.ne N)
    rw [show star (1 / (N : ℂ)) = 1 / N by simp]
    field_simp
  rw [taskKernel_add_eq_fourier, conjTranspose_smul, conjTranspose_mul, conjTranspose_mul,
    Matrix.smul_mul, Matrix.mul_smul, smul_smul, hF _ _ conjTranspose_fourierMatrix_mul,
    smul_smul, hc, hD]

/-- The eigenvalues of `K* K` are the `|ĝ(k)|²`: its characteristic polynomial is
`∏_k (X − |ĝ(k)|²)`. -/
theorem charpoly_conjTranspose_mul_taskKernel_add (g : ZMod N → ℂ) :
    ((taskKernel (· + ·) g)ᴴ * taskKernel (· + ·) g).charpoly =
      ∏ k, (X - C (((‖ZMod.dft g k‖ ^ 2 : ℝ)) : ℂ)) := by
  have hN : (N : ℂ) ≠ 0 := Nat.cast_ne_zero.mpr (NeZero.ne N)
  rw [conjTranspose_mul_taskKernel_add_eq_fourier, Matrix.mul_assoc, ← Matrix.smul_mul,
    charpoly_mul_comm, Matrix.mul_smul, Matrix.mul_assoc, fourierMatrix_mul_conjTranspose,
    Matrix.mul_smul, Matrix.mul_one, smul_smul, one_div_mul_cancel hN, one_smul,
    charpoly_diagonal]

/-- The singular values of `K(a, b) = g(a + b)` are the `|ĝ(k)|`: `‖K‖_S1 = ∑_k |ĝ(k)|`. -/
theorem nuclearNorm_taskKernel_add_eq_sum_dft (g : ZMod N → ℂ) :
    nuclearNorm (taskKernel (· + ·) g) = ∑ k, ‖ZMod.dft g k‖ := by
  have h := sum_comp_eigenvalues_of_charpoly
    (isHermitian_conjTranspose_mul_self (taskKernel (· + ·) g))
    (charpoly_conjTranspose_mul_taskKernel_add g) Real.sqrt
  simp only [Real.sqrt_sq_eq_abs, abs_norm] at h
  simpa only [nuclearNorm, singularValues] using h

/-- `‖K‖²_S2 = ∑_k |ĝ(k)|²` for `K(a, b) = g(a + b)`. -/
theorem frobeniusSq_taskKernel_add_eq_sum_dft (g : ZMod N → ℂ) :
    frobeniusSq (taskKernel (· + ·) g) = ∑ k, ‖ZMod.dft g k‖ ^ 2 := by
  have h := sum_comp_eigenvalues_of_charpoly
    (isHermitian_conjTranspose_mul_self (taskKernel (· + ·) g))
    (charpoly_conjTranspose_mul_taskKernel_add g) fun t => √t ^ 2
  simp only [Real.sqrt_sq_eq_abs, abs_norm] at h
  rw [← sum_singularValues_sq]
  simpa only [singularValues] using h

/-- `r_eff [g(a + b)] = (∑_k |ĝ(k)|)² / ∑_k |ĝ(k)|²`. -/
theorem effectiveRank_taskKernel_add_eq_dft (g : ZMod N → ℂ) :
    effectiveRank (taskKernel (· + ·) g) =
      (∑ k, ‖ZMod.dft g k‖) ^ 2 / ∑ k, ‖ZMod.dft g k‖ ^ 2 := by
  rw [effectiveRank, nuclearNorm_taskKernel_add_eq_sum_dft, frobeniusSq_taskKernel_add_eq_sum_dft,
    div_pow, Real.sq_sqrt (sum_nonneg fun k _ => sq_nonneg _)]

/-- The transform of `g(s) = (ψ(r s) + ψ(−r s))/2`, which is `cos(2π r s/N)`, is
`(N/2)([k = r] + [k = −r])`. -/
lemma dft_cos (r k : ZMod N) :
    ZMod.dft (fun s => (ZMod.stdAddChar (r * s) + ZMod.stdAddChar (-(r * s))) / 2) k =
      (((N / 2 : ℝ) * ((if r = k then 1 else 0) + if -r = k then 1 else 0) : ℝ) : ℂ) := by
  have h (j : ZMod N) : ZMod.stdAddChar (-(j * k)) •
      ((ZMod.stdAddChar (r * j) + ZMod.stdAddChar (-(r * j))) / 2) =
      (ZMod.stdAddChar (j * (r - k)) + ZMod.stdAddChar (j * (-r - k))) / 2 := by
    rw [smul_eq_mul, mul_div_assoc', mul_add, ← AddChar.map_add_eq_mul,
      ← AddChar.map_add_eq_mul, show -(j * k) + r * j = j * (r - k) by ring,
      show -(j * k) + -(r * j) = j * (-r - k) by ring]
  rw [ZMod.dft_apply, sum_congr rfl fun j _ => h j, ← sum_div, sum_add_distrib,
    sum_stdAddChar_mul, sum_stdAddChar_mul]
  simp only [sub_eq_zero]
  split_ifs <;> push_cast <;> ring

/-- The transform of a constant `c` is `N c [k = 0]`. -/
lemma dft_const (c : ℂ) (k : ZMod N) :
    ZMod.dft (fun _ => c) k = if k = 0 then (N : ℂ) * c else 0 := by
  simp only [ZMod.dft_apply, smul_eq_mul, ← sum_mul, ← mul_neg, sum_stdAddChar_mul, neg_eq_zero]
  split_ifs <;> simp

/-- A constant readout `c ≠ 0` has `r_eff = 1`, so the note's `2.0` is not a minimum over all
readouts of `a + b` or `(a + b)²`: its search runs over centered readouts. -/
theorem effectiveRank_taskKernel_add_const {c : ℂ} (hc : c ≠ 0) :
    effectiveRank (taskKernel (· + ·) fun _ : ZMod N => c) = 1 := by
  have hN : (N : ℂ) ≠ 0 := Nat.cast_ne_zero.mpr (NeZero.ne N)
  rw [effectiveRank_taskKernel_add_eq_dft]
  simp only [dft_const, apply_ite norm, norm_zero, apply_ite (· ^ 2), ne_eq,
    OfNat.ofNat_ne_zero, not_false_eq_true, zero_pow, sum_ite_eq', mem_univ, ite_true]
  exact div_self (pow_ne_zero 2 (norm_ne_zero_iff.mpr (mul_ne_zero hN hc)))

example := effectiveRank_taskKernel_add_const (N := 5) one_ne_zero

end Transformer.Grokking.FlatSpectrum
