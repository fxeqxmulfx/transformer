module

public import Transformer.Grokking.FlatSpectrum.Training.SignKernel
public import Transformer.Grokking.FlatSpectrum.Training.Scope.TwoByTwo

/-!
# The `2 × 2` kernel of signs: a matrix of its real parts and its norm

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), item XE16, the
direction "⟹". Let `X` have the real parts of `t K` for the kernel of signs
`K = [[1, 1], [−1, 1]]` and `t ≥ 0`, and at most its nuclear norm, `‖X‖_S1 ≤ 2 √2 t`. Then

    X = t K + i a J,   J = [[1, −1], [1, 1]],   |a| ≤ t

(`exists_eq_add_smul_signRot`). Write `X = t K + i B` with a real `B`. The bound
`‖X‖_S2² + 2 |det X| ≤ ‖X‖_S1² ≤ 8 t²` of `TwoByTwo`, with `‖X‖_S2² = 4 t² + ‖B‖²` and
`Re det X = 2 t² − det B`, gives `‖B‖² ≤ 2 det B`, so `B = [[a, −b], [b, a]]`. Then
`|det X| = Re det X`, so `Im det X = 2 t (a − b)` is zero and `B = a J`, and
`0 ≤ Re det X = 2 t² − 2 a²`. The step from the three bounds to `B` is real arithmetic
(`im_eq_of_det_bounds`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training.Scope

open Matrix

/-- The matrix `J = [[1, −1], [1, 1]]`. -/
def signRot : Matrix (Fin 2) (Fin 2) ℂ := !![1, -1; 1, 1]

/-- The real core of the segment: `b` are the imaginary parts of `X = t K + i B`, `N` is
`|det X|`, and the hypotheses are the bound of `TwoByTwo` at `‖X‖_S1² ≤ 8 t²`, `Re det X ≤ N`
and `N² = (Re det X)² + (Im det X)²`. -/
lemma im_eq_of_det_bounds {t b00 b01 b10 b11 N : ℝ} (ht : 0 ≤ t) (hN : 0 ≤ N)
    (h : 4 * t ^ 2 + (b00 ^ 2 + b01 ^ 2 + b10 ^ 2 + b11 ^ 2) + 2 * N ≤ 8 * t ^ 2)
    (hre : 2 * t ^ 2 - (b00 * b11 - b01 * b10) ≤ N)
    (hsq : N ^ 2 = (2 * t ^ 2 - (b00 * b11 - b01 * b10)) ^ 2 +
      (t * (b00 + b11 + b01 - b10)) ^ 2) :
    b01 = -b00 ∧ b10 = b00 ∧ b11 = b00 ∧ b00 ^ 2 ≤ t ^ 2 := by
  have hid : (b00 - b11) ^ 2 + (b01 + b10) ^ 2 =
      b00 ^ 2 + b01 ^ 2 + b10 ^ 2 + b11 ^ 2 - 2 * (b00 * b11 - b01 * b10) := by ring
  have s1 := sq_nonneg (b00 - b11)
  have s2 := sq_nonneg (b01 + b10)
  have e1 : b00 - b11 = 0 := pow_eq_zero_iff two_ne_zero |>.1 (by linarith)
  have e2 : b01 + b10 = 0 := pow_eq_zero_iff two_ne_zero |>.1 (by linarith)
  obtain rfl : b00 = b11 := by linarith
  obtain rfl : b10 = -b01 := by linarith
  have hR : 2 * t ^ 2 - (b00 * b00 - b01 * -b01) = 2 * t ^ 2 - b00 ^ 2 - b01 ^ 2 := by ring
  have hS : b00 ^ 2 + b01 ^ 2 + (-b01) ^ 2 + b00 ^ 2 = 2 * b00 ^ 2 + 2 * b01 ^ 2 := by ring
  rw [hR] at hre hsq
  rw [hS] at h
  have hNR : N = 2 * t ^ 2 - b00 ^ 2 - b01 ^ 2 := by linarith
  rw [hNR] at hsq
  have him : t * (b00 + b00 + b01 - -b01) = 0 := pow_eq_zero_iff two_ne_zero |>.1 (by linarith)
  have hb : b01 = -b00 := by
    rcases ht.eq_or_lt with h0 | hpos
    · have z0 : b00 ^ 2 = 0 := by nlinarith [sq_nonneg b00, sq_nonneg b01]
      have z1 : b01 ^ 2 = 0 := by nlinarith [sq_nonneg b00, sq_nonneg b01]
      rw [pow_eq_zero_iff two_ne_zero |>.1 z0, pow_eq_zero_iff two_ne_zero |>.1 z1, neg_zero]
    · rcases mul_eq_zero.1 him with h0 | h0
      · exact absurd h0 hpos.ne'
      · linarith
  refine ⟨hb, by rw [hb, neg_neg], rfl, ?_⟩
  rw [hb] at hNR
  nlinarith

/-- **A matrix with the real parts of `t K` and at most its nuclear norm lies on the segment**
(`review/flat_spectrum.md`, item XE16). -/
lemma exists_eq_add_smul_signRot {t : ℝ} (ht : 0 ≤ t) {X : Matrix (Fin 2) (Fin 2) ℂ}
    (r00 : (X 0 0).re = t) (r01 : (X 0 1).re = t) (r10 : (X 1 0).re = -t)
    (r11 : (X 1 1).re = t) (hn : nuclearNorm X ≤ t * (2 * √2)) :
    ∃ a : ℝ, |a| ≤ t ∧ X = t • signKernel + ((a : ℂ) * Complex.I) • signRot := by
  have h := frobeniusSq_add_two_mul_norm_det_le X
  have h8 : nuclearNorm X ^ 2 ≤ 8 * t ^ 2 := by
    calc nuclearNorm X ^ 2 ≤ (t * (2 * √2)) ^ 2 := pow_le_pow_left₀ (nuclearNorm_nonneg X) hn 2
      _ = 8 * t ^ 2 := by rw [mul_pow, mul_pow, Real.sq_sqrt zero_le_two]; ring
  have hF : frobeniusSq X = 4 * t ^ 2 +
      ((X 0 0).im ^ 2 + (X 0 1).im ^ 2 + (X 1 0).im ^ 2 + (X 1 1).im ^ 2) := by
    conv_lhs => rw [Matrix.eta_fin_two X]
    rw [frobeniusSq_fin_two, Complex.sq_norm, Complex.sq_norm, Complex.sq_norm, Complex.sq_norm,
      Complex.normSq_apply, Complex.normSq_apply, Complex.normSq_apply, Complex.normSq_apply,
      r00, r01, r10, r11]
    ring
  have hdre : X.det.re = 2 * t ^ 2 - ((X 0 0).im * (X 1 1).im - (X 0 1).im * (X 1 0).im) := by
    rw [Matrix.det_fin_two, Complex.sub_re, Complex.mul_re, Complex.mul_re, r00, r01, r10, r11]
    ring
  have hdim : X.det.im = t * ((X 0 0).im + (X 1 1).im + (X 0 1).im - (X 1 0).im) := by
    rw [Matrix.det_fin_two, Complex.sub_im, Complex.mul_im, Complex.mul_im, r00, r01, r10, r11]
    ring
  have hre := (le_abs_self _).trans (Complex.abs_re_le_norm X.det)
  have hsq := Complex.sq_norm X.det
  rw [Complex.normSq_apply, hdre, hdim, ← pow_two, ← pow_two] at hsq
  rw [hdre] at hre
  rw [hF] at h
  obtain ⟨e01, e10, e11, ha⟩ := im_eq_of_det_bounds ht (norm_nonneg X.det) (h.trans h8) hre hsq
  refine ⟨(X 0 0).im, abs_le_of_sq_le_sq ha ht, ?_⟩
  ext i j
  fin_cases i <;> fin_cases j
  · exact Complex.ext (by simp [signKernel, signRot, r00]) (by simp [signKernel, signRot])
  · exact Complex.ext (by simp [signKernel, signRot, r01]) (by simp [signKernel, signRot, e01])
  · exact Complex.ext (by simp [signKernel, signRot, r10]) (by simp [signKernel, signRot, e10])
  · exact Complex.ext (by simp [signKernel, signRot, r11]) (by simp [signKernel, signRot, e11])

/-! The witnesses: `t = 1`, `B = 0`, `N = 2`, and the kernel of signs itself. -/

example := im_eq_of_det_bounds (t := 1) (b00 := 0) (b01 := 0) (b10 := 0) (b11 := 0) (N := 2)
  zero_le_one zero_le_two (by norm_num) (by norm_num) (by norm_num)

example := exists_eq_add_smul_signRot (t := 1) zero_le_one (X := signKernel)
  (by simp [signKernel]) (by simp [signKernel]) (by simp [signKernel]) (by simp [signKernel])
  (by rw [nuclearNorm_signKernel, one_mul])

end Transformer.Grokking.FlatSpectrum.Training.Scope
