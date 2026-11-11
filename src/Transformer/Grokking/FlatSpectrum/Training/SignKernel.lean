module

public import Transformer.Grokking.FlatSpectrum.Training.SignPath

/-!
# A flat kernel of signs and a second matrix with its real parts and its norm

Source: `review/flat_spectrum.md`, *Training with cross-entropy* (added), items XE11 and XE13.
The witness of the code `±1` (gap `g = 2`): the kernel of signs

    K = [[1, 1], [−1, 1]],    K Kᴴ = 2,    ‖K‖_S1 = 2 √2

(`signKernel`, `isFlat_signKernel`, `nuclearNorm_signKernel`), with all four cells seen once
and the labels it shows (`signLabel`, `signKernel_apply`). The matrix

    K' = [[1 − i, 1 + i], [−(1 + i), 1 − i]] = (1, −i)ᵀ (1 − i, 1 + i)

(`signTwin`, `signTwin_eq_vecMulVec`) has the real parts of `K` (`signTwin_re`) and is not `K`
(`signTwin_ne`). It has rank one and `‖K'‖² = 8`, so `‖K'‖_S1 ≤ √8 = ‖K‖_S1` by R3
(`nuclearNorm_signTwin_le`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.Training

open Matrix

/-- The `2 × 2` kernel of signs `[[1, 1], [−1, 1]]`. -/
def signKernel : Matrix (Fin 2) (Fin 2) ℂ := !![1, 1; -1, 1]

/-- The labels that `signKernel` shows with the code `±1`: `false` at the cell `(1, 0)`. -/
def signLabel (s : Fin 2 × Fin 2) : Bool := !(s.1 == 1 && s.2 == 0)

/-- The matrix `[[1 − i, 1 + i], [−(1 + i), 1 − i]]`, with the real parts of `signKernel`. -/
def signTwin : Matrix (Fin 2) (Fin 2) ℂ :=
  !![1 - Complex.I, 1 + Complex.I; -(1 + Complex.I), 1 - Complex.I]

/-- The kernel of signs shows its labels, with every cell seen once. -/
lemma signKernel_apply (s : Fin 2 × Fin 2) :
    signKernel (id s).1 (id s).2 = signCode (signLabel s) := by
  obtain ⟨i, j⟩ := s
  fin_cases i <;> fin_cases j <;> simp [signKernel, signLabel, signCode]

/-- The kernel of signs is flat: `Kᴴ K = 2`. -/
lemma isFlat_signKernel : IsFlat signKernel 2 := by
  refine Or.inl ?_
  ext i j
  fin_cases i <;> fin_cases j <;>
    simp [signKernel, Matrix.mul_apply, Fin.sum_univ_two] <;> norm_num

lemma nuclearNorm_signKernel : nuclearNorm signKernel = 2 * √2 := by
  simpa using isFlat_signKernel.nuclearNorm_eq two_ne_zero

lemma signTwin_re (i j : Fin 2) : (signTwin i j).re = (signKernel i j).re := by
  fin_cases i <;> fin_cases j <;> simp [signTwin, signKernel]

lemma signTwin_ne : signTwin ≠ signKernel := fun h => by
  have h1 := congrArg (fun M : Matrix (Fin 2) (Fin 2) ℂ => (M 0 0).im) h
  simp [signTwin, signKernel] at h1

/-- The second matrix has rank one: it is `(1, −i)ᵀ (1 − i, 1 + i)`. -/
lemma signTwin_eq_vecMulVec :
    signTwin = vecMulVec ![1, -Complex.I] ![1 - Complex.I, 1 + Complex.I] := by
  ext i j
  fin_cases i <;> fin_cases j <;> simp [signTwin, vecMulVec_apply]
  · linear_combination -Complex.I_mul_I
  · linear_combination Complex.I_mul_I

lemma frobeniusSq_signTwin : frobeniusSq signTwin = 8 := by
  simp [frobeniusSq, Fin.sum_univ_two, signTwin, Complex.sq_norm, Complex.normSq_apply]
  norm_num

/-- The second matrix has at most the nuclear norm of the kernel of signs, by R3. -/
lemma nuclearNorm_signTwin_le : nuclearNorm signTwin ≤ nuclearNorm signKernel := by
  have h1 := nuclearNorm_sq_le_rank_mul_frobeniusSq signTwin
  have h2 : (signTwin.rank : ℝ) ≤ 1 := by
    rw [signTwin_eq_vecMulVec]
    exact_mod_cast rank_vecMulVec_le _ _
  rw [frobeniusSq_signTwin] at h1
  rw [nuclearNorm_signKernel]
  refine le_of_pow_le_pow_left₀ two_ne_zero (by positivity) ?_
  rw [mul_pow, Real.sq_sqrt zero_le_two]
  nlinarith

end Transformer.Grokking.FlatSpectrum.Training
