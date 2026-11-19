module

public import Transformer.KERPLE.SectionA_DiagonalDominance
public import Transformer.KERPLE.SectionA_GaussianKernels

/-!
# Appendix A.3: Gaussian-like kernels on integer positions

Chi, Fan, Ramadge, Rudnicky, arXiv:2205.09921v2, Appendix A.3: "Note we must constrain the power
in (0,2]; otherwise, the function is not positive definite. See Fact 5 for details." With positions
in `ℝ` this holds (`not_isPDKernel_gaussBias`, `not_isPDKernel_weightKernel_gauss`). Attention
evaluates the kernels at the integer positions `m, n ∈ ℕ`, and there the constraint is not needed
at scales `r ≥ 1`: for every `p ≥ 2`, `exp(−r|m − n|^p)` is PD on `ℕ`
(`isPDKernel_gaussKernel_nat`, by diagonal dominance), and so are (3-para-bias) and (2-para-wht)
(`isPDKernel_gaussBias_nat`, `isPDKernel_weightKernel_gauss_nat`). Fact 5's `exp(−|x|³)` is PD
on `ℕ` and not on `ℝ` (`gaussKernel_three_real_nat`).
-/

@[expose] public section

namespace Transformer.KERPLE

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- **On the positions `ℕ`, `exp(−r|m − n|^p)` is PD for `r ≥ 1` and every `p ≥ 2`**, by diagonal
dominance; on `ℝ` it is not PD for `p > 2` (Fact 5). -/
theorem isPDKernel_gaussKernel_nat {r p : ℝ} (hr : 1 ≤ r) (hp : 2 ≤ p) :
    IsPDKernel fun m n : ℕ => gaussKernel r p m n :=
  isPDKernel_of_sum_erase_le
    (fun m n => by
      simp only [gaussKernel]
      rw [abs_sub_comm])
    (fun m n => (Real.exp_pos _).le)
    (fun m => by simp [gaussKernel, (by linarith : (0 : ℝ) < p).ne'])
    fun s m _ => sum_erase_gaussKernel_le hr hp s m

/-- The hypotheses of `isPDKernel_gaussKernel_nat` hold for the power `4`. -/
example : IsPDKernel fun m n : ℕ => gaussKernel 1 4 m n :=
  isPDKernel_gaussKernel_nat le_rfl (by norm_num)

/-- **(3-para-bias) is PD on the positions `ℕ`** for `r₁ ≥ 0`, `r₂ ≥ 1` and every power `r₃ ≥ 2`,
against "we must constrain the power in (0,2]; otherwise, the function is not positive
definite". -/
theorem isPDKernel_gaussBias_nat {r₁ r₂ r₃ : ℝ} (hr₁ : 0 ≤ r₁) (hr₂ : 1 ≤ r₂) (hr₃ : 2 ≤ r₃) :
    IsPDKernel (compositeKernel (E := E) fun m n : ℕ => r₁ * gaussKernel r₂ r₃ m n) :=
  (isPDKernel_compositeKernel_iff _).mpr ((isPDKernel_gaussKernel_nat hr₂ hr₃).smul hr₁)

/-- The hypotheses of `isPDKernel_gaussBias_nat` hold for the power `3`. -/
example : IsPDKernel (compositeKernel (E := EuclideanSpace ℝ (Fin 2))
    fun m n : ℕ => 1 * gaussKernel 1 3 m n) :=
  isPDKernel_gaussBias_nat zero_le_one le_rfl (by norm_num)

/-- **(2-para-wht) is PD on the positions `ℕ`** for `r ≥ 1` and every power `p ≥ 2`, against "we
must constrain the power in (0,2]". -/
theorem isPDKernel_weightKernel_gauss_nat {r p : ℝ} (hr : 1 ≤ r) (hp : 2 ≤ p) :
    IsPDKernel (weightKernel (E := E) fun m n : ℕ => gaussKernel r p m n) :=
  isPDKernel_weightKernel (isPDKernel_gaussKernel_nat hr hp)

/-- The hypotheses of `isPDKernel_weightKernel_gauss_nat` hold for the power `3`. -/
example : IsPDKernel (weightKernel (E := EuclideanSpace ℝ (Fin 2))
    fun m n : ℕ => gaussKernel 1 3 m n) :=
  isPDKernel_weightKernel_gauss_nat le_rfl (by norm_num)

/-- **The power constraint is needed on `ℝ` and not on `ℕ`**: `exp(−|x − y|³)` is not PD on `ℝ`
(Fact 5), and it is PD on the positions `ℕ`. -/
theorem gaussKernel_three_real_nat :
    ¬IsPDKernel (gaussKernel 1 3) ∧ IsPDKernel fun m n : ℕ => gaussKernel 1 3 m n :=
  ⟨not_isPDFunction_exp_neg_mul_rpow one_pos (by norm_num),
    isPDKernel_gaussKernel_nat le_rfl (by norm_num)⟩

end Transformer.KERPLE
