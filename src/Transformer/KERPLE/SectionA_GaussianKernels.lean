module

public import Transformer.KERPLE.Section5_ComplicatedKernels

/-!
# Appendix A.3: Gaussian-like kernels

Chi, Fan, Ramadge, Rudnicky, arXiv:2205.09921v2, Appendix A.3 (Experiments on Gaussian-like
Kernels): four composite kernels put the Gaussian-like kernel `exp(−r|m − n|^p)` (`gaussKernel`)
in the bias or in the weight,

* (2-para-bias) `q_mᵀk_n + r₁ exp(−r₂|m − n|²)` and (3-para-bias)
  `q_mᵀk_n + r₁ exp(−r₂|m − n|^{r₃})`, with `r₁, r₂ > 0` and `0 < r₃ ≤ 2`;
* (1-para-wht) `q_mᵀk_n · exp(−r₁|m − n|²)` and (2-para-wht) `q_mᵀk_n · exp(−r₁|m − n|^{r₂})`,
  with `r₁ > 0` and `0 < r₂ ≤ 2` (`weightKernel`).

"Note we must constrain the power in (0,2]; otherwise, the function is not positive definite. See
Fact 5 for details." With positions in `ℝ`, all four are PD for powers in `(0, 2]`
(`isPDKernel_gaussBias`, `isPDKernel_weightKernel_gauss`), and for powers `> 2` the bias
versions are not PD (`not_isPDKernel_gaussBias`), nor are the weight versions on a nontrivial
query space (`not_isPDKernel_weightKernel_gauss`). The kernels are bounded, with values in
`(0, 1]` (`exp_neg_mul_rpow_mem_Ioc`).
-/

@[expose] public section

namespace Transformer.KERPLE

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- **(3-para-bias) is PD** with positions in `ℝ`, for `r₁, r₂ > 0` and `0 < r₃ ≤ 2`: its bias
`r₁ exp(−r₂|m − n|^{r₃})` is PD by Fact 5. (2-para-bias) is the case `r₃ = 2`. -/
theorem isPDKernel_gaussBias {r₁ r₂ r₃ : ℝ} (hr₁ : 0 < r₁) (hr₂ : 0 < r₂) (hr₃ : 0 < r₃)
    (hr₃2 : r₃ ≤ 2) :
    IsPDKernel (compositeKernel (E := E) fun x y : ℝ => r₁ * gaussKernel r₂ r₃ x y) :=
  (isPDKernel_compositeKernel_iff _).mpr ((isPDKernel_gaussKernel hr₂.le hr₃ hr₃2).smul hr₁.le)

/-- The hypotheses of `isPDKernel_gaussBias` hold for (2-para-bias), `r₃ = 2`. -/
example : IsPDKernel (compositeKernel (E := EuclideanSpace ℝ (Fin 2))
    fun x y : ℝ => 1 * gaussKernel 1 2 x y) :=
  isPDKernel_gaussBias one_pos one_pos two_pos le_rfl

/-- **For a power `r₃ > 2`, (3-para-bias) is not PD** with positions in `ℝ`, for `r₁, r₂ > 0`:
the composite kernel is PD only if its bias is, and `exp(−r₂|x|^{r₃})` is not PD by Fact 5. -/
theorem not_isPDKernel_gaussBias {r₁ r₂ r₃ : ℝ} (hr₁ : 0 < r₁) (hr₂ : 0 < r₂) (hr₃ : 2 < r₃) :
    ¬IsPDKernel (compositeKernel (E := E) fun x y : ℝ => r₁ * gaussKernel r₂ r₃ x y) := fun h =>
  not_isPDFunction_exp_neg_mul_rpow hr₂ hr₃ <|
    (((isPDKernel_compositeKernel_iff _).mp h).smul (inv_nonneg.mpr hr₁.le)).congr fun x y => by
      rw [inv_mul_cancel_left₀ hr₁.ne']
      rfl

/-- The hypotheses of `not_isPDKernel_gaussBias` hold for the power `3`. -/
example : ¬IsPDKernel (compositeKernel (E := EuclideanSpace ℝ (Fin 2))
    fun x y : ℝ => 1 * gaussKernel 1 3 x y) :=
  not_isPDKernel_gaussBias one_pos one_pos (by norm_num)

/-- **For a power `p > 2`, (2-para-wht) is not PD** with positions in `ℝ`, for `r > 0` and a
nontrivial query space: at `q = k = e ≠ 0` it is `‖e‖² exp(−r|m − n|^p)`, not PD by Fact 5. -/
theorem not_isPDKernel_weightKernel_gauss [Nontrivial E] {r p : ℝ} (hr : 0 < r) (hp : 2 < p) :
    ¬IsPDKernel (weightKernel (E := E) (gaussKernel r p)) := fun h => by
  obtain ⟨e, he⟩ := exists_ne (0 : E)
  have hee : 0 < inner ℝ e e := real_inner_self_pos.mpr he
  refine not_isPDFunction_exp_neg_mul_rpow hr hp
    (((h.comp fun x : ℝ => (e, x)).smul (inv_nonneg.mpr hee.le)).congr fun x y => ?_)
  simp only [weightKernel]
  rw [inv_mul_cancel_left₀ hee.ne']
  rfl

/-- The hypotheses of `not_isPDKernel_weightKernel_gauss` hold for the power `3` on `ℝ²`. -/
example : ¬IsPDKernel (weightKernel (E := EuclideanSpace ℝ (Fin 2)) (gaussKernel 1 3)) :=
  not_isPDKernel_weightKernel_gauss one_pos (by norm_num)

end Transformer.KERPLE
