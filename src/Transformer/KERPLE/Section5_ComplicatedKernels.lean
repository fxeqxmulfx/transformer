module

public import Transformer.KERPLE.Section4_BiasKernel
public import Transformer.KERPLE.SectionA_PDFunctions

/-!
# §5.3: complicated kernels

Chi, Fan, Ramadge, Rudnicky, arXiv:2205.09921v2, §5.3 (Experiments on Complicated Kernels).

* The tensor product property: "if `k₁` is a kernel on `𝒳` and `k₂` is a kernel on `𝒴`, then
  `k((x, y), (x', y')) = k₁(x, x') k₂(y, y')` is a kernel on `𝒳 × 𝒴`", for PD kernels
  (`IsPDKernel.tensor`); so a PD weight `k_w` makes `q_mᵀk_n · k_w(m, n)` PD
  (`isPDKernel_weightKernel`).
* (bias+wht): `q_mᵀk_n · exp(−r₃|m − n|^{r₄}) + c − r₁|m − n|^{r₂}` with `r₁, r₃ > 0` and
  `0 < r₂, r₄ ≤ 2` (`biasWhtKernel`). Its weight `q_mᵀk_n · exp(−r₃|m − n|^{r₄})` is PD
  (`isPDKernel_weightKernel_gauss`) and the whole kernel is CPD (`isCPDKernel_biasWhtKernel`), but
  the "bias kernel `c − r₁|m − n|^{r₂}`" is PD for no `c` (`not_isPDKernel_power_bias`), and so is
  the composite kernel, already at `q = 0` on the positions `ℕ` (`not_isPDKernel_biasWhtKernel`).
* (3-para-log): `q_mᵀk_n + c − r₁ log(1 + r₂|m − n|^{r₃})` with `r₁, r₂ > 0` and `0 < r₃ ≤ 2`
  (`log3Kernel`): CPD by Corollary 1(b) (`isCPDKernel_compositeKernel_log3`); "when `r₃ = 1`, it is
  reduced to the logarithmic variant" (`log3Kernel_one`); its bias kernel is PD for no `c` on the
  positions `ℕ` (`not_isPDKernel_log3_bias`).
-/

@[expose] public section

namespace Transformer.KERPLE

open Filter

variable {X E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- **The tensor product property** (§5.3): if `k₁` is PD on `X` and `k₂` is PD on `Y`, then
`k((x, y), (x', y')) = k₁(x, x') k₂(y, y')` is PD on `X × Y`. -/
theorem IsPDKernel.tensor {Y : Type*} {k₁ : X → X → ℝ} {k₂ : Y → Y → ℝ} (h₁ : IsPDKernel k₁)
    (h₂ : IsPDKernel k₂) : IsPDKernel fun a b : X × Y => k₁ a.1 b.1 * k₂ a.2 b.2 :=
  (h₁.comp Prod.fst).mul (h₂.comp Prod.snd)

/-- The hypotheses of `IsPDKernel.tensor` hold for the inner product and the Gaussian kernel. -/
example : IsPDKernel fun a b : EuclideanSpace ℝ (Fin 2) × ℝ =>
    inner ℝ a.1 b.1 * Real.exp (-|a.2 - b.2| ^ (2 : ℝ)) :=
  isPDKernel_inner.tensor (isPDFunction_exp_neg_rpow two_pos le_rfl)

/-- The weight composite kernel `k^comp([q, m], [k, n]) = qᵀk · k_w(m, n)` of §5.3 and
Appendix A.3: the kernel `k_w` weights the inner product instead of adding to it (Eq. 4). -/
def weightKernel (kw : X → X → ℝ) (a b : E × X) : ℝ := inner ℝ a.1 b.1 * kw a.2 b.2

/-- A PD weight kernel `k_w` makes `qᵀk · k_w(m, n)` PD, by the tensor product property. -/
theorem isPDKernel_weightKernel {kw : X → X → ℝ} (h : IsPDKernel kw) :
    IsPDKernel (weightKernel (E := E) kw) :=
  isPDKernel_inner.tensor h

/-- The Gaussian-like kernel `exp(−r|m − n|^p)` of (bias+wht) (§5.3) and of Appendix A.3. -/
noncomputable def gaussKernel (r p x y : ℝ) : ℝ := Real.exp (-r * |x - y| ^ p)

/-- `exp(−r|m − n|^p)` is PD for `r ≥ 0` and `0 < p ≤ 2`: Fact 5. -/
theorem isPDKernel_gaussKernel {r p : ℝ} (hr : 0 ≤ r) (hp : 0 < p) (hp2 : p ≤ 2) :
    IsPDKernel (gaussKernel r p) :=
  isPDFunction_exp_neg_mul_rpow hr hp hp2

/-- **The weight `q_mᵀk_n · exp(−r|m − n|^p)` is PD** for `r ≥ 0` and `0 < p ≤ 2`: the weight of
(bias+wht) (§5.3), and (1-para-wht) and (2-para-wht) (Appendix A.3). -/
theorem isPDKernel_weightKernel_gauss {r p : ℝ} (hr : 0 ≤ r) (hp : 0 < p) (hp2 : p ≤ 2) :
    IsPDKernel (weightKernel (E := E) (gaussKernel r p)) :=
  isPDKernel_weightKernel (isPDKernel_gaussKernel hr hp hp2)

/-- The hypotheses of `isPDKernel_weightKernel_gauss` hold for (1-para-wht) at `r₁ = 1`. -/
example : IsPDKernel (weightKernel (E := EuclideanSpace ℝ (Fin 2)) (gaussKernel 1 2)) :=
  isPDKernel_weightKernel_gauss zero_le_one two_pos le_rfl

/-- The (bias+wht) composite kernel of §5.3:
`q_mᵀk_n · exp(−r₃|m − n|^{r₄}) + c − r₁|m − n|^{r₂}`. -/
noncomputable def biasWhtKernel (c r₁ r₂ r₃ r₄ : ℝ) (a b : E × ℝ) : ℝ :=
  weightKernel (gaussKernel r₃ r₄) a b + biasKernel c (powerKernel r₁ r₂) a.2 b.2

/-- **(bias+wht) is CPD** for `r₁, r₃ > 0` and `0 < r₂, r₄ ≤ 2`: a PD weight plus the CPD bias
`c − r₁|m − n|^{r₂}`. -/
theorem isCPDKernel_biasWhtKernel {r₁ r₂ r₃ r₄ : ℝ} (hr₁ : 0 < r₁) (hr₂ : 0 < r₂) (hr₂2 : r₂ ≤ 2)
    (hr₃ : 0 < r₃) (hr₄ : 0 < r₄) (hr₄2 : r₄ ≤ 2) (c : ℝ) :
    IsCPDKernel (biasWhtKernel (E := E) c r₁ r₂ r₃ r₄) :=
  (isPDKernel_weightKernel_gauss hr₃.le hr₄ hr₄2).isCPDKernel.add
    (((isCPDKernel_powerKernel hr₁ hr₂ hr₂2).add_const c).comp Prod.snd)

/-- The hypotheses of `isCPDKernel_biasWhtKernel` hold for ALiBi's bias and a Gaussian weight. -/
example : IsCPDKernel (biasWhtKernel (E := EuclideanSpace ℝ (Fin 2)) 0 1 1 1 2) :=
  isCPDKernel_biasWhtKernel one_pos one_pos one_le_two one_pos two_pos le_rfl 0

/-- **(bias+wht) is PD for no `c`**, for `r₁, r₂ > 0`: at `q = 0` on the positions `ℕ` it is the
bias kernel `c − r₁|m − n|^{r₂}`, PD for no `c` (`not_isPDKernel_power_bias`). -/
theorem not_isPDKernel_biasWhtKernel {r₁ r₂ : ℝ} (hr₁ : 0 < r₁) (hr₂ : 0 < r₂) (c r₃ r₄ : ℝ) :
    ¬IsPDKernel fun m n : ℕ => biasWhtKernel (E := E) c r₁ r₂ r₃ r₄ (0, (m : ℝ)) (0, (n : ℝ)) :=
  fun h => not_isPDKernel_power_bias hr₁ hr₂ c (h.congr fun m n => by
    simp [biasWhtKernel, weightKernel])

/-- The hypotheses of `not_isPDKernel_biasWhtKernel` hold for ALiBi's bias, `r₁ = r₂ = 1`. -/
example : ¬IsPDKernel fun m n : ℕ =>
    biasWhtKernel (E := EuclideanSpace ℝ (Fin 2)) 100 1 1 1 2 (0, (m : ℝ)) (0, (n : ℝ)) :=
  not_isPDKernel_biasWhtKernel one_pos one_pos 100 1 2

/-- A CPD bias kernel makes the composite kernel `qᵀk + k(m, n)` of Eq. 4 CPD. -/
theorem isCPDKernel_compositeKernel {kb : X → X → ℝ} (h : IsCPDKernel kb) :
    IsCPDKernel (compositeKernel (E := E) kb) :=
  (isPDKernel_inner.comp Prod.fst).isCPDKernel.add (h.comp Prod.snd)

/-- The kernel `k̃(m, n) = −r₁ log(1 + r₂|m − n|^{r₃})` of (3-para-log) (§5.3). -/
noncomputable def log3Kernel (r₁ r₂ r₃ x y : ℝ) : ℝ := -r₁ * Real.log (1 + r₂ * |x - y| ^ r₃)

/-- **(3-para-log) reduces to the logarithmic variant at `r₃ = 1`** (§5.3). -/
theorem log3Kernel_one (r₁ r₂ : ℝ) : log3Kernel r₁ r₂ 1 = logKernel r₁ r₂ := by
  funext x y
  simp only [log3Kernel, logKernel, Real.rpow_one]

/-- `−r₁ log(1 + r₂|m − n|^{r₃})` is CPD for `r₁, r₂ > 0` and `0 < r₃ ≤ 2`: Corollary 1(b). -/
theorem isCPDKernel_log3Kernel {r₁ r₂ r₃ : ℝ} (hr₁ : 0 < r₁) (hr₂ : 0 < r₂) (hr₃ : 0 < r₃)
    (hr₃2 : r₃ ≤ 2) : IsCPDKernel (log3Kernel r₁ r₂ r₃) :=
  (isCPDKernel_neg_mul_log_one_add (E := ℝ) hr₂.le hr₁.le hr₃ hr₃2).congr fun x y => by
    simp only [log3Kernel, Real.norm_eq_abs]

/-- **(3-para-log) is CPD**: `q_mᵀk_n + c − r₁ log(1 + r₂|m − n|^{r₃})` for `r₁, r₂ > 0` and
`0 < r₃ ≤ 2`. -/
theorem isCPDKernel_compositeKernel_log3 {r₁ r₂ r₃ : ℝ} (hr₁ : 0 < r₁) (hr₂ : 0 < r₂)
    (hr₃ : 0 < r₃) (hr₃2 : r₃ ≤ 2) (c : ℝ) :
    IsCPDKernel (compositeKernel (E := E) (biasKernel c (log3Kernel r₁ r₂ r₃))) :=
  isCPDKernel_compositeKernel ((isCPDKernel_log3Kernel hr₁ hr₂ hr₃ hr₃2).add_const c)

/-- The hypotheses of `isCPDKernel_compositeKernel_log3` hold at `r₃ = 2`. -/
example : IsCPDKernel
    (compositeKernel (E := EuclideanSpace ℝ (Fin 2)) (biasKernel 0 (log3Kernel 1 1 2))) :=
  isCPDKernel_compositeKernel_log3 one_pos one_pos two_pos le_rfl 0

/-- **The bias kernel of (3-para-log) is PD for no `c`** on the positions `ℕ`, for `r₁, r₂, r₃ > 0`:
`r₁ log(1 + r₂ N^{r₃})` exceeds every `2c`. -/
theorem not_isPDKernel_log3_bias {r₁ r₂ r₃ : ℝ} (hr₁ : 0 < r₁) (hr₂ : 0 < r₂) (hr₃ : 0 < r₃)
    (c : ℝ) : ¬IsPDKernel fun m n : ℕ => biasKernel c (log3Kernel r₁ r₂ r₃) m n := by
  intro h
  have hlim : Tendsto (fun N : ℕ => r₁ * Real.log (1 + r₂ * (N : ℝ) ^ r₃)) atTop atTop :=
    (Real.tendsto_log_atTop.comp (tendsto_atTop_add_const_left _ 1
      (((tendsto_rpow_atTop hr₃).comp tendsto_natCast_atTop_atTop).const_mul_atTop
        hr₂))).const_mul_atTop hr₁
  obtain ⟨N, hN⟩ := (hlim.eventually_gt_atTop (2 * c)).exists
  have := IsPDKernel.neg_two_mul_le (k := fun m n : ℕ => log3Kernel r₁ r₂ r₃ m n) h
    (fun x => by simp [log3Kernel, hr₃.ne']) N 0
  simp only [log3Kernel, Nat.cast_zero, sub_zero, Nat.abs_cast] at this
  linarith

/-- The hypotheses of `not_isPDKernel_log3_bias` hold for `r₁ = r₂ = r₃ = 1`. -/
example : ¬IsPDKernel fun m n : ℕ => biasKernel 5 (log3Kernel 1 1 1) m n :=
  not_isPDKernel_log3_bias one_pos one_pos one_pos 5

end Transformer.KERPLE
