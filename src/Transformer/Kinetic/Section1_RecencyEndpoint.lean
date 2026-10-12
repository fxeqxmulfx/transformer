/-
# The endpoint interpretation of the kinetic recency claim

ArXiv:2605.09213v1, §1, `thm:U-shape` (ii), and its proof in §6.
The Bessel formula is stated for `σ₀ < σ`. Its totalized Lean expression
vanishes at `σ₀ = σ`, both because the graphon is zero on the diagonal
and because `Y=0`. That value is not its extension from the left.

The proof in the manuscript uses the power series `ψ_c`, with `ψ_c(0)=c`,
and computes the limiting slope through `Y(σ₀) → 0`. Consequently the
recency assertion is carried by the limit of the derivative on `(0,1)`,
as recorded in `Hardy.u_shape`, rather than the derivative of the
totalized formula at `1`. The latter assertion is refuted below: a
nonnegative function vanishing at its right endpoint cannot have a
positive left derivative there.
-/

import Transformer.Kinetic.Hardy
import Mathlib.Analysis.Calculus.Deriv.Slope

open Filter Set Real Topology

namespace Transformer.Kinetic

/-- The defining series of `I_n` is nonnegative at nonnegative arguments.
Source: arXiv:2605.09213v1, §1, `eq:Bessel-full` and the standard Bessel series. -/
theorem besselI_nonneg (n : ℕ) {x : ℝ} (hx : 0 ≤ x) : 0 ≤ besselI n x := by
  unfold besselI
  apply tsum_nonneg
  intro k
  exact div_nonneg (pow_nonneg (div_nonneg hx (by norm_num)) _)
    (mul_nonneg (by positivity) (by positivity))

/-- The directed graphon is nonnegative for nonnegative bias and output
position. Source: arXiv:2605.09213v1, §1, `eq:kernel-Klambda`. -/
theorem graphon_nonneg {lam σ : ℝ} (hlam : 0 ≤ lam) (hσ : 0 ≤ σ) (σ₀ : ℝ) :
    0 ≤ graphon lam σ σ₀ := by
  unfold graphon
  split_ifs with hpos hzero
  · exact inv_nonneg.mpr hσ
  · exact div_nonneg (mul_nonneg hlam (Real.exp_pos _).le)
      (sub_nonneg.mpr (Real.exp_le_one_iff.mpr (neg_nonpos.mpr (mul_nonneg hlam hσ))))
  · exact le_rfl

/-- The totalized Bessel profile is nonnegative. No sign assumption on
`a` or `t` is needed for this fact, since both square roots are nonnegative.
Source: arXiv:2605.09213v1, §1, `eq:Bessel-full`, `eq:Bessel-output`. -/
theorem hardyProfile_nonneg {lam σ : ℝ} (hlam : 0 ≤ lam) (hσ : 0 ≤ σ)
    (a t σ₀ : ℝ) : 0 ≤ hardyProfile lam a t σ σ₀ := by
  unfold hardyProfile
  exact mul_nonneg (mul_nonneg (graphon_nonneg hlam hσ σ₀) (Real.sqrt_nonneg _))
    (besselI_nonneg 1 (mul_nonneg (by norm_num) (Real.sqrt_nonneg _)))

/-- The literal soft-correction formula is nonnegative at every source
position, for nonnegative bias. Source: arXiv:2605.09213v1, §1, `eq:Sb-def`. -/
theorem softCorrection_nonneg (β M t σ₀ : ℝ) {lam : ℝ} (hlam : 0 ≤ lam) :
    0 ≤ softCorrection β lam M t σ₀ := by
  unfold softCorrection
  exact tsum_nonneg fun n => mul_nonneg (Real.exp_pos _).le
    (hardyProfile_nonneg hlam zero_le_one (aCoeff β (n + 1)) t σ₀)

/-- All sign hypotheses above are satisfied, including the zero-bias
branch of the graphon, at nonnegative argument `x=0` and output `σ=1`. -/
example : (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 ∧
    0 ≤ besselI 1 0 ∧ 0 ≤ graphon 0 1 (1 / 2) ∧
    0 ≤ hardyProfile 0 1 1 1 (1 / 2) ∧ 0 ≤ softCorrection 1 0 1 1 (1 / 2) :=
  ⟨le_rfl, zero_le_one, besselI_nonneg 1 le_rfl,
    graphon_nonneg le_rfl zero_le_one _, hardyProfile_nonneg le_rfl zero_le_one _ _ _,
    softCorrection_nonneg _ _ _ _ le_rfl⟩

/-- The totalized change of variables vanishes on the diagonal, including
both bias branches. Source: arXiv:2605.09213v1, §1, `eq:Ylambda` and
the `λ=0` limit in `eq:Bessel-output`. -/
theorem Yfun_diagonal (lam σ : ℝ) : Yfun lam σ σ = 0 := by
  have hlog (x : ℝ) : Real.log (x / x) = 0 := by
    by_cases hx : x = 0
    · rw [hx, zero_div, Real.log_zero]
    · rw [div_self hx, Real.log_one]
  unfold Yfun
  split_ifs <;> exact hlog _

/-- The literal Bessel profile vanishes on the diagonal because its
square-root prefactor is totalized to zero. Source: arXiv:2605.09213v1,
§1, `eq:Bessel-full`, whose stated domain excludes the diagonal. -/
theorem hardyProfile_diagonal_zero (lam a t σ : ℝ) : hardyProfile lam a t σ σ = 0 := by
  rw [hardyProfile, Yfun_diagonal, div_zero, Real.sqrt_zero, mul_zero, zero_mul]

/-- The formula on the diagonal is zero by its totalized Bessel profile.
Source: arXiv:2605.09213v1, §1, `eq:Sb-def` using the literal expression
`eq:Bessel-full` outside its stated domain `σ₀ < σ`. This is not a claim
about the continuous extension used in the source's proof of recency. -/
theorem softCorrection_one (β lam M t : ℝ) : softCorrection β lam M t 1 = 0 := by
  unfold softCorrection
  simp only [hardyProfile_diagonal_zero, mul_zero, tsum_zero]

/-- A nonnegative function that vanishes at a right endpoint has
nonpositive left derivative there. This is the obstruction to reading
`thm:U-shape` (ii) as an endpoint derivative of the totalized formula.
Source: arXiv:2605.09213v1, §1, `thm:U-shape` (ii). -/
theorem nonpos_left_derivative_of_nonneg {f : ℝ → ℝ} {b D : ℝ}
    (hzero : f b = 0) (hnonneg : ∀ x ≤ b, 0 ≤ f x)
    (hd : HasDerivWithinAt f D (Iic b) b) : D ≤ 0 := by
  have hlim := hasDerivWithinAt_iff_tendsto_slope.mp hd
  rw [Iic_sdiff_right] at hlim
  apply le_of_tendsto hlim
  filter_upwards [self_mem_nhdsWithin] with x hx
  rw [slope_def_field, hzero, sub_zero]
  exact div_nonpos_of_nonneg_of_nonpos (hnonneg x hx.le) (sub_nonpos.mpr hx.le)

/-- The scalar derivative hypotheses hold on the constant zero curve,
with a genuine left neighborhood at `b=1` and derivative `D=0`. -/
example : (fun _ : ℝ => (0 : ℝ)) 1 = 0 ∧
    (∀ x ≤ (1 : ℝ), 0 ≤ (fun _ : ℝ => (0 : ℝ)) x) ∧
    HasDerivWithinAt (fun _ : ℝ => (0 : ℝ)) 0 (Iic 1) 1 :=
  ⟨rfl, fun _ _ => le_rfl, (hasDerivAt_const 1 (0 : ℝ)).hasDerivWithinAt⟩

/-- **The earlier endpoint-derivative reading of recency is false.**
For every nonnegative `λ`, the totalized formula has no positive derivative
from the left at `σ₀=1`. The printed `S'_t(1⁻)>0` is therefore recorded as
the left limit of the interior derivative in `u_shape`, consistently with
the manuscript's proof using `ψ_c(0)=c`.
Source: arXiv:2605.09213v1, §1, `thm:U-shape` (ii), proof in §6. -/
theorem not_positive_endpoint_derivative (β M t : ℝ) {lam : ℝ} (hlam : 0 ≤ lam) :
    ¬ ∃ D : ℝ, 0 < D ∧
      HasDerivWithinAt (fun σ₀ => softCorrection β lam M t σ₀) D (Iic 1) 1 := by
  rintro ⟨D, hD, hd⟩
  have hnonpos := nonpos_left_derivative_of_nonneg (softCorrection_one β lam M t)
    (fun x _ => softCorrection_nonneg β M t x hlam) hd
  exact not_lt_of_ge hnonpos hD

/-- The refutation applies at the source's positive bias, `λ=1`. -/
example : (0 : ℝ) ≤ 1 := zero_le_one

/-- **The endpoint obstruction occurs under all numerical hypotheses of
the U-shape theorem.** At `β=λ=M=1`, a positive time satisfying both
smallness bounds exists, yet the positive derivative at the totalized
endpoint is impossible. This refutes that former Lean interpretation of
recency, while leaving the manuscript's interior limiting slope open.
Source: arXiv:2605.09213v1, §1, `thm:U-shape` (ii), `eq:affine-smallness`. -/
theorem exists_admissible_recency_endpoint_counterexample :
    ∃ t : ℝ, (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ 0 < t ∧
      (∀ n : ℕ, 1 ≤ n → t * aCoeff 1 n ≤
        min (3 - Real.sqrt 3) (2 * (1 - Real.exp (-(1 : ℝ))))) ∧
      ¬ ∃ D : ℝ, 0 < D ∧
        HasDerivWithinAt (fun σ₀ => softCorrection 1 1 1 t σ₀) D (Iic 1) 1 := by
  obtain ⟨t, ht, hsmall⟩ := exists_affine_smallness 1 1 one_pos
  exact ⟨t, one_pos, one_pos, one_pos, ht, hsmall,
    not_positive_endpoint_derivative 1 1 t zero_le_one⟩

end Transformer.Kinetic
