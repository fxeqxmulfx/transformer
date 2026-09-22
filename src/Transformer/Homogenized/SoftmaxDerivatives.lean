/-
# Homogenized Transformers — derivatives of the softmax

Formalization of `claim:softmax_is_smooth` of arXiv:2604.01978v1, *Homogenized
Transformers*: the partial derivatives of the softmax `χ` are polynomials in
`χ` itself, hence bounded uniformly in the scores and in the number of them.
This is what the source combines with `eq:faa_di_bruno` to bound the
derivatives of `π^A_{i→j}` in `eq:pi_derivative_bound`, and thence `G(·,θ)` in
`lem:Kernel_regularity`.

The softmax is `Perspective.softmaxWeight` of arXiv:2312.10794v5; there is one
softmax in this tree, not one per paper.

**What the source says and what is carried here.**

* The displayed formula of `claim:softmax_is_smooth` is not a statement: the
  multi-index `r_1, …, r_k` is free on the left-hand side and summed over on
  the right, and the `r_q'` of the indicator `1_{r_q = r_{q'} ∀ (q,q') ∈ I}` is
  bound by nothing.  What the claim is *for* — the only thing the source ever
  uses it for, two displays later in `eq:pi_derivative_bound` — is that each
  partial derivative of `χ_j` is `χ_j` times a bounded function of the scores,
  with a bound depending on the order alone: `|∂^k χ_j| ≤ C_k χ_j`.  That is
  what `softmax_is_smooth` states, and proves, in `SoftmaxSmooth.lean`.

* The induction the source proves the claim by,

    `∂χ_j/∂z_r = χ_j χ_r - 1_{r=j} χ_j`,

  has the sign the wrong way round: differentiating `e^{z_j}/Σ_k e^{z_k}` gives
  `1_{r=j} χ_j - χ_j χ_r`, which is positive at `r = j` — as it must be, the
  weight of a score increasing with that score.  `fderiv_softmaxWeight` states
  and proves the corrected identity.  Nothing downstream turns on the sign: the
  claim's constants `C^k_I` are left unspecified by the source.

Source: arXiv:2604.01978v1, `claim:softmax_is_smooth`, `eq:pi_derivative_bound`.
-/

import Transformer.Homogenized.Basic
import Transformer.Perspective.Softmax
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

open scoped BigOperators
open Real

namespace Transformer
namespace Homogenized

variable {m : ℕ}

/-- **The differential of a softmax weight.**  Writing `χ = σ(z)` for the
weights, `σ(·)_j` is differentiable at every `z` with

  `Dσ(z)_j = χ_j (e_j^* - Σ_k χ_k e_k^*)`,

`e_k^*` the `k`-th coordinate functional.  Evaluating at `e_r` gives the
partial derivatives; see `fderiv_softmaxWeight`.

Source: arXiv:2604.01978v1, `claim:softmax_is_smooth` (corrected). -/
theorem hasFDerivAt_softmaxWeight (hm : 0 < m) (u : Idx m → ℝ) (j : Idx m) :
    HasFDerivAt (fun v : Idx m → ℝ => Perspective.softmaxWeight v j)
      (Perspective.softmaxWeight u j •
        ((ContinuousLinearMap.proj j : (Idx m → ℝ) →L[ℝ] ℝ) -
          ∑ k : Idx m, Perspective.softmaxWeight u k •
            (ContinuousLinearMap.proj k : (Idx m → ℝ) →L[ℝ] ℝ))) u := by
  have hZ : (0 : ℝ) < ∑ k : Idx m, Real.exp (u k) := Perspective.softmaxPartition_pos hm u
  have hnum : HasFDerivAt (fun v : Idx m → ℝ => Real.exp (v j))
      (Real.exp (u j) • (ContinuousLinearMap.proj j : (Idx m → ℝ) →L[ℝ] ℝ)) u :=
    HasFDerivAt.exp (ContinuousLinearMap.hasFDerivAt
      (f := (ContinuousLinearMap.proj j : (Idx m → ℝ) →L[ℝ] ℝ)) (x := u))
  have hden : HasFDerivAt (fun v : Idx m → ℝ => ∑ k : Idx m, Real.exp (v k))
      (∑ k : Idx m, Real.exp (u k) • (ContinuousLinearMap.proj k : (Idx m → ℝ) →L[ℝ] ℝ)) u :=
    HasFDerivAt.fun_sum fun k _ =>
      HasFDerivAt.exp (ContinuousLinearMap.hasFDerivAt
        (f := (ContinuousLinearMap.proj k : (Idx m → ℝ) →L[ℝ] ℝ)) (x := u))
  have hinv : HasFDerivAt (fun v : Idx m → ℝ => (∑ k : Idx m, Real.exp (v k))⁻¹)
      ((-((∑ k : Idx m, Real.exp (u k)) ^ 2)⁻¹) •
        ∑ k : Idx m, Real.exp (u k) • (ContinuousLinearMap.proj k : (Idx m → ℝ) →L[ℝ] ℝ)) u :=
    (hasDerivAt_inv hZ.ne').comp_hasFDerivAt u hden
  have hmul := hnum.fun_mul hinv
  simp only [Perspective.softmaxWeight, div_eq_mul_inv]
  convert hmul using 1
  ext w
  have hs : ∑ k : Idx m, Real.exp (u k) * (∑ l : Idx m, Real.exp (u l))⁻¹ * w k
      = (∑ k : Idx m, Real.exp (u k) * w k) * (∑ l : Idx m, Real.exp (u l))⁻¹ := by
    rw [Finset.sum_mul]
    exact Finset.sum_congr rfl fun k _ => by ring
  simp only [smul_apply, sub_apply, add_apply, FunLike.coe_sum, Finset.sum_apply,
    ContinuousLinearMap.proj_apply, smul_eq_mul, mul_sub, hs]
  field_simp
  ring

/-- The hypothesis of `hasFDerivAt_softmaxWeight` is satisfiable. -/
example : 0 < 1 := one_pos

/-- **The partial derivatives of the softmax**, the identity the source runs
its induction on:

  `∂χ_j/∂z_r (z) = 1_{r=j} χ_j(z) - χ_j(z) χ_r(z)`.

The source displays `χ_j χ_r - 1_{r=j} χ_j`, with the signs exchanged; see the
module docstring.

Source: arXiv:2604.01978v1, `claim:softmax_is_smooth` (corrected). -/
theorem fderiv_softmaxWeight (hm : 0 < m) (u : Idx m → ℝ) (j r : Idx m) :
    fderiv ℝ (fun v : Idx m → ℝ => Perspective.softmaxWeight v j) u (Pi.single r 1)
      = (if r = j then Perspective.softmaxWeight u j else 0)
        - Perspective.softmaxWeight u j * Perspective.softmaxWeight u r := by
  rw [(hasFDerivAt_softmaxWeight hm u j).fderiv]
  have hproj : ∀ k : Idx m,
      (ContinuousLinearMap.proj k : (Idx m → ℝ) →L[ℝ] ℝ) (Pi.single r (1 : ℝ))
        = if k = r then (1 : ℝ) else 0 := by
    intro k
    simp [Pi.single_apply, eq_comm]
  simp only [smul_apply, sub_apply, FunLike.coe_sum, Finset.sum_apply, smul_eq_mul, hproj,
    mul_ite, mul_one, mul_zero, Finset.sum_ite_eq', Finset.mem_univ, ite_true]
  rcases eq_or_ne r j with rfl | hrj
  · rw [ite_eq_left rfl, ite_eq_left rfl]; ring
  · rw [ite_eq_right hrj, ite_eq_right (Ne.symm hrj)]; ring

/-- The hypothesis of `fderiv_softmaxWeight` is satisfiable. -/
example : 0 < 1 := one_pos

/-- A softmax weight is at most `1`. -/
theorem softmaxWeight_le_one (hm : 0 < m) (u : Idx m → ℝ) (j : Idx m) :
    Perspective.softmaxWeight u j ≤ 1 := by
  rw [← Perspective.sum_softmaxWeight hm u]
  exact Finset.single_le_sum (fun k _ => Perspective.softmaxWeight_nonneg u k)
    (Finset.mem_univ j)

/-- The hypothesis of `softmaxWeight_le_one` is satisfiable. -/
example : 0 < 1 := one_pos

/-- A softmax weight is a smooth function of the scores. -/
theorem contDiff_softmaxWeight (hm : 0 < m) (j : Idx m) :
    ContDiff ℝ (⊤ : ℕ∞) (fun v : Idx m → ℝ => Perspective.softmaxWeight v j) := by
  simp only [Perspective.softmaxWeight]
  exact ContDiff.div (by fun_prop) (by fun_prop)
    fun v => (Perspective.softmaxPartition_pos hm v).ne'

/-- The hypothesis of `contDiff_softmaxWeight` is satisfiable. -/
example : 0 < 1 := one_pos

/-- The monomial `χ_{a_1} ⋯ χ_{a_p}` in the softmax weights. -/
noncomputable def softmaxMonomial {p : ℕ} (a : Fin p → Idx m) (v : Idx m → ℝ) : ℝ :=
  ∏ i, Perspective.softmaxWeight v (a i)

/-- A monomial in the softmax weights is smooth. -/
theorem contDiff_softmaxMonomial (hm : 0 < m) {p : ℕ} (a : Fin p → Idx m) :
    ContDiff ℝ (⊤ : ℕ∞) (softmaxMonomial a) :=
  contDiff_prod fun i _ => contDiff_softmaxWeight hm (a i)

/-- The hypothesis of `contDiff_softmaxMonomial` is satisfiable. -/
example : 0 < 1 := one_pos

/-- A monomial in the softmax weights lies in `[0,1]`. -/
theorem abs_softmaxMonomial_le_one (hm : 0 < m) {p : ℕ} (a : Fin p → Idx m)
    (v : Idx m → ℝ) : |softmaxMonomial a v| ≤ 1 := by
  rw [softmaxMonomial,
    abs_of_nonneg (Finset.prod_nonneg fun i _ => Perspective.softmaxWeight_nonneg v (a i))]
  exact Finset.prod_le_one₀ (fun i _ => Perspective.softmaxWeight_nonneg v (a i))
    fun i _ => softmaxWeight_le_one hm v (a i)

/-- The hypothesis of `abs_softmaxMonomial_le_one` is satisfiable. -/
example : 0 < 1 := one_pos

end Homogenized
end Transformer
