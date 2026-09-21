/-
# Mean-Field Dynamics — two explicit trajectories on a great circle (§5 of 2512.01868v4)

For orthonormal `e, f`, two closed-form solutions on the circle they span:

* `hasDerivAt_massless` — `x(t) = (cosh t)⁻¹ (sinh t · e + f)` solves
  `ẋ = Proj_x e`: a particle attending only to a fixed `e`, which is what
  `eq: SA` does to a cluster of mass `0` next to a cluster of mass `1`;
* `hasDerivAt_pairPath` — `y_ī(s) = (cosh 2s)^{-1/2}(sinh s · e + cosh s · f)`,
  `y_j̄(s) = (cosh 2s)^{-1/2}(sinh s · f + cosh s · e)` solve the pairing
  dynamics `ẏ_ī = Proj_{y_ī} y_j̄`, `ẏ_j̄ = Proj_{y_j̄} y_ī` of
  `thm: agazzi_merge`, with `⟨y_ī, y_j̄⟩ = tanh 2s`.

They are the witnesses of `MeanField.not_agazzi_merge`.
-/

import Transformer.MeanField.Merging
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Analysis.Complex.ExponentialBounds

open Real

namespace Transformer
namespace MeanField

variable (d : ℕ)

/-- Inner products in an orthonormal pair `e, f`. -/
theorem inner_frame (e f : EucSpace d) (he : ‖e‖ = 1) (hf : ‖f‖ = 1)
    (hef : inner (𝕜 := ℝ) e f = 0) (a b a' b' : ℝ) :
    inner (𝕜 := ℝ) (a • e + b • f) (a' • e + b' • f) = a * a' + b * b' := by
  have hfe : inner (𝕜 := ℝ) f e = 0 := by rw [real_inner_comm]; exact hef
  simp only [inner_add_left, inner_add_right, real_inner_smul_left, real_inner_smul_right,
    real_inner_self_eq_norm_sq, he, hf, hef, hfe]
  ring

/-- Norms in an orthonormal pair `e, f`. -/
theorem norm_frame (e f : EucSpace d) (he : ‖e‖ = 1) (hf : ‖f‖ = 1)
    (hef : inner (𝕜 := ℝ) e f = 0) (c a b : ℝ) (h : c ^ 2 * (a ^ 2 + b ^ 2) = 1) :
    ‖c • (a • e + b • f)‖ = 1 := by
  have h2 : ‖c • (a • e + b • f)‖ ^ 2 = 1 := by
    rw [← real_inner_self_eq_norm_sq, real_inner_smul_left, real_inner_smul_right,
      inner_frame d e f he hf hef, ← h]
    ring
  have := norm_nonneg (c • (a • e + b • f))
  nlinarith [sq_nonneg (‖c • (a • e + b • f)‖ - 1)]

/-- **The massless particle.** `x(t) = (cosh t)⁻¹ (sinh t · e + f)` solves
`ẋ = Proj_x e`, from `x(0) = f` towards `e`.

Source: arXiv:2512.01868v4, §5, `eq: SA` at `μ = 0 · δ_x + 1 · δ_e`. -/
theorem hasDerivAt_massless (e f : EucSpace d) (he : ‖e‖ = 1)
    (hef : inner (𝕜 := ℝ) e f = 0) (t : ℝ) :
    HasDerivAt (fun t => (cosh t)⁻¹ • (sinh t • e + f))
      (proj d ((cosh t)⁻¹ • (sinh t • e + f)) e) t := by
  have hc : cosh t ≠ 0 := (cosh_pos t).ne'
  have h1 := ((hasDerivAt_cosh t).inv hc).smul
    (((hasDerivAt_sinh t).smul_const e).add_const f)
  convert h1 using 1
  have hfe : inner (𝕜 := ℝ) f e = 0 := by rw [real_inner_comm]; exact hef
  have hin : inner (𝕜 := ℝ) ((cosh t)⁻¹ • (sinh t • e + f)) e = (cosh t)⁻¹ * sinh t := by
    simp only [real_inner_smul_left, inner_add_left, real_inner_self_eq_norm_sq, he, hfe]
    ring
  rw [proj, hin]
  simp only [Pi.inv_apply]
  match_scalars <;> field_simp
  ring

/-- **The pairing dynamics in closed form.**  With `r(s) = (cosh 2s)^{-1/2}`,
`y_ī = r (sinh s · e + cosh s · f)` satisfies `ẏ_ī = Proj_{y_ī} y_j̄` for
`y_j̄ = r (cosh s · e + sinh s · f)`; the same with `e, f` exchanged gives the
equation of `y_j̄`.

Source: arXiv:2512.01868v4, §5, `thm: agazzi_merge`, the limit system. -/
theorem hasDerivAt_pairPath (e f : EucSpace d) (he : ‖e‖ = 1) (hf : ‖f‖ = 1)
    (hef : inner (𝕜 := ℝ) e f = 0) (s : ℝ) :
    HasDerivAt (fun s => (√(cosh (2 * s)))⁻¹ • (sinh s • e + cosh s • f))
      (proj d ((√(cosh (2 * s)))⁻¹ • (sinh s • e + cosh s • f))
        ((√(cosh (2 * s)))⁻¹ • (cosh s • e + sinh s • f))) s := by
  have hc : 0 < cosh (2 * s) := cosh_pos _
  have hr : √(cosh (2 * s)) ≠ 0 := (Real.sqrt_pos.2 hc).ne'
  have hch : HasDerivAt (fun s => cosh (2 * s)) (sinh (2 * s) * 2) s := by
    simpa [mul_comm] using ((hasDerivAt_id s).const_mul 2).cosh
  have h1 := ((hch.sqrt hc.ne').inv hr).smul
    (((hasDerivAt_sinh s).smul_const e).add ((hasDerivAt_cosh s).smul_const f))
  convert h1 using 1
  have hfe : inner (𝕜 := ℝ) f e = 0 := by rw [real_inner_comm]; exact hef
  have hin : inner (𝕜 := ℝ) ((√(cosh (2 * s)))⁻¹ • (sinh s • e + cosh s • f))
      ((√(cosh (2 * s)))⁻¹ • (cosh s • e + sinh s • f))
      = (√(cosh (2 * s)))⁻¹ * (√(cosh (2 * s)))⁻¹ * (2 * sinh s * cosh s) := by
    simp only [real_inner_smul_left, real_inner_smul_right, inner_add_left, inner_add_right,
      real_inner_self_eq_norm_sq, he, hf, hef, hfe]
    ring
  rw [proj, hin]
  have hsq : √(cosh (2 * s)) ^ 2 = cosh (2 * s) := Real.sq_sqrt hc.le
  simp only [Pi.inv_apply, Pi.add_apply, cosh_two_mul, sinh_two_mul] at hsq ⊢
  match_scalars <;> field_simp <;> rw [hsq] <;> ring

/-- `tanh 2s ≤ 1/2` for `s ≤ 1/4`, written without division:
`2 sinh s cosh s ≤ (cosh² s + sinh² s)/2`, since `e^{4s} ≤ e < 3`.  The bound
`⟨y_ī, y_j̄⟩ ≤ 1 - ε` of `thm: agazzi_merge` at `ε = 1/2`, `T = 1/4`.

Source: arXiv:2512.01868v4, §5, `thm: agazzi_merge`. -/
theorem two_sinh_mul_cosh_le (s : ℝ) (hs : s ≤ 1 / 4) :
    2 * sinh s * cosh s ≤ (cosh s ^ 2 + sinh s ^ 2) / 2 := by
  rw [sinh_eq, cosh_eq]
  have ha := exp_pos s
  have hab : exp s * exp (-s) = 1 := by rw [← exp_add, add_neg_cancel, exp_zero]
  have h4 : exp s ^ 4 ≤ 3 := by
    rw [← exp_nat_mul]
    calc exp ((4 : ℕ) * s) ≤ exp 1 := exp_le_exp.2 (by push_cast; linarith)
      _ ≤ 3 := exp_one_lt_three.le
  have hb : exp (-s) = (exp s)⁻¹ := exp_neg s
  rw [hb]
  field_simp
  nlinarith [sq_nonneg (exp s)]

/-- The hypotheses of this file are satisfiable: the standard basis of `ℝ²` is
orthonormal, `c = 1`, `a = 0`, `b = 1` has `c² (a² + b²) = 1`, and `0 ≤ 1/4`. -/
example :
    ‖(EuclideanSpace.single 0 1 : EucSpace 2)‖ = 1 ∧
      ‖(EuclideanSpace.single 1 1 : EucSpace 2)‖ = 1 ∧
      inner (𝕜 := ℝ) (EuclideanSpace.single 0 1 : EucSpace 2)
        (EuclideanSpace.single 1 1 : EucSpace 2) = 0 ∧
      (1 : ℝ) ^ 2 * (0 ^ 2 + 1 ^ 2) = 1 ∧ (0 : ℝ) ≤ 1 / 4 := by
  refine ⟨by simp, by simp, by simp [EuclideanSpace.inner_single_left], by norm_num, by norm_num⟩

end MeanField
end Transformer
