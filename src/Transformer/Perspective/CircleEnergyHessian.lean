/-
# Critical points and second derivatives of `𝖤_β` on the space of angles

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5, §7 and
Appendix B.

On the Euclidean space `EucSpace n` of angles, where `𝖤_β` is
`eucTorusEnergy` (`Perspective.CircleSAGradient`), the critical points of
`-𝖤_β` are those of `IsCriticalTorusEnergy`
(`gradient_neg_eucTorusEnergy_eq_zero_iff`), and `𝖤_β` is unchanged when
every angle turns by the same amount (`eucTorusEnergy_add_mem_span_one`).
Along every line, `𝖤_β` has the second derivative

  `d²/ds² 𝖤_β(θ + s v)|_{s=0} = -(1/(2n²)) Σ_i Σ_j (v_i - v_j)² g_β(θ_i - θ_j)`

(`secondDerivTorusEnergy_line`); `secondDeriv_torusEnergy_block` computes it
for the block directions `1_𝒮`.  A second derivative is unique
(`secondDerivTorusEnergy_unique`), and it is the quadratic form of the
derivative of the gradient (`inner_fderiv_gradient_eucTorusEnergy`, from
`inner_fderiv_gradient_eq_of_hasDerivAt`).
-/

import Transformer.Perspective.CircleSAGradient
import Transformer.Perspective.CircleCritical
import Transformer.SaddleAvoidance.GradientDescent
import Mathlib.Analysis.InnerProductSpace.Calculus

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

local notation "⟪" x ", " y "⟫" => inner ℝ x y

variable (n : ℕ)

/-- **The critical points of `-𝖤_β` on `EucSpace n` are those of
`IsCriticalTorusEnergy`**: both say that every partial derivative of
`hasDerivAt_torusEnergy` vanishes (`gradient_eucTorusEnergy_apply`). -/
theorem gradient_neg_eucTorusEnergy_eq_zero_iff {β : ℝ} (hβ : β ≠ 0) (ψ : EucSpace n) :
    gradient (fun φ => -eucTorusEnergy n β φ) ψ = 0 ↔ IsCriticalTorusEnergy n β ψ.ofLp := by
  rw [gradient_neg_eucTorusEnergy, neg_eq_zero]
  refine ⟨fun h i => ?_, fun h => ?_⟩
  · have hi := gradient_eucTorusEnergy_apply n β hβ ψ i
    rw [h, WithLp.ofLp_zero, Pi.zero_apply] at hi
    exact hi ▸ hasDerivAt_torusEnergy n β hβ ψ.ofLp i
  · ext i
    rw [gradient_eucTorusEnergy_apply n β hβ ψ i, WithLp.ofLp_zero, Pi.zero_apply]
    exact (hasDerivAt_torusEnergy n β hβ ψ.ofLp i).unique (h i)

/-- **`𝖤_β` is unchanged when every angle turns by the same amount**: the
translations by the line `ℝ𝟙` leave each difference `θ_i - θ_j` fixed. -/
theorem eucTorusEnergy_add_mem_span_one (β : ℝ) {w : EucSpace n}
    (hw : w ∈ Submodule.span ℝ {WithLp.toLp 2 fun _ : Idx n => (1 : ℝ)}) (ψ : EucSpace n) :
    eucTorusEnergy n β (ψ + w) = eucTorusEnergy n β ψ := by
  obtain ⟨a, rfl⟩ := Submodule.mem_span_singleton.1 hw
  unfold eucTorusEnergy torusEnergy
  congr 1
  refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun j _ => ?_
  simp

/-- **The second derivative of `𝖤_β` along a line**, for every direction `v`:

  `d²/ds² 𝖤_β(θ + s v)|_{s=0} = -(1/(2n²)) Σ_i Σ_j (v_i - v_j)² g_β(θ_i - θ_j)`,

the computation of `secondDeriv_torusEnergy_block` (Appendix B of
arXiv:2312.10794v5, `eq: taylor2`) with `1_𝒮` replaced by `v`. -/
theorem secondDerivTorusEnergy_line {β : ℝ} (hβ : β ≠ 0) (θ v : Angles n) :
    SecondDerivTorusEnergy n β θ v
      (-(1 / (2 * (n : ℝ) ^ 2)) * ∑ i, ∑ j, (v i - v j) ^ 2 * g_β_2d β (θ i - θ j)) := by
  have hfun : (fun u : ℝ => torusEnergy n β (fun i => θ i + u * v i))
      = fun u : ℝ => (1 / (2 * β * (n : ℝ) ^ 2)) *
          ∑ a : Idx n, ∑ b : Idx n, Real.exp (β * Real.cos ((θ a - θ b) + u * (v a - v b))) := by
    funext u
    simp only [torusEnergy]
    congr 1
    refine Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun b _ => ?_
    ring_nf
  refine ⟨fun s => (1 / (2 * β * (n : ℝ) ^ 2)) *
      ∑ a : Idx n, ∑ b : Idx n,
        Real.exp (β * Real.cos ((θ a - θ b) + s * (v a - v b))) *
          (-β * Real.sin ((θ a - θ b) + s * (v a - v b))) * (v a - v b), fun s => ?_, ?_⟩
  · rw [hfun]
    exact HasDerivAt.const_mul _ (HasDerivAt.fun_sum fun a _ => HasDerivAt.fun_sum fun b _ =>
      hasDerivAt_expCosLine β (θ a - θ b) (v a - v b) s)
  · refine (HasDerivAt.const_mul (1 / (2 * β * (n : ℝ) ^ 2))
      (HasDerivAt.fun_sum fun a (_ : a ∈ Finset.univ) =>
        HasDerivAt.fun_sum fun b (_ : b ∈ Finset.univ) =>
          hasDerivAt_expCosLine_deriv β (θ a - θ b) (v a - v b))).congr_deriv ?_
    simp only [Finset.mul_sum]
    refine Finset.sum_congr rfl fun a _ => Finset.sum_congr rfl fun b _ => ?_
    field_simp

/-- **A second derivative along a line is unique.** -/
theorem secondDerivTorusEnergy_unique {β : ℝ} {θ v : Angles n} {c c' : ℝ}
    (h : SecondDerivTorusEnergy n β θ v c) (h' : SecondDerivTorusEnergy n β θ v c') : c = c' := by
  obtain ⟨f, hf, hc⟩ := h
  obtain ⟨g, hg, hc'⟩ := h'
  obtain rfl : f = g := funext fun s => (hf s).unique (hg s)
  exact hc.unique hc'

/-- **The second derivative along a line is the Hessian quadratic form.**
For `F` of class `C²`, if `s ↦ F(z + s h)` has the derivative `f'(s)` and
`f'` the derivative `c` at `0`, then `⟪D(∇F)(z) h, h⟫ = c`: by the chain rule
`f'(s) = ⟪∇F(z + s h), h⟫`, whose derivative at `0` is `⟪D(∇F)(z) h, h⟫`. -/
theorem inner_fderiv_gradient_eq_of_hasDerivAt {E : Type*} [NormedAddCommGroup E]
    [InnerProductSpace ℝ E] [FiniteDimensional ℝ E] {F : E → ℝ} (hF : ContDiff ℝ 2 F) {z h : E}
    {f' : ℝ → ℝ} {c : ℝ} (h1 : ∀ s, HasDerivAt (fun u : ℝ => F (z + u • h)) (f' s) s)
    (h2 : HasDerivAt f' c 0) : ⟪fderiv ℝ (gradient F) z h, h⟫ = c := by
  have hline : ∀ s : ℝ, HasDerivAt (fun u : ℝ => z + u • h) h s := fun s => by
    simpa using ((hasDerivAt_id s).smul_const h).const_add z
  have hf' : f' = fun s => ⟪gradient F (z + s • h), h⟫ := funext fun s => (h1 s).unique (by
    rw [inner_gradient_left]
    exact ((hF.differentiable (by norm_num)) _).hasFDerivAt.comp_hasDerivAt s (hline s))
  have hG := (((SaddleAvoidance.contDiff_gradient hF).differentiable one_ne_zero) z).hasFDerivAt
  have hc := (hG.comp_hasDerivAt_of_eq 0 (hline 0) (by simp)).inner ℝ (hasDerivAt_const 0 h)
  rw [hf'] at h2
  rw [h2.unique hc]
  simp

/-- **The second derivative of Appendix B is the Hessian quadratic form of
`eucTorusEnergy`**: if `SecondDerivTorusEnergy n β θ v c`, then
`⟪D(∇𝖤_β)(θ) v, v⟫ = c`. -/
theorem inner_fderiv_gradient_eucTorusEnergy {β : ℝ} {θ v : Angles n} {c : ℝ}
    (h : SecondDerivTorusEnergy n β θ v c) :
    ⟪fderiv ℝ (gradient (eucTorusEnergy n β)) (WithLp.toLp 2 θ) (WithLp.toLp 2 v),
      WithLp.toLp 2 v⟫ = c := by
  obtain ⟨f', h1, h2⟩ := h
  refine inner_fderiv_gradient_eq_of_hasDerivAt (contDiff_eucTorusEnergy n β) (fun s => ?_) h2
  convert h1 s using 1
  funext u
  simp only [eucTorusEnergy, WithLp.ofLp_add, WithLp.ofLp_smul]
  rfl

/-- The hypotheses of the theorems above on `𝖤_β` hold for `β = 1`, `n = 2`:
`1 ≠ 0`, `(1, 1)` lies on the line `ℝ𝟙`, and the second derivative of
Appendix B along the block `{0}` exists at `(0, π)`
(`secondDeriv_torusEnergy_block`), twice for `secondDerivTorusEnergy_unique`. -/
example : (1 : ℝ) ≠ 0 ∧
    WithLp.toLp 2 (fun _ : Idx 2 => (1 : ℝ)) ∈ Submodule.span ℝ {WithLp.toLp 2 fun _ : Idx 2 => (1 : ℝ)} ∧
    ∃ c, SecondDerivTorusEnergy 2 1 ![0, π] (blockDir 2 {0}) c :=
  ⟨one_ne_zero, Submodule.mem_span_singleton_self _,
    ⟨_, secondDeriv_torusEnergy_block 2 1 one_ne_zero (by norm_num) _ _⟩⟩

/-- The hypotheses of `inner_fderiv_gradient_eq_of_hasDerivAt` hold for
`F(x) = x²` on the line at `z = 0` along `h = 1`, with `f'(s) = 2s` and
`c = 2`. -/
example : ContDiff ℝ 2 (fun x : ℝ => x ^ 2) ∧
    (∀ s : ℝ, HasDerivAt (fun u : ℝ => (0 + u • (1 : ℝ)) ^ 2) (2 * s) s) ∧
    HasDerivAt (fun s : ℝ => 2 * s) 2 0 :=
  ⟨contDiff_id.pow 2, fun s => by simpa using hasDerivAt_pow 2 s,
    by simpa using (hasDerivAt_id (0 : ℝ)).const_mul (2 : ℝ)⟩

end Perspective
end Transformer
