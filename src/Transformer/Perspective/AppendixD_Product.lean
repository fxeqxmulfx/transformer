/-
# Appendix D — the differential inequality for `α` and its integration

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

One of the survey's equations: `e:productcloseto1`, the integrated form of the
differential inequality for `α`.

It is the Grönwall step, and it is proved from `e:diffineqalpha`
(`Perspective.diff_ineq_alpha`, in `Perspective.AppendixD_AlphaDeriv`) and
from `e:1/n` (`Perspective.alpha_at_one_over_n`) carried as explicit
hypotheses, on the time range the survey integrates over, `t ≥ 1/n`.
`product_close_to_one_of_limit` is the same estimate for the survey's `x⋆`,
the common limit of the particles, with `e:diffineqalpha` discharged.

What the survey builds on top of `e:productcloseto1` is in
`Perspective.AppendixD_Assembly`.
-/

import Transformer.Perspective.AppendixD_AlphaDeriv
import Mathlib.Analysis.Calculus.MeanValue

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- The consensus configuration of `n` copies of `basePoint 0` in `𝕊^0`, with
its minimum-inner-product function `α ≡ 1`: the common witness of the `SA` and
`IsMinInner` hypotheses of Appendix D. -/
theorem isMinInner_const_consensus (m : ℕ) (hm : 0 < m) :
    IsMinInner 1 m (fun _ _ => basePoint 0) (basePoint 0) (fun _ => 1) := by
  have hx : ‖((basePoint 0 : SSphere 1) : EucSpace 1)‖ = 1 :=
    mem_sphere_zero_iff_norm.mp (basePoint 0).2
  have hxx : inner (𝕜 := ℝ) (((basePoint 0 : SSphere 1)) : EucSpace 1)
      (((basePoint 0 : SSphere 1)) : EucSpace 1) = 1 := by
    rw [real_inner_self_eq_norm_mul_norm, hx]; ring
  exact fun _ => ⟨fun _ => le_of_eq hxx.symm, ⟨⟨0, hm⟩, hxx.symm⟩⟩

/-- **Equation (e:productcloseto1).**

  `1 - α(t) ≤ exp( (1 - γ_β(1/n) t) / (2 n e^{2β}) )`.

This is `e:diffineqalpha` integrated by Grönwall, starting from `e:1/n`.

**What the source says and what is changed here.**  Three changes, the first
two forced by the survey's own derivation.

*The two inputs are carried as hypotheses.*  `hdiff` is `e:diffineqalpha`
(`diff_ineq_alpha`, proved in `Perspective.AppendixD_AlphaDeriv` under the
hypotheses the survey's derivation uses), and `hone` is `e:1/n`
(`alpha_at_one_over_n`, proved).  What is proved below is the deduction, and
its dependence is visible in its signature.

*The range is `t ≥ 1/n`.*  The survey does not display it; it obtains the
estimate by "integrating `e:diffineqalpha` from `1/n` to `t`", and that is the
range on which it holds.

*`γ_β(1/n)` enters only through `hγpos`, `hγle` and `hone`.*  The solution of
`eq: ybeta` is not otherwise used, so the hypothesis `ybetaODE_SA n β γ` is
dropped; what is proved is the stronger statement in which `γ_β(1/n)` is any
number in `(0, 1]` that `e:1/n` holds for.  A cosine of an angle is such a
number.

Source: arXiv:2312.10794v5, Appendix D, `e:productcloseto1`. -/
theorem product_close_to_one (hn : 0 < n) (β : ℝ) (X : ℝ → SphereTuple d n)
    (γ α : ℝ → ℝ) (x_star : SSphere d) (hα : IsMinInner d n X x_star α)
    (hγpos : 0 < γ ((n : ℝ)⁻¹)) (hγle : γ ((n : ℝ)⁻¹) ≤ 1)
    (hone : (1/2 : ℝ) * γ ((n : ℝ)⁻¹) ≤ α ((n : ℝ)⁻¹))
    (hdiff : ∀ t : ℝ, (n : ℝ)⁻¹ ≤ t →
      ∃ c : ℝ, HasDerivAt α c t ∧
        ((n : ℝ) * Real.exp (2 * β))⁻¹ * α ((n : ℝ)⁻¹) * (1 - α t) ≤ c) :
    ∀ t : ℝ, (n : ℝ)⁻¹ ≤ t →
      1 - α t
        ≤ Real.exp ((1 - γ ((n : ℝ)⁻¹) * t) / (2 * (n : ℝ) * Real.exp (2 * β))) := by
  have hnR : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hn1 : (1 : ℝ) ≤ (n : ℝ) := by exact_mod_cast hn
  have hEpos : (0 : ℝ) < Real.exp (2 * β) := Real.exp_pos _
  have hxs : ‖((x_star : SSphere d) : EucSpace d)‖ = 1 :=
    mem_sphere_zero_iff_norm.mp x_star.2
  -- `α` is a minimum of inner products of unit vectors, so `1 - α ≥ 0`.
  have hαle : ∀ s : ℝ, α s ≤ 1 := by
    intro s
    obtain ⟨j, hj⟩ := (hα s).2
    rw [hj]
    have hcs := real_inner_le_norm ((X s j : EucSpace d)) ((x_star : EucSpace d))
    rwa [mem_sphere_zero_iff_norm.mp (X s j).2, hxs, one_mul] at hcs
  set kap : ℝ := γ ((n : ℝ)⁻¹) / (2 * (n : ℝ) * Real.exp (2 * β)) with hkapdef
  have hkappos : 0 < kap := by rw [hkapdef]; positivity
  -- the rate in `e:diffineqalpha` is at least `kap`, by `e:1/n`
  have hrate : kap ≤ ((n : ℝ) * Real.exp (2 * β))⁻¹ * α ((n : ℝ)⁻¹) := by
    have hcomm : ((n : ℝ) * Real.exp (2 * β))⁻¹ * α ((n : ℝ)⁻¹)
        = α ((n : ℝ)⁻¹) / ((n : ℝ) * Real.exp (2 * β)) := by ring
    rw [hkapdef, hcomm, div_le_div_iff₀ (by positivity) (by positivity)]
    nlinarith [mul_nonneg
      (by linarith : (0 : ℝ) ≤ 2 * α ((n : ℝ)⁻¹) - γ ((n : ℝ)⁻¹))
      (by positivity : (0 : ℝ) ≤ (n : ℝ) * Real.exp (2 * β))]
  -- `(1 - α) e^{kap ·}` is nonincreasing past `1/n`
  have hzderiv : ∀ s : ℝ, (n : ℝ)⁻¹ ≤ s →
      ∃ c : ℝ, HasDerivAt (fun r : ℝ => (1 - α r) * Real.exp (kap * r)) c s ∧ c ≤ 0 := by
    intro s hs
    obtain ⟨c, hc, hbound⟩ := hdiff s hs
    have hexp : HasDerivAt (fun r : ℝ => Real.exp (kap * r))
        (Real.exp (kap * s) * kap) s := by
      simpa using ((hasDerivAt_id s).const_mul kap).exp
    refine ⟨(0 - c) * Real.exp (kap * s) + (1 - α s) * (Real.exp (kap * s) * kap),
      ((hasDerivAt_const s (1 : ℝ)).sub hc).mul hexp, ?_⟩
    have hle : kap * (1 - α s) ≤ c := by
      have h2 : kap * (1 - α s)
          ≤ ((n : ℝ) * Real.exp (2 * β))⁻¹ * α ((n : ℝ)⁻¹) * (1 - α s) :=
        mul_le_mul_of_nonneg_right hrate (by linarith [hαle s])
      linarith
    nlinarith [mul_nonneg (Real.exp_pos (kap * s)).le (sub_nonneg.mpr hle)]
  have hz_anti : AntitoneOn (fun r : ℝ => (1 - α r) * Real.exp (kap * r))
      (Set.Ici ((n : ℝ)⁻¹)) := by
    refine antitoneOn_of_deriv_nonpos (convex_Ici _) ?_ ?_ ?_
    · intro s hs
      obtain ⟨c, hc, -⟩ := hzderiv s (Set.mem_Ici.mp hs)
      exact hc.continuousAt.continuousWithinAt
    · intro s hs
      rw [interior_Ici] at hs
      obtain ⟨c, hc, -⟩ := hzderiv s (Set.mem_Ioi.mp hs).le
      exact hc.differentiableAt.differentiableWithinAt
    · intro s hs
      rw [interior_Ici] at hs
      obtain ⟨c, hc, hcle⟩ := hzderiv s (Set.mem_Ioi.mp hs).le
      rw [hc.deriv]
      exact hcle
  intro t ht
  have hkey : (1 - α t) * Real.exp (kap * t)
      ≤ (1 - α ((n : ℝ)⁻¹)) * Real.exp (kap * (n : ℝ)⁻¹) :=
    hz_anti (Set.mem_Ici.mpr le_rfl) (Set.mem_Ici.mpr ht) ht
  have hEt : (0 : ℝ) < Real.exp (kap * t) := Real.exp_pos _
  have hstep : 1 - α t ≤ Real.exp (kap * ((n : ℝ)⁻¹ - t)) := by
    refine le_of_mul_le_mul_right ?_ hEt
    have hprod : Real.exp (kap * ((n : ℝ)⁻¹ - t)) * Real.exp (kap * t)
        = Real.exp (kap * (n : ℝ)⁻¹) := by
      rw [← Real.exp_add]; congr 1; ring
    rw [hprod]
    calc (1 - α t) * Real.exp (kap * t)
        ≤ (1 - α ((n : ℝ)⁻¹)) * Real.exp (kap * (n : ℝ)⁻¹) := hkey
      _ ≤ 1 * Real.exp (kap * (n : ℝ)⁻¹) := by
          nlinarith [Real.exp_pos (kap * (n : ℝ)⁻¹)]
      _ = Real.exp (kap * (n : ℝ)⁻¹) := one_mul _
  refine hstep.trans (Real.exp_le_exp.mpr ?_)
  have hinv : (n : ℝ)⁻¹ ≤ 1 := by
    rw [inv_eq_one_div, div_le_one hnR]; exact hn1
  have hnum : γ ((n : ℝ)⁻¹) * ((n : ℝ)⁻¹ - t) ≤ 1 - γ ((n : ℝ)⁻¹) * t := by
    nlinarith [hγpos, hinv, hγle]
  calc kap * ((n : ℝ)⁻¹ - t)
      = γ ((n : ℝ)⁻¹) * ((n : ℝ)⁻¹ - t) / (2 * (n : ℝ) * Real.exp (2 * β)) := by
        rw [hkapdef]; ring
    _ ≤ (1 - γ ((n : ℝ)⁻¹) * t) / (2 * (n : ℝ) * Real.exp (2 * β)) := by gcongr

/-- The hypotheses of `product_close_to_one` are satisfiable: the two particles
of `isMinInner_const_consensus`, where `α ≡ 1`, and `γ ≡ 1`, for which `e:1/n`
reads `1/2 ≤ 1` and `e:diffineqalpha` reads `0 ≤ 0`. -/
example :
    IsMinInner 1 2 (fun _ _ => basePoint 0) (basePoint 0) (fun _ => 1) ∧
      (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 1 ∧ (1/2 : ℝ) * 1 ≤ 1 ∧
      ∀ t : ℝ, (((2 : ℕ) : ℝ))⁻¹ ≤ t →
        ∃ c : ℝ, HasDerivAt (fun _ : ℝ => (1 : ℝ)) c t ∧
          (((2 : ℕ) : ℝ) * Real.exp (2 * (0 : ℝ)))⁻¹ * 1 * (1 - 1) ≤ c := by
  refine ⟨isMinInner_const_consensus 2 two_pos, one_pos, le_rfl, by norm_num,
    fun t _ => ⟨0, hasDerivAt_const t 1, by norm_num⟩⟩

/-- **Equation (e:productcloseto1) for the limit `x⋆`.**

  `1 - α(t) ≤ exp( (1 - γ_β(1/n) t) / (2 n e^{2β}) )`,  `t ≥ 1/n`,

for `α(t) = min_i ⟨x_i(t), x⋆⟩` and `x⋆` the common limit of particles that
start in an open hemisphere — the survey's setting.  `e:diffineqalpha` is
discharged by `diff_ineq_alpha`; `e:1/n` stays a hypothesis, `hone`, and is
`alpha_at_one_over_n` under the high-dimensional estimates of Appendix D.  As
in `diff_ineq_alpha`, `α` is assumed differentiable (`hdα`), and `β ≥ 0`.

Source: arXiv:2312.10794v5, Appendix D, `e:productcloseto1`. -/
theorem product_close_to_one_of_limit (hn : 0 < n) (β : ℝ) (hβ : 0 ≤ β)
    (X : ℝ → SphereTuple d n) (hX : SA d n β X) (w : SSphere d)
    (hw : ∀ i : Idx n, 0 < inner (𝕜 := ℝ) ((X 0 i : EucSpace d)) ((w : EucSpace d)))
    (x_star : SSphere d)
    (hlim : ∀ i : Idx n, Filter.Tendsto (fun s => (X s i : EucSpace d)) Filter.atTop
      (nhds (x_star : EucSpace d)))
    (α : ℝ → ℝ) (hα : IsMinInner d n X x_star α) (hdα : ∀ s : ℝ, DifferentiableAt ℝ α s)
    (γ : ℝ → ℝ) (hγpos : 0 < γ ((n : ℝ)⁻¹)) (hγle : γ ((n : ℝ)⁻¹) ≤ 1)
    (hone : (1/2 : ℝ) * γ ((n : ℝ)⁻¹) ≤ α ((n : ℝ)⁻¹)) :
    ∀ t : ℝ, (n : ℝ)⁻¹ ≤ t →
      1 - α t
        ≤ Real.exp ((1 - γ ((n : ℝ)⁻¹) * t) / (2 * (n : ℝ) * Real.exp (2 * β))) :=
  product_close_to_one d n hn β X γ α x_star hα hγpos hγle hone
    (diff_ineq_alpha d n hn β hβ X x_star α hX hα w hw hlim (by linarith) hdα)

/-- The hypotheses of `product_close_to_one_of_limit` are satisfiable: two
particles at rest at `basePoint 0`, their own limit and hemisphere, `α ≡ 1`,
`γ ≡ 1`. -/
example :
    ∀ t : ℝ, (((2 : ℕ) : ℝ))⁻¹ ≤ t →
      1 - (fun _ : ℝ => (1 : ℝ)) t
        ≤ Real.exp ((1 - (fun _ : ℝ => (1 : ℝ)) ((((2 : ℕ) : ℝ))⁻¹) * t)
          / (2 * ((2 : ℕ) : ℝ) * Real.exp (2 * (0 : ℝ)))) := by
  have hx : ‖((basePoint 0 : SSphere 1) : EucSpace 1)‖ = 1 :=
    mem_sphere_zero_iff_norm.mp (basePoint 0).2
  have hxx : inner (𝕜 := ℝ) (((basePoint 0 : SSphere 1)) : EucSpace 1)
      (((basePoint 0 : SSphere 1)) : EucSpace 1) = 1 := by
    rw [real_inner_self_eq_norm_mul_norm, hx]; ring
  exact product_close_to_one_of_limit 1 2 two_pos 0 le_rfl (fun _ _ => basePoint 0)
    (SA_const_consensus 1 2 two_pos 0 (basePoint 0)) (basePoint 0)
    (fun _ => by rw [hxx]; norm_num) (basePoint 0) (fun _ => tendsto_const_nhds)
    (fun _ => 1) (isMinInner_const_consensus 2 two_pos) (fun _ => differentiableAt_const 1)
    (fun _ => 1) one_pos le_rfl (by norm_num)

end Perspective
end Transformer
