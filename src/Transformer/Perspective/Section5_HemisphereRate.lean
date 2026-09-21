/-
# §6.1 — Step 2 of cone collapse: the differential inequality

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*, §6.1, proof of
`lem: hemisphere.clustering`, step 2.

`e:diffineqalpha.step2`, for the `x⋆` that step 1 produces: the common limit
of particles starting in an open hemisphere.  The cone of step 2 is
`Perspective.limit_mem_cone`, and the estimate shared with Appendix D is
`Perspective.dot_alpha_ge`; what is left here is the survey's choice of `t₀`,
past which `α ≥ 1/2`.  The integration of the inequality into the exponential
rate is `Perspective.hemisphere_clustering`.
-/

import Transformer.Perspective.AppendixD_DotAlpha

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- **Equation (e:diffineqalpha.step2).**

  `α̇(t) ≥ (1/(2 n e^{2β})) (1 - α(t))`   for `t ≥ t₀`,

where `α(t) = min_i ⟨x_i(t), x⋆⟩`, `x⋆` is the common limit of step 1
(`eq: qual.conv`), and `t₀ > 0` is a time past which `α ≥ 1/2`.

**What the source says and what is changed here.**  The survey fixes `t₀` in
the text ("there exists some `t₀ > 0` such that `α(t) ≥ 1/2` for all
`t ≥ t₀`"); here it is the existential of the conclusion, and it is produced
from the limit exactly as the survey does.  The setting is the lemma's: the
particles start in the open hemisphere around `w` (`hw`) and converge to `x⋆`
(`hlim`, which is `eq: qual.conv`).  Two hypotheses are not in the survey's
display:

*`hβ : 0 ≤ β`.*  The lemma assumes `β > 0`; the bound `a_{ij} ≥ n^{-1}
e^{-2β}` needs only `β ≥ 0`.

*`hdα`, differentiability.*  `α` is a minimum of finitely many smooth curves
and the survey's `α̇` is a one-sided derivative along a minimising index; the
statement asserts an honest `HasDerivAt`, so differentiability is carried, as
in `diff_ineq_alpha`.

Source: arXiv:2312.10794v5, §6.1, proof of `lem: hemisphere.clustering`,
step 2, `e:dotalpha.step2`, `e:mineqalpha.step2`, `e:diffineqalpha.step2`. -/
theorem step2_alpha_diff_ineq (hn : 0 < n) (β : ℝ) (hβ : 0 ≤ β)
    (X : ℝ → SphereTuple d n) (x_star : SSphere d) (α : ℝ → ℝ)
    (hX : SA d n β X) (hα : IsMinInner d n X x_star α)
    (w : SSphere d)
    (hw : ∀ i : Idx n, 0 < inner (𝕜 := ℝ) ((X 0 i : EucSpace d)) ((w : EucSpace d)))
    (hlim : ∀ i : Idx n, Filter.Tendsto (fun s => (X s i : EucSpace d)) Filter.atTop
      (nhds (x_star : EucSpace d)))
    (hdα : ∀ s : ℝ, DifferentiableAt ℝ α s) :
    ∃ t₀ : ℝ, 0 < t₀ ∧ ∀ t : ℝ, t₀ ≤ t →
      ∃ c : ℝ, HasDerivAt α c t ∧
        (1 - α t) / (2 * (n : ℝ) * Real.exp (2 * β)) ≤ c := by
  have hxs : inner (𝕜 := ℝ) ((x_star : EucSpace d)) ((x_star : EucSpace d)) = 1 := by
    rw [real_inner_self_eq_norm_mul_norm, mem_sphere_zero_iff_norm.mp x_star.2]; ring
  -- `α ≥ 1/2` from some time on, by `eq: qual.conv`
  have hev : ∀ᶠ s in Filter.atTop, ∀ i : Idx n,
      (1 / 2 : ℝ) < inner (𝕜 := ℝ) ((X s i : EucSpace d)) ((x_star : EucSpace d)) := by
    refine Filter.eventually_all.mpr fun i => ?_
    have ht := (hlim i).inner (𝕜 := ℝ) (tendsto_const_nhds (x := (x_star : EucSpace d)))
    rw [hxs] at ht
    exact ht.eventually (lt_mem_nhds (by norm_num))
  obtain ⟨T, hT⟩ := Filter.eventually_atTop.mp hev
  refine ⟨max T 1, lt_max_of_lt_right one_pos, fun t ht => ?_⟩
  have hαhalf : (1 / 2 : ℝ) ≤ α t := by
    obtain ⟨i, hi⟩ := (hα t).2
    rw [hi]
    exact (hT t ((le_max_left _ _).trans ht) i).le
  have ht0 : (0 : ℝ) ≤ t := zero_le_one.trans ((le_max_right _ _).trans ht)
  have hαle : α t ≤ 1 := by
    obtain ⟨i, hi⟩ := (hα t).2
    rw [hi]
    have hcs := real_inner_le_norm ((X t i : EucSpace d)) ((x_star : EucSpace d))
    rwa [mem_sphere_zero_iff_norm.mp (X t i).2,
      mem_sphere_zero_iff_norm.mp x_star.2, one_mul] at hcs
  obtain ⟨c, hc, hle⟩ := dot_alpha_ge d n hn β hβ X x_star α hX hα w hw hlim hdα t ht0
    (by linarith)
  refine ⟨c, hc, le_trans ?_ hle⟩
  have hnR : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have heq : (1 - α t) / (2 * (n : ℝ) * Real.exp (2 * β))
      = ((n : ℝ) * Real.exp (2 * β))⁻¹ * (1 / 2 * (1 - α t)) := by
    field_simp
  rw [heq]
  exact mul_le_mul_of_nonneg_left
    (mul_le_mul_of_nonneg_right hαhalf (by linarith)) (by positivity)

/-- The hypotheses of `step2_alpha_diff_ineq` are satisfiable: two particles
sitting together at `basePoint 0`, with `w = x⋆` their position, where `α ≡ 1`
and both sides of `e:diffineqalpha.step2` vanish. -/
example :
    ∃ t₀ : ℝ, 0 < t₀ ∧ ∀ t : ℝ, t₀ ≤ t →
      ∃ c : ℝ, HasDerivAt (fun _ : ℝ => (1 : ℝ)) c t ∧
        (1 - 1) / (2 * ((2 : ℕ) : ℝ) * Real.exp (2 * (0 : ℝ))) ≤ c := by
  have hx : ‖((basePoint 0 : SSphere 1) : EucSpace 1)‖ = 1 :=
    mem_sphere_zero_iff_norm.mp (basePoint 0).2
  have hxx : inner (𝕜 := ℝ) (((basePoint 0 : SSphere 1)) : EucSpace 1)
      (((basePoint 0 : SSphere 1)) : EucSpace 1) = 1 := by
    rw [real_inner_self_eq_norm_mul_norm, hx]; ring
  exact step2_alpha_diff_ineq 1 2 two_pos 0 le_rfl (fun _ _ => basePoint 0) (basePoint 0)
    (fun _ => 1) (SA_const_consensus 1 2 two_pos 0 (basePoint 0))
    (fun _ => ⟨fun _ => le_of_eq hxx.symm, ⟨0, hxx.symm⟩⟩) (basePoint 0)
    (fun _ => by rw [hxx]; norm_num) (fun _ => tendsto_const_nhds)
    (fun _ => differentiableAt_const 1)

end Perspective
end Transformer
