/-
# Appendix D — the differential inequality for `α`

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

`e:dotalpha`, `e:mineqalpha` and `e:diffineqalpha`: the lower bound on the
derivative of `α(t) = min_i ⟨x_i(t), x⋆⟩` that Appendix D integrates into
`e:productcloseto1` (`Perspective.product_close_to_one`).

The survey writes the three equations for the `x⋆` that `lem: hemisphere.clustering`
produces — the common limit of the particles, which start in an open
hemisphere — and it imports from step 2 of that lemma the fact that `x⋆` lies
in the cone of the particles.  Both are hypotheses of `diff_ineq_alpha` in
that form: the hemisphere and the limit, from which the cone is proved
(`Perspective.limit_mem_cone`).  The estimate `e:dotalpha` + `e:mineqalpha`
it shares with step 2 of the lemma is `Perspective.dot_alpha_ge`.
-/

import Transformer.Perspective.AppendixD_DotAlpha

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- **Equation (e:diffineqalpha).** *The differential inequality for `α`.*

  `α̇(t) ≥ (1/(n e^{2β})) α(1/n) (1 - α(t))`   for `t ≥ 1/n`.

**What the source says and what is changed here.**  The conclusion is the
survey's, verbatim.  The hypotheses are the survey's setting, each used in its
own derivation without appearing in the displayed equation.

*`hβ : 0 ≤ β`.*  The step "`a_{ij}(t) ≥ n^{-1} e^{-2β}`" is `e^{β⟨x_i,x_j⟩} ≥
e^{-β}` over `Z_{β,i} ≤ n e^{β}`, and both need `β ≥ 0`.

*`hw` and `hlim`: `x⋆` is the limit of `lem: hemisphere.clustering`.*  The
particles start in the open hemisphere around some `w`, and `x⋆` is their
common limit.  Step 2 of the lemma then puts `x⋆` in the cone of the particles
at every time (`limit_mem_cone`), which is what `e:mineqalpha` uses: with
`η x⋆` in the hull, `min_j ⟨x_i,x_j⟩ ≤ η α(t) ≤ α(t)`.

*`hpos : 0 < α(1/n)`.*  This is what the survey draws from `e:1/n`,
`α(1/n) ≥ γ_β(1/n)/2 > 0`, which rests on the high-dimensional estimates of
Appendix D (`Perspective.alpha_at_one_over_n`).  From it step 1, run from time
`1/n` along `x⋆`, makes `α` non-decreasing — "we gather that `α(t) ≥ α(1/n)`
for `t ≥ 1/n`" — and hence `0 ≤ α(t)`, without which `⟨x_j,x⋆⟩ - ⟨x_i,x_j⟩ α(t)
≥ α(t)(1 - ⟨x_i,x_j⟩)` points the wrong way.

*`hdα`, differentiability.*  `α` is a minimum of finitely many smooth curves,
so it need not be differentiable where the minimising index changes; the
survey's `α̇` is a Dini derivative.  The statement asserts an honest
`HasDerivAt`, so differentiability is carried as a hypothesis — at a time
where `α` *is* differentiable, Fermat's theorem forces its derivative to agree
with that of the attaining curve, which is what the proof uses.

Source: arXiv:2312.10794v5, Appendix D, `e:dotalpha`, `e:mineqalpha`,
`e:diffineqalpha`. -/
theorem diff_ineq_alpha (hn : 0 < n) (β : ℝ) (hβ : 0 ≤ β)
    (X : ℝ → SphereTuple d n) (x_star : SSphere d) (α : ℝ → ℝ)
    (hX : SA d n β X) (hα : IsMinInner d n X x_star α)
    (w : SSphere d)
    (hw : ∀ i : Idx n, 0 < inner (𝕜 := ℝ) ((X 0 i : EucSpace d)) ((w : EucSpace d)))
    (hlim : ∀ i : Idx n, Filter.Tendsto (fun s => (X s i : EucSpace d)) Filter.atTop
      (nhds (x_star : EucSpace d)))
    (hpos : 0 < α ((n : ℝ)⁻¹))
    (hdα : ∀ s : ℝ, DifferentiableAt ℝ α s) :
    ∀ t : ℝ, (n : ℝ)⁻¹ ≤ t →
      ∃ c : ℝ, HasDerivAt α c t ∧
        ((n : ℝ) * Real.exp (2 * β))⁻¹ * α ((n : ℝ)⁻¹) * (1 - α t) ≤ c := by
  have hnR : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hninv : (0 : ℝ) < (n : ℝ)⁻¹ := inv_pos.mpr hnR
  -- step 1 from time `1/n` along `x⋆`: `α` does not decrease
  have hmono : MonotoneOn (fun s => α (s + (n : ℝ)⁻¹)) (Set.Ici (0 : ℝ)) :=
    hemisphere_step1_monotone d n β x_star (fun s => X (s + (n : ℝ)⁻¹))
      (fun s => α (s + (n : ℝ)⁻¹)) (SA_shift d n β X hX _) (fun s => hα _)
      (fun i => by rw [zero_add]; exact hpos.trans_le ((hα _).1 i))
  intro t ht
  have ht0 : (0 : ℝ) ≤ t := le_trans hninv.le ht
  have hαstep : α ((n : ℝ)⁻¹) ≤ α t := by
    have := hmono (Set.mem_Ici.mpr le_rfl) (Set.mem_Ici.mpr (sub_nonneg.mpr ht))
      (sub_nonneg.mpr ht)
    simpa using this
  have hαt0 : 0 ≤ α t := hpos.le.trans hαstep
  obtain ⟨c, hc, hle⟩ := dot_alpha_ge d n hn β hβ X x_star α hX hα w hw hlim hdα t ht0 hαt0
  refine ⟨c, hc, le_trans ?_ hle⟩
  have hαle : α t ≤ 1 := by
    obtain ⟨i, hi⟩ := (hα t).2
    rw [hi]
    have hcs := real_inner_le_norm ((X t i : EucSpace d)) ((x_star : EucSpace d))
    rwa [mem_sphere_zero_iff_norm.mp (X t i).2,
      mem_sphere_zero_iff_norm.mp x_star.2, one_mul] at hcs
  rw [mul_assoc]
  exact mul_le_mul_of_nonneg_left
    (mul_le_mul_of_nonneg_right hαstep (by linarith)) (by positivity)

/-- The hypotheses of `diff_ineq_alpha` are satisfiable: two particles sitting
together at `basePoint 0`, with `w = x⋆` their position, where `α ≡ 1` and both
sides of `e:diffineqalpha` vanish. -/
example :
    ∀ t : ℝ, (((2 : ℕ) : ℝ))⁻¹ ≤ t →
      ∃ c : ℝ, HasDerivAt (fun _ : ℝ => (1 : ℝ)) c t ∧
        (((2 : ℕ) : ℝ) * Real.exp (2 * (0 : ℝ)))⁻¹ * 1 * (1 - 1) ≤ c := by
  have hx : ‖((basePoint 0 : SSphere 1) : EucSpace 1)‖ = 1 :=
    mem_sphere_zero_iff_norm.mp (basePoint 0).2
  have hxx : inner (𝕜 := ℝ) (((basePoint 0 : SSphere 1)) : EucSpace 1)
      (((basePoint 0 : SSphere 1)) : EucSpace 1) = 1 := by
    rw [real_inner_self_eq_norm_mul_norm, hx]; ring
  exact diff_ineq_alpha 1 2 two_pos 0 le_rfl (fun _ _ => basePoint 0) (basePoint 0)
    (fun _ => 1) (SA_const_consensus 1 2 two_pos 0 (basePoint 0))
    (fun _ => ⟨fun _ => le_of_eq hxx.symm, ⟨0, hxx.symm⟩⟩) (basePoint 0)
    (fun _ => by rw [hxx]; norm_num) (fun _ => tendsto_const_nhds) one_pos
    (fun _ => differentiableAt_const 1)

end Perspective
end Transformer
