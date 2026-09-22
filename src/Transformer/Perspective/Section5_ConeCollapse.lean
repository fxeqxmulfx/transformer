/-
# §6.1 — Cone collapse: the last step of its proof

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*, §6.1,
`lem: hemisphere.clustering`.

The survey proves the lemma in two steps: step 1 produces the limit point
`x⋆` (`eq: qual.conv`), step 2 the differential inequality
`α̇ ≥ (1 - α)/(2 n e^{2β})` for `α(t) = min_i ⟨x_i(t), x⋆⟩`
(`e:diffineqalpha.step2`).  The lemma itself is `cone_collapse`
(`Perspective.Section5_ExpRate`), proved there by another route: the chart
`x ↦ x / ⟨x, w⟩` of the hemisphere (`Perspective.ConeLimit`), which needs
neither `x⋆` in advance nor the differential inequality.  The survey's route
is kept as well: step 2 is proved from the limit in
`Perspective.Section5_HemisphereRate`, with the differentiability of `α`
carried.  Integrating the differential inequality gives the exponential rate,
and that is `exp_rate_of_diffineqalpha` below — the last step of the proof,
not the lemma: the conclusion of the two steps is its hypothesis.
-/

import Transformer.Perspective.Section3_Gronwall
import Transformer.Perspective.Section5_HighD
import Transformer.Perspective.Section5_ExpRate
import Mathlib.Analysis.Calculus.Deriv.MeanValue

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

open Perspective

variable (d n : ℕ)

/-- **The last step of the proof of `lem: hemisphere.clustering`.**

If every `SA` solution from `X₀` satisfies `e:diffineqalpha.step2` past some
`t₀` — `α̇ ≥ (1 - α)/(2 n e^{2β})` for `α(t) = min_i ⟨x_i(t), x⋆⟩` — then it
converges to `x⋆` exponentially: `‖x_i(t) - x⋆‖ ≤ C e^{-λ t}`.  This is not
the lemma (`cone_collapse`): the hemisphere hypothesis, and steps 1 and 2 that
turn it into the differential inequality, are what `hstep` stands for.

The proof: with `λ₀ = 1/(2 n e^{2β})`, the function
`(1 - α(t)) e^{λ₀ t}` has non-positive derivative past `t₀`, hence
`1 - α(t) ≤ 2 e^{λ₀(t₁ - t)}` for `t₁ = max(t₀, 0)`, and
`‖x_i(t) - x⋆‖² = 2 - 2⟨x_i(t), x⋆⟩ ≤ 2(1 - α(t))` turns that into the rate
`C = 2 e^{λ₀ t₁/2}`, `λ = λ₀/2`.  Before `t₁` the crude bound
`‖x_i(t) - x⋆‖ ≤ 2` is already below `C e^{-λ t}`.

Source: arXiv:2312.10794v5, §6.1, proof of `lem: hemisphere.clustering`,
from `e:diffineqalpha.step2` to the end. -/
theorem exp_rate_of_diffineqalpha
    (β : ℝ) (X₀ : SphereTuple d n) (x_star : SSphere d) (t₀ : ℝ)
    (hstep : ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → Perspective.SA d n β X →
      ∃ α : ℝ → ℝ, IsMinInner d n X x_star α ∧
        ∀ t : ℝ, t₀ ≤ t → ∃ α' : ℝ, HasDerivAt α α' t ∧
          (1 - α t) / (2 * (n : ℝ) * Real.exp (2 * β)) ≤ α') :
    ∃ C lam : ℝ, 0 < C ∧ 0 < lam ∧
      ∀ X : ℝ → SphereTuple d n, X 0 = X₀ → Perspective.SA d n β X →
        ∀ i : Idx n, ∀ t : ℝ, 0 ≤ t →
          ‖((X t i : EucSpace d)) - ((x_star : EucSpace d))‖
            ≤ C * Real.exp (-(lam * t)) := by
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · exact ⟨1, 1, one_pos, one_pos, fun X _ _ i => i.elim0⟩
  have hnR : (0 : ℝ) < n := Nat.cast_pos.mpr hn
  have hsq2 : ∀ x : ℝ, (Real.exp x) ^ 2 = Real.exp (2 * x) := fun x => by
    rw [two_mul, Real.exp_add]; ring
  set lam0 : ℝ := 1 / (2 * (n : ℝ) * Real.exp (2 * β)) with hlam0def
  have hlam0 : 0 < lam0 := by rw [hlam0def]; positivity
  set t₁ : ℝ := max t₀ 0 with ht₁def
  have ht₀t₁ : t₀ ≤ t₁ := le_max_left _ _
  refine ⟨2 * Real.exp (lam0 / 2 * t₁), lam0 / 2, by positivity, by positivity, ?_⟩
  intro X hX0 hXSA i t ht
  have hxi : ‖(X t i : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp (X t i).2
  have hxs : ‖(x_star : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x_star.2
  have hgoal_eq : 2 * Real.exp (lam0 / 2 * t₁) * Real.exp (-(lam0 / 2 * t))
      = 2 * Real.exp (lam0 / 2 * (t₁ - t)) := by
    have hexp : lam0 / 2 * t₁ + -(lam0 / 2 * t) = lam0 / 2 * (t₁ - t) := by ring
    rw [mul_assoc, ← Real.exp_add, hexp]
  rw [hgoal_eq]
  have hcrude : ‖((X t i : EucSpace d)) - ((x_star : EucSpace d))‖ ≤ 2 := by
    calc ‖((X t i : EucSpace d)) - ((x_star : EucSpace d))‖
        ≤ ‖(X t i : EucSpace d)‖ + ‖(x_star : EucSpace d)‖ := norm_sub_le _ _
      _ = 2 := by rw [hxi, hxs]; norm_num
  rcases le_or_gt t t₁ with hle | hlt
  · -- Before `t₁` the crude bound already suffices.
    have hnn : (0 : ℝ) ≤ lam0 / 2 * (t₁ - t) := mul_nonneg (by positivity) (by linarith)
    have h1 := Real.exp_le_exp.mpr hnn
    rw [Real.exp_zero] at h1
    linarith
  · -- Past `t₁` the differential inequality integrates to the rate.
    obtain ⟨α, hα, hdiff⟩ := hstep X hX0 hXSA
    have hαge : ∀ s : ℝ, -1 ≤ α s := by
      intro s
      obtain ⟨j, hj⟩ := (hα s).2
      rw [hj]
      have hcs := abs_real_inner_le_norm ((X s j : EucSpace d)) ((x_star : EucSpace d))
      rw [mem_sphere_zero_iff_norm.mp (X s j).2, hxs, one_mul] at hcs
      exact neg_le_of_abs_le hcs
    have hzderiv : ∀ s : ℝ, t₀ ≤ s →
        ∃ c : ℝ, HasDerivAt (fun r : ℝ => (1 - α r) * Real.exp (lam0 * r)) c s ∧ c ≤ 0 := by
      intro s hs
      obtain ⟨α', hα', hbound⟩ := hdiff s hs
      have hexp : HasDerivAt (fun r : ℝ => Real.exp (lam0 * r))
          (Real.exp (lam0 * s) * lam0) s := by
        simpa using ((hasDerivAt_id s).const_mul lam0).exp
      refine ⟨(0 - α') * Real.exp (lam0 * s) + (1 - α s) * (Real.exp (lam0 * s) * lam0),
        ((hasDerivAt_const s (1 : ℝ)).sub hα').mul hexp, ?_⟩
      have heq : lam0 * (1 - α s) = (1 - α s) / (2 * (n : ℝ) * Real.exp (2 * β)) := by
        rw [hlam0def]; ring
      have hle : lam0 * (1 - α s) ≤ α' := by rw [heq]; exact hbound
      nlinarith [mul_nonneg (Real.exp_pos (lam0 * s)).le (sub_nonneg.mpr hle)]
    have hz_anti : AntitoneOn (fun r : ℝ => (1 - α r) * Real.exp (lam0 * r))
        (Set.Ici t₁) := by
      refine antitoneOn_of_deriv_nonpos (convex_Ici t₁) ?_ ?_ ?_
      · intro s hs
        obtain ⟨c, hc, -⟩ := hzderiv s (ht₀t₁.trans (Set.mem_Ici.mp hs))
        exact hc.continuousAt.continuousWithinAt
      · intro s hs
        rw [interior_Ici] at hs
        obtain ⟨c, hc, -⟩ := hzderiv s (ht₀t₁.trans (Set.mem_Ioi.mp hs).le)
        exact hc.differentiableAt.differentiableWithinAt
      · intro s hs
        rw [interior_Ici] at hs
        obtain ⟨c, hc, hcle⟩ := hzderiv s (ht₀t₁.trans (Set.mem_Ioi.mp hs).le)
        rw [hc.deriv]
        exact hcle
    have hkey : (1 - α t) * Real.exp (lam0 * t) ≤ (1 - α t₁) * Real.exp (lam0 * t₁) :=
      hz_anti (Set.mem_Ici.mpr le_rfl) (Set.mem_Ici.mpr hlt.le) hlt.le
    have hEt : (0 : ℝ) < Real.exp (lam0 * t) := Real.exp_pos _
    have hdecay : 1 - α t ≤ 2 * Real.exp (lam0 * (t₁ - t)) := by
      refine le_of_mul_le_mul_right ?_ hEt
      have hprod : 2 * Real.exp (lam0 * (t₁ - t)) * Real.exp (lam0 * t)
          = 2 * Real.exp (lam0 * t₁) := by
        have hexp : lam0 * (t₁ - t) + lam0 * t = lam0 * t₁ := by ring
        rw [mul_assoc, ← Real.exp_add, hexp]
      rw [hprod]
      calc (1 - α t) * Real.exp (lam0 * t)
          ≤ (1 - α t₁) * Real.exp (lam0 * t₁) := hkey
        _ ≤ 2 * Real.exp (lam0 * t₁) := by
            nlinarith [Real.exp_pos (lam0 * t₁), hαge t₁]
    have hsq : ‖((X t i : EucSpace d)) - ((x_star : EucSpace d))‖ ^ 2 ≤ 2 * (1 - α t) := by
      rw [norm_sub_sq_real, hxi, hxs]
      nlinarith [(hα t).1 i]
    have hrhs : (2 * Real.exp (lam0 / 2 * (t₁ - t))) ^ 2
        = 4 * Real.exp (lam0 * (t₁ - t)) := by
      rw [mul_pow, hsq2]
      have hexp : 2 * (lam0 / 2 * (t₁ - t)) = lam0 * (t₁ - t) := by ring
      rw [hexp]
      norm_num
    have hle2 : ‖((X t i : EucSpace d)) - ((x_star : EucSpace d))‖ ^ 2
        ≤ (2 * Real.exp (lam0 / 2 * (t₁ - t))) ^ 2 := by
      rw [hrhs]; linarith
    nlinarith [hle2, norm_nonneg ((X t i : EucSpace d) - (x_star : EucSpace d)),
      Real.exp_pos (lam0 / 2 * (t₁ - t))]

/-- The hypothesis of `exp_rate_of_diffineqalpha` is satisfiable, and by a genuine
solution: at `n = 1` a lone token stands still (`const_of_SA_one`), so every
solution from `X₀` sits at `x⋆ = X₀ 0` for all time, `α ≡ 1` is the minimum
`min_i ⟨x_i(t), x⋆⟩`, and the differential inequality reads `0 ≤ 0`. -/
example :
    ∀ X : ℝ → SphereTuple 1 1, X 0 = (fun _ => basePoint 0) →
      Perspective.SA 1 1 1 X →
      ∃ α : ℝ → ℝ, IsMinInner 1 1 X (basePoint 0) α ∧
        ∀ t : ℝ, (0 : ℝ) ≤ t → ∃ α' : ℝ, HasDerivAt α α' t ∧
          (1 - α t) / (2 * (1 : ℝ) * Real.exp (2 * 1)) ≤ α' := by
  intro X hX0 hXSA
  have hconst : ∀ (t : ℝ) (i : Idx 1),
      (X t i : EucSpace 1) = ((basePoint 0 : SSphere 1) : EucSpace 1) := by
    intro t i
    rw [const_of_SA_one 1 1 X hXSA t i, hX0]
  have hxx : inner (𝕜 := ℝ) ((basePoint 0 : EucSpace 1))
      ((basePoint 0 : EucSpace 1)) = (1 : ℝ) := by
    rw [real_inner_self_eq_norm_mul_norm,
      mem_sphere_zero_iff_norm.mp (basePoint 0 : SSphere 1).2]
    ring
  refine ⟨fun _ => 1, fun t => ⟨fun i => ?_, ⟨0, ?_⟩⟩, fun t _ => ⟨0, hasDerivAt_const t 1, ?_⟩⟩
  · rw [hconst t i, hxx]
  · rw [hconst t 0, hxx]
  · norm_num
