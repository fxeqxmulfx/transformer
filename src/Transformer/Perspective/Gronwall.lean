/-
# The scalar Grönwall step

The one inequality the exponential estimates of arXiv:2312.10794v5 are
integrated with: if `γ̇ ≥ λ (1 - γ)` from time `a` on, then `1 - γ` decays at
rate `λ` from its value at `a`.

It is stated for a bare differentiable function and a bare rate, with the
differential inequality as the hypothesis, so that each estimate supplies its
own `λ` and its own reason for the rate.
-/

import Transformer.Basic
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

open Real

namespace Transformer
namespace Perspective

/-- **Grönwall, in the form the exponential estimates need.**

If `γ̇(t) ≥ λ (1 - γ(t))` for every `t ≥ a`, then

  `1 - γ(t) ≤ (1 - γ(a)) e^{λ (a - t)}`   for every `t ≥ a`.

The proof is that `r ↦ (1 - γ(r)) e^{λ r}` has nonpositive derivative on
`[a, ∞)`, hence is nonincreasing there. -/
theorem decay_of_deriv_ge (γ : ℝ → ℝ) (a lam : ℝ)
    (hd : ∀ t : ℝ, a ≤ t → ∃ c : ℝ, HasDerivAt γ c t ∧ lam * (1 - γ t) ≤ c) :
    ∀ t : ℝ, a ≤ t → 1 - γ t ≤ (1 - γ a) * Real.exp (lam * (a - t)) := by
  have hzderiv : ∀ t : ℝ, a ≤ t →
      ∃ c : ℝ, HasDerivAt (fun r : ℝ => (1 - γ r) * Real.exp (lam * r)) c t ∧ c ≤ 0 := by
    intro t ht
    obtain ⟨R, hR, hRle⟩ := hd t ht
    have hexp : HasDerivAt (fun r : ℝ => Real.exp (lam * r))
        (Real.exp (lam * t) * lam) t := by
      simpa using ((hasDerivAt_id t).const_mul lam).exp
    refine ⟨(0 - R) * Real.exp (lam * t) + (1 - γ t) * (Real.exp (lam * t) * lam),
      ((hasDerivAt_const t (1 : ℝ)).sub hR).mul hexp, ?_⟩
    nlinarith [mul_nonneg (Real.exp_pos (lam * t)).le (sub_nonneg.mpr hRle)]
  have hz_anti : AntitoneOn (fun r : ℝ => (1 - γ r) * Real.exp (lam * r))
      (Set.Ici a) := by
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
  have hkey := hz_anti (Set.mem_Ici.mpr le_rfl) (Set.mem_Ici.mpr ht) ht
  refine le_of_mul_le_mul_right ?_ (Real.exp_pos (lam * t))
  have hprod : (1 - γ a) * Real.exp (lam * (a - t)) * Real.exp (lam * t)
      = (1 - γ a) * Real.exp (lam * a) := by
    rw [mul_assoc, ← Real.exp_add]; congr 2; ring
  rw [hprod]
  exact hkey

/-- The hypothesis of `decay_of_deriv_ge` is satisfiable, and the conclusion it
gives is sharp there: `γ(t) = 1 - e^{-t}` has `γ̇ = 1 - γ` exactly, and the
bound it yields at `a = 0`, `λ = 1` is `e^{-t}`, the value itself. -/
example :
    ∀ t : ℝ, (0 : ℝ) ≤ t →
      ∃ c : ℝ, HasDerivAt (fun s : ℝ => 1 - Real.exp (-s)) c t ∧
        1 * (1 - (1 - Real.exp (-t))) ≤ c := by
  intro t _
  refine ⟨Real.exp (-t), ?_, by simp⟩
  have h : HasDerivAt (fun s : ℝ => 1 - Real.exp (-s)) (-(Real.exp (-t) * -1)) t :=
    ((hasDerivAt_neg t).exp).const_sub 1
  exact h.congr_deriv (by ring)

end Perspective
end Transformer
