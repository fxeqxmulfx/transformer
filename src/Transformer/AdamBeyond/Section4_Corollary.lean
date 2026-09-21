import Transformer.AdamBeyond.Section4_Regret

/-
# Adam and beyond — §4: Corollary 1 and the `β₁/t` remark

Corollary 1 of arXiv:1904.09237 and the remark after it, both consequences of
Theorem 4 (`amsgrad_regret`); the deviations are recorded in the docstring of
`Section4_AMSGrad`.  Their hypotheses are witnessed there.

Source: arXiv:1904.09237, §4, Corollary 1 (cor:t1-cor) and the paragraph
after it; appendix, the closing remark of §"Proof of Theorem 4".
-/

open Finset

namespace Transformer
namespace AdamBeyond

open AMSGrad

variable {d : ℕ}

/-- **Corollary 1**, with the factor `d/α` of its second term restored.  Under
the assumptions of Theorem 4 with `β_{1,t} = β₁λ^{t-1}`, `0 ≤ λ < 1`, for
every `x* ∈ F`,

  `R_T ≤ D²√T/(α(1-β₁)) Σᵢ √v̂_{T,i} + dβ₁D²G/(α(1-β₁)²(1-λ)²)`
  `     + α√(1 + log T)/((1-β₁)²(1-γ)√(1-β₂)) Σᵢ ‖g_{1:T,i}‖₂`.

The source's second term is `β₁D²G/((1-β₁)²(1-λ)²)`; see the module docstring.

Source: arXiv:1904.09237, §4, Corollary 1 (cor:t1-cor). -/
theorem amsgrad_regret_lambda {S : Setup d} {F : Set (Vec d)} {D G : ℝ}
    (hS : IsOnlineConvex S F D G) {α β₁ lam : ℝ} (hα : 0 < α)
    (hαt : S.α = fun t : ℕ => α / Real.sqrt t) (hβ₁ : 0 ≤ β₁) (hβ₁' : β₁ < 1)
    (hl : 0 ≤ lam) (hl' : lam < 1) (hb : ∀ t, S.β₁ t = β₁ * lam ^ (t - 1))
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (hγ : β₁ / Real.sqrt S.β₂ < 1)
    (T : ℕ) {xstar : Vec d} (hxstar : xstar ∈ F) :
    S.regret amsgradRule xstar T ≤
      D ^ 2 * Real.sqrt T / (α * (1 - β₁)) * ∑ i, Real.sqrt (S.vhat amsgradRule T i)
      + d * β₁ * D ^ 2 * G / (α * (1 - β₁) ^ 2 * (1 - lam) ^ 2)
      + α * Real.sqrt (1 + Real.log T) /
          ((1 - β₁) ^ 2 * (1 - β₁ / Real.sqrt S.β₂) * Real.sqrt (1 - S.β₂))
          * ∑ i, S.gnorm amsgradRule T i := by
  sorry

/-- **`β_{1,t} = β₁/t` gives `O(√T)` regret.**  Under the assumptions of
Theorem 4 with `β_{1,t} = β₁/t`, there is `K` with `R_T ≤ K√T` for every `T`
and every `x* ∈ F`.

Source: arXiv:1904.09237, §4, the paragraph after Corollary 1; appendix, the
closing remark of §"Proof of Theorem 4". -/
theorem amsgrad_regret_inv {S : Setup d} {F : Set (Vec d)} {D G : ℝ}
    (hS : IsOnlineConvex S F D G) {α β₁ : ℝ} (hα : 0 < α)
    (hαt : S.α = fun t : ℕ => α / Real.sqrt t) (hβ₁ : 0 ≤ β₁) (hβ₁' : β₁ < 1)
    (hb : ∀ t, S.β₁ t = β₁ / t)
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (hγ : β₁ / Real.sqrt S.β₂ < 1) :
    ∃ K : ℝ, ∀ T : ℕ, ∀ xstar ∈ F, S.regret amsgradRule xstar T ≤ K * Real.sqrt T := by
  sorry

end AdamBeyond
end Transformer
