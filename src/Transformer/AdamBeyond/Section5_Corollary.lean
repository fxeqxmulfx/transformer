import Transformer.AdamBeyond.Section5_Regret

/-
# Adam and beyond — §5: Corollary 2

Corollary 2 of arXiv:1904.09237 and the remark that `β_{1,t} = β₁/t` still
gives `O(√T)`, both from Theorem 5 (`adamNC_regret`, `Section5_Regret`).

**What the source says and what is carried here.**

* Corollary 2 is missing its left-hand side "`R_T ≤`", restored here.  Its
  second term is `β₁D²G/((1-β₁)²(1-λ)²)`; Theorem 5 gives it with the factor
  `d/α`, as for Corollary 1, restored here.  Its `ζ` is left implicit; the
  setting satisfies both conditions with `ζ = α` (`adamNC_inv_cond`), which is
  the `ζ` stated.  With `v_{T,i} = ‖g_{1:T,i}‖²/T` the first term of
  Theorem 5 is `D²/(2α(1-β₁)) Σᵢ ‖g_{1:T,i}‖₂`, as the source writes.

* The remark "one can use `β_{1t} = β₁/t` and still ensure a data-dependent
  regret of `O(√T)`" is stated as `R_T ≤ K√T` for every `T` and `x* ∈ F`.

Not transcribed: "it is easy to generalize this result for similar settings
of `β_{2t}`", which names no statement.

Source: arXiv:1904.09237, §5, Corollary 2 and the paragraph after it.
-/

open Finset

namespace Transformer
namespace AdamBeyond

open AMSGrad

variable {d : ℕ}

/-- **Corollary 2**, with "`R_T ≤`" and the factor `d/α` of its second term
restored and `ζ = α`.  For AdamNC with `β_{2,t} = 1 - 1/t`, `α_t = α/√t` and
`β_{1,t} = β₁λ^{t-1}`, `0 ≤ β₁ < 1`, `0 ≤ λ < 1`, for every `x* ∈ F`,

  `R_T ≤ D²/(2α(1-β₁)) Σᵢ ‖g_{1:T,i}‖₂ + dβ₁D²G/(α(1-β₁)²(1-λ)²)`
  `     + 2α/(1-β₁)³ Σᵢ ‖g_{1:T,i}‖₂`.

Source: arXiv:1904.09237, §5, Corollary 2. -/
theorem adamNC_regret_lambda {S : Setup d} {F : Set (Vec d)} {D G : ℝ}
    (hS : IsOnlineConvex S F D G) (hβ₂0 : S.β₂ = 0) {α β₁ lam : ℝ} (hα : 0 < α)
    (hαt : S.α = fun t : ℕ => α / Real.sqrt t) (hβ₁ : 0 ≤ β₁) (hβ₁' : β₁ < 1)
    (hl : 0 ≤ lam) (hl' : lam < 1) (hb : ∀ t, S.β₁ t = β₁ * lam ^ (t - 1))
    (T : ℕ) {xstar : Vec d} (hxstar : xstar ∈ F) :
    S.regret (adamNCRule fun t : ℕ => 1 - 1 / (t : ℝ)) xstar T ≤
      D ^ 2 / (2 * α * (1 - β₁)) * ∑ i, S.gnorm (adamNCRule fun t : ℕ => 1 - 1 / (t : ℝ)) T i
      + d * β₁ * D ^ 2 * G / (α * (1 - β₁) ^ 2 * (1 - lam) ^ 2)
      + 2 * α / (1 - β₁) ^ 3 * ∑ i, S.gnorm (adamNCRule fun t : ℕ => 1 - 1 / (t : ℝ)) T i := by
  sorry

/-- **`β_{1,t} = β₁/t` gives `O(√T)` regret for AdamNC.**  With
`β_{2,t} = 1 - 1/t`, `α_t = α/√t` and `β_{1,t} = β₁/t`, `0 ≤ β₁ < 1`, there is
`K` with `R_T ≤ K√T` for every `T` and every `x* ∈ F`.

Source: arXiv:1904.09237, §5, the paragraph after Corollary 2. -/
theorem adamNC_regret_inv {S : Setup d} {F : Set (Vec d)} {D G : ℝ}
    (hS : IsOnlineConvex S F D G) (hβ₂0 : S.β₂ = 0) {α β₁ : ℝ} (hα : 0 < α)
    (hαt : S.α = fun t : ℕ => α / Real.sqrt t) (hβ₁ : 0 ≤ β₁) (hβ₁' : β₁ < 1)
    (hb : ∀ t, S.β₁ t = β₁ / t) :
    ∃ K : ℝ, ∀ T : ℕ, ∀ xstar ∈ F,
      S.regret (adamNCRule fun t : ℕ => 1 - 1 / (t : ℝ)) xstar T ≤ K * Real.sqrt T := by
  sorry

/-- The hypotheses of Corollary 2 and of the `β₁/t` remark are satisfiable:
the zero cost on `[-1, 1]`, `β₂ = 0`, `α = 1`, `β₁ = 0`, `λ = 0`, `x* = 0`. -/
example :
    let S := zeroSetup (d := 1) (fun t : ℕ => 1 / Real.sqrt t) (fun _ => 0) 0
    IsOnlineConvex S (Set.Icc (fun _ => -1) (fun _ => 1)) 2 0 ∧ S.β₂ = 0 ∧ (0 : ℝ) < 1 ∧
      S.α = (fun t : ℕ => 1 / Real.sqrt t) ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 ∧
      (∀ t, S.β₁ t = 0 * (0 : ℝ) ^ (t - 1)) ∧ (∀ t, S.β₁ t = 0 / t) ∧
      (0 : Vec 1) ∈ Set.Icc (fun _ => -1) (fun _ => 1) :=
  ⟨isOnlineConvex_zero _ _ _, rfl, one_pos, rfl, le_rfl, one_pos, fun _ => by simp [zeroSetup],
    fun _ => by simp [zeroSetup], fun _ => by norm_num, fun _ => by norm_num⟩

end AdamBeyond
end Transformer
