import Transformer.AdamBeyond.Section5_Lemma

/-
# Adam and beyond — §5: the regret of AdamNC

Theorem 5 of arXiv:1904.09237 (the lemma of its proof is in
`Section5_Lemma`), Corollary 2, and the remark that `β_{1,t} = β₁/t` still
gives `O(√T)`.

**What the source says and what is carried here.**

* AdamNC is the run with `β₂ = 0` and the rule `adamNCRule β₂`
  (`Section5_AdamNC`); `v_t` of the source is `v̂_t` of the run.

* Condition 1 of Theorem 5 reads `(1/α_T) √v_{t,i} ≥ (1/ζ) ‖g_{1:t,i}‖₂` for
  all `t ∈ [T]`.  The proof uses it at every `t` with `α_t` in place of `α_T`
  ("using similar argument for all time steps"), and with `α_T` it does not
  give that, since `1/α_t ≤ 1/α_T`; condition 1 is stated with `α_t`.  Its sum
  `Σ_j Π_k β_{2(t-k+1)}(1-β_{2j}) g²_{j,i}` is `v_{t,i}` (`adamNC_vhat_sum`).
  Implicit in the source and made explicit: `0 ≤ β_{2,t} ≤ 1`,
  `0 ≤ β_{1,t}`, `β₁ < 1`, `α > 0`.

* Corollary 2 is missing its left-hand side "`R_T ≤`", restored here.  Its
  second term is `β₁D²G/((1-β₁)²(1-λ)²)`; Theorem 5 gives it with the factor
  `d/α`, as for Corollary 1, restored here.  Its `ζ` is left implicit; the
  setting satisfies both conditions with `ζ = α` (`adamNC_inv_cond`), which is
  the `ζ` stated.  With `v_{T,i} = ‖g_{1:T,i}‖²/T` the first term of
  Theorem 5 is `D²/(2α(1-β₁)) Σᵢ ‖g_{1:T,i}‖₂`, as the source writes.

* The remark "one can use `β_{1t} = β₁/t` and still ensure a data-dependent
  regret of `O(√T)`" is stated as `R_T ≤ K√T` for every `T` and `x* ∈ F`.

Not transcribed: "one can generalize the result to deal with the case where
[`Γ_t ⪰ 0`] is violated as long as the violation is not too large or
frequent", and "it is easy to generalize this result for similar settings of
`β_{2t}`", which name no statement.

Source: arXiv:1904.09237, §5, Theorem 5 (thm:adamnc-proof), Corollary 2 and
the paragraph after it; appendix, §"Proof of Theorem 5" and its lemma.
-/

open Finset

namespace Transformer
namespace AdamBeyond

open AMSGrad

variable {d : ℕ}

/-- **Theorem 5**, with condition 1 at `α_t`.  For AdamNC with
`0 ≤ β_{2,t} ≤ 1`, `α_t = α/√t`, `0 ≤ β_{1,t} ≤ β₁ = β_{1,1} < 1`, under the
standing assumptions, if for some `ζ > 0`
1. `√v_{t,i}/α_t ≥ ‖g_{1:t,i}‖₂/ζ` for all `t ∈ [T]` and `i`, and
2. `√v_{t,i}/α_t ≥ √v_{t-1,i}/α_{t-1}` for all `t ∈ {2, …, T}` and `i`,

then for every `x* ∈ F`

  `R_T ≤ D²/(2α(1-β₁)) Σᵢ √T √v_{T,i} + D²/(1-β₁)² Σ_{t=1}^T Σᵢ β_{1,t}√v_{t,i}/α_t`
  `     + 2ζ/(1-β₁)³ Σᵢ ‖g_{1:T,i}‖₂`.

Source: arXiv:1904.09237, §5, Theorem 5 (thm:adamnc-proof). -/
theorem adamNC_regret {S : Setup d} {F : Set (Vec d)} {D G : ℝ} (hS : IsOnlineConvex S F D G)
    (hβ₂0 : S.β₂ = 0) {β₂ : ℕ → ℝ} (hβ₂ : ∀ t, 1 ≤ t → 0 ≤ β₂ t ∧ β₂ t ≤ 1)
    {α : ℝ} (hα : 0 < α) (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁ : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1) (hβ₁' : S.β₁ 1 < 1)
    (T : ℕ) {ζ : ℝ} (hζ : 0 < ζ)
    (h₁ : ∀ t ∈ Icc 1 T, ∀ i,
      S.gnorm (adamNCRule β₂) t i / ζ ≤ Real.sqrt (S.vhat (adamNCRule β₂) t i) / S.α t)
    (h₂ : ∀ t ∈ Icc 2 T, ∀ i,
      Real.sqrt (S.vhat (adamNCRule β₂) (t - 1) i) / S.α (t - 1) ≤
        Real.sqrt (S.vhat (adamNCRule β₂) t i) / S.α t)
    {xstar : Vec d} (hxstar : xstar ∈ F) :
    S.regret (adamNCRule β₂) xstar T ≤
      D ^ 2 / (2 * α * (1 - S.β₁ 1)) *
          ∑ i, Real.sqrt T * Real.sqrt (S.vhat (adamNCRule β₂) T i)
      + D ^ 2 / (1 - S.β₁ 1) ^ 2 *
          ∑ t ∈ Icc 1 T, ∑ i, S.β₁ t * Real.sqrt (S.vhat (adamNCRule β₂) t i) / S.α t
      + 2 * ζ / (1 - S.β₁ 1) ^ 3 * ∑ i, S.gnorm (adamNCRule β₂) T i := by
  sorry

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

/-- The hypotheses of Theorem 5 are satisfiable: the zero cost on
`[-1, 1]`, `β₂ = 0`, `β_{2,t} = 1 - 1/t`, `α = 1`, `β_{1,t} = 0`, `ζ = 1`,
`T = 2`, `x* = 0`; the conditions hold by `adamNC_inv_cond`. -/
example :
    let S := zeroSetup (d := 1) (fun t : ℕ => 1 / Real.sqrt t) (fun _ => 0) 0
    let R : Rule 1 := adamNCRule fun t : ℕ => 1 - 1 / (t : ℝ)
    IsOnlineConvex S (Set.Icc (fun _ => -1) (fun _ => 1)) 2 0 ∧ S.β₂ = 0 ∧
      (∀ t, 1 ≤ t → 0 ≤ 1 - 1 / (t : ℝ) ∧ 1 - 1 / (t : ℝ) ≤ 1) ∧ (0 : ℝ) < 1 ∧
      S.α = (fun t : ℕ => 1 / Real.sqrt t) ∧ (∀ t, 1 ≤ t → 0 < S.α t) ∧
      (∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1) ∧ S.β₁ 1 < 1 ∧
      (∀ t ∈ Icc 1 2, ∀ i, S.gnorm R t i / 1 ≤ Real.sqrt (S.vhat R t i) / S.α t) ∧
      (∀ t ∈ Icc 2 2, ∀ i,
        Real.sqrt (S.vhat R (t - 1) i) / S.α (t - 1) ≤ Real.sqrt (S.vhat R t i) / S.α t) ∧
      (0 : Vec 1) ∈ Set.Icc (fun _ => -1) (fun _ => 1) := by
  intro S R
  have hc := fun i => adamNC_inv_cond (S := S) rfl one_pos rfl i
  refine ⟨isOnlineConvex_zero _ _ _, rfl, fun t ht => ⟨?_, ?_⟩, one_pos, rfl,
    fun t ht => by simp [S, zeroSetup]; omega, fun _ _ => ⟨le_rfl, le_rfl⟩,
    by simp [S, zeroSetup], fun t ht i => (hc i).1 t (mem_Icc.mp ht).1,
    fun t ht i => (hc i).2 t (mem_Icc.mp ht).1, fun _ => by norm_num, fun _ => by norm_num⟩
  · have : (1 : ℝ) ≤ t := by exact_mod_cast ht
    rw [sub_nonneg]; exact div_le_one_of_le₀ this (by positivity)
  · have : (0 : ℝ) ≤ 1 / t := by positivity
    linarith

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
