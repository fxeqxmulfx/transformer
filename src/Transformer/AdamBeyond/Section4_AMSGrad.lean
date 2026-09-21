import Transformer.AMSGrad.Section1_TheoremA

/-
# Adam and beyond — §4: the regret of AMSGrad

§4 of arXiv:1904.09237 introduces AMSGrad (Algorithm 2), `v̂_t = max(v̂_{t-1}, v_t)`,
which is `Transformer.AMSGrad.amsgradRule`, and bounds its regret: Theorem 4,
the lemma of its proof (appendix, §"Proof of Theorem 4"), Corollary 1, and the
remark that `β_{1,t} = β₁/t` still gives `O(√T)`.

**What the source says and what is carried here.**

* This is the revised Theorem 4: its second term carries `D²/(1-β₁)²`, where
  the first version, quoted by arXiv:1904.03590 as its Theorem A
  (`Transformer.AMSGrad.theorem_A`), had `D²/(2(1-β₁))`; the footnote of the
  appendix records the missing factor `2/(1-β₁)`.  The hypotheses the source
  leaves implicit are made explicit, as in Theorem A: `0 ≤ β_{1,t}`,
  `β₁ < 1` and `0 < β₂ < 1`, without which `1/(1-β₁)` and `√(1-β₂)` are not
  bounds, and `α > 0`.  The source restricts to `T ≥ 1`; at `T = 0` both sides
  vanish, so the bound is stated for every `T`.

* The lemma is stated for AMSGrad under the assumptions of Theorem 4 other
  than the diameter and gradient bounds, which its proof does not use.  With
  `α_t = α/√t` it is `α` times the sum over `i` of Lemma 4.4 of
  arXiv:1904.03590 (`Transformer.AMSGrad.mainlem`).

* Corollary 1, with `β_{1,t} = β₁λ^{t-1}`, states its second term as
  `β₁D²G/((1-β₁)²(1-λ)²)`.  Theorem 4 gives
  `D²/(1-β₁)² Σ_t Σ_i β₁λ^{t-1}√t √v̂_{t,i}/α ≤ dβ₁D²G/(α(1-β₁)²(1-λ)²)`,
  using `√v̂_{t,i} ≤ G` and `Σ_t λ^{t-1}√t ≤ 1/(1-λ)²`: the factor `d/α` is
  missing from the source and is restored here.  `0 ≤ λ < 1` is made explicit.

* The remark after Corollary 1 is stated as: with `β_{1,t} = β₁/t` there is
  `K` with `R_T ≤ K√T` for every `T` and every `x* ∈ F`.  The third term of
  Theorem 4 is `O(√(T log T))`; the `O(√T)` rests on the closing remark of the
  appendix proof, that `Σ_t |g_{t,i}|/√t ≤ 2G√T` may replace the harmonic
  bound.

Not transcribed: the remark that averaging the `v_t` instead of taking their
maximum "can be shown to have similar convergence as AdaGrad", which names no
bound.

Source: arXiv:1904.09237, §4, Algorithm 2, Theorem 4 (thm:amsgrad-proof),
Corollary 1 (cor:t1-cor) and the paragraph after it; appendix, §"Proof of
Theorem 4", its lemma and closing remark.
-/

open Finset

namespace Transformer
namespace AdamBeyond

open AMSGrad

variable {d : ℕ}

/-- **Theorem 4.**  For AMSGrad with `α_t = α/√t`, `0 ≤ β_{1,t} ≤ β₁ = β_{1,1} < 1`,
`0 < β₂ < 1` and `γ = β₁/√β₂ < 1`, under the standing assumptions, for every
`x* ∈ F`,

  `R_T ≤ D²√T/(α(1-β₁)) Σᵢ √v̂_{T,i} + D²/(1-β₁)² Σ_{t=1}^T Σᵢ β_{1,t}√v̂_{t,i}/α_t`
  `     + α√(1 + log T)/((1-β₁)²(1-γ)√(1-β₂)) Σᵢ ‖g_{1:T,i}‖₂`.

Source: arXiv:1904.09237, §4, Theorem 4 (thm:amsgrad-proof). -/
theorem amsgrad_regret {S : Setup d} {F : Set (Vec d)} {D G : ℝ} (hS : IsOnlineConvex S F D G)
    {α : ℝ} (hα : 0 < α) (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁ : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1) (hβ₁' : S.β₁ 1 < 1)
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (hγ : S.β₁ 1 / Real.sqrt S.β₂ < 1)
    (T : ℕ) {xstar : Vec d} (hxstar : xstar ∈ F) :
    S.regret amsgradRule xstar T ≤
      D ^ 2 * Real.sqrt T / (α * (1 - S.β₁ 1)) * ∑ i, Real.sqrt (S.vhat amsgradRule T i)
      + D ^ 2 / (1 - S.β₁ 1) ^ 2 *
          ∑ t ∈ Icc 1 T, ∑ i, S.β₁ t * Real.sqrt (S.vhat amsgradRule t i) / S.α t
      + α * Real.sqrt (1 + Real.log T) /
          ((1 - S.β₁ 1) ^ 2 * (1 - S.β₁ 1 / Real.sqrt S.β₂) * Real.sqrt (1 - S.β₂))
          * ∑ i, S.gnorm amsgradRule T i := by
  sorry

/-- **The lemma of the proof of Theorem 4.**  For AMSGrad with `α_t = α/√t`,
`0 ≤ β_{1,t} ≤ β₁ = β_{1,1} < 1`, `0 < β₂ < 1` and `γ = β₁/√β₂ < 1`,

  `Σ_{t=1}^T α_t ‖V̂_t^{-1/4} m_t‖² ≤ α√(1 + log T)/((1-β₁)(1-γ)√(1-β₂)) Σᵢ ‖g_{1:T,i}‖₂`.

Source: arXiv:1904.09237, appendix, §"Proof of Theorem 4", Lemma. -/
theorem amsgrad_moment_sum {S : Setup d} {α : ℝ} (hα : 0 < α)
    (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁ : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1) (hβ₁' : S.β₁ 1 < 1)
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (hγ : S.β₁ 1 / Real.sqrt S.β₂ < 1) (T : ℕ) :
    ∑ t ∈ Icc 1 T, S.α t * ∑ i, S.m amsgradRule t i ^ 2 / Real.sqrt (S.vhat amsgradRule t i) ≤
      α * Real.sqrt (1 + Real.log T) /
          ((1 - S.β₁ 1) * (1 - S.β₁ 1 / Real.sqrt S.β₂) * Real.sqrt (1 - S.β₂))
          * ∑ i, S.gnorm amsgradRule T i := by
  sorry

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

/-- The hypotheses of Theorem 4, its lemma, Corollary 1 and the `β₁/t` remark
are satisfiable: the zero cost on `[-1, 1]`, `α = 1`, `β_{1,t} = 0` (`β₁ = 0`,
`λ = 0`), `β₂ = 1/2`, `x* = 0`. -/
example :
    let S := zeroSetup (d := 1) (fun t : ℕ => 1 / Real.sqrt t) (fun _ => 0) (1 / 2)
    IsOnlineConvex S (Set.Icc (fun _ => -1) (fun _ => 1)) 2 0 ∧ (0 : ℝ) < 1 ∧
      S.α = (fun t : ℕ => 1 / Real.sqrt t) ∧ (∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1) ∧
      S.β₁ 1 < 1 ∧ 0 < S.β₂ ∧ S.β₂ < 1 ∧ S.β₁ 1 / Real.sqrt S.β₂ < 1 ∧
      (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 ∧ (∀ t, S.β₁ t = 0 * (0 : ℝ) ^ (t - 1)) ∧
      (∀ t, S.β₁ t = 0 / t) ∧ (0 : ℝ) / Real.sqrt S.β₂ < 1 ∧
      (0 : Vec 1) ∈ Set.Icc (fun _ => -1) (fun _ => 1) :=
  ⟨isOnlineConvex_zero _ _ _, one_pos, rfl, fun _ _ => ⟨le_rfl, le_rfl⟩,
    by simp [zeroSetup], by norm_num [zeroSetup], by norm_num [zeroSetup],
    by simp [zeroSetup], le_rfl, one_pos, fun _ => by simp [zeroSetup],
    fun _ => by simp [zeroSetup], by simp, fun _ => by norm_num, fun _ => by norm_num⟩

end AdamBeyond
end Transformer
