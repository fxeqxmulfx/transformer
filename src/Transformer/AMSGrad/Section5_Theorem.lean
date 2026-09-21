import Transformer.AMSGrad.Section5_AdamX
import Transformer.AMSGrad.Section2_Prelim

/-
# AdamX — the regret bound and its corollaries

§5 of arXiv:1904.03590v4: Theorem 5.1 (`mainthm2`), Corollary 5.5 (`ge_cor`),
Corollary 5.6 (`bound`) and the sums (last1), (last2) of its proof.

**What the source says and what is carried here.**

* Theorem 5.1 bounds the second term by
  `dD²G/(2α(1-β₁)) Σ_{t=2}^T β_{1,t}√(t-1)`.  Its own proof gives
  `dD²G/(2α(1-β₁)²) Σ_{t=2}^T β_{1,t}√(t-1)`: the factor
  `β_{1,t}√v̂_{t-1}/(2α_{t-1}(1-β_{1,t}))` contributes one `1/(1-β₁)` and
  Lemma 5.3, `√v̂_{t-1} ≤ G/(1-β₁)`, a second one.  The constant is corrected.
  The first term `dD²G√T/(2α(1-β₁))` stands as stated, although the proof's
  last line writes `(1-β₁)²`: Lemma 5.2 gives `√v̂_T ≤ (1-β_{1,T})G/(1-β₁)`,
  which cancels `1-β_{1,T}`.  As in §4, `γ < 1` replaces `γ ≤ 1`,
  `0 ≤ β_{1,t}`, `0 < β₂ < 1` and `α > 0` are made explicit, and the regret is
  bounded against every `x* ∈ F`.

* Corollaries 5.5 and 5.6 assert `lim R(T)/T = 0`; as for Corollary 4.5 only
  the upper half follows from an upper bound and is stated.  The lower half is
  false: with `β_{1,t} = 0` AdamX is AMSGrad (`adamX_eq_amsgrad`), and
  `not_cor_lower` is a run with `β_{1,t} = 0` and `R(T) ≤ -T/2` infinitely often.

* Corollary 5.6's second setting is `β_{1,t} = 1/t`, which gives
  `β_{1,1} = 1`, against the standing `β₁ = β_{1,1} < 1`, and makes AdamX
  divide by `1 - β_{1,1} = 0`.  It is corrected to `β_{1,t} = β₁/t`, the
  setting of Theorem 4.1; its proof (last2) is unchanged.

Source: arXiv:1904.03590v4, §5, Theorem 5.1, Corollaries 5.5, 5.6.
-/

open Finset Filter Topology

namespace Transformer
namespace AMSGrad

variable {d : ℕ}

/-- **Theorem 5.1.**  With `α_t = α/√t`, `0 ≤ β_{1,t} ≤ β₁ = β_{1,1} < 1`,
`0 < β₂ < 1` and `γ = β₁/√β₂ < 1`, AdamX has, for `T ≥ 1` and `x* ∈ F`,

  `R(T) ≤ dD²G√T/(2α(1-β₁)) + dD²G/(2α(1-β₁)²) Σ_{t=2}^T β_{1,t}√(t-1)`
  `      + α√(ln T + 1)/((1-β₁)²√(1-β₂)(1-γ)) Σᵢ ‖g_{1:T,i}‖₂`.

The source has `(1-β₁)` for `(1-β₁)²` in the second term; see the module
docstring.  Lemma 5.4 is `mainlem le_adamXRule`.
Source: arXiv:1904.03590v4, §5, Theorem 5.1. -/
theorem mainthm2 {S : Setup d} {F : Set (Vec d)} {D G : ℝ} (hS : IsOnlineConvex S F D G)
    {α : ℝ} (hα : 0 < α) (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁ : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1) (hβ₁' : S.β₁ 1 < 1)
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (hγ : S.β₁ 1 / Real.sqrt S.β₂ < 1)
    {T : ℕ} (hT : 1 ≤ T) {xstar : Vec d} (hx : xstar ∈ F) :
    S.regret (adamXRule S.β₁) xstar T ≤
      d * D ^ 2 * G / (2 * α * (1 - S.β₁ 1)) * Real.sqrt T
      + d * D ^ 2 * G / (2 * α * (1 - S.β₁ 1) ^ 2) *
          ∑ t ∈ Icc 2 T, S.β₁ t * Real.sqrt ((t : ℝ) - 1)
      + α * Real.sqrt (Real.log T + 1) /
          ((1 - S.β₁ 1) ^ 2 * Real.sqrt (1 - S.β₂) * (1 - S.β₁ 1 / Real.sqrt S.β₂))
          * ∑ i, S.gnorm (adamXRule S.β₁) T i := by
  sorry

/-- **Corollary 5.5, the upper half.**  Under the assumptions of `mainthm2`, if
`Σ_{t=2}^T β_{1,t}√(t-1)/T → 0`, then for every `ε > 0`, eventually
`R(T)/T ≤ ε` for all `x* ∈ F`.
Source: arXiv:1904.03590v4, §5, Corollary 5.5. -/
theorem ge_cor {S : Setup d} {F : Set (Vec d)} {D G : ℝ} (hS : IsOnlineConvex S F D G)
    {α : ℝ} (hα : 0 < α) (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁ : ∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1) (hβ₁' : S.β₁ 1 < 1)
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (hγ : S.β₁ 1 / Real.sqrt S.β₂ < 1)
    (hlim : Tendsto (fun T : ℕ => (∑ t ∈ Icc 2 T, S.β₁ t * Real.sqrt ((t : ℝ) - 1)) / T)
      atTop (𝓝 0))
    {ε : ℝ} (hε : 0 < ε) :
    ∀ᶠ T : ℕ in atTop, ∀ xstar ∈ F, S.regret (adamXRule S.β₁) xstar T / T ≤ ε := by
  sorry

/-- **Corollary 5.6, `β_{1,t} = β₁λ^{t-1}`, the upper half.**  AdamX with
`α_t = α/√t`, `0 ≤ β₁ < 1`, `0 < λ < 1`, `0 < β₂ < 1` and `γ < 1`: for every
`ε > 0`, eventually `R(T)/T ≤ ε` for all `x* ∈ F`.
Source: arXiv:1904.03590v4, §5, Corollary 5.6. -/
theorem bound_lambda {S : Setup d} {F : Set (Vec d)} {D G : ℝ} (hS : IsOnlineConvex S F D G)
    {α β₁ lam : ℝ} (hα : 0 < α) (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁ : 0 ≤ β₁) (hβ₁' : β₁ < 1) (hl : 0 < lam) (hl' : lam < 1)
    (hb : ∀ t, S.β₁ t = β₁ * lam ^ (t - 1))
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (hγ : β₁ / Real.sqrt S.β₂ < 1)
    {ε : ℝ} (hε : 0 < ε) :
    ∀ᶠ T : ℕ in atTop, ∀ xstar ∈ F, S.regret (adamXRule S.β₁) xstar T / T ≤ ε := by
  sorry

/-- **Corollary 5.6, `β_{1,t} = β₁/t`, the upper half.**  As `bound_lambda`
with `β_{1,t} = β₁/t`; the source has `β_{1,t} = 1/t`, see the module
docstring.
Source: arXiv:1904.03590v4, §5, Corollary 5.6. -/
theorem bound_inv {S : Setup d} {F : Set (Vec d)} {D G : ℝ} (hS : IsOnlineConvex S F D G)
    {α β₁ : ℝ} (hα : 0 < α) (hαt : S.α = fun t : ℕ => α / Real.sqrt t)
    (hβ₁ : 0 ≤ β₁) (hβ₁' : β₁ < 1) (hb : ∀ t, S.β₁ t = β₁ / t)
    (hβ₂ : 0 < S.β₂) (hβ₂' : S.β₂ < 1) (hγ : β₁ / Real.sqrt S.β₂ < 1)
    {ε : ℝ} (hε : 0 < ε) :
    ∀ᶠ T : ℕ in atTop, ∀ xstar ∈ F, S.regret (adamXRule S.β₁) xstar T / T ≤ ε := by
  sorry

/-- The hypotheses of Theorem 5.1 and Corollaries 5.5, 5.6 are satisfiable:
the zero cost on `[-1, 1]`, `α = 1`, `β₁ = 0`, `λ = 1/2`, `β₂ = 1/2`, `ε = 1`. -/
example :
    let S := zeroSetup (d := 1) (fun t : ℕ => 1 / Real.sqrt t) (fun _ => 0) (1 / 2)
    IsOnlineConvex S (Set.Icc (fun _ => -1) (fun _ => 1)) 2 0 ∧ (0 : ℝ) < 1 ∧
      S.α = (fun t : ℕ => 1 / Real.sqrt t) ∧ (∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1) ∧
      S.β₁ 1 < 1 ∧ 0 < S.β₂ ∧ S.β₂ < 1 ∧ S.β₁ 1 / Real.sqrt S.β₂ < 1 ∧
      Tendsto (fun T : ℕ => (∑ t ∈ Icc 2 T, S.β₁ t * Real.sqrt ((t : ℝ) - 1)) / T)
        atTop (𝓝 0) ∧
      (∀ t, S.β₁ t = 0 * (1 / 2 : ℝ) ^ (t - 1)) ∧ (∀ t, S.β₁ t = 0 / t) ∧ (0 : ℝ) < 1 :=
  ⟨isOnlineConvex_zero _ _ _, one_pos, rfl, fun _ _ => ⟨le_rfl, le_rfl⟩,
    by simp [zeroSetup], by norm_num [zeroSetup], by norm_num [zeroSetup],
    by simp [zeroSetup], by simp [zeroSetup],
    fun _ => by simp [zeroSetup], fun _ => by simp [zeroSetup], one_pos⟩

/-! ### The sums (last1) and (last2) -/

/-- **(last1).**  For `β₁ ≤ 1` and `0 ≤ λ < 1`,
`Σ_{t=2}^T β₁λ^{t-1}√(t-1) ≤ 1/(1-λ)²`.
Source: arXiv:1904.03590v4, §5, proof of Corollary 5.6, (last1). -/
theorem sum_lambda_le {β₁ lam : ℝ} (hβ₁' : β₁ ≤ 1) (hl : 0 ≤ lam)
    (hl' : lam < 1) (T : ℕ) :
    ∑ t ∈ Icc 2 T, β₁ * lam ^ (t - 1) * Real.sqrt ((t : ℝ) - 1) ≤ 1 / (1 - lam) ^ 2 := by
  have hg : ∀ n : ℕ, 0 ≤ ((n : ℝ) + 1) * lam ^ n := fun n => by positivity
  have key : ∀ T : ℕ, ∑ t ∈ Icc 2 T, β₁ * lam ^ (t - 1) * Real.sqrt ((t : ℝ) - 1) ≤
      ∑ n ∈ range T, ((n : ℝ) + 1) * lam ^ n := by
    intro T
    induction T with
    | zero => simp
    | succ T ih =>
      rw [sum_range_succ]
      rcases Nat.eq_zero_or_pos T with rfl | hT
      · simp
      rw [sum_Icc_succ_top (by omega)]
      have hs : Real.sqrt ((((T + 1 : ℕ) : ℝ)) - 1) ≤ (T : ℝ) + 1 := by
        push_cast; rw [add_sub_cancel_right]
        nlinarith [Real.sq_sqrt (Nat.cast_nonneg (α := ℝ) T), Real.sqrt_nonneg (T : ℝ),
          sq_nonneg (Real.sqrt (T : ℝ) - 1)]
      have hp : 0 ≤ lam ^ (T + 1 - 1) := by positivity
      have : β₁ * lam ^ (T + 1 - 1) * Real.sqrt ((((T + 1 : ℕ) : ℝ)) - 1) ≤
          ((T : ℝ) + 1) * lam ^ T := by
        rw [Nat.add_sub_cancel] at hp ⊢
        nlinarith [mul_le_mul_of_nonneg_left hs hp, Real.sqrt_nonneg ((((T + 1 : ℕ) : ℝ)) - 1),
          mul_nonneg (mul_nonneg (sub_nonneg.mpr hβ₁') hp)
            (Real.sqrt_nonneg ((((T + 1 : ℕ) : ℝ)) - 1))]
      linarith
  have hsum : Summable fun n : ℕ => ((n : ℝ) + 1) * lam ^ n := by
    have hn : ‖lam‖ < 1 := by rwa [Real.norm_of_nonneg hl]
    have h := (hasSum_coe_mul_geometric_of_norm_lt_one hn).add (hasSum_geometric_of_lt_one hl hl')
    have he : (fun n : ℕ => ((n : ℝ) + 1) * lam ^ n) = fun n : ℕ => (n : ℝ) * lam ^ n + lam ^ n := by
      funext n; ring
    rw [he]; exact h.summable
  calc _ ≤ _ := key T
    _ ≤ ∑' n : ℕ, ((n : ℝ) + 1) * lam ^ n := hsum.sum_le_tsum _ fun n _ => hg n
    _ = 1 / (1 - lam) ^ 2 := taylor_deriv hl hl'

/-- **(last2).**  For `β₁ ≤ 1`, `Σ_{t=2}^T (β₁/t)√(t-1) ≤ 2√T`.  The source
takes `β₁ = 1`.
Source: arXiv:1904.03590v4, §5, proof of Corollary 5.6, (last2). -/
theorem sum_inv_le {β₁ : ℝ} (hβ₁' : β₁ ≤ 1) (T : ℕ) :
    ∑ t ∈ Icc 2 T, β₁ / t * Real.sqrt ((t : ℝ) - 1) ≤ 2 * Real.sqrt T := by
  have hterm : ∀ t ∈ Icc 2 T, β₁ / t * Real.sqrt ((t : ℝ) - 1) ≤ 1 / Real.sqrt t := by
    intro t ht
    have ht1 : (1 : ℝ) ≤ t := by exact_mod_cast (show 1 ≤ t by simp at ht; omega)
    have hx : (0 : ℝ) < t := by linarith
    have hr : 0 < Real.sqrt (t : ℝ) := Real.sqrt_pos.mpr hx
    have h1 : Real.sqrt ((t : ℝ) - 1) ≤ Real.sqrt t := Real.sqrt_le_sqrt (by linarith)
    rw [div_mul_eq_mul_div, div_le_div_iff₀ hx hr]
    nlinarith [mul_le_mul_of_nonneg_right h1 hr.le, Real.mul_self_sqrt hx.le,
      mul_nonneg (sub_nonneg.mpr hβ₁') (mul_nonneg (Real.sqrt_nonneg ((t : ℝ) - 1)) hr.le)]
  calc _ ≤ ∑ t ∈ Icc 2 T, 1 / Real.sqrt t := sum_le_sum hterm
    _ ≤ ∑ t ∈ Icc 1 T, 1 / Real.sqrt t :=
      sum_le_sum_of_subset_of_nonneg (Icc_subset_Icc_left (by norm_num))
        fun _ _ _ => by positivity
    _ ≤ 2 * Real.sqrt T := sum_inv_sqrt_le T

/-- The hypotheses of `sum_lambda_le` and `sum_inv_le` are satisfiable. -/
example : (1 : ℝ) ≤ 1 ∧ (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) < 1 := by norm_num

end AMSGrad
end Transformer
