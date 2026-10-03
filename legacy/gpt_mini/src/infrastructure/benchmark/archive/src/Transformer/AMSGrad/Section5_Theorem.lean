import Transformer.AMSGrad.Section5_Bounds

/-
# AdamX — the regret bound and its corollaries

§5 of arXiv:1904.03590v4: Theorem 5.1 (`mainthm2`).  Corollaries 5.5 and 5.6
are `Section5_Corollary`.

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

Source: arXiv:1904.03590v4, §5, Theorem 5.1.
-/

open Finset

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
  have hα' : ∀ t, 1 ≤ t → 0 < S.α t := fun t ht => by
    rw [hαt]; exact div_pos hα (Real.sqrt_pos.2 (by exact_mod_cast ht))
  have hp := prepare_lem hS hα' hβ₁ hβ₁' hT
    (fun _ _ _ h => m_eq_zero_of_vhat (le_adamXRule S.β₁) hβ₂ hβ₂' h) hx
  have h1 := eqmain_le_of hS (R := adamXRule S.β₁) (t₀ := 0) hα hαt hβ₁ hβ₁'
    (fun t ht i => vtnew_div hS hβ₁ hβ₁' hβ₂.le hβ₂'.le t ht i)
    (fun t ht i => adamX_mono hβ₁ hβ₁' hβ₂.le hβ₂'.le t ht i) hT hx
  rw [show ∑ t ∈ Icc (1 : ℕ) 0, Real.sqrt (t : ℝ) = 0 by simp, zero_add] at h1
  have h2 := eqsecond_le (le_adamXRule S.β₁) hα.le hαt hβ₁ hβ₁' hβ₂ hβ₂' hγ T
  have h3 := eqthird_adamX_le hS hα hαt hβ₁ hβ₁' hβ₂.le hβ₂'.le T hx
  linarith

/-- The hypotheses of Theorem 5.1 are satisfiable: the zero cost on `[-1, 1]`,
`α = 1`, `β_{1,t} = 0`, `β₂ = 1/2`, `T = 1`, `x* = 0`. -/
example :
    let S := zeroSetup (d := 1) (fun t : ℕ => 1 / Real.sqrt t) (fun _ => 0) (1 / 2)
    IsOnlineConvex S (Set.Icc (fun _ => -1) (fun _ => 1)) 2 0 ∧ (0 : ℝ) < 1 ∧
      S.α = (fun t : ℕ => 1 / Real.sqrt t) ∧ (∀ t, 1 ≤ t → 0 ≤ S.β₁ t ∧ S.β₁ t ≤ S.β₁ 1) ∧
      S.β₁ 1 < 1 ∧ 0 < S.β₂ ∧ S.β₂ < 1 ∧ S.β₁ 1 / Real.sqrt S.β₂ < 1 ∧ 1 ≤ 1 ∧
      (0 : Vec 1) ∈ Set.Icc (fun _ => -1) (fun _ => 1) :=
  ⟨isOnlineConvex_zero _ _ _, one_pos, rfl, fun _ _ => ⟨le_rfl, le_rfl⟩,
    by simp [zeroSetup], by norm_num [zeroSetup], by norm_num [zeroSetup],
    by simp [zeroSetup], le_rfl, ⟨fun _ => by norm_num, fun _ => by norm_num⟩⟩

end AMSGrad
end Transformer
