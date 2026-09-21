import Transformer.AdamBeyond.Section3_GenRegret

/-
# Adam and beyond — §3: Theorem 2, for every `β₁ < √β₂`

Theorem 2 extends the counterexample of Theorem 1 to every `β₁, β₂ ∈ [0, 1)`
with `β₁ < √β₂`, with the costs `C x` once every `C` steps and `-x` otherwise:
every block of `C` steps starts at `x = 1` (`gen_block`), and costs at least `2`
against `x* = -1` (`gen_regret`).  Theorem 3, its stochastic version, is
`counter_example_stochastic`.

**What the source says and what is carried here.**

* `β₂ ≥ 0` is not a hypothesis: it follows from `0 ≤ β₁ < √β₂`.

* Theorem 2 is stated for every `α > 0` in `α_t = α/√t`, which its proof
  handles uniformly; "`R_T/T ↛ 0`" is `c T ≤ R_T` for all large `T`, against a
  fixed `x* ∈ F`.  The statement is the source's.

* The proof departs from the source's in three places.  The problem is
  preceded by `C` steps with `f_t = 0`, during which Adam does not move
  (`m = v = 0`, and the step `0/√0` is `0`); the blocks then start at `t ≥ C`,
  where `√(t + C) ≤ √(2t)`, and at `x = 1`, which replaces the source's `T'`,
  `τ` and the divergence of `Σ 1/√t` that brings the iterate to `1`.
  `C = 2N` is taken with `β₂^N C² ≤ 1` and `N √(1-β₂)(1-γ) ≥ 4`,
  `γ = β₁/√β₂`, in place of eq:p-condition, and `-1 ≤ m_t ≤ C`,
  `0 ≤ v_t ≤ C²` along the whole run replace `m_{kC} ≤ 0`.  At the `j`-th step
  of a block `v ≥ (1-β₂)β₂^{j-1}C²`: the source's text (`β₂^{i-2}`, `j = i - 1`)
  has the right exponent, its display (`β₂^{i-1}`) is off by one.

Source: arXiv:1904.09237, §3, Theorem 2; Appendix, its proof.
-/

open Filter

namespace Transformer
namespace AdamBeyond

open AMSGrad

/-- **Theorem 2.**  For every `β₁, β₂ ∈ [0, 1)` with `β₁ < √β₂` and every
`α > 0`, Adam with `β_{1,t} = β₁`, `β₂` and `α_t = α/√t` has non-zero average
regret on an online convex problem.  arXiv:1904.09237, §3, Theorem 2. -/
theorem counter_example_gen {β₁ β₂ : ℝ} (hβ₁ : 0 ≤ β₁) (hβ₂ : β₂ < 1) (hγ : β₁ < Real.sqrt β₂)
    {α : ℝ} (hα : 0 < α) :
    ∃ (S : Setup 1) (F : Set (Vec 1)) (D G : ℝ), IsOnlineConvex S F D G ∧
      S.α = (fun t : ℕ => α / Real.sqrt t) ∧ (∀ t, S.β₁ t = β₁) ∧ S.β₂ = β₂ ∧
      ∃ xstar ∈ F, ∃ c > 0, ∀ᶠ T : ℕ in atTop, c * T ≤ S.regret adamRule xstar T := by
  obtain ⟨N, hN1, hN, hγN⟩ := gen_constants hβ₁ hβ₂ hγ
  have hs : 0 < Real.sqrt β₂ := lt_of_le_of_lt hβ₁ hγ
  have hβ₂0 : 0 < β₂ := Real.sqrt_pos.1 hs
  have hβ₁' : β₁ ≤ 1 := by
    have := Real.sqrt_lt_sqrt hβ₂0.le hβ₂
    rw [Real.sqrt_one] at this
    linarith
  have hC : 2 ≤ 2 * N := by omega
  refine ⟨genSetup (2 * N) β₁ β₂ α, Set.Icc (fun _ => -1) (fun _ => 1), 2, (2 * N : ℕ),
    isOnlineConvex_gen (by omega) _ _ _, rfl, fun _ => rfl, rfl, fun _ => -1,
    ⟨fun _ => le_rfl, fun _ => by norm_num⟩, 1 / ((2 * N : ℕ) : ℝ), by positivity,
    eventually_atTop.2 ⟨4 * (2 * N), fun T hT => ?_⟩⟩
  exact gen_regret hC hβ₁ hβ₁' hβ₂0 hβ₂ hα rfl hN ((div_lt_one hs).2 hγ) hγN hT

/-- The hypotheses of Theorem 2 are satisfiable (`β₁ = 0`, `β₂ = 1/2`, `α = 1`). -/
example : (0 : ℝ) ≤ 0 ∧ (1 / 2 : ℝ) < 1 ∧ (0 : ℝ) < Real.sqrt (1 / 2) ∧ (0 : ℝ) < 1 :=
  ⟨le_rfl, by norm_num, Real.sqrt_pos.mpr (by norm_num), one_pos⟩

end AdamBeyond
end Transformer
