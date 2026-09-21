import Transformer.AdamBeyond.Section3_Counter

/-
# Adam and beyond — §3: Theorem 2, for every `β₁ < √β₂`

Theorem 2 extends the counterexample of Theorem 1 to every `β₁, β₂ ∈ [0, 1)`
with `β₁ < √β₂`, with the costs `C x` once every `C` steps and `-x` otherwise.
Theorem 3, its stochastic version, is `counter_example_stochastic`.

**What the source says and what is carried here.**

* `β₂ ≥ 0` is not a hypothesis: it follows from `0 ≤ β₁ < √β₂`.

* Theorem 2 is stated for every `α > 0` in `α_t = α/√t`, which its proof
  handles uniformly; "`R_T/T ↛ 0`" is `c T ≤ R_T` for all large `T`, against a
  fixed `x* ∈ F`.  Its proof lower-bounds `v_{t+i-1}` by
  `(1-β₂)β₂^{i-2}C²` in the text and `(1-β₂)β₂^{i-1}C²` in the display; the
  statement is left open.

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
  sorry

/-- The hypotheses of Theorem 2 are satisfiable (`β₁ = 0`, `β₂ = 1/2`, `α = 1`). -/
example : (0 : ℝ) ≤ 0 ∧ (1 / 2 : ℝ) < 1 ∧ (0 : ℝ) < Real.sqrt (1 / 2) ∧ (0 : ℝ) < 1 :=
  ⟨le_rfl, by norm_num, Real.sqrt_pos.mpr (by norm_num), one_pos⟩

end AdamBeyond
end Transformer
