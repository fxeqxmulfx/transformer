import Transformer.AdamBeyond.Section3_StochStep

/-
# Adam and beyond — §3: Theorem 3

The costs are i.i.d.: `C x` with probability `p = (1 + δ)/(C + 1)` and `-x`
otherwise, whose mean `F(x) = δ x` is minimized over `[-1, 1]` at `x* = -1`,
while Adam drifts upward in expectation (`stoch_step`) from `x₁ = 0`, so that
`E[x_t] ≥ 0` and `E[F(x_t)] - F(-1) ≥ δ` for every `t`.

**What the source says and what is carried here.**

* `β₂ ≥ 0` is not a hypothesis: it follows from `0 ≤ β₁ < √β₂`.
* Theorem 3 is run, as its proof assumes, without projection and from
  `x₁ = 0`; the source's footnote defers the projected case.  "Adam does not
  converge" is carried as the proof's conclusion: `E[F(x_t)] - F(-1) ≥ δ` for
  every `t ≥ 1`, with `δ = 1` and `C = max(C₀, 2)`.

Source: arXiv:1904.09237, §3, Theorem 3; Appendix, its proof.
-/

open MeasureTheory ProbabilityTheory

namespace Transformer
namespace AdamBeyond

open AMSGrad

/-- **Theorem 3.**  For every `β₁, β₂ ∈ [0, 1)` with `β₁ < √β₂` there is a
stochastic convex problem, `f_t = C x` with probability `p` and `-x` otherwise,
with mean `F(x) = δ x`, on which Adam stays `δ` away from the optimum in
expectation: `E[F(x_t)] - F(-1) ≥ δ` for every `t ≥ 1`.
arXiv:1904.09237, §3, Theorem 3. -/
theorem counter_example_stochastic {β₁ β₂ : ℝ} (hβ₁ : 0 ≤ β₁) (hβ₂ : β₂ < 1)
    (hγ : β₁ < Real.sqrt β₂) :
    ∃ C p δ : ℝ, 0 < p ∧ p < 1 ∧ 0 < δ ∧ p * C - (1 - p) = δ ∧
      ∀ (Ω : Type) [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
        (b : ℕ → Ω → Bool), IsBernoulliSeq μ b p → ∀ α : ℕ → ℝ, (∀ t, 0 ≤ α t) →
        ∀ t, 1 ≤ t → Integrable (stochX C β₁ β₂ α b t) μ ∧
          δ ≤ ∫ ω, δ * stochX C β₁ β₂ α b t ω ∂μ - δ * (-1) := by
  obtain ⟨C₀, hC₀⟩ := stoch_step hβ₁ hβ₂ hγ one_pos
  set C := max C₀ 2 with hCdef
  have hC2 : 2 ≤ C := le_max_right _ _
  have hu : 0 < C + 1 := by linarith
  refine ⟨C, (1 + 1) / (C + 1), 1, by positivity, by rw [div_lt_one hu]; linarith, one_pos,
    ?_, fun Ω _ μ _ b hb α hα t ht => ?_⟩
  · field_simp
    ring
  have hstep := hC₀ C (le_max_left _ _) Ω μ b hb α hα
  have key : Integrable (stochX C β₁ β₂ α b t) μ ∧ 0 ≤ ∫ ω, stochX C β₁ β₂ α b t ω ∂μ := by
    induction t, ht using Nat.le_induction with
    | base =>
      have e : stochX C β₁ β₂ α b 1 = fun _ => 0 := funext fun ω => stochX_one C β₁ β₂ α b ω
      rw [e]
      exact ⟨integrable_const _, by simp⟩
    | succ t ht ih =>
      obtain ⟨hI, hD⟩ := hstep t ht
      have e : stochX C β₁ β₂ α b (t + 1) =
          fun ω => (stochX C β₁ β₂ α b (t + 1) ω - stochX C β₁ β₂ α b t ω) +
            stochX C β₁ β₂ α b t ω := funext fun ω => by ring
      rw [e, integral_add hI ih.1]
      exact ⟨hI.add ih.1, add_nonneg hD ih.2⟩
  refine ⟨key.1, ?_⟩
  simp only [one_mul]
  linarith [key.2]

/-- The hypotheses of Theorem 3 and of `stoch_step` are satisfiable
(`β₁ = 0`, `β₂ = 1/2`, `δ = 1`); `exists_isBernoulliSeq` supplies the coins
for every `p ∈ [0, 1]`. -/
example : (0 : ℝ) ≤ 0 ∧ (1 / 2 : ℝ) < 1 ∧ (0 : ℝ) < Real.sqrt (1 / 2) ∧ (0 : ℝ) < 1 :=
  ⟨le_rfl, by norm_num, Real.sqrt_pos.mpr (by norm_num), one_pos⟩

end AdamBeyond
end Transformer
