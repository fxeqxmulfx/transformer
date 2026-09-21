import Transformer.AdamBeyond.Section3_Counter
import Mathlib.Probability.Independence.InfinitePi
import Mathlib.Probability.Distributions.Bernoulli

/-
# Adam and beyond — §3: Theorems 2 and 3, for every `β₁ < √β₂`

Theorem 2 extends the counterexample of Theorem 1 to every `β₁, β₂ ∈ [0, 1)`
with `β₁ < √β₂`, with the costs `C x` once every `C` steps and `-x` otherwise;
Theorem 3 makes the costs i.i.d.: `C x` with probability `p = (1 + δ)/(C + 1)`
and `-x` otherwise, whose mean `F(x) = δ x` is minimized at `x* = -1`, while
Adam drifts upward in expectation (the lemma of its proof, `stoch_step`).

**What the source says and what is carried here.**

* `β₂ ≥ 0` is not a hypothesis: it follows from `0 ≤ β₁ < √β₂`.

* Theorem 2 is stated for every `α > 0` in `α_t = α/√t`, which its proof
  handles uniformly; "`R_T/T ↛ 0`" is `c T ≤ R_T` for all large `T`, against a
  fixed `x* ∈ F`.  Its proof lower-bounds `v_{t+i-1}` by
  `(1-β₂)β₂^{i-2}C²` in the text and `(1-β₂)β₂^{i-1}C²` in the display; the
  statement is left open.

* Theorem 3 is run, as its proof assumes, without projection and from
  `x₁ = 0`; the source's footnote defers the projected case.  The coins are an
  `IsBernoulliSeq`, which exists for every `p ∈ [0, 1]`
  (`exists_isBernoulliSeq`).  "Adam does not converge" is carried as the
  proof's conclusion: `E[F(x_t)] - F(-1) ≥ δ` for every `t ≥ 1`.

* The lemma's `E[T₃]` bound reads `√(β₂(1+δ)C² + 1 - β₂)`, while its own
  estimate of `E[v_{t-1}]` is `≤ (1+δ)C`; the final display uses `C`, which the
  argument needs.  The lemma is stated for step sizes `α_t ≥ 0`.

Source: arXiv:1904.09237, §3, Theorems 2 and 3; Appendix, their proofs and
the lemma in the proof of Theorem 3.
-/

open Filter MeasureTheory ProbabilityTheory

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

/-- The hypotheses of Theorems 2 and 3 and of `stoch_step` are satisfiable
(`β₁ = 0`, `β₂ = 1/2`, `α = δ = 1`); `exists_isBernoulliSeq` supplies the coins
for every `p ∈ [0, 1]`, and its own hypotheses hold at `p = 0`. -/
example : (0 : ℝ) ≤ 0 ∧ (1 / 2 : ℝ) < 1 ∧ (0 : ℝ) < Real.sqrt (1 / 2) ∧ (0 : ℝ) < 1 :=
  ⟨le_rfl, by norm_num, Real.sqrt_pos.mpr (by norm_num), one_pos⟩

/-! ### The stochastic problem -/

/-- `b` is an i.i.d. sequence of coins, each `true` with probability `p`, under `μ`.
arXiv:1904.09237, Appendix, proof of Theorem 3. -/
structure IsBernoulliSeq {Ω : Type*} [MeasurableSpace Ω] (μ : Measure Ω) (b : ℕ → Ω → Bool)
    (p : ℝ) : Prop where
  /-- Each coin is a random variable. -/
  measurable : ∀ t, Measurable (b t)
  /-- The coins are independent. -/
  indep : iIndepFun b μ
  /-- Each coin is `true` with probability `p`. -/
  prob : ∀ t, μ {ω | b t ω = true} = ENNReal.ofReal p

/-- An i.i.d. sequence of Bernoulli(`p`) coins exists for every `p ∈ [0, 1]`. -/
theorem exists_isBernoulliSeq {p : ℝ} (h₀ : 0 ≤ p) (h₁ : p ≤ 1) :
    ∃ (μ : Measure (ℕ → Bool)), IsProbabilityMeasure μ ∧
      IsBernoulliSeq μ (fun t ω => ω t) p := by
  set P : Measure Bool := bernoulliMeasure true false ⟨p, h₀, h₁⟩
  refine ⟨Measure.infinitePi fun _ => P, inferInstance, fun t => measurable_pi_apply t,
    iIndepFun_infinitePi (X := fun _ => id) fun _ => measurable_id, fun t => ?_⟩
  have hs : {ω : ℕ → Bool | ω t = true} = (fun ω : ℕ → Bool => ω t) ⁻¹' {true} := rfl
  rw [hs, ← Measure.map_apply (measurable_pi_apply t) (measurableSet_singleton _)]
  have hm : Measure.map (fun ω : ℕ → Bool => ω t) (Measure.infinitePi fun _ => P) = P :=
    Measure.infinitePi_map_eval _ t
  rw [hm, bernoulliMeasure_apply_of_mem_of_notMem _ (measurableSet_singleton _) rfl (by simp)]
  simp [ENNReal.ofReal, unitInterval.toNNReal, Real.toNNReal_of_nonneg h₀]
  rfl

/-- The run of Theorem 3 for the outcome `ω`: `f_t(x) = C x` if `b_t(ω)`, else
`-x`; no projection; `x₁ = 0`.  arXiv:1904.09237, Appendix, proof of Theorem 3. -/
noncomputable def stochSetup {Ω : Type*} (C β₁ β₂ : ℝ) (α : ℕ → ℝ) (b : ℕ → Ω → Bool)
    (ω : Ω) : Setup 1 :=
  ⟨fun _ y => y, fun t x => (if b t ω then C else -1) * x 0, α, fun _ => β₁, β₂, fun _ => 0⟩

/-- The iterate `x_t(ω)` of Adam on the stochastic problem. -/
noncomputable def stochX {Ω : Type*} (C β₁ β₂ : ℝ) (α : ℕ → ℝ) (b : ℕ → Ω → Bool) (t : ℕ)
    (ω : Ω) : ℝ :=
  (stochSetup C β₁ β₂ α b ω).x adamRule t 0

/-- **The lemma of the proof of Theorem 3.**  For `C` large enough, depending on
`β₁`, `β₂`, `δ`, Adam's step `Δ_t = x_{t+1} - x_t` has `E[Δ_t] ≥ 0`.
arXiv:1904.09237, Appendix, proof of Theorem 3, Lemma. -/
theorem stoch_step {β₁ β₂ : ℝ} (hβ₁ : 0 ≤ β₁) (hβ₂ : β₂ < 1) (hγ : β₁ < Real.sqrt β₂) {δ : ℝ}
    (hδ : 0 < δ) : ∃ C₀ : ℝ, ∀ C ≥ C₀, ∀ (Ω : Type) [MeasurableSpace Ω] (μ : Measure Ω)
      [IsProbabilityMeasure μ] (b : ℕ → Ω → Bool), IsBernoulliSeq μ b ((1 + δ) / (C + 1)) →
      ∀ α : ℕ → ℝ, (∀ t, 0 ≤ α t) → ∀ t, 1 ≤ t →
        Integrable (fun ω => stochX C β₁ β₂ α b (t + 1) ω - stochX C β₁ β₂ α b t ω) μ ∧
        0 ≤ ∫ ω, (stochX C β₁ β₂ α b (t + 1) ω - stochX C β₁ β₂ α b t ω) ∂μ := by
  sorry

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
  sorry

end AdamBeyond
end Transformer
