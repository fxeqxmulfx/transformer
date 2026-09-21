import Transformer.AdamBeyond.Section3_Run
import Mathlib.Probability.Independence.InfinitePi
import Mathlib.Probability.Distributions.Bernoulli

/-
# Adam and beyond — §3: the stochastic problem of Theorem 3

The problem of the proof of Theorem 3: the costs are i.i.d., `C x` with
probability `p = (1 + δ)/(C + 1)` and `-x` otherwise, so that the gradient is
`C` or `-1` whatever the iterate.  For one outcome of the coins the run is an
ordinary run of Adam, `stochSetup`; this file carries what holds outcome by
outcome: the gradients, the recursions of `m_t` and `v_t`, their closed forms,
and the step `x_{t+1} - x_t = -α_t m_t/√v_t`.

**What the source says and what is carried here.**

* The run is, as the proof of Theorem 3 assumes, without projection and from
  `x₁ = 0`; the source's footnote defers the projected case.
* The coins are an `IsBernoulliSeq`, which exists for every `p ∈ [0, 1]`
  (`exists_isBernoulliSeq`).

Source: arXiv:1904.09237, Appendix, proof of Theorem 3.
-/

open MeasureTheory ProbabilityTheory Finset

namespace Transformer
namespace AdamBeyond

open AMSGrad

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

/-- The hypotheses of `exists_isBernoulliSeq` are satisfiable: `p = 0`. -/
example : (0 : ℝ) ≤ 0 ∧ (0 : ℝ) ≤ 1 := ⟨le_rfl, zero_le_one⟩

/-- The gradient of the cost of a coin `c`: `C` if `c`, else `-1`.
arXiv:1904.09237, Appendix, proof of Theorem 3. -/
def coinGrad (C : ℝ) (c : Bool) : ℝ := if c then C else -1

/-- The run of Theorem 3 for the outcome `ω`: `f_t(x) = C x` if `b_t(ω)`, else
`-x`; no projection; `x₁ = 0`.  arXiv:1904.09237, Appendix, proof of Theorem 3. -/
noncomputable def stochSetup {Ω : Type*} (C β₁ β₂ : ℝ) (α : ℕ → ℝ) (b : ℕ → Ω → Bool)
    (ω : Ω) : Setup 1 :=
  ⟨fun _ y => y, fun t x => coinGrad C (b t ω) * x 0, α, fun _ => β₁, β₂, fun _ => 0⟩

/-- The iterate `x_t(ω)` of Adam on the stochastic problem. -/
noncomputable def stochX {Ω : Type*} (C β₁ β₂ : ℝ) (α : ℕ → ℝ) (b : ℕ → Ω → Bool) (t : ℕ)
    (ω : Ω) : ℝ :=
  (stochSetup C β₁ β₂ α b ω).x adamRule t 0

section Run

variable {Ω : Type*} (C β₁ β₂ : ℝ) (α : ℕ → ℝ) (b : ℕ → Ω → Bool) (ω : Ω)

/-- The gradient `g_t` is `C` or `-1`, whatever the iterate. -/
theorem stoch_g (t : ℕ) :
    (stochSetup C β₁ β₂ α b ω).g adamRule t 0 = coinGrad C (b t ω) := by
  simp only [Setup.g, stochSetup]
  rw [grad_linear]

/-- `m_{n+1} = β₁ m_n + (1 - β₁) g_{n+1}`. -/
theorem stoch_m_succ (n : ℕ) :
    (stochSetup C β₁ β₂ α b ω).m adamRule (n + 1) 0 =
      β₁ * (stochSetup C β₁ β₂ α b ω).m adamRule n 0 + (1 - β₁) * coinGrad C (b (n + 1) ω) := by
  rw [← stoch_g C β₁ β₂ α b ω]
  rfl

/-- `v_{n+1} = β₂ v_n + (1 - β₂) g²_{n+1}`. -/
theorem stoch_v_succ (n : ℕ) :
    (stochSetup C β₁ β₂ α b ω).v adamRule (n + 1) 0 =
      β₂ * (stochSetup C β₁ β₂ α b ω).v adamRule n 0
        + (1 - β₂) * coinGrad C (b (n + 1) ω) ^ 2 := by
  rw [← stoch_g C β₁ β₂ α b ω]
  rfl

/-- The step: `x_{n+2} - x_{n+1} = -α_{n+1} m_{n+1}/√v_{n+1}`, with no projection. -/
theorem stochX_succ (n : ℕ) :
    stochX C β₁ β₂ α b (n + 2) ω - stochX C β₁ β₂ α b (n + 1) ω =
      -(α (n + 1) * ((stochSetup C β₁ β₂ α b ω).m adamRule (n + 1) 0 /
        Real.sqrt ((stochSetup C β₁ β₂ α b ω).v adamRule (n + 1) 0))) := by
  change stochX C β₁ β₂ α b (n + 1) ω - α (n + 1) *
      ((stochSetup C β₁ β₂ α b ω).m adamRule (n + 1) 0 /
        Real.sqrt ((stochSetup C β₁ β₂ α b ω).v adamRule (n + 1) 0))
    - stochX C β₁ β₂ α b (n + 1) ω = _
  ring

/-- `x₁ = 0`. -/
theorem stochX_one : stochX C β₁ β₂ α b 1 ω = 0 := rfl

/-- `m_n = (1 - β₁) Σ_{j ≤ n} β₁^{n-j} g_j`. -/
theorem stoch_m_eq (n : ℕ) :
    (stochSetup C β₁ β₂ α b ω).m adamRule n 0 =
      (1 - β₁) * ∑ j ∈ Icc 1 n, β₁ ^ (n - j) * coinGrad C (b j ω) := by
  rw [m_sum (b := β₁) (fun _ => rfl)]
  simp only [stoch_g]

/-- `v_n = (1 - β₂) Σ_{j ≤ n} β₂^{n-j} g_j²`. -/
theorem stoch_v_eq (n : ℕ) :
    (stochSetup C β₁ β₂ α b ω).v adamRule n 0 =
      (1 - β₂) * ∑ j ∈ Icc 1 n, β₂ ^ (n - j) * coinGrad C (b j ω) ^ 2 := by
  rw [v_sum]
  simp only [stoch_g]
  rfl

end Run

end AdamBeyond
end Transformer
