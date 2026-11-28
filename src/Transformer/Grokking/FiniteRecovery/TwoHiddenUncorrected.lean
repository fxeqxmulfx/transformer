module

public import Transformer.Grokking.FiniteRecovery.TwoHiddenAnswers

/-! On the same identifiable three-by-three phase table, changing the
moment decays, weight decay, or schedule preserves the two-answer
obstruction for ordinary AdamW without bias correction. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.Argand.TwoHiddenAnswers

open MeasureTheory Set Filter
open scoped NNReal

local notation "Draw" => (ComplexParameter Token 4 16 6 → ℂ)

noncomputable def uncorrectedScore (a b ε decay : ℝ) (rate : ℕ → ℝ)
    (stream : ℕ → List (Fin 5)) (z : Draw) (t : ℕ) (x : Cell) : Token → ℝ :=
  (Model.fromWeights (uncorrectedAdamRun 1 a b ε decay (batch stream) rate z t).weights).logits 1 (prompt x) 0

theorem uncorrected_probe_bound (a b ε decay : ℝ) (hε : 0 < ε) (rate : ℕ → ℝ)
    (stream : ℕ → List (Fin 5)) (grid : Finset ℕ) :
    let μ := independentGaussianDraws (fun _ : ComplexParameter Token 4 16 6 => (1/50 : ℝ≥0)^2)
    let s := uncorrectedScore a b ε decay rate stream
    μ.real {z | ∃ t ∈ grid, StrictOn (fun k => s z t (probe k)) (label ∘ probe)} ≤ (grid.card : ℝ)/2756 := by
  apply equivariant_probe_bound
  · intro t p
    exact ((strict_prompt_weights_open 1 (prompt ∘ probe) (pairLabels p)).preimage
      ((EqualMoment.continuous_uncorrectedRun decay rate (fun j => batchGradient 1 (batch stream j)) a b
        hε (fun j => continuous_batchGradient 1 (batch stream j)) t).1.comp
        (continuous_sgdInitialWeights true))).measurableSet
  · intro e fix z t k c
    obtain ⟨pr,bs⟩ := fixed_data e fix stream
    simpa only [uncorrectedScore,bs,pr] using uncorrectedAdam_logits_relabel e 1 t a b ε decay rate
      (batch stream) z (prompt (probe k)) 0 c

/-- Simultaneous permanent hidden success has probability at most 1/2756
under the actual uncorrected update for any fixed observed minibatch stream. -/
theorem uncorrected_permanent_bound (a b ε decay : ℝ) (hε : 0 < ε) (rate : ℕ → ℝ)
    (stream : ℕ → List (Fin 5)) :
    let μ := independentGaussianDraws (fun _ : ComplexParameter Token 4 16 6 => (1/50 : ℝ≥0)^2)
    let s := uncorrectedScore a b ε decay rate stream
    μ.real {z | ∀ᶠ t in atTop, ∀ x ∉ range observedCell, ∀ c, c ≠ label x →
      s z t x c < s z t x (label x)} ≤ 1/2756 := by
  intro μ s
  apply eventual_success_mass_le μ (fun t => {z | ∀ x ∉ range observedCell, ∀ c, c ≠ label x →
      s z t x c < s z t x (label x)}) _ (by norm_num)
  intro t
  have sub : {z | ∀ x ∉ range observedCell, ∀ c, c ≠ label x → s z t x c < s z t x (label x)}
      ⊆ {z | StrictOn (fun k => s z t (probe k)) (label ∘ probe)} := fun _ hz => two_probes_strict hz
  exact (measureReal_mono (μ := μ) sub).trans (by simpa [μ,s] using uncorrected_probe_bound a b ε decay hε rate stream {t})

example (stream : ℕ → List (Fin 5)) :=
  uncorrected_permanent_bound (9/10) (19/20) (1/100000000) (1/10) (by norm_num) (fun t => 3*wsdRate 20000 t) stream

end Transformer.Grokking.FiniteRecovery.Argand.TwoHiddenAnswers
