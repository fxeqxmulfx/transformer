module

public import Transformer.Grokking.FiniteRecovery.ArgandFirstNoise
public import Mathlib.MeasureTheory.Measure.Continuity

/-! A small first-update perturbation preserves each strict probability
threshold attained by ordinary AdamW. The same outcome specifies the
native initialization, minibatch order and standardized noise across
perturbation strengths. No independence between these inputs is needed
for this comparison. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery

open Filter MeasureTheory Set
open scoped ENNReal Topology

/-- If every success in a reference event eventually succeeds in a
sequence of trials on the same probability space, every threshold strictly
below its measure is eventually exceeded by the trial measures. -/
theorem eventually_preserves_mass {D : Type*} [MeasurableSpace D] (μ : Measure D) (U : Set D)
    (V : ℕ → Set D) (hs : ∀ x ∈ U, ∀ᶠ n in atTop, x ∈ V n)
    {q : ℝ≥0∞} (hq : q < μ U) : ∀ᶠ n in atTop, q < μ (V n) := by
  let tails (n : ℕ) : Set D := {x | ∀ k ≥ n, x ∈ V k}
  have hm : Monotone tails := fun _ _ h _ hx k hk => hx k (h.trans hk)
  have hcover : U ⊆ ⋃ n, tails n := by
    intro x hx
    obtain ⟨n,hn⟩ := eventually_atTop.mp (hs x hx)
    exact mem_iUnion.mpr ⟨n,hn⟩
  have hlim := tendsto_measure_iUnion_atTop (μ := μ) hm
  filter_upwards [hlim.eventually (lt_mem_nhds (hq.trans_le (measure_mono hcover)))] with n hn
  exact hn.trans_le (measure_mono fun x hx => (hx : x ∈ tails n) n le_rfl)

namespace Argand

variable {A X D : Type*} [Fintype A] [Finite X] [MeasurableSpace D] {H d L T : ℕ}

/-- For any sequence of noise amplitudes tending to zero, the probability
of strict recovery eventually exceeds any floor strictly below the ordinary
one-run probability. In particular the floor can be one half or three
quarters when the stock probability exceeds that floor. Each seed may
choose its own fixed minibatch stream, and X can be the hidden corpus. -/
theorem firstNoise_preserves_probability (μ : Measure D) (r n : ℕ) (rate : ℕ → ℝ)
    (batch : D → ℕ → List (LabeledPosition A)) (prompt : X → Word A (T+1)) (f : X → A)
    (draw : D → ComplexParameter A H d L → ℂ) (noise : D → Weights A H d L)
    (scale : ℕ → ℝ) (hs : Tendsto scale atTop (𝓝 0)) {q : ℝ≥0∞}
    (hstock : q < μ {s | StrictOn (fun x => (Model.fromWeights (adamWRun r (batch s) rate
      (draw s) (n+1)).weights).logits r (prompt x) (Fin.last T)) f}) :
    ∀ᶠ k in atTop, q < μ {s | StrictOn (firstNoiseScores r (n+1) rate (batch s) prompt
      (draw s) (scale k • noise s)) f} := by
  apply eventually_preserves_mass μ _ _ _ hstock
  intro s h
  have ht : Tendsto (fun k => scale k • noise s) atTop (𝓝 0) := by
    simpa only [zero_smul] using hs.smul_const (noise s)
  exact ht.eventually (firstNoise_preserves_strict_near_zero r n rate (batch s) prompt f (draw s) h)

end Argand

example (U : Set Bool) (μ : Measure Bool) {q : ℝ≥0∞} (h : q < μ U) :=
  eventually_preserves_mass μ U (fun _ => U) (fun _ hx => Eventually.of_forall (fun _ => hx)) h

end Transformer.Grokking.FiniteRecovery
