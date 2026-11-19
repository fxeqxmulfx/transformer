module

public import Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Hilbert.Domain

/-!
# Finite original-class queries from mixtures of both learned branches

Source: `review/convex/moments.md`, CM107. CM98's actual nonzero bias components
remain available even when the current model contains Hilbert components.
The exact full-class logits and finite CE identity are still CM98's formulas.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Hilbert

variable {O : Type*} [DecidableEq O]

/-- CM107: preserve the original class probes as actual enlarged-family points. -/
theorem bias_mem (B : ℝ) (target : O) (s : ℝ) (hs : |s| ≤ 1) :
    B • Rotary.score (Rotary.bias target s) ∈ domain O B :=
  old_domain_subset B (Rotary.generator_mem B _ (Rotary.bias_unit target s hs))

/-- CM107: the unchanged full-class finite query is feasible from either branch. -/
theorem query_feasible (B eta : ℝ) (r : List O → O → ℝ) (target : O) (s : ℝ)
    (hr : r ∈ domain O B) (hs : |s| ≤ 1) (he : 0 ≤ eta) (he1 : eta ≤ 1) :
    Rotary.query B eta r target s ∈ domain O B :=
  domain_convex B hr (bias_mem B target s hs) (by linarith) he (by ring)

example : (2 : ℝ) • Rotary.score (Rotary.bias true (-1)) ∈ domain Bool 2 :=
  bias_mem _ _ _ (by norm_num)

example : Rotary.query 2 (1 / 2)
    ((2 : ℝ) • componentScore (.inr (witness (3 / 5) (4 / 5)))) true (-1) ∈
      domain Bool 2 :=
  query_feasible _ _ _ _ _
    (generator_mem 2 (.inr (witness (3 / 5) (4 / 5))) witness_unit)
    (by norm_num) (by norm_num) (by norm_num)

example (xs : List Bool) (c : Bool) : Rotary.query 2 (1 / 2)
    ((2 : ℝ) • componentScore (.inr (witness (3 / 5) (4 / 5)))) true (-1) xs c =
      score (witness (3 / 5) (4 / 5)) xs c + if c = true then -1 else 0 := by
  rw [Rotary.query_logits]
  simp only [componentScore, Pi.smul_apply, smul_eq_mul]
  split_ifs <;> ring

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedRecurrent.Hilbert
