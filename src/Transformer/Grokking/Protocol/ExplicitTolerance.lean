module

public import Transformer.Grokking.Protocol.EventualSuccess
public import Transformer.Grokking.Protocol.LabelBound

/-!
# An explicit bound on the probability that a run eventually answers every held-out equation

Source: arXiv:2201.02177v1 (Power et al.), Figure 1; `experiments/grok_division/PROTOCOL.md`,
*The split* and *Randomness*; `review/grok_division_convergence.md`, section 2. The paper
states no probability; the event is `eventuallyRight` of
`Transformer.Grokking.Protocol.EventualSuccess`.

`EventualSuccess` bounds the probability of the event by `1` less half the probability that
the split is not labelled, and `LabelBound` puts that probability above `7 · 10⁻³⁰`. So, under
Adam and under AdamW at every weight decay, a run eventually answers every held-out equation
with a probability of at most `1 - 7 · 10⁻³⁰ / 2` (`law_eventuallyRight_le_explicit`), and the
target with a tolerance below `7 · 10⁻³⁰ / 2` is false
(`not_le_law_eventuallyRight_explicit`). A true statement about this protocol with no
condition on the split has a tolerance of at least that number.

The number is not sharp: the probability that a split is not labelled is about `7.5 · 10⁻²⁸`
by a count that is not formalized here, a hundred times the bound used. Nothing here bounds
the probability of the event from below.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open MeasureTheory

/-- **A run eventually answers every held-out equation with a probability of at most
`1 - 7 · 10⁻³⁰ / 2`**, under Adam and under AdamW at every weight decay. -/
theorem law_eventuallyRight_le_explicit (decay : ℝ) :
    law (97 * 96) 4656 (initMeasure paperSizes) (eventuallyRight decay) ≤
      1 - 7 / 10 ^ 30 / 2 :=
  (law_eventuallyRight_le decay).trans
    (tsub_le_tsub_left (ENNReal.div_le_div_right le_uniformPerm_compl_labelled 2) 1)

/-- **The target with a tolerance below `7 · 10⁻³⁰ / 2` is false**: the runs that eventually
answer every held-out equation do not have probability `1 - δ` or more. -/
theorem not_le_law_eventuallyRight_explicit (decay : ℝ) {δ : ENNReal}
    (hδ : δ < 7 / 10 ^ 30 / 2) :
    ¬ 1 - δ ≤ law (97 * 96) 4656 (initMeasure paperSizes) (eventuallyRight decay) :=
  not_le_law_eventuallyRight decay
    (hδ.trans_le (ENNReal.div_le_div_right le_uniformPerm_compl_labelled 2))

/-! The witness: the tolerance `10⁻³⁰`, which is below `7 · 10⁻³⁰ / 2`. -/

example (decay : ℝ) : ¬ 1 - 1 / 10 ^ 30 ≤
    law (97 * 96) 4656 (initMeasure paperSizes) (eventuallyRight decay) := by
  refine not_le_law_eventuallyRight_explicit decay ?_
  rw [ENNReal.lt_div_iff_mul_lt (Or.inl two_ne_zero) (Or.inl ENNReal.ofNat_ne_top),
    ← ENNReal.mul_div_right_comm, one_mul]
  exact ENNReal.div_lt_div_right (pow_ne_zero _ (by norm_num))
    (ENNReal.pow_ne_top (by norm_num)) (by norm_num)

end Transformer.Grokking.Protocol
