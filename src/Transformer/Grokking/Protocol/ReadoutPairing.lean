module

public import Transformer.Grokking.Protocol.Decision
public import Transformer.Grokking.Protocol.ReadoutGradient

/-!
# The readout paired with its gradient, and what a margin makes of it

Source: openai/grok at 3d64b1d8, `grok/transformer.py`, `Transformer` (the readout
`self.linear`), and `grok/training.py`, `TrainableTransformer._step`;
`experiments/grok_division/PROTOCOL.md`, *The model* and *The loss and the accuracy*.

The logits are linear in the readout, and by `sourceGrad_out` the readout paired with its
gradient, the sum over the readout of parameter times gradient, is the mean over the targets of

    expected logit under the softmax − logit of the target

(`sum_out_mul_sourceGrad`). Where the target leads every other token by a margin `μ ≥ 0` and
the logits are at most `Z` in absolute value, that difference is at most `− μ e^{−2Z} / V`
for `V` tokens (`expected_logit_le_of_margin`): every other token has a probability of at
least `e^{−2Z} / V`. So on a batch that is fitted with the margin `μ` the pairing is at most
`− μ e^{−2Z} / V` (`sum_out_mul_sourceGrad_le_of_margin`), a negative number when `μ > 0`
that stays away from `0` as long as the logits are bounded.

This is a statement about the gradient of one batch at given parameters; `ReadoutStationary`
uses it along a run.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Transformer.Landscape.Neural (sum_exp_pos)
open Transformer.Grokking.Geometry (probability probability_pos sum_probability)

variable {S : Sizes}

/-- **The readout paired with its gradient** is the mean over the targets of the expected
logit less the logit of the target. -/
theorem sum_out_mul_sourceGrad (B : List (Example S)) (θ : Param S → ℝ) :
    ∑ c, ∑ i, θ (.out c i) * sourceGrad B θ (.out c i) = targetMean B fun e pc =>
      ∑ c, probability (logits θ (relu S) e.tokens pc.1) c * logits θ (relu S) e.tokens pc.1 c -
        logits θ (relu S) e.tokens pc.1 pc.2 := by
  simp only [sourceGrad_out, sum_mul_targetMean, sum_targetMean]
  congr 1
  funext e pc
  have h : ∀ c, ∑ i, θ (.out c i) * (decodeTo θ (relu S) e.tokens S.blocks pc.1 i *
      (probability (logits θ (relu S) e.tokens pc.1) c - if c = pc.2 then 1 else 0)) =
      logits θ (relu S) e.tokens pc.1 c *
        (probability (logits θ (relu S) e.tokens pc.1) c - if c = pc.2 then 1 else 0) :=
    fun c => by simp only [logits, Finset.sum_mul, mul_assoc]
  rw [Finset.sum_congr rfl fun c _ => h c]
  simp only [mul_sub, Finset.sum_sub_distrib, mul_ite, mul_one, mul_zero, Finset.sum_ite_eq',
    Finset.mem_univ, ite_true]
  congr 1
  exact Finset.sum_congr rfl fun c _ => mul_comm _ _

/-- **A margin keeps the expected logit below the logit of the target**: if the target leads
every other class by `μ ≥ 0`, the logits are at most `Z` in absolute value and there is
another class, the expected logit is below the target's by at least `μ e^{−2Z} / V`. -/
theorem expected_logit_le_of_margin {V : Type*} [Fintype V] {z : V → ℝ} {y : V} {μ Z : ℝ}
    (hμ : 0 ≤ μ) (hm : ∀ c, c ≠ y → μ ≤ z y - z c) (hZ : ∀ c, |z c| ≤ Z) (hV : ∃ c, c ≠ y) :
    ∑ c, probability z c * z c - z y ≤ -(μ * Real.exp (-(2 * Z)) / Fintype.card V) := by
  obtain ⟨c₀, hc₀⟩ := hV
  have hsum : ∑ c, probability z c * (z y - z c) = z y - ∑ c, probability z c * z c := by
    simp only [mul_sub, Finset.sum_sub_distrib, ← Finset.sum_mul, sum_probability z y, one_mul]
  have hterm : ∀ c ∈ (Finset.univ : Finset V), 0 ≤ probability z c * (z y - z c) := fun c _ => by
    rcases eq_or_ne c y with rfl | hc
    · simp
    · exact mul_nonneg (probability_pos z c).le (hμ.trans (hm c hc))
  have h1 : probability z c₀ * (z y - z c₀) ≤ ∑ c, probability z c * (z y - z c) :=
    Finset.single_le_sum hterm (Finset.mem_univ c₀)
  have hcard : (0 : ℝ) < Fintype.card V := Nat.cast_pos.2 (Fintype.card_pos_iff.2 ⟨y⟩)
  have hp : Real.exp (-(2 * Z)) / Fintype.card V ≤ probability z c₀ := by
    have hden : ∑ q, Real.exp (z q) ≤ Fintype.card V * Real.exp Z :=
      (Finset.sum_le_sum fun q _ => Real.exp_le_exp.2 (le_of_abs_le (hZ q))).trans_eq
        (by rw [Finset.sum_const, Finset.card_univ, nsmul_eq_mul])
    have hnum : Real.exp (-Z) ≤ Real.exp (z c₀) := Real.exp_le_exp.2 (neg_le_of_abs_le (hZ c₀))
    have hone : Real.exp (-Z) * Real.exp Z = 1 := by
      rw [← Real.exp_add, neg_add_cancel, Real.exp_zero]
    rw [probability, div_le_div_iff₀ hcard (sum_exp_pos z y)]
    calc Real.exp (-(2 * Z)) * ∑ q, Real.exp (z q)
        ≤ Real.exp (-(2 * Z)) * (Fintype.card V * Real.exp Z) :=
          mul_le_mul_of_nonneg_left hden (Real.exp_pos _).le
      _ = Real.exp (-Z) * Fintype.card V := by
          rw [show -(2 * Z) = -Z + -Z by ring, Real.exp_add]
          linear_combination (Real.exp (-Z) * (Fintype.card V : ℝ)) * hone
      _ ≤ Real.exp (z c₀) * Fintype.card V := mul_le_mul_of_nonneg_right hnum hcard.le
  have h2 : μ * Real.exp (-(2 * Z)) / Fintype.card V ≤ probability z c₀ * (z y - z c₀) := by
    rw [mul_div_assoc, mul_comm]
    exact mul_le_mul hp (hm c₀ hc₀) hμ (probability_pos z c₀).le
  linarith

/-- **On a batch fitted with a margin the readout pairs negatively with its gradient**: if
every target of the batch leads by `μ ≥ 0` and the logits at its positions are at most `Z` in
absolute value, the pairing is at most `− μ e^{−2Z} / vocab`. -/
theorem sum_out_mul_sourceGrad_le_of_margin (hV : 2 ≤ S.vocab) {B : List (Example S)}
    (hN : 0 < targetCount B) {θ : Param S → ℝ} {μ Z : ℝ} (hμ : 0 ≤ μ)
    (hZ : ∀ e ∈ B, ∀ pc ∈ e.targets, ∀ c, |logits θ (relu S) e.tokens pc.1 c| ≤ Z)
    (hfit : ∀ e ∈ B, ∀ pc ∈ e.targets, μ ≤ margin (logits θ (relu S) e.tokens pc.1) pc.2) :
    ∑ c, ∑ i, θ (.out c i) * sourceGrad B θ (.out c i) ≤
      -(μ * Real.exp (-(2 * Z)) / S.vocab) := by
  have hnt : Nontrivial (Fin S.vocab) := Fin.nontrivial_iff_two_le.2 hV
  rw [sum_out_mul_sourceGrad]
  refine targetMean_le hN fun e he pc hpc => ?_
  simpa using expected_logit_le_of_margin hμ
    (fun c hc => (hfit e he pc hpc).trans (margin_le _ hc)) (hZ e he pc hpc) (exists_ne pc.2)

/-! The witness: two classes with the logits `2` and `0`, the first the target. -/

example : ∑ c, probability ![(2 : ℝ), 0] c * ![(2 : ℝ), 0] c - ![(2 : ℝ), 0] 0 ≤
    -(2 * Real.exp (-(2 * 2)) / Fintype.card (Fin 2)) :=
  expected_logit_le_of_margin (by norm_num)
    (fun c hc => by fin_cases c <;> simp_all)
    (fun c => by fin_cases c <;> simp) ⟨1, by decide⟩

end Transformer.Grokking.Protocol
