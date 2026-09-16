/-
# Property: Output is a probability distribution

After softmax over the logits, the output for each token position is a
probability distribution over the vocabulary:
  - each entry is in `[0, 1]`,
  - the entries sum to `1`.

This is independent of the model parameters, depending only on `softmax`.
-/

import Transformer.GPTMini.Model
import Mathlib.Analysis.SpecialFunctions.Exp

open scoped BigOperators
open Real

namespace Transformer
namespace GPTMini
namespace Properties

variable (cfg : Config) (params : ModelParams cfg) (eps : ℝ)

/-- **Softmax output is non-negative.** -/
theorem softmaxOutput_nonneg
    {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size)
    (i : Fin T) (v : Fin cfg.vocab_size) :
    0 ≤ softmaxOutput cfg params eps positions tokens i v := by
  unfold softmaxOutput
  apply div_nonneg
  · exact le_of_lt (Real.exp_pos _)
  · apply Finset.sum_nonneg
    intros _ _
    exact le_of_lt (Real.exp_pos _)

/-- **Softmax sum is positive.** -/
theorem softmaxOutput_denom_pos
    {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) (i : Fin T)
    (hvocab : 0 < cfg.vocab_size) :
    0 < ∑ w : Fin cfg.vocab_size,
          Real.exp (forward cfg params eps positions tokens i w) := by
  have : Nonempty (Fin cfg.vocab_size) := ⟨⟨0, hvocab⟩⟩
  apply Finset.sum_pos
  · intros _ _; exact Real.exp_pos _
  · exact Finset.univ_nonempty

/-- **Softmax output is strictly positive.**

`exp` never vanishes, so no token is ever assigned probability exactly zero —
this is what makes `log (prob_i v)` well behaved in the entropy bounds. -/
theorem softmaxOutput_pos
    {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size)
    (i : Fin T) (v : Fin cfg.vocab_size)
    (hvocab : 0 < cfg.vocab_size) :
    0 < softmaxOutput cfg params eps positions tokens i v :=
  div_pos (Real.exp_pos _)
    (softmaxOutput_denom_pos cfg params eps positions tokens i hvocab)

/-- **Softmax output sums to 1.**

`Σ_v prob_i(v) = 1`. -/
theorem softmaxOutput_sum_one
    {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) (i : Fin T)
    (hvocab : 0 < cfg.vocab_size) :
    (∑ v : Fin cfg.vocab_size,
        softmaxOutput cfg params eps positions tokens i v) = 1 := by
  unfold softmaxOutput
  rw [← Finset.sum_div]
  exact div_self
    (ne_of_gt (softmaxOutput_denom_pos cfg params eps positions tokens i hvocab))

/-- **Softmax output is bounded above by 1.** -/
theorem softmaxOutput_le_one
    {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size)
    (i : Fin T) (v : Fin cfg.vocab_size)
    (hvocab : 0 < cfg.vocab_size) :
    softmaxOutput cfg params eps positions tokens i v ≤ 1 := by
  unfold softmaxOutput
  rw [div_le_one (softmaxOutput_denom_pos cfg params eps positions tokens i hvocab)]
  apply Finset.single_le_sum (f := fun w => Real.exp (forward cfg params eps positions tokens i w))
    (fun w _ => le_of_lt (Real.exp_pos _))
  exact Finset.mem_univ _

/-- **Output is a valid probability distribution.**

The softmax output `(prob_i(v))_v` is in the probability simplex:
non-negative and sums to one. -/
theorem softmaxOutput_is_distribution
    {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) (i : Fin T)
    (hvocab : 0 < cfg.vocab_size) :
    (∀ v : Fin cfg.vocab_size,
        0 ≤ softmaxOutput cfg params eps positions tokens i v)
    ∧
    (∑ v : Fin cfg.vocab_size,
        softmaxOutput cfg params eps positions tokens i v) = 1 := by
  refine ⟨?_, softmaxOutput_sum_one cfg params eps positions tokens i hvocab⟩
  intro v
  exact softmaxOutput_nonneg cfg params eps positions tokens i v

end Properties
end GPTMini
end Transformer
