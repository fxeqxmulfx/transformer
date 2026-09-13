/-
# Property: Output entropy is bounded below

The softmax output distribution has Shannon entropy bounded below by a
function of the depth `L`, the parameter norms, and the per-head
temperature `α_max`:

  `H(prob_i(·)) ≥ log(vocab_size) - C(L, ‖W_*‖, α_max)`.

The bound is non-vacuous when the parameter norms and temperatures are
not too large; in particular it implies that the output never degenerates
to a delta on a single token (avoids the "rank collapse" pathology).

This is an *anti-collapse* property: it ensures expressive output even at
deep layers.

The proof uses:
  - logits boundedness: `|logits_i(v)| ≤ M(params, α_max)` (from
    `forward_lipschitz_embedding` and the bounded residual stream)
  - softmax with bounded logits has positive entropy
-/

import Transformer.GPTMini.Model
import Transformer.GPTMini.Properties.OutputSimplex
import Mathlib.Analysis.SpecialFunctions.Log.Basic

open scoped BigOperators
open Real

namespace Transformer
namespace GPTMini
namespace Properties

variable (cfg : Config) (params : ModelParams cfg) (eps : ℝ)

/-- **Shannon entropy of the softmax output.**

  `H_i = -Σ_v prob_i(v) · log prob_i(v)`. -/
noncomputable def softmaxEntropy
    {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) (i : Fin T) : ℝ :=
  -∑ v : Fin cfg.vocab_size,
      let p := softmaxOutput cfg params eps positions tokens i v
      if p = 0 then 0 else p * Real.log p

/-- **Entropy is non-negative.**

For any probability distribution `(p_v)_v` with `p_v ∈ [0, 1]` and
`Σ p_v = 1`, `-Σ p_v log p_v ≥ 0`. -/
theorem softmaxEntropy_nonneg
    {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) (i : Fin T)
    (hvocab : 0 < cfg.vocab_size) :
    0 ≤ softmaxEntropy cfg params eps positions tokens i := by
  unfold softmaxEntropy
  rw [neg_nonneg]
  apply Finset.sum_nonpos
  intros v _
  by_cases hp : softmaxOutput cfg params eps positions tokens i v = 0
  · simp [hp]
  · simp [hp]
    apply mul_nonpos_of_nonneg_of_nonpos
      (softmaxOutput_nonneg cfg params eps positions tokens i v)
    apply Real.log_nonpos
    · exact softmaxOutput_nonneg cfg params eps positions tokens i v
    · exact softmaxOutput_le_one cfg params eps positions tokens i v hvocab

/-- **Entropy is at most `log(vocab_size)`** (maximum at uniform distribution). -/
theorem softmaxEntropy_le_log_vocab
    {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) (i : Fin T)
    (hvocab : 0 < cfg.vocab_size) :
    softmaxEntropy cfg params eps positions tokens i ≤ Real.log cfg.vocab_size := by
  sorry

/-- **Lower bound on entropy** (anti-collapse).

Given a uniform upper bound `M` on `|logits_i(v)|`, the entropy is bounded
below:

  `H_i ≥ log(vocab_size) - 2M`.

(This is a standard estimate: bounded logits give `prob_i(v) ≥
exp(-2M)/V`, so each term `-prob · log prob ≥ exp(-2M)/V · (2M + log V)`.) -/
theorem softmaxEntropy_lower_bound
    (M : ℝ) (hM : 0 < M)
    {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) (i : Fin T)
    (h_logits_bounded : ∀ v : Fin cfg.vocab_size,
        |forward cfg params eps positions tokens i v| ≤ M)
    (hvocab : 0 < cfg.vocab_size) :
    Real.log cfg.vocab_size - 2 * M ≤
      softmaxEntropy cfg params eps positions tokens i := by
  sorry

end Properties
end GPTMini
end Transformer
