/-
# Property: the output entropy, bounded by the embedding alone

`Properties.Entropy` bounds the entropy of the output below by
`log V - 2M` given a bound `M` on the logits.  Here `M` is discharged: the
final RMSNorm puts the representation in the ball of radius `√d_model`
(`final_representation_norm_le`), and the unembedding is tied to the
embedding, so every logit is at most `√d_model · max_v ‖E_v‖` in absolute
value.  Hence

  `H(prob_i) ≥ log V - 2 √d_model · max_v ‖E_v‖`

whatever the depth, the blocks, the temperatures and the input: only the
embedding table enters.

Source: `reference/model.py` (`GPTMini.forward`: `norm_final`, the tied
`unembed`).
-/

import Transformer.GPTMini.Properties.Entropy
import Transformer.GPTMini.Properties.StreamGrowth

open scoped BigOperators
open Real

namespace Transformer
namespace GPTMini
namespace Properties

variable (cfg : Config) (params : ModelParams cfg) (eps : ℝ)

/-- **Every logit is bounded by the embedding.**  `|logit_i(v)| ≤ √d_model R`
as soon as every embedding vector has norm at most `R`.

Source: `reference/model.py` (`GPTMini.forward`, tied `unembed`). -/
theorem abs_forward_le (heps : 0 < eps) (R : ℝ)
    (hR : ∀ v : Fin cfg.vocab_size, ‖params.embedding v‖ ≤ R)
    {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) (i : Fin T) (v : Fin cfg.vocab_size) :
    |forward cfg params eps positions tokens i v| ≤ Real.sqrt (cfg.d_model : ℝ) * R :=
  (abs_real_inner_le_norm _ _).trans
    (mul_le_mul (final_representation_norm_le cfg params eps heps positions tokens i)
      (hR v) (norm_nonneg _) (Real.sqrt_nonneg _))

/-- **The output never collapses, at any depth.**  With every embedding vector
of norm at most `R`,

  `H(prob_i) ≥ log V - 2 √d_model R`.

Source: `reference/model.py` (`GPTMini.forward`); the entropy estimate is
`softmaxEntropy_lower_bound`. -/
theorem softmaxEntropy_ge_of_embedding (heps : 0 < eps) (R : ℝ)
    (hR : ∀ v : Fin cfg.vocab_size, ‖params.embedding v‖ ≤ R)
    {T : ℕ} (positions : Fin T → ℝ)
    (tokens : Fin T → Fin cfg.vocab_size) (i : Fin T)
    (hvocab : 0 < cfg.vocab_size) :
    Real.log cfg.vocab_size - 2 * (Real.sqrt (cfg.d_model : ℝ) * R) ≤
      softmaxEntropy cfg params eps positions tokens i :=
  softmaxEntropy_lower_bound cfg params eps _ positions tokens i
    (abs_forward_le cfg params eps heps R hR positions tokens i) hvocab

/-- The hypotheses are satisfiable for every parameter set: `eps = 10⁻⁶` of
`reference/model.py`, `R = Σ_v ‖E_v‖`, and a positive vocabulary. -/
example (params : ModelParams Config.default) :
    (0 : ℝ) < 1e-6 ∧
      (∀ v, ‖params.embedding v‖ ≤ ∑ w, ‖params.embedding w‖) ∧
      0 < Config.default.vocab_size :=
  ⟨by norm_num,
    fun v => Finset.single_le_sum (f := fun w => ‖params.embedding w‖)
      (fun _ _ => norm_nonneg _) (Finset.mem_univ v),
    Config.default.vocab_pos⟩

end Properties
end GPTMini
end Transformer
