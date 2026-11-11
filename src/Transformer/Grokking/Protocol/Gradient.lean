module

public import Transformer.Grokking.Protocol.Model
public import Transformer.Landscape.Neural.Loss
public import Mathlib.Analysis.Calculus.Deriv.Basic

/-!
# The loss of a batch and the gradient that the source's backward pass returns

Source: openai/grok at 3d64b1d8, `grok/training.py`, `TrainableTransformer._step`:
`F.cross_entropy(y_hat_rhs, y_rhs, reduction="mean")` over the positions to the right of `=`,
followed by PyTorch's backward pass, whose rule for `relu` passes the incoming gradient where
the unit's input is positive and passes `0` elsewhere, at input `0` too;
`experiments/grok_division/PROTOCOL.md`, *The loss and the accuracy*. arXiv:2201.02177v1
(Power et al.), appendix *Additional experimental details*, says only that loss and accuracy
are calculated on the answer part of the equation.

An example (`Example`) is the tokens read and the positions trained on with their targets.
The loss of a batch (`batchLoss`) is the mean of `crossEntropy` over all the targets of the
batch, as `reduction="mean"` computes it; it is nonnegative (`batchLoss_nonneg`).

**The derivative convention.** The loss is not differentiable in the parameters where the
input of a hidden unit is `0`, and the source does not differentiate it there: its backward
pass differentiates the computation in which every hidden unit is the linear map chosen by
the sign of its input at the current parameters. `gate θ₀ tokens` is that choice: a unit passes
its input if the input is positive at `θ₀` and passes `0` otherwise. With the gates frozen the
only operations left are sums, products, exponentials, the logarithm of a positive sum, and
quotients by the positive roots and sums of `layerNorm_root_pos` and `sum_weight`. At `θ₀`
itself the frozen pass is the model's (`decodeTo_gate`, `logits_gate`), so the frozen loss
(`frozenLoss`) is the loss there (`frozenLoss_self`). The gradient of the backward pass
(`sourceGrad`) is, coordinate by coordinate, the derivative of the frozen loss. This is the
source's rule at every parameter, the exceptional ones included; no claim is made here that it
is a derivative of `batchLoss` at them.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Transformer.Landscape.Neural (crossEntropy crossEntropy_nonneg)

variable {S : Sizes}

/-- **The gates of the backward pass** at the parameters `θ₀` for the tokens read: a hidden
unit passes its input if that input is positive at `θ₀`, and passes `0` otherwise. -/
noncomputable def gate (θ₀ : Param S → ℝ) (tokens : Fin S.context → Fin S.vocab) : Activation S :=
  fun l t u v => if 0 < preactivation θ₀ (decodeTo θ₀ (relu S) tokens l) l t u then v else 0

lemma ite_pos_eq_max (v : ℝ) : (if 0 < v then v else 0) = max 0 v := by
  split_ifs with hv
  · exact (max_eq_right hv.le).symm
  · exact (max_eq_left (not_lt.1 hv)).symm

/-- **At the parameters where they are frozen the gates change nothing**: every block's output
is the model's. -/
theorem decodeTo_gate (θ : Param S → ℝ) (tokens : Fin S.context → Fin S.vocab) (n : ℕ) :
    decodeTo θ (gate θ tokens) tokens n = decodeTo θ (relu S) tokens n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [decodeTo]
    split_ifs with hn
    · funext t
      rw [ih]
      simp only [block, gate, relu, ite_pos_eq_max]
    · exact ih

/-- The logits of the frozen pass at the parameters where it is frozen are the model's. -/
theorem logits_gate (θ : Param S → ℝ) (tokens : Fin S.context → Fin S.vocab) :
    logits θ (gate θ tokens) tokens = logits θ (relu S) tokens := by
  funext t c
  simp only [logits, decodeTo_gate]

/-- A supervised example. -/
structure Example (S : Sizes) where
  /-- The tokens read. -/
  tokens : Fin S.context → Fin S.vocab
  /-- The positions trained on, each with its target. -/
  targets : List (Fin S.context × Fin S.vocab)

/-- The cross-entropies of an example's targets, summed, with the activations `act`. -/
noncomputable def exampleLoss (θ : Param S → ℝ) (act : Activation S) (e : Example S) : ℝ :=
  (e.targets.map fun pc => crossEntropy (logits θ act e.tokens pc.1) pc.2).sum

/-- The number of targets of a batch. -/
def targetCount (B : List (Example S)) : ℕ := (B.map fun e => e.targets.length).sum

/-- **The source's loss of a batch**: the mean cross-entropy over all its targets. -/
noncomputable def batchLoss (θ : Param S → ℝ) (B : List (Example S)) : ℝ :=
  (B.map (exampleLoss θ (relu S))).sum / targetCount B

/-- **The loss of a batch with the gates frozen at `θ₀`**: every example's hidden units have
the gates of `θ₀` for its tokens. -/
noncomputable def frozenLoss (θ₀ θ : Param S → ℝ) (B : List (Example S)) : ℝ :=
  (B.map fun e => exampleLoss θ (gate θ₀ e.tokens) e).sum / targetCount B

/-- **The frozen loss at the parameters where it is frozen is the loss.** -/
theorem frozenLoss_self (θ : Param S → ℝ) (B : List (Example S)) :
    frozenLoss θ θ B = batchLoss θ B := by
  have h : ∀ e : Example S, exampleLoss θ (gate θ e.tokens) e = exampleLoss θ (relu S) e :=
    fun e => by simp only [exampleLoss, logits_gate]
  simp only [frozenLoss, batchLoss, h]

/-- **The gradient that the source's backward pass returns** at `θ` for a batch: its coordinate
`p` is the derivative in that coordinate of the loss with the gates frozen at `θ`. -/
noncomputable def sourceGrad (B : List (Example S)) (θ : Param S → ℝ) (p : Param S) : ℝ :=
  deriv (fun s => frozenLoss θ (Function.update θ p s) B) (θ p)

lemma exampleLoss_nonneg (θ : Param S → ℝ) (act : Activation S) (e : Example S) :
    0 ≤ exampleLoss θ act e :=
  List.sum_nonneg fun _ hx => by
    obtain ⟨pc, -, rfl⟩ := List.mem_map.1 hx
    exact crossEntropy_nonneg _ _

/-- **The loss of a batch is nonnegative.** -/
theorem batchLoss_nonneg (θ : Param S → ℝ) (B : List (Example S)) : 0 ≤ batchLoss θ B :=
  div_nonneg (List.sum_nonneg fun _ hx => by
    obtain ⟨e, -, rfl⟩ := List.mem_map.1 hx
    exact exampleLoss_nonneg θ _ e) (Nat.cast_nonneg _)

end Transformer.Grokking.Protocol
