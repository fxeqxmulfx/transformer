module

public import Transformer.Grokking.Protocol.Continuity
public import Transformer.Grokking.Protocol.Differentiability
public import Transformer.Grokking.Protocol.Division

/-!
# Off the kinks the gradient of the backward pass is the gradient of the loss

Source: openai/grok at 3d64b1d8, `grok/training.py`, `TrainableTransformer._step`, and
PyTorch's backward pass, whose rule for `relu` passes the incoming gradient where the unit's
input is positive and `0` elsewhere; `experiments/grok_division/PROTOCOL.md`, *The loss and
the accuracy*; `review/grok_division_convergence.md`, bridge 3.

`sourceGrad` differentiates the loss with the gates frozen at the current parameters. Call
parameters *off the kinks* for a row of tokens when no hidden unit has the input `0` there
(`OffKinks`). The inputs of the hidden units are continuous in the parameters (`Continuity`),
so off the kinks they keep their signs along a coordinate, near the parameters: the gates
are locally constant (`eventually_gate_update_eq`), and the loss with frozen gates is the loss
(`eventually_frozenLoss_eq`). Hence off the kinks of a batch the loss of the batch has a
partial derivative in every coordinate, and it is the coordinate of `sourceGrad`
(`hasDerivAt_batchLoss_sourceGrad`, `sourceGrad_eq_deriv_batchLoss`).

At a kink the two may differ, and the source's rule is `sourceGrad` there too. Nothing is
proved here of how often a run meets a kink. The witnesses: with every parameter `1` each
hidden unit has the input `width` (`preactivation_one`: the normalized entries sum to zero and
the biases to `width`), so those parameters are off the kinks; with every parameter `0` every
hidden unit has the input `0`.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Filter Topology

variable {S : Sizes}

/-- **Off the kinks**: no hidden unit has the input `0` at these parameters, for the tokens
read. -/
def OffKinks (θ : Param S → ℝ) (tokens : Fin S.context → Fin S.vocab) : Prop :=
  ∀ (l : Fin S.blocks) (t : Fin S.context) (u : Fin S.hidden),
    preactivation θ (decodeTo θ (relu S) tokens l) l t u ≠ 0

/-- **Off the kinks the gates are locally constant along a coordinate.** -/
lemma eventually_gate_update_eq {θ : Param S → ℝ} {tokens : Fin S.context → Fin S.vocab}
    (h : OffKinks θ tokens) (p : Param S) :
    ∀ᶠ s in 𝓝 (θ p), gate (Function.update θ p s) tokens = gate θ tokens := by
  have hline : ∀ q, Continuous fun s : ℝ => Function.update θ p s q := fun q =>
    (differentiable_update θ p q).continuous
  have hpre : ∀ (l : Fin S.blocks) (t : Fin S.context) (u : Fin S.hidden),
      Continuous fun s : ℝ => preactivation (Function.update θ p s)
        (decodeTo (Function.update θ p s) (relu S) tokens l) l t u := fun l t u =>
    continuous_preactivation hline (continuous_decodeTo continuous_relu hline tokens l) l t u
  have hsign : ∀ (l : Fin S.blocks) (t : Fin S.context) (u : Fin S.hidden),
      ∀ᶠ s in 𝓝 (θ p), (0 < preactivation (Function.update θ p s)
        (decodeTo (Function.update θ p s) (relu S) tokens l) l t u ↔
        0 < preactivation θ (decodeTo θ (relu S) tokens l) l t u) := fun l t u => by
    have hat := (hpre l t u).continuousAt (x := θ p)
    simp only [ContinuousAt, Function.update_eq_self] at hat
    rcases lt_or_gt_of_ne (h l t u) with hneg | hpos
    · filter_upwards [hat.eventually (gt_mem_nhds hneg)] with s hs
      exact ⟨fun h' => absurd h' (not_lt.2 hs.le), fun h' => absurd h' (not_lt.2 hneg.le)⟩
    · filter_upwards [hat.eventually (lt_mem_nhds hpos)] with s hs
      exact ⟨fun _ => hpos, fun _ => hs⟩
  filter_upwards [eventually_all.2 fun l => eventually_all.2 fun t =>
    eventually_all.2 fun u => hsign l t u] with s hs
  funext l t u v
  simp only [gate, hs l t u]

/-- **Off the kinks of a batch the loss with frozen gates is the loss**, along a coordinate
and near the parameters. -/
lemma eventually_frozenLoss_eq {B : List (Example S)} {θ : Param S → ℝ}
    (h : ∀ e ∈ B, OffKinks θ e.tokens) (p : Param S) :
    (fun s => frozenLoss θ (Function.update θ p s) B) =ᶠ[𝓝 (θ p)]
      fun s => batchLoss (Function.update θ p s) B := by
  have hall : ∀ᶠ s in 𝓝 (θ p), ∀ e ∈ {e | e ∈ B},
      gate (Function.update θ p s) e.tokens = gate θ e.tokens :=
    (eventually_all_finite (List.finite_toSet B)).2 fun e he => eventually_gate_update_eq (h e he) p
  filter_upwards [hall] with s hs
  simp only [frozenLoss, batchLoss]
  congr 1
  refine congrArg List.sum (List.map_congr_left fun e he => ?_)
  rw [← hs e he]
  simp only [exampleLoss, logits_gate]

/-- **Off the kinks the gradient of the backward pass is the gradient of the loss**: the loss
of the batch has a partial derivative in every coordinate, the coordinate of `sourceGrad`. -/
theorem hasDerivAt_batchLoss_sourceGrad {B : List (Example S)} {θ : Param S → ℝ}
    (h : ∀ e ∈ B, OffKinks θ e.tokens) (p : Param S) :
    HasDerivAt (fun s => batchLoss (Function.update θ p s) B) (sourceGrad B θ p) (θ p) :=
  (hasDerivAt_frozenLoss_sourceGrad B θ p).congr_of_eventuallyEq (eventually_frozenLoss_eq h p).symm

/-- Off the kinks `sourceGrad` is the partial derivative of the loss. -/
theorem sourceGrad_eq_deriv_batchLoss {B : List (Example S)} {θ : Param S → ℝ}
    (h : ∀ e ∈ B, OffKinks θ e.tokens) (p : Param S) :
    sourceGrad B θ p = deriv (fun s => batchLoss (Function.update θ p s) B) (θ p) :=
  (hasDerivAt_batchLoss_sourceGrad h p).deriv.symm

/-- The entries of a normalized vector with a constant gain and a constant bias sum to `n`
times the bias: the centred entries sum to zero. -/
lemma sum_layerNorm_const {n : ℕ} (g b : ℝ) (z : Fin n → ℝ) :
    ∑ i, layerNorm (fun _ => g) (fun _ => b) z i = n * b := by
  unfold layerNorm
  have hc : ∑ i, (z i - (∑ j, z j) / n) = 0 := by
    rcases Nat.eq_zero_or_pos n with rfl | hn
    · simp
    · have hn' : (n : ℝ) ≠ 0 := (Nat.cast_pos.2 hn).ne'
      rw [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
        nsmul_eq_mul, mul_div_cancel₀ _ hn', sub_self]
  rw [Finset.sum_add_distrib, ← Finset.sum_mul, ← Finset.sum_div, hc, zero_div, zero_mul,
    zero_add, Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

/-- **With every parameter `1` each hidden unit has the input `width`.** -/
lemma preactivation_one (x : Fin S.context → Fin S.width → ℝ) (l : Fin S.blocks)
    (t : Fin S.context) (u : Fin S.hidden) :
    preactivation (fun _ => 1) x l t u = S.width := by
  simp only [preactivation, attended, one_mul]
  exact (sum_layerNorm_const 1 1 _).trans (mul_one _)

/-- **The parameters that are all `1` are off the kinks**, for a decoder of positive width. -/
theorem offKinks_one (hw : 0 < S.width) (tokens : Fin S.context → Fin S.vocab) :
    OffKinks (fun _ => (1 : ℝ)) tokens := fun l t u => by
  rw [preactivation_one]
  exact (Nat.cast_pos.2 hw).ne'

/-! The witnesses: at the parameters that are all `1` the loss of the supervision of
`3 / 5 = 20` has the partial derivatives of the backward pass; the parameters `0` are at a
kink. -/

example (p : Param paperSizes) :
    HasDerivAt (fun s => batchLoss (Function.update (fun _ => 1) p s) [supervised (20, 4)])
      (sourceGrad [supervised (20, 4)] (fun _ => 1) p) 1 :=
  hasDerivAt_batchLoss_sourceGrad (θ := fun _ => 1)
    (fun e _ => offKinks_one (S := paperSizes) (by decide) e.tokens) p

example (p : Param paperSizes) : sourceGrad [supervised (20, 4)] (fun _ => 1) p =
    deriv (fun s => batchLoss (Function.update (fun _ => 1) p s) [supervised (20, 4)]) 1 :=
  sourceGrad_eq_deriv_batchLoss (θ := fun _ => 1)
    (fun e _ => offKinks_one (S := paperSizes) (by decide) e.tokens) p

example : ∀ᶠ s in 𝓝 (1 : ℝ), gate (Function.update (fun _ => 1) (.out 0 0) s) (read (20, 4)) =
    gate (S := paperSizes) (fun _ => 1) (read (20, 4)) :=
  eventually_gate_update_eq (offKinks_one (S := paperSizes) (by decide) _) _

example (tokens : Fin 6 → Fin 239) : ¬ OffKinks (0 : Param paperSizes → ℝ) tokens := fun h =>
  h 0 0 0 (by simp [preactivation])

end Transformer.Grokking.Protocol
