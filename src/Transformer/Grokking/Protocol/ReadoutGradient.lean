module

public import Mathlib.Analysis.SpecialFunctions.ExpDeriv
public import Mathlib.Analysis.SpecialFunctions.Log.Deriv
public import Transformer.Grokking.Geometry.ProbabilityLoss
public import Transformer.Grokking.Protocol.ReadoutBound
public import Transformer.Grokking.Protocol.TargetMean

/-!
# What the backward pass returns for the readout

Source: openai/grok at 3d64b1d8, `grok/transformer.py`, `Transformer` (the readout
`self.linear`, without a bias and not tied to the embedding), and `grok/training.py`,
`TrainableTransformer._step` (the mean cross-entropy and its backward pass);
`experiments/grok_division/PROTOCOL.md`, *The model* and *The loss and the accuracy*.

No block reads the readout (`decodeTo_update_out`), so along one readout parameter a logit of
its token moves linearly and the others do not move (`logits_update_out`). The coordinate of
`sourceGrad` at the readout parameter of the token `c` and the feature `i` is therefore the
mean over the targets of

    feature i · (probability of c − [c is the target])

(`sourceGrad_out`), the formula of the textbook; the gates frozen by `sourceGrad` do not enter
it. Two consequences are used by `ReadoutStationary`:

* the coordinate is at most a bound of the feature in absolute value
  (`abs_sourceGrad_out_le`), hence at most `√width · G + B` when the second normalizations have
  gains at most `G` and biases at most `B` (`abs_sourceGrad_out_le_of_norm`);
* the readout paired with its gradient is the mean over the targets of the expected logit
  less the logit of the target (`sum_out_mul_sourceGrad`, in `ReadoutPairing`).
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Transformer.Landscape.Neural (crossEntropy sum_exp_pos)
open Transformer.Grokking.Geometry (probability probability_pos probability_le_one)

variable {S : Sizes}

/-- **No block reads the readout**: changing a readout parameter changes no block's output. -/
theorem decodeTo_update_out (θ : Param S → ℝ) (c : Fin S.vocab) (i : Fin S.width) (s : ℝ)
    (act : Activation S) (tokens : Fin S.context → Fin S.vocab) (n : ℕ) :
    decodeTo (Function.update θ (.out c i) s) act tokens n = decodeTo θ act tokens n := by
  induction n with
  | zero =>
    funext t j
    simp only [decodeTo, Function.update_apply, reduceCtorEq, ite_false]
  | succ n ih =>
    simp only [decodeTo]
    split_ifs with hn
    · funext t
      rw [ih]
      simp only [block, attended, attention, weight, score, preactivation,
        Function.update_apply, reduceCtorEq, ite_false]
    · exact ih

/-- **Along a readout parameter one logit moves linearly**: that of its token, at the rate of
its feature. -/
theorem logits_update_out (θ : Param S → ℝ) (c : Fin S.vocab) (i : Fin S.width) (s : ℝ)
    (act : Activation S) (tokens : Fin S.context → Fin S.vocab) (t : Fin S.context)
    (q : Fin S.vocab) :
    logits (Function.update θ (.out c i) s) act tokens t q = logits θ act tokens t q +
      if q = c then (s - θ (.out c i)) * decodeTo θ act tokens S.blocks t i else 0 := by
  simp only [logits, decodeTo_update_out]
  split_ifs with hq
  · subst hq
    have h : ∀ j, Function.update θ (.out q i) s (.out q j) *
        decodeTo θ act tokens S.blocks t j = θ (.out q j) * decodeTo θ act tokens S.blocks t j +
          if j = i then (s - θ (.out q i)) * decodeTo θ act tokens S.blocks t i else 0 :=
      fun j => by
        rcases eq_or_ne j i with rfl | hj
        · rw [Function.update_self, ite_eq_left rfl]
          ring
        · rw [Function.update_of_ne (by simpa using hj), ite_eq_right hj, add_zero]
    simp only [h, Finset.sum_add_distrib, Finset.sum_ite_eq', Finset.mem_univ, ite_true]
  · rw [add_zero]
    refine Finset.sum_congr rfl fun j _ => ?_
    rw [Function.update_of_ne (by simp [hq])]

/-- The derivative of the cross-entropy in one logit that moves at the rate `φ`. -/
lemma hasDerivAt_crossEntropy_coordinate {V : Type*} [Fintype V] [DecidableEq V] (z : V → ℝ)
    (y c : V) (φ a : ℝ) :
    HasDerivAt (fun s => crossEntropy (fun q => z q + if q = c then (s - a) * φ else 0) y)
      (φ * (probability z c - if c = y then 1 else 0)) a := by
  have hline : HasDerivAt (fun s : ℝ => (s - a) * φ) φ a := by
    simpa using ((hasDerivAt_id' a).sub_const a).mul_const φ
  have hterm : ∀ q ∈ (Finset.univ : Finset V), HasDerivAt
      (fun s => Real.exp (z q + if q = c then (s - a) * φ else 0))
      (if q = c then Real.exp (z q) * φ else 0) a := fun q _ => by
    split_ifs with hq
    · simpa using (hline.const_add (z q)).exp
    · simpa using hasDerivAt_const a (Real.exp (z q))
  have hy : HasDerivAt (fun s => z y + if y = c then (s - a) * φ else 0)
      (if y = c then φ else 0) a := by
    split_ifs with hyc
    · exact hline.const_add (z y)
    · simpa using hasDerivAt_const a (z y)
  have hpos : (∑ q, Real.exp (z q + if q = c then (a - a) * φ else 0)) ≠ 0 := by
    simpa using (sum_exp_pos z y).ne'
  refine (((HasDerivAt.fun_sum hterm).log hpos).fun_sub hy).congr_deriv ?_
  simp only [Finset.sum_ite_eq', Finset.mem_univ, ite_true, sub_self, zero_mul, ite_self,
    add_zero, probability]
  rcases eq_or_ne c y with rfl | hcy
  · rw [ite_eq_left rfl, ite_eq_left rfl]
    ring
  · rw [ite_eq_right hcy, ite_eq_right (Ne.symm hcy)]
    ring

/-- **The gradient of a readout parameter** that the backward pass returns: the mean over the
targets of the feature times the probability of the parameter's token less `1` at the
target. -/
theorem sourceGrad_out (B : List (Example S)) (θ : Param S → ℝ) (c : Fin S.vocab)
    (i : Fin S.width) :
    sourceGrad B θ (.out c i) = targetMean B fun e pc =>
      decodeTo θ (relu S) e.tokens S.blocks pc.1 i *
        (probability (logits θ (relu S) e.tokens pc.1) c - if c = pc.2 then 1 else 0) := by
  refine HasDerivAt.deriv (hasDerivAt_targetMean B fun e _ pc _ => ?_)
  have hfun : (fun s => crossEntropy
      (logits (Function.update θ (.out c i) s) (gate θ e.tokens) e.tokens pc.1) pc.2) =
      fun s => crossEntropy (fun q => logits θ (relu S) e.tokens pc.1 q +
        if q = c then (s - θ (.out c i)) * decodeTo θ (relu S) e.tokens S.blocks pc.1 i
        else 0) pc.2 := by
    funext s
    congr 1
    funext q
    rw [logits_update_out, logits_gate, decodeTo_gate]
  rw [hfun]
  exact hasDerivAt_crossEntropy_coordinate _ _ _ _ _

/-- **A readout coordinate of the gradient is at most a bound of its feature.** -/
theorem abs_sourceGrad_out_le (B : List (Example S)) (θ : Param S → ℝ) (c : Fin S.vocab)
    (i : Fin S.width) {F : ℝ} (hF : 0 ≤ F)
    (h : ∀ e ∈ B, ∀ pc ∈ e.targets, |decodeTo θ (relu S) e.tokens S.blocks pc.1 i| ≤ F) :
    |sourceGrad B θ (.out c i)| ≤ F := by
  rw [sourceGrad_out]
  refine abs_targetMean_le hF fun e he pc hpc => ?_
  have hp := probability_pos (logits θ (relu S) e.tokens pc.1) c
  have hp1 := probability_le_one (logits θ (relu S) e.tokens pc.1) c
  have h1 : |probability (logits θ (relu S) e.tokens pc.1) c - if c = pc.2 then 1 else 0| ≤ 1 := by
    rw [abs_le]
    split_ifs <;> constructor <;> linarith
  rw [abs_mul]
  exact (mul_le_mul (h e he pc hpc) h1 (abs_nonneg _) hF).trans_eq (mul_one F)

/-- A readout coordinate of the gradient is bounded by the second normalizations. -/
theorem abs_sourceGrad_out_le_of_norm (hS : 0 < S.blocks) (B : List (Example S))
    {θ : Param S → ℝ} {G M : ℝ} (hG : ∀ l i, |θ (.gain2 l i)| ≤ G)
    (hM : ∀ l i, |θ (.bias2 l i)| ≤ M) (c : Fin S.vocab) (i : Fin S.width) :
    |sourceGrad B θ (.out c i)| ≤ √S.width * G + M := by
  rcases isEmpty_or_nonempty (Fin S.blocks) with he | ⟨⟨l⟩⟩
  · exact absurd hS (by simpa using Fin.isEmpty_iff.1 he)
  · have h0 : 0 ≤ √S.width * G + M :=
      add_nonneg (mul_nonneg (Real.sqrt_nonneg _) ((abs_nonneg _).trans (hG l i)))
        ((abs_nonneg _).trans (hM l i))
    exact abs_sourceGrad_out_le B θ c i h0 fun e _ pc _ => abs_decodeTo_le hS hG hM _ _ _ i

/-! The witnesses: the supervision of `3 / 5 = 20` at the parameters that are all `1`, where
every feature is at most `√128 + 1`. -/

example (c : Fin 239) (i : Fin 128) :
    |sourceGrad [supervised (20, 4)] (fun _ => 1) (.out c i)| ≤ √(128 : ℕ) * 1 + 1 :=
  abs_sourceGrad_out_le (S := paperSizes) _ _ c i (by positivity) fun e _ pc _ =>
    abs_decodeTo_le (S := paperSizes) (by decide) (by simp) (by simp) _ _ _ i

example (c : Fin 239) (i : Fin 128) :
    |sourceGrad [supervised (20, 4)] (fun _ => 1) (.out c i)| ≤ √(128 : ℕ) * 1 + 1 :=
  abs_sourceGrad_out_le_of_norm (S := paperSizes) (by decide) _ (by simp) (by simp) c i

end Transformer.Grokking.Protocol
