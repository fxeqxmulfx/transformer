module

public import Mathlib.Analysis.SpecialFunctions.ExpDeriv
public import Mathlib.Analysis.SpecialFunctions.Log.Deriv
public import Mathlib.Analysis.SpecialFunctions.Sqrt
public import Transformer.Grokking.Protocol.Gradient

/-!
# The gradient of the backward pass is a derivative

Source: openai/grok at 3d64b1d8, `grok/training.py`, `TrainableTransformer._step`, and
PyTorch's backward pass; `experiments/grok_division/PROTOCOL.md`, *The loss and the accuracy*;
`review/grok_division_convergence.md`, bridge 3.

`sourceGrad` is defined as a `deriv`, and a `deriv` is `0` where the function has no
derivative. This file shows that this does not happen: with the gates frozen, the loss of a
batch is differentiable along every coordinate of the parameters
(`differentiable_frozenLoss`), so each coordinate of `sourceGrad` is the derivative of the
frozen loss in that coordinate (`hasDerivAt_frozenLoss_sourceGrad`), at every parameter, the
exceptional ones included.

The proof follows the forward pass along a differentiable line of parameters: sums and
products, the exponential, the quotient by a sum of exponentials and the quotient by the root
of `layerNorm`, which is positive, the logarithm of a positive sum, and activations that are
differentiable, as a frozen gate is: the identity or zero (`differentiable_gate`). The
source's activation `max 0` is not, and the statements here are not about it.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

open Transformer.Landscape.Neural (crossEntropy sum_exp_pos)

variable {S : Sizes}

/-- Layer normalization is differentiable in the gain, the bias and the vector. -/
lemma differentiable_layerNorm {n : ℕ} {g b z : ℝ → Fin n → ℝ}
    (hg : ∀ i, Differentiable ℝ fun a => g a i) (hb : ∀ i, Differentiable ℝ fun a => b a i)
    (hz : ∀ i, Differentiable ℝ fun a => z a i) (i : Fin n) :
    Differentiable ℝ fun a => layerNorm (g a) (b a) (z a) i := by
  have hmean : Differentiable ℝ fun a => (∑ j, z a j) / (n : ℝ) :=
    (Differentiable.fun_sum fun j _ => hz j).div_const _
  have hvar : Differentiable ℝ fun a =>
      (∑ j, (z a j - (∑ j', z a j') / n) ^ 2) / (n : ℝ) + 1 / 10 ^ 5 :=
    ((Differentiable.fun_sum fun j _ => ((hz j).fun_sub hmean).fun_pow 2).div_const _).fun_add
      (differentiable_const _)
  exact ((((hz i).fun_sub hmean).fun_div (hvar.sqrt fun a => by positivity) fun a =>
    (layerNorm_root_pos (z a)).ne').fun_mul (hg i)).fun_add (hb i)

variable {θ : ℝ → Param S → ℝ} {x : ℝ → Fin S.context → Fin S.width → ℝ}

lemma differentiable_score (hθ : ∀ p, Differentiable ℝ fun a => θ a p)
    (hx : ∀ t i, Differentiable ℝ fun a => x a t i) (l : Fin S.blocks) (h : Fin S.heads)
    (t j : Fin S.context) : Differentiable ℝ fun a => score (θ a) (x a) l h t j :=
  (Differentiable.fun_sum fun _ _ =>
    (Differentiable.fun_sum fun i _ => (hθ _).fun_mul (hx t i)).fun_mul
      (Differentiable.fun_sum fun i _ => (hθ _).fun_mul (hx j i))).div_const _

lemma differentiable_weight (hθ : ∀ p, Differentiable ℝ fun a => θ a p)
    (hx : ∀ t i, Differentiable ℝ fun a => x a t i) (l : Fin S.blocks) (h : Fin S.heads)
    (t j : Fin S.context) : Differentiable ℝ fun a => weight (θ a) (x a) l h t j := by
  unfold weight
  split_ifs with hj
  · exact (differentiable_score hθ hx l h t j).exp.fun_div
      (Differentiable.fun_sum fun j' _ => (differentiable_score hθ hx l h t j').exp)
      fun a => (Finset.sum_pos (fun _ _ => Real.exp_pos _) ⟨t, Finset.mem_Iic.2 le_rfl⟩).ne'
  · exact differentiable_const _

lemma differentiable_attended (hθ : ∀ p, Differentiable ℝ fun a => θ a p)
    (hx : ∀ t i, Differentiable ℝ fun a => x a t i) (l : Fin S.blocks) (t : Fin S.context)
    (i : Fin S.width) : Differentiable ℝ fun a => attended (θ a) (x a) l t i :=
  differentiable_layerNorm (fun _ => hθ _) (fun _ => hθ _) (fun o => (hx t o).fun_add
    (Differentiable.fun_sum fun h _ => Differentiable.fun_sum fun _ _ => (hθ _).fun_mul
      (Differentiable.fun_sum fun j _ => (differentiable_weight hθ hx l h t j).fun_mul
        (Differentiable.fun_sum fun i _ => (hθ _).fun_mul (hx j i))))) i

/-- A block is differentiable along a line of parameters, with differentiable activations. -/
lemma differentiable_block {act : Activation S} (hact : ∀ l t u, Differentiable ℝ (act l t u))
    (hθ : ∀ p, Differentiable ℝ fun a => θ a p) (hx : ∀ t i, Differentiable ℝ fun a => x a t i)
    (l : Fin S.blocks) (t : Fin S.context) (i : Fin S.width) :
    Differentiable ℝ fun a => block (θ a) act (x a) l t i :=
  differentiable_layerNorm (fun _ => hθ _) (fun _ => hθ _)
    (fun o => (differentiable_attended hθ hx l t o).fun_add (Differentiable.fun_sum fun u _ =>
      (hθ _).fun_mul ((hact l t u).fun_comp (Differentiable.fun_sum fun i _ =>
        (hθ _).fun_mul (differentiable_attended hθ hx l t i))))) i

/-- **Every block's output is differentiable along a line of parameters**, with differentiable
activations. -/
theorem differentiable_decodeTo {act : Activation S}
    (hact : ∀ l t u, Differentiable ℝ (act l t u)) (hθ : ∀ p, Differentiable ℝ fun a => θ a p)
    (tokens : Fin S.context → Fin S.vocab) (n : ℕ) :
    ∀ t i, Differentiable ℝ fun a => decodeTo (θ a) act tokens n t i := by
  induction n with
  | zero => exact fun t i => (hθ _).fun_add (differentiable_const _)
  | succ n ih =>
    intro t i
    simp only [decodeTo]
    split_ifs with hn
    · exact differentiable_block hact hθ ih ⟨n, hn⟩ t i
    · exact ih t i

/-- **Every logit is differentiable along a line of parameters**, with differentiable
activations. -/
theorem differentiable_logits {act : Activation S}
    (hact : ∀ l t u, Differentiable ℝ (act l t u)) (hθ : ∀ p, Differentiable ℝ fun a => θ a p)
    (tokens : Fin S.context → Fin S.vocab) (t : Fin S.context) (c : Fin S.vocab) :
    Differentiable ℝ fun a => logits (θ a) act tokens t c :=
  Differentiable.fun_sum fun i _ =>
    (hθ _).fun_mul (differentiable_decodeTo hact hθ tokens S.blocks t i)

/-- **A frozen gate is differentiable**: it is the identity or zero. -/
lemma differentiable_gate (θ₀ : Param S → ℝ) (tokens : Fin S.context → Fin S.vocab)
    (l : Fin S.blocks) (t : Fin S.context) (u : Fin S.hidden) :
    Differentiable ℝ (gate θ₀ tokens l t u) := by
  unfold gate
  split_ifs
  · exact differentiable_fun_id
  · exact differentiable_const 0

/-- A coordinate line of parameters is differentiable in every coordinate. -/
lemma differentiable_update (θ : Param S → ℝ) (p q : Param S) :
    Differentiable ℝ fun s : ℝ => Function.update θ p s q := by
  by_cases h : q = p
  · subst h
    simp
  · simp [Function.update_of_ne h]

lemma differentiable_list_sum_map {α : Type*} (l : List α) {f : α → ℝ → ℝ}
    (h : ∀ a ∈ l, Differentiable ℝ (f a)) :
    Differentiable ℝ fun s => (l.map fun a => f a s).sum := by
  induction l with
  | nil => simp
  | cons a l ih =>
    simp only [List.map_cons, List.sum_cons]
    exact (h a List.mem_cons_self).fun_add (ih fun b hb => h b (List.mem_cons_of_mem a hb))

/-- **The loss with frozen gates is differentiable along every coordinate.** -/
theorem differentiable_frozenLoss (θ₀ θ : Param S → ℝ) (p : Param S) (B : List (Example S)) :
    Differentiable ℝ fun s => frozenLoss θ₀ (Function.update θ p s) B :=
  (differentiable_list_sum_map B fun e _ => differentiable_list_sum_map e.targets fun pc _ =>
    ((Differentiable.fun_sum fun q _ => (differentiable_logits (differentiable_gate θ₀ e.tokens)
      (differentiable_update θ p) e.tokens pc.1 q).exp).log fun _ =>
        (sum_exp_pos _ pc.2).ne').fun_sub
      (differentiable_logits (differentiable_gate θ₀ e.tokens) (differentiable_update θ p)
        e.tokens pc.1 pc.2)).div_const _

/-- **The gradient of the backward pass is a derivative**: each of its coordinates is the
derivative of the loss with frozen gates in that coordinate, at every parameter. -/
theorem hasDerivAt_frozenLoss_sourceGrad (B : List (Example S)) (θ : Param S → ℝ)
    (p : Param S) :
    HasDerivAt (fun s => frozenLoss θ (Function.update θ p s) B) (sourceGrad B θ p) (θ p) :=
  (differentiable_frozenLoss θ θ p B (θ p)).hasDerivAt

/-! The witnesses: a normalization of two entries of a real variable, and the logits of the
paper's decoder with the hidden units switched off, along a line of parameters. -/

example : Differentiable ℝ fun a : ℝ => layerNorm ![a, 1] ![0, a] ![a ^ 2, 3] 0 :=
  differentiable_layerNorm (fun i => by fin_cases i <;> simp)
    (fun i => by fin_cases i <;> simp)
    (fun i => by fin_cases i <;> simp) 0

example (θ : Param ⟨239, 128, 4, 32, 512, 2, 6⟩ → ℝ) (p : Param ⟨239, 128, 4, 32, 512, 2, 6⟩)
    (tokens : Fin 6 → Fin 239) (t : Fin 6) (i : Fin 128) (c : Fin 239) :
    (Differentiable ℝ fun s => decodeTo (Function.update θ p s) (fun _ _ _ _ => 0) tokens 2 t i) ∧
      Differentiable ℝ fun s => logits (Function.update θ p s) (fun _ _ _ _ => 0) tokens t c :=
  ⟨differentiable_decodeTo (fun _ _ _ => differentiable_const 0) (differentiable_update θ p)
      tokens 2 t i,
    differentiable_logits (fun _ _ _ => differentiable_const 0) (differentiable_update θ p)
      tokens t c⟩

end Transformer.Grokking.Protocol
