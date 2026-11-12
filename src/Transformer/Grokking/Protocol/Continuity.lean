module

public import Transformer.Grokking.Protocol.Model

/-!
# The decoder is continuous in its parameters

Source: openai/grok at 3d64b1d8, `grok/transformer.py`;
`experiments/grok_division/PROTOCOL.md`, *The model*. The forward pass is that of
`Transformer.Grokking.Protocol.Model`.

Every operation of the forward pass is continuous where it is used: sums and products, the
exponential, the quotient by a sum of exponentials, which is positive, and the quotient by the
root of `layerNorm`, which is positive because `10⁻⁵` is added to the variance
(`layerNorm_root_pos`). So with continuous activations, as `max 0` is, every block's output
and every logit is a continuous function of the parameters (`continuous_decodeTo`,
`continuous_logits`), for all tokens read. The statements are for parameters that depend
continuously on a point of any space, so that they compose.

Nothing is said here of the gradient of the backward pass: its gates are indicators of the
sign of a hidden input (`Gradient`).
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

variable {S : Sizes} {α : Type*} [TopologicalSpace α]

/-- Layer normalization is continuous in the gain, the bias and the vector. -/
lemma continuous_layerNorm {n : ℕ} {g b z : α → Fin n → ℝ} (hg : ∀ i, Continuous fun a => g a i)
    (hb : ∀ i, Continuous fun a => b a i) (hz : ∀ i, Continuous fun a => z a i) (i : Fin n) :
    Continuous fun a => layerNorm (g a) (b a) (z a) i := by
  have hmean : Continuous fun a => (∑ j, z a j) / (n : ℝ) :=
    (continuous_finsetSum _ fun j _ => hz j).div_const _
  have hvar : Continuous fun a =>
      (∑ j, (z a j - (∑ j', z a j') / n) ^ 2) / (n : ℝ) + 1 / 10 ^ 5 :=
    ((continuous_finsetSum _ fun j _ => ((hz j).sub hmean).pow 2).div_const _).add
      continuous_const
  exact ((((hz i).sub hmean).div (Real.continuous_sqrt.comp hvar) fun a =>
    (layerNorm_root_pos (z a)).ne').mul (hg i)).add (hb i)

variable {θ : α → Param S → ℝ} {x : α → Fin S.context → Fin S.width → ℝ}

lemma continuous_score (hθ : ∀ p, Continuous fun a => θ a p)
    (hx : ∀ t i, Continuous fun a => x a t i) (l : Fin S.blocks) (h : Fin S.heads)
    (t j : Fin S.context) : Continuous fun a => score (θ a) (x a) l h t j :=
  (continuous_finsetSum _ fun _ _ =>
    (continuous_finsetSum _ fun i _ => (hθ _).mul (hx t i)).mul
      (continuous_finsetSum _ fun i _ => (hθ _).mul (hx j i))).div_const _

lemma continuous_weight (hθ : ∀ p, Continuous fun a => θ a p)
    (hx : ∀ t i, Continuous fun a => x a t i) (l : Fin S.blocks) (h : Fin S.heads)
    (t j : Fin S.context) : Continuous fun a => weight (θ a) (x a) l h t j := by
  unfold weight
  split_ifs with hj
  · exact (Real.continuous_exp.comp (continuous_score hθ hx l h t j)).div
      (continuous_finsetSum _ fun j' _ =>
        Real.continuous_exp.comp (continuous_score hθ hx l h t j'))
      fun a => (Finset.sum_pos (fun _ _ => Real.exp_pos _) ⟨t, Finset.mem_Iic.2 le_rfl⟩).ne'
  · exact continuous_const

lemma continuous_attention (hθ : ∀ p, Continuous fun a => θ a p)
    (hx : ∀ t i, Continuous fun a => x a t i) (l : Fin S.blocks) (t : Fin S.context)
    (o : Fin S.width) : Continuous fun a => attention (θ a) (x a) l t o :=
  continuous_finsetSum _ fun h _ => continuous_finsetSum _ fun _ _ => (hθ _).mul
    (continuous_finsetSum _ fun j _ => (continuous_weight hθ hx l h t j).mul
      (continuous_finsetSum _ fun i _ => (hθ _).mul (hx j i)))

lemma continuous_attended (hθ : ∀ p, Continuous fun a => θ a p)
    (hx : ∀ t i, Continuous fun a => x a t i) (l : Fin S.blocks) (t : Fin S.context)
    (i : Fin S.width) : Continuous fun a => attended (θ a) (x a) l t i :=
  continuous_layerNorm (fun _ => hθ _) (fun _ => hθ _)
    (fun i => (hx t i).add (continuous_attention hθ hx l t i)) i

lemma continuous_preactivation (hθ : ∀ p, Continuous fun a => θ a p)
    (hx : ∀ t i, Continuous fun a => x a t i) (l : Fin S.blocks) (t : Fin S.context)
    (u : Fin S.hidden) : Continuous fun a => preactivation (θ a) (x a) l t u :=
  continuous_finsetSum _ fun i _ => (hθ _).mul (continuous_attended hθ hx l t i)

/-- A block is continuous in the parameters and its input, with continuous activations. -/
lemma continuous_block {act : Activation S} (hact : ∀ l t u, Continuous (act l t u))
    (hθ : ∀ p, Continuous fun a => θ a p) (hx : ∀ t i, Continuous fun a => x a t i)
    (l : Fin S.blocks) (t : Fin S.context) (i : Fin S.width) :
    Continuous fun a => block (θ a) act (x a) l t i :=
  continuous_layerNorm (fun _ => hθ _) (fun _ => hθ _)
    (fun o => (continuous_attended hθ hx l t o).add (continuous_finsetSum _ fun u _ =>
      (hθ _).mul ((hact l t u).comp (continuous_preactivation hθ hx l t u)))) i

/-- **Every block's output is continuous in the parameters**, with continuous activations. -/
theorem continuous_decodeTo {act : Activation S} (hact : ∀ l t u, Continuous (act l t u))
    (hθ : ∀ p, Continuous fun a => θ a p) (tokens : Fin S.context → Fin S.vocab) (n : ℕ) :
    ∀ t i, Continuous fun a => decodeTo (θ a) act tokens n t i := by
  induction n with
  | zero => exact fun t i => (hθ _).add continuous_const
  | succ n ih =>
    intro t i
    simp only [decodeTo]
    split_ifs with hn
    · exact continuous_block hact hθ ih ⟨n, hn⟩ t i
    · exact ih t i

/-- **Every logit is continuous in the parameters**, with continuous activations. -/
theorem continuous_logits {act : Activation S} (hact : ∀ l t u, Continuous (act l t u))
    (hθ : ∀ p, Continuous fun a => θ a p) (tokens : Fin S.context → Fin S.vocab)
    (t : Fin S.context) (c : Fin S.vocab) : Continuous fun a => logits (θ a) act tokens t c :=
  continuous_finsetSum _ fun i _ => (hθ _).mul (continuous_decodeTo hact hθ tokens S.blocks t i)

/-- The source's activation is continuous. -/
lemma continuous_relu (l : Fin S.blocks) (t : Fin S.context) (u : Fin S.hidden) :
    Continuous (relu S l t u) :=
  continuous_const.max continuous_id

/-- **The logits of the model are continuous in the parameters.** -/
theorem continuous_logits_relu (tokens : Fin S.context → Fin S.vocab) (t : Fin S.context)
    (c : Fin S.vocab) : Continuous fun θ : Param S → ℝ => logits θ (relu S) tokens t c :=
  continuous_logits continuous_relu continuous_apply tokens t c

/-! The witnesses: a normalization of two entries of a real variable; the output of the last
block of the paper's decoder with the source's activation; and its logits with the hidden
units switched off. -/

example : Continuous fun a : ℝ => layerNorm ![a, 1] ![0, a] ![a ^ 2, 3] 0 :=
  continuous_layerNorm (fun i => by fin_cases i <;> simp <;> fun_prop)
    (fun i => by fin_cases i <;> simp <;> fun_prop)
    (fun i => by fin_cases i <;> simp <;> fun_prop) 0

example (tokens : Fin 6 → Fin 239) (t : Fin 6) (i : Fin 128) :
    Continuous fun θ : Param ⟨239, 128, 4, 32, 512, 2, 6⟩ → ℝ =>
      decodeTo θ (relu _) tokens 2 t i :=
  continuous_decodeTo (S := ⟨239, 128, 4, 32, 512, 2, 6⟩) (θ := fun θ => θ) continuous_relu
    continuous_apply tokens 2 t i

example (tokens : Fin 6 → Fin 239) (t : Fin 6) (c : Fin 239) :
    Continuous fun θ : Param ⟨239, 128, 4, 32, 512, 2, 6⟩ → ℝ =>
      logits θ (fun _ _ _ _ => 0) tokens t c :=
  continuous_logits (S := ⟨239, 128, 4, 32, 512, 2, 6⟩) (act := fun _ _ _ _ => 0) (θ := fun θ => θ)
    (fun _ _ _ => continuous_const) continuous_apply tokens t c

end Transformer.Grokking.Protocol
