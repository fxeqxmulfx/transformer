module

public import Transformer.Grokking.Protocol.Model

/-!
# The decoder is causal

Source: arXiv:2201.02177v1 (Power et al.), appendix *Additional experimental details*, "a
standard decoder-only transformer … with causal attention masking"; openai/grok at 3d64b1d8,
`grok/transformer.py`, `AttentionHead.forward`, which fills the scores of later positions with
`-inf` before the softmax; `experiments/grok_division/PROTOCOL.md`, *The model*.

In `weight` a position weighs only the positions up to its own. So the output of every block
at a position depends on the tokens up to that position only (`decodeTo_congr`), and so do the
logits there (`logits_congr`), for any activations. This is what makes the logits after `=`
a function of the two operands, and the supervision of a later position invisible to them.

The witness is a decoder with two tokens and two positions: its logits at the first position
are the same for the rows `0 0` and `0 1`.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

variable {S : Sizes} {θ : Param S → ℝ} {act : Activation S}
  {x x' : Fin S.context → Fin S.width → ℝ} {l : Fin S.blocks} {h : Fin S.heads}
  {t : Fin S.context}

lemma weight_congr (hx : ∀ j ≤ t, x j = x' j) (j : Fin S.context) :
    weight θ x l h t j = weight θ x' l h t j := by
  have hs : ∀ j ≤ t, score θ x l h t j = score θ x' l h t j := fun j hj => by
    simp only [score, hx t le_rfl, hx j hj]
  unfold weight
  split_ifs with hj
  · rw [hs j hj, Finset.sum_congr rfl fun j' hj' => by rw [hs j' (Finset.mem_Iic.1 hj')]]
  · rfl

lemma block_congr (hx : ∀ j ≤ t, x j = x' j) : block θ act x l t = block θ act x' l t := by
  have ha : attention θ x l t = attention θ x' l t := by
    funext o
    unfold attention
    refine Finset.sum_congr rfl fun h _ => Finset.sum_congr rfl fun r _ => ?_
    rw [Finset.sum_congr rfl fun j hj => by rw [weight_congr hx, hx j (Finset.mem_Iic.1 hj)]]
  have hb : attended θ x l t = attended θ x' l t := by simp only [attended, ha, hx t le_rfl]
  simp only [block, preactivation, hb]

/-- **The pass is causal**: the output of every block at a position does not change with later
tokens. -/
theorem decodeTo_congr {tokens tokens' : Fin S.context → Fin S.vocab}
    (ht : ∀ s ≤ t, tokens s = tokens' s) (n : ℕ) :
    ∀ s ≤ t, decodeTo θ act tokens n s = decodeTo θ act tokens' n s := by
  induction n with
  | zero => exact fun s hs => by simp only [decodeTo, ht s hs]
  | succ n ih =>
    intro s hs
    simp only [decodeTo]
    split_ifs with hn
    · exact block_congr fun j hj => ih j (hj.trans hs)
    · exact ih s hs

/-- **The logits at a position do not change with later tokens.** -/
theorem logits_congr {tokens tokens' : Fin S.context → Fin S.vocab}
    (ht : ∀ s ≤ t, tokens s = tokens' s) : logits θ act tokens t = logits θ act tokens' t := by
  funext c
  simp only [logits, decodeTo_congr ht S.blocks t le_rfl]


/-! The witnesses. -/

/-- A decoder with two tokens, width `1`, one head, one hidden unit, one block, two positions. -/
abbrev tinySizes : Sizes := ⟨2, 1, 1, 1, 1, 1, 2⟩

lemma eq_zero_of_le_zero {s : Fin 2} (hs : s ≤ 0) : s = 0 := le_antisymm hs (Fin.zero_le s)

example (θ : Param tinySizes → ℝ) :
    logits θ (relu tinySizes) (![0, 0] : Fin 2 → Fin 2) (0 : Fin 2) =
      logits θ (relu tinySizes) (![0, 1] : Fin 2 → Fin 2) (0 : Fin 2) :=
  logits_congr fun s hs => by rw [eq_zero_of_le_zero hs]; rfl

example (θ : Param tinySizes → ℝ) :=
  decodeTo_congr (θ := θ) (act := relu tinySizes) (tokens := (![0, 0] : Fin 2 → Fin 2))
    (tokens' := (![0, 1] : Fin 2 → Fin 2)) (t := (0 : Fin 2))
    (fun s hs => by rw [eq_zero_of_le_zero hs]; rfl) 1

/-- Two inputs of a block that agree at the first position. -/
example (θ : Param tinySizes → ℝ) :
    block θ (relu tinySizes) (![![0], ![1]] : Fin 2 → Fin 1 → ℝ) (0 : Fin 1) (0 : Fin 2) =
      block θ (relu tinySizes) (![![0], ![2]] : Fin 2 → Fin 1 → ℝ) (0 : Fin 1) (0 : Fin 2) :=
  block_congr fun s hs => by rw [eq_zero_of_le_zero hs]; rfl

example (θ : Param tinySizes → ℝ) (j : Fin 2) :
    weight θ (![![0], ![1]] : Fin 2 → Fin 1 → ℝ) (0 : Fin 1) (0 : Fin 1) (0 : Fin 2) j =
      weight θ (![![0], ![2]] : Fin 2 → Fin 1 → ℝ) (0 : Fin 1) (0 : Fin 1) (0 : Fin 2) j :=
  weight_congr (S := tinySizes) (θ := θ) (x := ![![0], ![1]]) (x' := ![![0], ![2]])
    (l := (0 : Fin 1)) (h := (0 : Fin 1)) (t := (0 : Fin 2))
    (fun s hs => by rw [eq_zero_of_le_zero hs]; rfl) j

end Transformer.Grokking.Protocol
