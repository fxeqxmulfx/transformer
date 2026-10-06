import Transformer.GPTMini.Causality.Block
import Transformer.GPTMini.Model

/-!
# Causality of the complete GPTMini forward

Source: GPTMini.forward in archived gpt_mini.py at f11b6e2 and
Transformer.forward at cbafbe9. All layers, final RMSNorm and tied
unembedding are included. Theorems quantify over every ModelParams;
they prove architectural properties rather than weight existence.
Prefix agreement also handles different sequence lengths, which is
needed to compare teacher-forced rows with separate prefix evaluation.

Position agreement is essential: changing an old RoPE position changes
its computation. Suffix tokens and their positions are unrestricted.
Task-dependent answer distinctions are separate from this causal law.
-/

namespace Transformer.GPTMini.Causality

open scoped BigOperators
noncomputable section

/-- Embedding lookup preserves agreement on raw finite-token prefixes.
Source: the actual embed operation in GPTMini.forward. -/
theorem embed_prefix (cfg : Config) (params : ModelParams cfg) {S R : ℕ}
    (tokens : Fin S → Fin cfg.vocab_size) (tl : Fin (S + R) → Fin cfg.vocab_size)
    (ht : PrefixEq tokens tl) : PrefixEq (embed cfg params tokens) (embed cfg params tl) :=
  map_prefix params.embedding tokens tl ht

example : PrefixEq (fun _ : Fin 2 => (1 : Fin 36))
    (fun j : Fin (2 + 1) => if j.val < 2 then (1 : Fin 36) else 9) := by
  intro j
  dsimp only
  change (if j.val < 2 then (1 : Fin 36) else 9) = 1
  rw [ite_eq_left j.isLt]

/-- Every hidden layer preserves the complete prefix, with no restriction on learned parameters.
Source: the actual hidden recursion implementing the ordered GPTMini block loop. -/
theorem hidden_prefix (cfg : Config) (params : ModelParams cfg) (eps : ℝ) {S R : ℕ}
    (positions : Fin S → ℝ) (pl : Fin (S + R) → ℝ)
    (tokens : Fin S → Fin cfg.vocab_size) (tl : Fin (S + R) → Fin cfg.vocab_size)
    (hp : PrefixEq positions pl) (ht : PrefixEq tokens tl) (L : ℕ) :
    PrefixEq (hidden cfg params eps positions tokens L) (hidden cfg params eps pl tl L) := by
  induction L with
  | zero => exact embed_prefix cfg params tokens tl ht
  | succ L ih =>
      simp only [hidden]
      split_ifs with hL
      · exact block_prefix cfg (params.blocks ⟨L, hL⟩) eps _ _ _ _ hp ih
      · exact ih

example : PrefixEq (fun j : Fin 2 => (j.val : ℝ))
      (fun j : Fin (2 + 1) => (j.val : ℝ)) ∧
    PrefixEq (fun _ : Fin 2 => (1 : Fin 36))
      (fun j : Fin (2 + 1) => if j.val < 2 then (1 : Fin 36) else 10) := by
  refine ⟨fun _ => rfl, ?_⟩
  intro j
  dsimp only
  change (if j.val < 2 then (1 : Fin 36) else 10) = 1
  rw [ite_eq_left j.isLt]

/-- Appending arbitrary future tokens cannot change any old vocabulary logit.
Source: the complete actual forward, including final normalization and tied readout. -/
theorem forward_prefix (cfg : Config) (params : ModelParams cfg) (eps : ℝ) {S R : ℕ}
    (positions : Fin S → ℝ) (pl : Fin (S + R) → ℝ)
    (tokens : Fin S → Fin cfg.vocab_size) (tl : Fin (S + R) → Fin cfg.vocab_size)
    (hp : PrefixEq positions pl) (ht : PrefixEq tokens tl)
    (i : Fin S) (v : Fin cfg.vocab_size) :
    forward cfg params eps pl tl (i.castAdd R) v = forward cfg params eps positions tokens i v := by
  have hx := hidden_prefix cfg params eps positions pl tokens tl hp ht cfg.n_layers i
  unfold forward unembed
  dsimp only
  rw [hx]

example : PrefixEq (fun j : Fin 2 => (j.val : ℝ))
      (fun j : Fin (2 + 1) => (j.val : ℝ)) ∧
    PrefixEq (fun _ : Fin 2 => (1 : Fin 36))
      (fun _ : Fin (2 + 1) => (1 : Fin 36)) := ⟨fun _ => rfl, fun _ => rfl⟩

/-- The full output probability distribution is preserved, not only its argmax.
Source: GPTMini.softmaxOutput applied to all vocabulary logits from forward_prefix. -/
theorem probabilities_prefix (cfg : Config) (params : ModelParams cfg) (eps : ℝ) {S R : ℕ}
    (positions : Fin S → ℝ) (pl : Fin (S + R) → ℝ)
    (tokens : Fin S → Fin cfg.vocab_size) (tl : Fin (S + R) → Fin cfg.vocab_size)
    (hp : PrefixEq positions pl) (ht : PrefixEq tokens tl)
    (i : Fin S) (v : Fin cfg.vocab_size) :
    softmaxOutput cfg params eps pl tl (i.castAdd R) v =
      softmaxOutput cfg params eps positions tokens i v := by
  unfold softmaxOutput
  have hall : ∀ w, forward cfg params eps pl tl (i.castAdd R) w =
      forward cfg params eps positions tokens i w :=
    forward_prefix cfg params eps positions pl tokens tl hp ht i
  simp only [hall]

example : PrefixEq (fun j : Fin 1 => (j.val : ℝ))
      (fun j : Fin (1 + 2) => (j.val : ℝ)) ∧
    PrefixEq (fun _ : Fin 1 => (1 : Fin 36))
      (fun _ : Fin (1 + 2) => (1 : Fin 36)) := ⟨fun _ => rfl, fun _ => rfl⟩

/-- Prefix preservation for an arbitrary pair of lengths, with explicit index transport.
Source: the same model theorem; this form is convenient for List.get and arbitrary truncation. -/
theorem forward_castLE (cfg : Config) (params : ModelParams cfg) (eps : ℝ) {S T : ℕ}
    (h : S ≤ T) (positions : Fin S → ℝ) (pl : Fin T → ℝ)
    (tokens : Fin S → Fin cfg.vocab_size) (tl : Fin T → Fin cfg.vocab_size)
    (hp : ∀ j, pl (j.castLE h) = positions j) (ht : ∀ j, tl (j.castLE h) = tokens j)
    (i : Fin S) (v : Fin cfg.vocab_size) :
    forward cfg params eps pl tl (i.castLE h) v = forward cfg params eps positions tokens i v := by
  obtain ⟨R, rfl⟩ := Nat.exists_eq_add_of_le h
  exact forward_prefix cfg params eps positions pl tokens tl hp ht i v

example : (2 : ℕ) ≤ 3 ∧
    (∀ j : Fin 2, ((j.castLE (by decide : 2 ≤ 3)).val : ℝ) = (j.val : ℝ)) ∧
    (∀ j : Fin 2,
      (fun k : Fin 3 => if k.val < 2 then (1 : Fin 36) else 9) (j.castLE (by decide)) = 1) := by
  refine ⟨by decide, fun _ => rfl, ?_⟩
  intro j
  dsimp only
  change (if j.val < 2 then (1 : Fin 36) else 9) = 1
  rw [ite_eq_left j.isLt]

/-- Changing tokens or positions strictly after i cannot affect logits at i in the full model.
Source: the causal mask composed through every actual block, proved by a common-prefix restriction. -/
theorem forward_causal (cfg : Config) (params : ModelParams cfg) (eps : ℝ) {T : ℕ}
    (px py : Fin T → ℝ) (x y : Fin T → Fin cfg.vocab_size) (i : Fin T)
    (hp : ∀ j, j.val ≤ i.val → px j = py j)
    (ht : ∀ j, j.val ≤ i.val → x j = y j) (v : Fin cfg.vocab_size) :
    forward cfg params eps px x i v = forward cfg params eps py y i v := by
  have h : i.val + 1 ≤ T := by omega
  let short : Fin (i.val + 1) → Fin T := Fin.castLE h
  let si : Fin (i.val + 1) := ⟨i.val, by omega⟩
  have hi : short si = i := by apply Fin.ext; rfl
  have hxp := forward_castLE cfg params eps h
    (fun j => px (short j)) px (fun j => x (short j)) x
    (fun _ => rfl) (fun _ => rfl) si v
  have hyp := forward_castLE cfg params eps h
    (fun j => px (short j)) py (fun j => x (short j)) y
    (fun j => (hp (short j) (by have hj := j.isLt; change j.val ≤ i.val; omega)).symm)
    (fun j => (ht (short j) (by have hj := j.isLt; change j.val ≤ i.val; omega)).symm) si v
  change forward cfg params eps px x (short si) v = _ at hxp
  change forward cfg params eps py y (short si) v = _ at hyp
  rw [hi] at hxp hyp
  exact hxp.trans hyp.symm

example : (∀ j : Fin 3, j.val ≤ (1 : Fin 3).val → (j.val : ℝ) = (j.val : ℝ)) ∧
    (∀ j : Fin 3, j.val ≤ (1 : Fin 3).val →
      (if j.val < 2 then (1 : Fin 36) else 9) =
        (if j.val < 2 then (1 : Fin 36) else 10)) := by
  refine ⟨fun _ _ => rfl, ?_⟩
  intro j hj
  have hlt : j.val < 2 := by omega
  rw [ite_eq_left hlt, ite_eq_left hlt]

end
end Transformer.GPTMini.Causality
