import Transformer.GPTMini.Semantics.Basic
import Transformer.Basis.Parity
import Mathlib.Algebra.BigOperators.Fin

/-!
# Counting raw ones with the actual uniform softmax head

Source: CausalMHA.forward at f11b6e2 and the parity prompt/bit IDs in
synthetic/bits.py at cbafbe9. Zero queries and keys give exactly uniform
finite softmax, even after RoPE. Values encoding ONE by a direction and
the other raw tokens by zero yield count(ONE) / context length in that
direction. At SEP the self-value is zero, so XSA preserves this signal.

This is an internal head computation, not a parity oracle in the model.
It supplies the exact count from which a subsequent decoder can compute
parity. Choosing FFN/readout weights and recognizing the EOS phase are
separate obligations; counting alone is not a full Basis solver.
-/

namespace Transformer.GPTMini.Semantics

open scoped BigOperators
open Transformer.Basis

/-- A raw ONE indicator in a chosen head direction; Source: Bits.inputs at cbafbe9. -/
def oneValue {d : ℕ} (direction : EucSpace d) (token : ℤ) : EucSpace d :=
  if token = oneBit then direction else 0

/-- Summing the token indicators counts actual raw ONE IDs, excluding delimiters and other tokens.
Source: the new internal scalar feature for the no-scratchpad parity prompt. -/
theorem one_values_sum {d : ℕ} (tokens : Tokens) (direction : EucSpace d) :
    (tokens.map (oneValue direction)).sum = (tokens.count oneBit : ℝ) • direction := by
  have h := List.sum_map_ite_eq tokens (fun _ => direction) (fun _ => 0) oneBit
  unfold oneValue
  simpa only [sub_zero, List.sum_map_zero, add_zero,
    Nat.cast_smul_eq_nsmul] using h

/-- Reading the raw list as the model's finite-position array preserves that same count.
Source: the exact List Int/Fin array representation used by GPTMini.TokenInterface. -/
theorem one_values_fin_sum {d : ℕ} (tokens : Tokens) (direction : EucSpace d) :
    (∑ j : Fin tokens.length, oneValue direction (tokens.get j)) =
      (tokens.count oneBit : ℝ) • direction := by
  rw [← List.sum_ofFn]
  change (List.ofFn (oneValue direction ∘ tokens.get)).sum = _
  rw [← List.map_ofFn, List.ofFn_get, one_values_sum]

/-- Zero queries and keys give exact uniform weights at the final causal position.
Source: CausalMHA.forward at f11b6e2, with its original finite-temperature softmax. -/
theorem uniform_weights (cfg : Config) {T : ℕ} (alpha eps : ℝ) (i j : Fin T)
    (hfinal : ∀ r : Fin T, r.val ≤ i.val) :
    causalAttnWeights cfg alpha eps (fun _ => 0) (fun _ => 0) i j = 1 / (T : ℝ) := by
  rw [causalAttnWeights, ite_eq_right (not_lt.mpr (hfinal j))]
  simp only [preScore, score, normL2, smul_zero, inner_zero_left, mul_zero, Real.exp_zero]
  simp only [ite_eq_left (hfinal _)]
  simp

example : ∀ r : Fin 2, r.val ≤ (1 : Fin 2).val := by
  intro r
  have hr := r.isLt
  change r.val ≤ 1
  omega

/-- With zero self-value, XSA leaves the exact arithmetic mean of all visible values.
Source: uniform softmax and the actual self-value subtraction at f11b6e2. -/
theorem head_uniform_average (cfg : Config) {T : ℕ} (alpha eps : ℝ)
    (v : Fin T → EucSpace cfg.head_dim) (positions : Fin T → ℝ) (i : Fin T)
    (hfinal : ∀ r : Fin T, r.val ≤ i.val) (hself : v i = 0) :
    attentionHead cfg alpha eps (fun _ => 0) (fun _ => 0) v positions i =
      (1 / (T : ℝ)) • ∑ j, v j := by
  unfold attentionHead xsaProjection
  simp only [rope_zero]
  simp only [hself, normL2, smul_zero, inner_zero_right, sub_zero]
  unfold attnOutput
  simp_rw [uniform_weights cfg alpha eps i _ hfinal]
  rw [← Finset.smul_sum]
  simp

example (direction : EucSpace 16) :
    (∀ r : Fin 2, r.val ≤ (1 : Fin 2).val) ∧
      (fun j : Fin 2 => if j.val = 0 then direction else 0) 1 = 0 := by
  constructor
  · intro r
    have hr := r.isLt
    change r.val ≤ 1
    omega
  · simp

/-- A raw prompt ending in a non-ONE token carries the exact normalized ONE count in its head output.
Source: the actual head, with faithful token-indicator values and zero Q/K. -/
theorem head_raw_one_count (cfg : Config) (alpha eps : ℝ)
    (head : ℤ) (body : Tokens) (last : ℤ) (direction : EucSpace cfg.head_dim)
    (hlast : last ≠ oneBit) :
    attentionHead cfg alpha eps (fun _ => 0) (fun _ => 0)
        (fun j : Fin (head :: (body ++ [last])).length =>
          oneValue direction ((head :: (body ++ [last])).get j))
        (fun j => (j.val : ℝ)) ⟨body.length + 1, by simp⟩ =
      (((head :: (body ++ [last])).count oneBit : ℝ) / (body.length + 2 : ℕ)) • direction := by
  have hself : oneValue direction ((head :: (body ++ [last])).get
      ⟨body.length + 1, by simp⟩) = 0 := by
    simp only [List.get_eq_getElem, List.getElem_cons_succ,
      List.getElem_append_right (le_refl body.length), Nat.sub_self, List.getElem_cons_zero]
    exact ite_eq_right hlast
  have hfinal (r : Fin (head :: (body ++ [last])).length) :
      r.val ≤ (⟨body.length + 1, by simp⟩ : Fin (head :: (body ++ [last])).length).val := by
    have hr := r.isLt
    simp only [List.length_cons, List.length_append, List.length_nil] at hr
    change r.val ≤ body.length + 1
    omega
  have h := head_uniform_average cfg alpha eps
    (fun j => oneValue direction ((head :: (body ++ [last])).get j))
    (fun j => (j.val : ℝ)) ⟨body.length + 1, by simp⟩ hfinal hself
  rw [one_values_fin_sum, smul_smul] at h
  simpa only [one_div_mul_eq_div, List.length_cons, List.length_append,
    List.length_nil] using h

example : (18 : ℤ) ≠ oneBit := by decide

/-- BOS and SEP do not add ones; the head's raw count is the bit word's mathematical count.
Source: Bits.prompt at cbafbe9 with ZERO=21, ONE=22, BOS=1 and SEP=18. -/
theorem parityPrompt_one_count (bits : List Bool) :
    (parityPrompt bits).count oneBit = bits.count true := by
  have hbody : (bitTokens bits).count oneBit = bits.count true := by
    induction bits with
    | nil => rfl
    | cons bit bits ih =>
        cases bit <;> simp [bitTokens, oneBit, zeroBit] at ih ⊢ <;> omega
  simp only [parityPrompt, List.count_append, List.count_cons, hbody]
  norm_num [bos, oneBit, sep]

/-- The actual softmax/XSA head preserves the complete ones-count feature of every raw parity prompt.
Source: Basis no-scratchpad parity at cbafbe9; no desired output label is assumed. -/
theorem head_parity_count (cfg : Config) (alpha eps : ℝ) (bits : List Bool)
    (direction : EucSpace cfg.head_dim) :
    attentionHead cfg alpha eps (fun _ => 0) (fun _ => 0)
        (fun j : Fin (parityPrompt bits).length => oneValue direction ((parityPrompt bits).get j))
        (fun j => (j.val : ℝ)) ⟨bits.length + 1, by simp [parityPrompt_length]⟩ =
      ((bits.count true : ℝ) / (bits.length + 2 : ℕ)) • direction := by
  have h := head_raw_one_count cfg alpha eps bos (bitTokens bits) sep direction (by decide)
  change attentionHead cfg alpha eps (fun _ => 0) (fun _ => 0)
    (fun j => oneValue direction ((parityPrompt bits).get j))
    (fun j => (j.val : ℝ)) ⟨(bitTokens bits).length + 1, by simp [parityPrompt, bitTokens]⟩ =
      (((parityPrompt bits).count oneBit : ℝ) / ((bitTokens bits).length + 2 : ℕ)) • direction at h
  rw [parityPrompt_one_count] at h
  simpa only [bitTokens, List.length_map] using h

/-- A unit direction lets the exact head signal recover the integer ones count.
Source: the preceding actual-head computation; length scaling removes the softmax average. -/
theorem head_parity_count_recovers (cfg : Config) (alpha eps : ℝ) (bits : List Bool)
    (direction : EucSpace cfg.head_dim) (hunit : ‖direction‖ = 1) :
    inner (𝕜 := ℝ)
        (attentionHead cfg alpha eps (fun _ => 0) (fun _ => 0)
          (fun j : Fin (parityPrompt bits).length => oneValue direction ((parityPrompt bits).get j))
          (fun j => (j.val : ℝ)) ⟨bits.length + 1, by simp [parityPrompt_length]⟩) direction *
        (bits.length + 2 : ℕ) = (bits.count true : ℝ) := by
  rw [head_parity_count, real_inner_smul_left, real_inner_self_eq_norm_sq, hunit]
  have hn : (bits.length + 2 : ℕ) ≠ 0 := by omega
  norm_num
  field_simp

example : ‖(EuclideanSpace.single (0 : Fin 16) (1 : ℝ))‖ = 1 := by
  simp [PiLp.norm_single]

/-- The count head alone loses the parity completion phase on two valid six-token prefixes.
Source: Parity.solve at cbafbe9; the first input needs EVEN, the second needs EOS.
The rest of the residual stream or other heads can retain this missing phase. -/
theorem parity_phase_count_collision (cfg : Config) (alpha eps : ℝ)
    (direction : EucSpace cfg.head_dim) :
    attentionHead cfg alpha eps (fun _ => 0) (fun _ => 0)
        (fun j : Fin 6 => oneValue direction (([1, 22, 22, 21, 21, 18] : Tokens).get j))
        (fun j => (j.val : ℝ)) 5 =
      attentionHead cfg alpha eps (fun _ => 0) (fun _ => 0)
        (fun j : Fin 6 => oneValue direction (([1, 22, 22, 21, 18, 24] : Tokens).get j))
        (fun j => (j.val : ℝ)) 5 ∧
    parityNext [1, 22, 22, 21, 21, 18] = evenToken ∧
      parityNext [1, 22, 22, 21, 18, 24] = eos := by
  have hx := head_raw_one_count cfg alpha eps 1 [22, 22, 21, 21] 18 direction (by decide)
  have hy := head_raw_one_count cfg alpha eps 1 [22, 22, 21, 18] 24 direction (by decide)
  norm_num [oneBit] at hx hy
  exact ⟨hx.trans hy.symm, by decide, by decide⟩

end Transformer.GPTMini.Semantics
