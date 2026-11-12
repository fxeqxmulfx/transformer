module

public import Transformer.Grokking.Protocol.Division

/-!
# The last normalization and the readout bound the logits

Source: arXiv:2201.02177v1 (Power et al.), appendix *Additional experimental details*, "a
standard decoder-only transformer"; openai/grok at 3d64b1d8, `grok/transformer.py`:
`DecoderBlock` (a block returns the output of its second `LayerNorm`) and `Transformer` (the
readout `self.linear`, without a bias); `experiments/grok_division/PROTOCOL.md`, *The model*.

A normalized vector of `n` entries has no entry above `√n` in absolute value, whatever was
normalized: the square of one centred entry is at most the sum of all of them, which is `n`
times the variance. So an entry of the output of `layerNorm` is at most `√n |gain| + |bias|`
(`abs_layerNorm_le`), and since a block is post-norm the output of the last block is bounded
by the gains and biases of the second normalizations alone (`abs_decodeTo_le`): no embedding,
attention or feed-forward parameter enters, and neither do the tokens read or the activations.
The logits are the readout of that output, so they are bounded by the readout and those gains
and biases (`abs_logits_le`).

This is a bound of the decoder at given parameters. It says nothing of a run by itself;
`ReadoutStationary` and `DecayedLogits` apply it along the runs of the optimizer.
-/

@[expose] public section

namespace Transformer.Grokking.Protocol

variable {S : Sizes}

/-- **An entry of a normalized vector is at most `√n |gain| + |bias|`**, whatever vector is
normalized. -/
theorem abs_layerNorm_le {n : ℕ} (g b z : Fin n → ℝ) (i : Fin n) :
    |layerNorm g b z i| ≤ √n * |g i| + |b i| := by
  have hn : (0 : ℝ) < n := Nat.cast_pos.2 i.pos
  have hroot := layerNorm_root_pos z
  unfold layerNorm
  generalize (∑ j, z j) / (n : ℝ) = m at hroot ⊢
  have hsq : (z i - m) ^ 2 ≤ n * ((∑ j, (z j - m) ^ 2) / n + 1 / 10 ^ 5) := by
    have h1 : (z i - m) ^ 2 ≤ ∑ j, (z j - m) ^ 2 :=
      Finset.single_le_sum (f := fun j => (z j - m) ^ 2) (fun j _ => sq_nonneg _)
        (Finset.mem_univ i)
    have h2 : (n : ℝ) * ((∑ j, (z j - m) ^ 2) / n + 1 / 10 ^ 5) =
        ∑ j, (z j - m) ^ 2 + n / 10 ^ 5 := by
      field_simp
    have h3 : (0 : ℝ) ≤ n / 10 ^ 5 := by positivity
    linarith
  generalize (∑ j, (z j - m) ^ 2) / (n : ℝ) + 1 / 10 ^ 5 = v at hroot hsq ⊢
  have hq : |(z i - m) / √v| ≤ √n := by
    rw [abs_div, abs_of_pos hroot, div_le_iff₀ hroot, ← Real.sqrt_mul (Nat.cast_nonneg n)]
    exact Real.abs_le_sqrt hsq
  calc |(z i - m) / √v * g i + b i| ≤ |(z i - m) / √v| * |g i| + |b i| := by
        rw [← abs_mul]
        exact abs_add_le _ _
    _ ≤ √n * |g i| + |b i| := by gcongr

/-- The output of a block is bounded by the gain and the bias of its second normalization. -/
lemma abs_block_le (θ : Param S → ℝ) (act : Activation S) (x : Fin S.context → Fin S.width → ℝ)
    (l : Fin S.blocks) (t : Fin S.context) (i : Fin S.width) :
    |block θ act x l t i| ≤ √S.width * |θ (.gain2 l i)| + |θ (.bias2 l i)| :=
  abs_layerNorm_le _ _ _ i

/-- **The output of the last block is bounded by the second normalizations alone**: if their
gains are at most `G` and their biases at most `B` in absolute value, every entry is at most
`√width · G + B`, for all tokens and all activations. -/
theorem abs_decodeTo_le (hS : 0 < S.blocks) {θ : Param S → ℝ} {G B : ℝ}
    (hG : ∀ l i, |θ (.gain2 l i)| ≤ G) (hB : ∀ l i, |θ (.bias2 l i)| ≤ B) (act : Activation S)
    (tokens : Fin S.context → Fin S.vocab) (t : Fin S.context) (i : Fin S.width) :
    |decodeTo θ act tokens S.blocks t i| ≤ √S.width * G + B := by
  obtain ⟨n, hn⟩ := Nat.exists_eq_add_one_of_ne_zero hS.ne'
  have hlt : n < S.blocks := by omega
  have h : decodeTo θ act tokens (n + 1) t i =
      block θ act (decodeTo θ act tokens n) ⟨n, hlt⟩ t i := by
    simp only [decodeTo, hlt, ↓reduceDIte]
  rw [hn, h]
  exact (abs_block_le θ act _ _ t i).trans
    (add_le_add (mul_le_mul_of_nonneg_left (hG _ i) (Real.sqrt_nonneg _)) (hB _ i))

/-- **The logits are bounded by the readout and the second normalizations**: with the readout
at most `W`, those gains at most `G` and those biases at most `B` in absolute value, every
logit is at most `width · W · (√width · G + B)`. -/
theorem abs_logits_le (hS : 0 < S.blocks) {θ : Param S → ℝ} {W G B : ℝ}
    (hW : ∀ c i, |θ (.out c i)| ≤ W) (hG : ∀ l i, |θ (.gain2 l i)| ≤ G)
    (hB : ∀ l i, |θ (.bias2 l i)| ≤ B) (act : Activation S)
    (tokens : Fin S.context → Fin S.vocab) (t : Fin S.context) (c : Fin S.vocab) :
    |logits θ act tokens t c| ≤ S.width * (W * (√S.width * G + B)) := by
  unfold logits
  refine (Finset.abs_sum_le_sum_abs _ _).trans ?_
  have hterm : ∀ i ∈ (Finset.univ : Finset (Fin S.width)),
      |θ (.out c i) * decodeTo θ act tokens S.blocks t i| ≤ W * (√S.width * G + B) :=
    fun i _ => by
      rw [abs_mul]
      exact mul_le_mul (hW c i) (abs_decodeTo_le hS hG hB act tokens t i) (abs_nonneg _)
        ((abs_nonneg _).trans (hW c i))
  refine (Finset.sum_le_sum hterm).trans (le_of_eq ?_)
  rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]

/-! The witnesses: a vector of two entries normalized with the gain `3` and the bias `1/2`,
and the decoder of the paper's sizes with every parameter `1`, whose last block returns at most
`√128 + 1` and whose logits after `=` are at most `128 (√128 + 1)`, which is less than `1577`. -/

example : |layerNorm ![3, 3] ![1 / 2, 1 / 2] ![7, -2] 0| ≤ √(2 : ℕ) * 3 + 1 / 2 := by
  simpa using abs_layerNorm_le ![3, 3] ![1 / 2, 1 / 2] ![7, -2] 0

example (tokens : Fin 6 → Fin 239) (t : Fin 6) (i : Fin 128) :
    |decodeTo (fun _ => 1) (relu paperSizes) tokens 2 t i| ≤ √(128 : ℕ) * 1 + 1 :=
  abs_decodeTo_le (S := paperSizes) (by decide) (by simp) (by simp) _ tokens t i

example (x : Equation) (c : Fin 239) :
    |answerLogits (fun _ => 1) x c| ≤ (128 : ℕ) * (1 * (√(128 : ℕ) * 1 + 1)) :=
  abs_logits_le (S := paperSizes) (by decide) (by simp) (by simp) (by simp) _ _ _ c

end Transformer.Grokking.Protocol
