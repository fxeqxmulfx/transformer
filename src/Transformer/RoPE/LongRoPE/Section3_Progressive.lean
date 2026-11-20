module

public import Transformer.RoPE.LongRoPE.Section3_Search

/-!
# LongRoPE §1, §3.3 and Appendix A.1: the progressive extension to 2048k

Ding et al., arXiv:2402.13753v1. Lengths count `1k = 1024` tokens, as the caption of Table 6
(Books3) states; `4k = 4096` is LLaMA2's window (§1).

* `new_positions`: §1, "an extension from 4k to >1000k introduces more than 90% new positions":
  more than `99.5%` of the positions of a window longer than `10^6` lie at or beyond `4096`;
* `progressive_ratios`: §3.3, the searches for `128k` and `256k` are "`32×` and `64×`" extensions,
  the secondary search from `256k` to `2048k` is `8×` and from `128k` is `16×` (§4.2, Appendix A.3),
  "The final extension ratio is `512×`", `64·8`; `2048k` is "beyond 2 million tokens" (§1, §3),
  and `400 + 600` fine-tuning steps are "up to only 1k fine-tuning steps" (abstract);
* `pi_crowding`: §3.3, "With a `512×` extension ratio, positions within the original 4k context
  window become particularly crowded": PI at `512×` gives every position of `[0, 4096)` an angle
  below `8θ_i`, where RoPE reaches `4095θ_i`;
* `causal_scores`, `causal_scores_subexponential`: Appendix A.1, "the GPU memory and computation
  time increase exponentially with the sequence length": exact causal attention over `n` tokens
  scores `n(n+1)/2` query–key pairs per head and layer, and this count divided by `c^n` tends to
  `0` for every `c > 1`.
-/

@[expose] public section

open Filter Topology

namespace Transformer.RoPE.LongRoPE

/-- §1: more than `99.5%` of the positions `[0, L')` of a window `L' > 10^6` are new beyond the
`4096` pretrained ones. -/
theorem new_positions {L' : ℝ} (hL' : 1000000 < L') : 0.995 < (L' - 4096) / L' := by
  rw [lt_div_iff₀ (by linarith)]
  linarith

example : 0.995 < ((1024000 : ℝ) - 4096) / 1024000 := new_positions (by norm_num)

/-- §3.3: the extension ratios of the progressive extension, its length and its fine-tuning
steps. -/
theorem progressive_ratios :
    YaRN.contextScale 4096 (128 * 1024) = 32 ∧ YaRN.contextScale 4096 (256 * 1024) = 64 ∧
      YaRN.contextScale (256 * 1024) (2048 * 1024) = 8 ∧
      YaRN.contextScale (128 * 1024) (2048 * 1024) = 16 ∧
      YaRN.contextScale 4096 (2048 * 1024) = 64 * 8 ∧
      YaRN.contextScale 4096 (2048 * 1024) = 512 ∧ (2 * 10 ^ 6 : ℝ) < 2048 * 1024 ∧
      400 + 600 ≤ 1000 := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, by norm_num, by norm_num⟩ <;>
    rw [YaRN.contextScale, max_eq_right (by norm_num)] <;> norm_num

/-- §3.3: at PI's `512×` every position of the original window `[0, 4096)` has an angle in
`[0, 8θ_i)`. -/
theorem pi_crowding {base : ℝ} (hbase : 0 < base) (d i : ℕ) {m : ℝ} (hm0 : 0 ≤ m)
    (hm : m < 4096) :
    0 ≤ unifiedAngle base d (fun _ => YaRN.contextScale 4096 (2048 * 1024)) m i ∧
      unifiedAngle base d (fun _ => YaRN.contextScale 4096 (2048 * 1024)) m i <
        8 * frequency base d i := by
  have hθ : 0 < frequency base d i := Real.rpow_pos_of_pos hbase _
  rw [pi_unifiedAngle hbase, progressive_ratios.2.2.2.2.2.1]
  exact ⟨mul_nonneg (div_nonneg hm0 (by norm_num)) hθ.le,
    mul_lt_mul_of_pos_right (by linarith) hθ⟩

example : 0 ≤ unifiedAngle 10000 128 (fun _ => YaRN.contextScale 4096 (2048 * 1024)) 4000 3 ∧
    unifiedAngle 10000 128 (fun _ => YaRN.contextScale 4096 (2048 * 1024)) 4000 3 <
      8 * frequency 10000 128 3 :=
  pi_crowding (by norm_num) 128 3 (by norm_num) (by norm_num)

/-- Exact causal attention over `n` tokens: the query at the position `m` scores the `m + 1` keys
`0, …, m`. -/
def causalScores (n : ℕ) : ℕ := ∑ m ∈ Finset.range n, (m + 1)

/-- Appendix A.1: causal attention over `n` tokens scores `n(n+1)/2` query–key pairs. -/
theorem causal_scores (n : ℕ) : causalScores n * 2 = n * (n + 1) := by
  induction n with
  | zero => rfl
  | succ k ih =>
    unfold causalScores at *
    rw [Finset.sum_range_succ, add_mul, ih]
    ring

/-- Appendix A.1: the count of causal scores is at most `n²`, so it is `o(c^n)` for every
`c > 1`. -/
theorem causal_scores_subexponential {c : ℝ} (hc : 1 < c) :
    Tendsto (fun n : ℕ => (causalScores n : ℝ) / c ^ n) atTop (𝓝 0) := by
  have hbound : ∀ n : ℕ, (causalScores n : ℝ) / c ^ n ≤ (n : ℝ) ^ 2 / c ^ n := by
    intro n
    have h : causalScores n ≤ n ^ 2 := by nlinarith [causal_scores n]
    exact div_le_div_of_nonneg_right (by exact_mod_cast h) (pow_nonneg (by linarith) n)
  exact squeeze_zero (fun n => div_nonneg (Nat.cast_nonneg _) (pow_nonneg (by linarith) n))
    hbound (tendsto_pow_const_div_const_pow_of_one_lt 2 hc)

example : Tendsto (fun n : ℕ => (causalScores n : ℝ) / 2 ^ n) atTop (𝓝 0) :=
  causal_scores_subexponential (by norm_num)

end Transformer.RoPE.LongRoPE
