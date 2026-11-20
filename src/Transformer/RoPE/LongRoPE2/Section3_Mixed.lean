module

public import Transformer.RoPE.LongRoPE2.Section3_Search
public import Transformer.RoPE.LongRoPE.Section3_Progressive

/-!
# LongRoPE2 §3.3, Fig. 5, Fig. 10 and Appendix B: mixed context window training

Shang et al., *LongRoPE2*, arXiv:2502.20082v1. Positions are natural numbers; a RoPE phase
depends on the distance between the query and the key only (RoFormer, Eq. (16)).

* `short_window_phases`: §3.3 and Fig. 5(a), "For segments labeled as short windows, the original
  RoPE is used with attention masks to prevent self-attention across different documents":
  inside a document of at most `L_train` tokens every causal phase of the original RoPE lies in
  `[0, L_train·θ)`;
* `long_window_phases`: §3.3 and Fig. 5(b), "For long-context segments, we apply the rescaled
  RoPE for full attention": in a segment of `L > L_train` tokens the original RoPE reaches the
  phase `(L − 1)θ ≥ L_train·θ`, and with `λ ≥ L/L_train` every causal phase stays below `L_train·θ`;
* `switch_once`, `cached_key_rotates`: Fig. 10 and Appendix B, "the switch between rescaled and
  original RoPE is triggered when the combined length of the input context and generated tokens
  exceeds the pre-trained context window. Switching to rescaled RoPE for long-context inference
  requires a one-time recalculation of the KV cache": a length beyond `L_train` keeps the rescaled
  frequency as it grows, and for `λ > 1` the rescaled RoPE rotates a key cached at a position
  `m > 0` with `mθ ≤ π` by another angle;
* `printed_inv_freq`: Fig. 10, `inv_freq = 1.0 / (base**torch.Tensor([0, 1, 2, 3, ..., rope_dim]))`:
  its entry `j ≥ 1` is `base^{−j}`, strictly below Eq. (5)'s `θ_j = base^{−2j/d}` for `d > 2`;
* `attention_cost`: Appendix B, "Given that GPU memory and computation time increase
  exponentially with sequence length", against Appendix A, "Attention computation and memory
  costs grow quadratically with context length": causal attention over `n` tokens scores
  `n(n+1)/2` pairs, between `n²/2` and `n²`, and `o(c^n)` for every `c > 1`.
-/

@[expose] public section

open Filter Topology

namespace Transformer.RoPE.LongRoPE2

/-- §3.3 and Fig. 5(a): inside a masked document of at most `L_train` tokens, every causal phase
of the original RoPE lies in `[0, L_train·θ)`. -/
theorem short_window_phases {train omega : ℝ} {start len q k : ℕ} (homega : 0 < omega)
    (hlen : (len : ℝ) ≤ train) (hq : q < start + len) (hk : start ≤ k) (hkq : k ≤ q) :
    0 ≤ ((q - k : ℕ) : ℝ) * omega ∧ ((q - k : ℕ) : ℝ) * omega < train * omega :=
  ⟨by positivity, mul_lt_mul_of_pos_right
    ((Nat.cast_lt.mpr (show q - k < len by omega)).trans_le hlen) homega⟩

example : 0 ≤ ((3000 - 2000 : ℕ) : ℝ) * frequency 10000 96 40 ∧
    ((3000 - 2000 : ℕ) : ℝ) * frequency 10000 96 40 < 2048 * frequency 10000 96 40 :=
  short_window_phases (start := 1500) (len := 1600) (Real.rpow_pos_of_pos (by norm_num) _)
    (by norm_num) (by norm_num) (by norm_num) (by norm_num)

/-- §3.3 and Fig. 5(b): a segment of `L > L_train` tokens reaches the phase `(L − 1)θ ≥ L_train·θ`
under the original RoPE, and with `λ ≥ L/L_train` every causal phase stays below `L_train·θ`. -/
theorem long_window_phases {omega factor : ℝ} {train L q k : ℕ} (homega : 0 < omega)
    (htrain : 0 < train) (hlong : train < L) (hfactor : (L : ℝ) / train ≤ factor)
    (hkq : k ≤ q) (hq : q < L) :
    (train : ℝ) * omega ≤ ((L - 1 : ℕ) : ℝ) * omega ∧
      ((q - k : ℕ) : ℝ) * scaledFrequency omega factor < train * omega := by
  have htrain' : (0 : ℝ) < train := by exact_mod_cast htrain
  have hL : (0 : ℝ) < L := by exact_mod_cast htrain.trans hlong
  have hfac : 0 < factor := (div_pos hL htrain').trans_le hfactor
  refine ⟨mul_le_mul_of_nonneg_right (by exact_mod_cast (show train ≤ L - 1 by omega))
    homega.le, ?_⟩
  calc ((q - k : ℕ) : ℝ) * scaledFrequency omega factor < L * scaledFrequency omega factor :=
        mul_lt_mul_of_pos_right (by exact_mod_cast (show q - k < L by omega))
          (div_pos homega hfac)
    _ ≤ train * omega := (phase_range_iff homega hfac htrain').mpr hfactor

example : ((2048 : ℕ) : ℝ) * frequency 10000 96 40 ≤
      ((131072 - 1 : ℕ) : ℝ) * frequency 10000 96 40 ∧
    ((131071 - 0 : ℕ) : ℝ) * scaledFrequency (frequency 10000 96 40) 64 <
      ((2048 : ℕ) : ℝ) * frequency 10000 96 40 :=
  long_window_phases (Real.rpow_pos_of_pos (by norm_num) _) (by norm_num) (by norm_num)
    (by norm_num) (by norm_num) (by norm_num)

/-- Fig. 10 and Appendix B: once the length exceeds `L_train`, every longer length keeps the
rescaled frequency. -/
theorem switch_once {train n n' omega factor : ℝ} (hn : train < n) (hn' : n ≤ n') :
    mixedFrequency train n' omega factor = scaledFrequency omega factor := by
  rw [mixedFrequency, ite_eq_right (not_le.mpr (hn.trans_le hn'))]

example : mixedFrequency 2048 4096 (frequency 10000 96 40) 64 =
    scaledFrequency (frequency 10000 96 40) 64 :=
  switch_once (n := 2049) (by norm_num) (by norm_num)

/-- Appendix B: for `λ > 1` the rescaled RoPE rotates a key cached at a position `m > 0` with
`mθ ≤ π` by another angle. -/
theorem cached_key_rotates {omega factor m : ℝ} (homega : 0 < omega) (hfactor : 1 < factor)
    (hm : 0 < m) (hpi : m * omega ≤ Real.pi) :
    Real.cos (m * omega) < Real.cos (m * scaledFrequency omega factor) := by
  have hθ : scaledFrequency omega factor < omega := by
    rw [scaledFrequency, div_lt_iff₀ (by linarith)]
    nlinarith
  exact Real.cos_lt_cos_of_nonneg_of_le_pi
    (mul_nonneg hm.le (div_pos homega (by linarith)).le) hpi (mul_lt_mul_of_pos_left hθ hm)

example : Real.cos (1 * frequency 10000 96 40) <
    Real.cos (1 * scaledFrequency (frequency 10000 96 40) 64) := by
  have h : frequency 10000 96 40 ≤ 1 :=
    Real.rpow_le_one_of_one_le_of_nonpos (by norm_num) (by norm_num)
  exact cached_key_rotates (Real.rpow_pos_of_pos (by norm_num) _) (by norm_num) one_pos
    (by linarith [Real.pi_gt_three])

/-- Fig. 10: the printed inverse frequency `1/base^j` lies strictly below Eq. (5)'s
`θ_j = base^{−2j/d}` at every `j ≥ 1` when `d > 2`. -/
theorem printed_inv_freq {base : ℝ} {d j : ℕ} (hbase : 1 < base) (hd : 2 < d) (hj : 0 < j) :
    1 / base ^ (j : ℝ) < frequency base d j := by
  have hd' : (2 : ℝ) < d := by exact_mod_cast hd
  have hj' : (0 : ℝ) < j := by exact_mod_cast hj
  rw [one_div, ← Real.rpow_neg (by linarith), frequency]
  apply Real.rpow_lt_rpow_of_exponent_lt hbase
  rw [neg_div, neg_lt_neg_iff, div_lt_iff₀ (by linarith)]
  nlinarith

example : 1 / (10000 : ℝ) ^ ((1 : ℕ) : ℝ) < frequency 10000 96 1 :=
  printed_inv_freq (by norm_num) (by norm_num) (by norm_num)

/-- Appendix B against Appendix A: causal attention over `n` tokens scores `n(n+1)/2` pairs,
between `n²/2` and `n²`, and `o(c^n)` for every `c > 1`. -/
theorem attention_cost (n : ℕ) {c : ℝ} (hc : 1 < c) :
    LongRoPE.causalScores n * 2 = n * (n + 1) ∧ n ^ 2 ≤ 2 * LongRoPE.causalScores n ∧
      LongRoPE.causalScores n ≤ n ^ 2 ∧
      Tendsto (fun m : ℕ => (LongRoPE.causalScores m : ℝ) / c ^ m) atTop (𝓝 0) := by
  have h := LongRoPE.causal_scores n
  exact ⟨h, by nlinarith, by nlinarith [Nat.le_mul_self n],
    LongRoPE.causal_scores_subexponential hc⟩

example : LongRoPE.causalScores 131072 * 2 = 131072 * (131072 + 1) ∧
    131072 ^ 2 ≤ 2 * LongRoPE.causalScores 131072 ∧ LongRoPE.causalScores 131072 ≤ 131072 ^ 2 ∧
    Tendsto (fun m : ℕ => (LongRoPE.causalScores m : ℝ) / 2 ^ m) atTop (𝓝 0) :=
  attention_cost 131072 (by norm_num)

end Transformer.RoPE.LongRoPE2
