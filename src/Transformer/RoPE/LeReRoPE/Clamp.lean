module

public import Transformer.RoPE.LeRoPE.Section7_Extrapolation
public import Transformer.RoPE.ReRoPE.Section2_Fusion

/-!
# LeRoPE's dominant band under ReRoPE's clamp

In no paper. LeRoPE's dominant band (Karypis et al., arXiv:2607.10134v1, §6) is `≤ 0` throughout
the training window `[0, L]`, and unmodified it turns positive within half a period beyond it
(§7, `Transformer.RoPE.LeRoPE.exists_pos_score_beyond_window`). ReRoPE (Su,
kexue.fm/archives/9708) scores the distance `d` at the position `min(d, w)`, Leaky ReRoPE at
`w + (d − w)/k` beyond the window `w` (`Transformer.RoPE.ReRoPE.rectifiedPosition`,
`Transformer.RoPE.ReRoPE.leakyPosition`):

* `score_rectified_nonpos`: with a window `w ≤ L` ReRoPE keeps the band `≤ 0` at every
  distance. Inside the window the positions are the distances themselves, so the band keeps its
  trained values there exactly, where the interpolation of Appendix D.2
  (`Transformer.RoPE.LeRoPE.score_ntkMultiplier_nonpos`) moves every distance `d ≠ 0`;
* `leakyPosition_mem_Icc`, `score_leaky_nonpos`: Leaky ReRoPE keeps the band `≤ 0` up to the
  distance `w + k (L − w)`, where its positions reach `L`;
* `exists_pos_score_leaky`: beyond that the band turns positive at some distance at most
  `w + k (L + π/θ − w)`. Leaky ReRoPE delays the flip `k`-fold; only the clamp removes it.
-/

@[expose] public section

namespace Transformer.RoPE.LeReRoPE

open Set Transformer.RoPE.LeRoPE Transformer.RoPE.ReRoPE

/-- **ReRoPE keeps the band at its trained values**: with a window `0 ≤ w ≤ L` the band scores
the distance `d ≥ 0` at `min(d, w) ∈ [0, w]`, so a band `≤ 0` throughout the training window
stays `≤ 0` at every distance. -/
theorem score_rectified_nonpos {a b θ L w : ℝ} (hw : 0 ≤ w) (hwL : w ≤ L)
    (hneg : ∀ d ∈ Icc 0 L, score a b θ d ≤ 0) {d : ℝ} (hd : 0 ≤ d) :
    score a b θ (rectifiedPosition w d) ≤ 0 := by
  have h := rectifiedPosition_mem_Icc hw hd
  exact hneg _ ⟨h.1, h.2.trans hwL⟩

/-- The hypotheses of `score_rectified_nonpos` hold for the band of Table 10's 2.52B run with the
window `1024`, at the distance `10⁶`. -/
example := score_rectified_nonpos (a := 0) (b := -1) (θ := 0.001507) (L := 2048) (w := 1024)
  (by norm_num) (by norm_num) score_table10_nonpos (d := 1000000) (by norm_num)

/-- **Leaky ReRoPE's positions stay in the training window** up to the distance `w + k (L − w)`,
for `0 ≤ w ≤ L` and `k > 0`. -/
theorem leakyPosition_mem_Icc {w k L d : ℝ} (hw : 0 ≤ w) (hwL : w ≤ L) (hk : 0 < k)
    (hd : d ∈ Icc 0 (w + k * (L - w))) : leakyPosition w k d ∈ Icc 0 L := by
  rw [leakyPosition]
  split_ifs with h
  · exact ⟨hd.1, by linarith⟩
  · have h0 : 0 ≤ (d - w) / k := div_nonneg (by linarith) hk.le
    have h1 : (d - w) / k ≤ L - w := by
      rw [div_le_iff₀ hk]
      linarith [hd.2, mul_comm k (L - w)]
    exact ⟨by linarith, by linarith⟩

/-- The hypotheses of `leakyPosition_mem_Icc` hold for the window `1024`, `k = 16` and
`L = 2048` at the distance `17408 = 1024 + 16 · 1024`. -/
example := leakyPosition_mem_Icc (w := 1024) (k := 16) (L := 2048) (d := 17408) (by norm_num)
  (by norm_num) (by norm_num) ⟨by norm_num, by norm_num⟩

/-- **Leaky ReRoPE keeps the band `≤ 0` up to the distance `w + k (L − w)`**. -/
theorem score_leaky_nonpos {a b θ L w k : ℝ} (hw : 0 ≤ w) (hwL : w ≤ L) (hk : 0 < k)
    (hneg : ∀ d ∈ Icc 0 L, score a b θ d ≤ 0) {d : ℝ} (hd : d ∈ Icc 0 (w + k * (L - w))) :
    score a b θ (leakyPosition w k d) ≤ 0 :=
  hneg _ (leakyPosition_mem_Icc hw hwL hk hd)

/-- The hypotheses of `score_leaky_nonpos` hold for the band of the examples with the window
`1024` and `k = 16`. -/
example := score_leaky_nonpos (a := 0) (b := -1) (θ := 0.001507) (L := 2048) (w := 1024)
  (k := 16) (by norm_num) (by norm_num) (by norm_num) score_table10_nonpos (d := 17408)
  ⟨by norm_num, by norm_num⟩

/-- **Leaky ReRoPE delays the flip `k`-fold, and does not remove it**: a band `≤ 0` throughout the
training window `[0, L]`, `L > 0`, and not zero is positive at the Leaky ReRoPE position of some
distance in `(w + k (L − w), w + k (L + π/θ − w)]`, for every window `w ≤ L` and `k > 0`. -/
theorem exists_pos_score_leaky {a b θ L w k : ℝ} (hθ : 0 < θ) (hL : 0 < L) (hwL : w ≤ L)
    (hk : 0 < k) (hab : a ≠ 0 ∨ b ≠ 0) (hneg : ∀ d ∈ Icc 0 L, score a b θ d ≤ 0) :
    ∃ d ∈ Ioc (w + k * (L - w)) (w + k * (L + Real.pi / θ - w)),
      0 < score a b θ (leakyPosition w k d) := by
  obtain ⟨e, he, hpos⟩ := exists_pos_score_beyond_window hθ hL hab hneg
  have hlt : k * (L - w) < k * (e - w) := mul_lt_mul_of_pos_left (by linarith [he.1]) hk
  have hle : k * (e - w) ≤ k * (L + Real.pi / θ - w) :=
    mul_le_mul_of_nonneg_left (by linarith [he.2]) hk.le
  have hpos' : 0 ≤ k * (e - w) := mul_nonneg hk.le (by linarith [he.1])
  refine ⟨w + k * (e - w), ⟨by linarith, by linarith⟩, ?_⟩
  have hpos_eq : w + (w + k * (e - w) - w) / k = e := by
    rw [show w + k * (e - w) - w = k * (e - w) by ring, mul_div_cancel_left₀ _ hk.ne']
    ring
  rwa [leakyPosition, ite_eq_right (by linarith), hpos_eq]

/-- The hypotheses of `exists_pos_score_leaky` hold for the band of the examples with the window
`1024` and `k = 16`. -/
example := exists_pos_score_leaky (a := 0) (b := -1) (θ := 0.001507) (L := 2048) (w := 1024)
  (k := 16) (by norm_num) (by norm_num) (by norm_num) (by norm_num) (Or.inr (by norm_num))
  score_table10_nonpos

end Transformer.RoPE.LeReRoPE
