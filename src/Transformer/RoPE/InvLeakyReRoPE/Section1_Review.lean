module

public import Transformer.RoPE.ReRoPE.Section2_Fusion

/-!
# InvLeaky ReRoPE, 回顾 (Review): the window and the largest position

Su Jianlin, *Transformer升级之路：13、逆用Leaky ReRoPE* ("The road to upgrading the Transformer,
13: Leaky ReRoPE used in reverse"), kexue.fm/archives/9728, 2023-08-14. The post's sections
回顾 (Review), 反转 (Inversion), 实验 (Experiments) and 小结 (Summary) are unnumbered; the files
count them 1–4. Its matrices (rope), (leaky-rerope) and (rerope) are those of part 12,
kexue.fm/archives/9708, formalized in `Transformer.RoPE.ReRoPE`: at the distance `d = i − j`
Leaky ReRoPE's position is `leakyPosition w k d`, `d` inside the window and `w + (d − w)/k`
outside it, and ReRoPE's is `rectifiedPosition w d = min(d, w)`, the limit `k → ∞`
(`ReRoPE.tendsto_leakyPosition`).

Review: "`w` is the window width, taken at about `1/4` to `1/2` of the training length; `k`
adjusts the largest length that can be handled, and it is generally best to make `w + (L−1−w)/k`
not exceed half the training length." Here `L` is the input length and `N` the training length.

* `leakyPosition_step`: outside the window the positions step by `1/k`;
* `leakyPosition_monotone`: for `k > 0` the positions grow with the distance, so the position
  `w + (L−1−w)/k` of the largest distance `L − 1` is the largest of the input
  (`forall_leakyPosition_le_iff`);
* `exists_leakyPosition_le_half_iff`: on an input longer than the window, some `k > 0` keeps every
  position at most `N/2` exactly when `w < N/2`. At the top `w = N/2` of the window's range no
  finite `k` follows the rule (`not_exists_leakyPosition_le_half`); ReRoPE does, its positions
  stay at most `w` (`ReRoPE.rectifiedPosition_mem_Icc`);
* `leakyPosition_w128_le_iff`: at the experiments' `N = 512`, `w = 128` and test length `4096`,
  the rule asks for `k ≥ 3967/128 ≈ 31`.
-/

@[expose] public section

namespace Transformer.RoPE.InvLeakyReRoPE

open ReRoPE

/-- Review, the matrix (leaky-rerope): outside the window Leaky ReRoPE's positions step by
`1/k`, the entries `w, w + 1/k, w + 2/k, …`; Inversion: the step outside the window is above
RoPE's `1` when `k < 1`. -/
theorem leakyPosition_step (w : ℝ) {k d : ℝ} (hd : w ≤ d) :
    leakyPosition w k (d + 1) - leakyPosition w k d = 1 / k := by
  rw [leakyPosition, leakyPosition, ite_eq_right (not_lt.mpr (by linarith)),
    ite_eq_right (not_lt.mpr hd)]
  ring

/-- InvLeaky ReRoPE's training step outside the window is `16`. -/
example : leakyPosition 128 (1 / 16) (200 + 1) - leakyPosition 128 (1 / 16) 200 = 16 := by
  rw [leakyPosition_step 128 (by norm_num)]
  norm_num

/-- Review: for `k > 0` Leaky ReRoPE's positions grow with the distance. -/
theorem leakyPosition_monotone (w : ℝ) {k : ℝ} (hk : 0 < k) : Monotone (leakyPosition w k) := by
  intro a b hab
  unfold leakyPosition
  split_ifs with ha hb hb
  · exact hab
  · have : 0 ≤ (b - w) / k := div_nonneg (by linarith [not_lt.mp hb]) hk.le
    linarith
  · linarith [not_lt.mp ha]
  · have : (a - w) / k ≤ (b - w) / k := div_le_div_of_nonneg_right (by linarith) hk.le
    linarith

example : Monotone (leakyPosition 128 16) :=
  leakyPosition_monotone 128 (by norm_num)

/-- Review: for `k > 0` and a largest distance `D ≥ w`, every distance `d ≤ D` has a position at
most `B` exactly when `w + (D − w)/k`, the position of `D`, is at most `B`. -/
theorem forall_leakyPosition_le_iff (w : ℝ) {k D : ℝ} (hk : 0 < k) (hwD : w ≤ D) (B : ℝ) :
    (∀ d, d ≤ D → leakyPosition w k d ≤ B) ↔ w + (D - w) / k ≤ B := by
  have hD : leakyPosition w k D = w + (D - w) / k := by
    rw [leakyPosition, ite_eq_right (not_lt.mpr hwD)]
  constructor
  · intro h
    rw [← hD]
    exact h D le_rfl
  · intro h d hd
    rw [← hD] at h
    exact (leakyPosition_monotone w hk hd).trans h

example : (∀ d, d ≤ 4095 → leakyPosition 128 32 d ≤ 256) ↔ (128 : ℝ) + (4095 - 128) / 32 ≤ 256 :=
  forall_leakyPosition_le_iff 128 (k := 32) (D := 4095) (by norm_num) (by norm_num) 256

/-- Review: on an input whose largest distance `D` exceeds the window `w`, some `k > 0` keeps the
positions of all distances `d ≤ D` at most half the training length `N` exactly when `w < N/2`. -/
theorem exists_leakyPosition_le_half_iff {w D : ℝ} (hwD : w < D) (N : ℝ) :
    (∃ k, 0 < k ∧ ∀ d, d ≤ D → leakyPosition w k d ≤ N / 2) ↔ w < N / 2 := by
  constructor
  · rintro ⟨k, hk, h⟩
    have h1 := h D le_rfl
    rw [leakyPosition, ite_eq_right (not_lt.mpr hwD.le)] at h1
    have : 0 < (D - w) / k := div_pos (by linarith) hk
    linarith
  · intro hw
    have hk : 0 < (D - w) / (N / 2 - w) := div_pos (by linarith) (by linarith)
    refine ⟨_, hk, (forall_leakyPosition_le_iff w hk hwD.le _).mpr ?_⟩
    rw [div_div_cancel₀ (by linarith)]
    linarith

example : ∃ k, 0 < k ∧ ∀ d, d ≤ 4095 → leakyPosition 128 k d ≤ 512 / 2 :=
  (exists_leakyPosition_le_half_iff (by norm_num) 512).mpr (by norm_num)

/-- Review: at the top `w = N/2` of the window's range, on an input longer than the window, no
finite `k > 0` keeps every position at most `N/2`. -/
theorem not_exists_leakyPosition_le_half {N D : ℝ} (hD : N / 2 < D) :
    ¬ ∃ k, 0 < k ∧ ∀ d, d ≤ D → leakyPosition (N / 2) k d ≤ N / 2 := by
  rw [exists_leakyPosition_le_half_iff hD]
  exact lt_irrefl _

/-- ReRoPE-w256 at the training length `512`: no Leaky ReRoPE with `w = 256` keeps the positions
of `4096` tokens at most `256`. -/
example : ¬ ∃ k, 0 < k ∧ ∀ d, d ≤ 4095 → leakyPosition (512 / 2) k d ≤ 512 / 2 :=
  not_exists_leakyPosition_le_half (by norm_num)

/-- Review, the experiments' numbers: with `N = 512`, `w = 128` and inputs of `4096` tokens, a
`k > 0` keeps every position at most `256` exactly when `k ≥ 3967/128`. -/
theorem leakyPosition_w128_le_iff {k : ℝ} (hk : 0 < k) :
    (∀ d, d ≤ 4095 → leakyPosition 128 k d ≤ 512 / 2) ↔ 3967 / 128 ≤ k := by
  rw [forall_leakyPosition_le_iff 128 hk (by norm_num),
    show (4095 : ℝ) - 128 = 3967 by norm_num]
  constructor
  · intro h
    have h1 : 3967 / k ≤ 128 := by linarith
    rw [div_le_iff₀ hk] at h1
    linarith
  · intro h
    have h1 : 3967 / k ≤ 128 := by
      rw [div_le_iff₀ hk]
      linarith
    linarith

example : ∀ d, d ≤ 4095 → leakyPosition 128 32 d ≤ 512 / 2 :=
  (leakyPosition_w128_le_iff (by norm_num)).mpr (by norm_num)

end Transformer.RoPE.InvLeakyReRoPE
