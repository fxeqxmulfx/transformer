module

public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic

/-!
# ReRoPE, 融合 (Fusion): the relative position matrices

Su Jianlin, *Transformer升级之路：12、无限外推的ReRoPE？* ("The road to upgrading the Transformer,
12: ReRoPE for infinite extrapolation?"), kexue.fm/archives/9708, 2023-08-07. The post's sections
重温 (Review), 融合 (Fusion), 计算 (Computation) and 实验 (Experiments) are unnumbered; the files
count them 1–4. The entry of a relative position matrix at row `i` and column `j` is a function of
the distance `d = i − j`, and positions are real.

* `leakyPosition w k d` is the matrix (leaky-rerope): positions spaced `1` inside the window,
  `d` for `d < w`, and spaced `1/k` outside it, `w + (d − w)/k`; `rectifiedPosition w d =
  min(d, w)` is the matrix (rerope);
* `exists_leakyPosition_lt`: "as long as `w` is less than the training length, by controlling `k`
  we can keep locality exactly and make all position encodings not exceed the training length";
  the positions stay strictly below it. Locality, `leakyPosition_of_lt`, holds for every `k`;
* `exists_interpolated_lt`, `interpolated_ne_self`: §1, Position Interpolation `d ↦ d/k` keeps the
  positions below the training length too, but moves every position `d ≠ 0`, "the position
  encodings near `0` are compressed to `1/k`";
* `tendsto_leakyPosition`: "when `k → ∞`, (leaky-rerope) simplifies to" (rerope);
* `rectifiedPosition_mem_Icc`: "whatever the input length, its position encodings do not exceed
  `w`"; `exists_lt_leakyPosition`: §4, with a finite `k` the positions of long enough inputs
  exceed the training length, "while ReRoPE has no such risk";
* `leakyPosition_one`, `leakyPosition_eq`, `rectifiedPosition_eq`: the matrices relate to RoPE's
  "as ReLU and Leaky ReLU relate to Linear". With `u = w − d`, RoPE's position is `w − u`, Leaky
  ReRoPE's `w − max(u, u/k)` and ReRoPE's `w − max(u, 0)`; `k = 1` is RoPE.
-/

@[expose] public section

open Filter Topology

namespace Transformer.RoPE.ReRoPE

/-- Fusion, matrix (leaky-rerope): the Leaky ReRoPE position at distance `d`, `d` inside the window
`d < w` and `w + (d − w)/k` outside it. -/
noncomputable def leakyPosition (w k d : ℝ) : ℝ :=
  if d < w then d else w + (d - w) / k

/-- Fusion, matrix (rerope): the ReRoPE position at distance `d`, `min(d, w)`. -/
noncomputable def rectifiedPosition (w d : ℝ) : ℝ :=
  min d w

/-- Fusion: Leaky ReRoPE keeps every position inside the window, "keeping locality exactly". -/
theorem leakyPosition_of_lt {w d : ℝ} (k : ℝ) (hd : d < w) : leakyPosition w k d = d := by
  rw [leakyPosition, ite_eq_left hd]

example : leakyPosition 512 16 100 = 100 :=
  leakyPosition_of_lt 16 (by norm_num)

/-- Fusion: if the window `w` is below the training length `T`, then for every length `L` some
`k ≥ 1` keeps locality and keeps the positions of all distances `d ≤ L` below `T`. -/
theorem exists_leakyPosition_lt {w T : ℝ} (hwT : w < T) (L : ℝ) :
    ∃ k, 1 ≤ k ∧ (∀ d, d < w → leakyPosition w k d = d) ∧
      ∀ d, d ≤ L → leakyPosition w k d < T := by
  have hTw : 0 < T - w := sub_pos.mpr hwT
  have hk1 : 1 ≤ max 1 ((L - w) / (T - w) + 1) := le_max_left _ _
  have hk2 : (L - w) / (T - w) + 1 ≤ max 1 ((L - w) / (T - w) + 1) := le_max_right _ _
  have hk0 : 0 < max 1 ((L - w) / (T - w) + 1) := lt_of_lt_of_le one_pos hk1
  refine ⟨_, hk1, fun d hd => leakyPosition_of_lt _ hd, fun d hd => ?_⟩
  rw [leakyPosition]
  split_ifs with h
  · linarith
  · have h3 : L - w < ((L - w) / (T - w) + 1) * (T - w) := by
      rw [add_mul, div_mul_cancel₀ _ hTw.ne']
      linarith
    have h4 : (d - w) / max 1 ((L - w) / (T - w) + 1) < T - w := by
      rw [div_lt_iff₀ hk0]
      nlinarith [mul_le_mul_of_nonneg_right hk2 hTw.le]
    linarith

example : ∃ k, 1 ≤ k ∧ (∀ d, d < 512 → leakyPosition 512 k d = d) ∧
    ∀ d, d ≤ 65536 → leakyPosition 512 k d < 4096 :=
  exists_leakyPosition_lt (by norm_num) 65536

/-- Review (§1), Position Interpolation: for every length `L` and training length `T > 0` some
`k ≥ 1` keeps the positions `d/k` of all distances `d ≤ L` below `T`. -/
theorem exists_interpolated_lt {T : ℝ} (hT : 0 < T) (L : ℝ) :
    ∃ k, 1 ≤ k ∧ ∀ d, d ≤ L → d / k < T := by
  have hk1 : 1 ≤ max 1 (L / T + 1) := le_max_left _ _
  have hk2 : L / T + 1 ≤ max 1 (L / T + 1) := le_max_right _ _
  have hk0 : 0 < max 1 (L / T + 1) := lt_of_lt_of_le one_pos hk1
  refine ⟨_, hk1, fun d hd => ?_⟩
  have h3 : L < (L / T + 1) * T := by
    rw [add_mul, div_mul_cancel₀ _ hT.ne']
    linarith
  rw [div_lt_iff₀ hk0]
  nlinarith [mul_le_mul_of_nonneg_right hk2 hT.le]

example : ∃ k : ℝ, 1 ≤ k ∧ ∀ d : ℝ, d ≤ 65536 → d / k < 4096 :=
  exists_interpolated_lt (by norm_num) 65536

/-- Review (§1): Position Interpolation with `k ≠ 1` moves every position `d ≠ 0`; "the position
encodings near `0` are compressed to `1/k`", and locality is lost. -/
theorem interpolated_ne_self {k d : ℝ} (hk : k ≠ 1) (hd : d ≠ 0) : d / k ≠ d := by
  intro h
  rcases eq_or_ne k 0 with rfl | hk0
  · rw [div_zero] at h
    exact hd h.symm
  · rw [div_eq_iff hk0] at h
    exact hk (mul_left_cancel₀ hd (by rw [mul_one]; exact h.symm))

example : (1 : ℝ) / 16 ≠ 1 :=
  interpolated_ne_self (by norm_num) one_ne_zero

/-- Fusion: "when `k → ∞`, the matrix (leaky-rerope) simplifies to" the matrix (rerope). -/
theorem tendsto_leakyPosition (w d : ℝ) :
    Tendsto (fun k => leakyPosition w k d) atTop (𝓝 (rectifiedPosition w d)) := by
  unfold leakyPosition rectifiedPosition
  by_cases h : d < w
  · simp only [ite_eq_left h, min_eq_left h.le]
    exact tendsto_const_nhds
  · simp only [ite_eq_right h, min_eq_right (not_lt.mp h)]
    have h0 : Tendsto (fun k : ℝ => (d - w) / k) atTop (𝓝 0) :=
      tendsto_const_nhds.div_atTop tendsto_id
    simpa using (tendsto_const_nhds (x := w)).add h0

/-- Fusion: "whatever the input length, the position encodings do not exceed `w`": ReRoPE's
positions of all distances `d ≥ 0` lie in `[0, w]`. -/
theorem rectifiedPosition_mem_Icc {w d : ℝ} (hw : 0 ≤ w) (hd : 0 ≤ d) :
    rectifiedPosition w d ∈ Set.Icc 0 w :=
  ⟨le_min hd hw, min_le_right d w⟩

example (d : ℕ) : rectifiedPosition 512 d ∈ Set.Icc 0 512 :=
  rectifiedPosition_mem_Icc (by norm_num) d.cast_nonneg

/-- Experiments (§4): "when `k` takes a finite value, the maximal length it can handle is finite
too": for every `k > 0` and training length `T`, the positions of all long enough distances exceed
`T`.
ReRoPE's stay at most `w` (`rectifiedPosition_mem_Icc`). -/
theorem exists_lt_leakyPosition (w T : ℝ) {k : ℝ} (hk : 0 < k) :
    ∃ L, ∀ d, L ≤ d → T < leakyPosition w k d := by
  refine ⟨w + k * (|T - w| + 1), fun d hd => ?_⟩
  have hpos : 0 < k * (|T - w| + 1) := mul_pos hk (by positivity)
  rw [leakyPosition, ite_eq_right (by linarith)]
  have h1 : |T - w| + 1 ≤ (d - w) / k := by
    rw [le_div_iff₀ hk]
    linarith
  linarith [le_abs_self (T - w)]

example : ∃ L, ∀ d, L ≤ d → 4096 < leakyPosition 128 16 d :=
  exists_lt_leakyPosition 128 4096 (by norm_num)

/-- Fusion: with `k = 1` the matrix (leaky-rerope) is RoPE's matrix (rope), the "Linear" case. -/
theorem leakyPosition_one (w d : ℝ) : leakyPosition w 1 d = d := by
  rw [leakyPosition]
  split_ifs <;> ring

/-- Fusion: Leaky ReRoPE is RoPE's position through a Leaky ReLU of slope `1/k`:
`leakyPosition w k d = w − max(u, u/k)` with `u = w − d`, for `k ≥ 1`. -/
theorem leakyPosition_eq (w d : ℝ) {k : ℝ} (hk : 1 ≤ k) :
    leakyPosition w k d = w - max (w - d) ((w - d) / k) := by
  have hk0 : 0 < k := lt_of_lt_of_le one_pos hk
  rw [leakyPosition]
  split_ifs with h
  · rw [max_eq_left (div_le_self (by linarith) hk)]
    ring
  · have h1 : w - d ≤ (w - d) / k := by
      rw [le_div_iff₀ hk0]
      nlinarith
    rw [max_eq_right h1]
    ring

example (d : ℝ) : leakyPosition 128 16 d = 128 - max (128 - d) ((128 - d) / 16) :=
  leakyPosition_eq 128 d (by norm_num)

/-- Fusion: ReRoPE is RoPE's position through a ReLU: `rectifiedPosition w d = w − max(u, 0)` with
`u = w − d`. -/
theorem rectifiedPosition_eq (w d : ℝ) : rectifiedPosition w d = w - max (w - d) 0 := by
  rw [rectifiedPosition]
  rcases le_total d w with h | h
  · rw [min_eq_left h, max_eq_left (by linarith)]
    ring
  · rw [min_eq_right h, max_eq_right (by linarith)]
    ring

end Transformer.RoPE.ReRoPE
