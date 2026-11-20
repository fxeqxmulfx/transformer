module

public import Transformer.RoPE.ReRoPE2D.Defs
public import Transformer.RoPE.ReRoPE.Section3_Computation

/-!
# 2D ReRoPE: the two-sided clamp on each axis

2D ReRoPE is in no paper; see `Transformer.RoPE.ReRoPE2D.Defs`. Su's matrix (rerope),
`rectifiedPosition w d = min(d, w)` (kexue.fm/archives/9708, 融合 (Fusion)), clamps the distance
`d = i − j ≥ 0` of a causal model. A vision encoder attends both ways, so `d` takes both signs,
and Su's matrix does not bound it from below (`exists_rectifiedPosition_lt`). 2D ReRoPE clamps
each axis's offset on both sides, each axis at its own window.

* `clampPosition w d = max(−w, min(d, w))`: in `[−w, w]` whatever `d` (`clampPosition_mem_Icc`;
  Su's `rectifiedPosition_mem_Icc` needs `d ≥ 0`), `d` inside the window
  (`clampPosition_of_mem_Icc`), Su's position from `−w` on (`clampPosition_eq_rectifiedPosition`),
  odd (`clampPosition_neg`);
* `clampOffset w x y` and `score θ ax w x y q k = qᵀ R^{−c(x−y)} k`: 2D ReRoPE's relative position
  of a query at `x` and a key at `y`, and its score, as Su's `q_iᵀ R^{−min(i−j, w)} k_j`
  (`Transformer.RoPE.ReRoPE.merge_rectified`);
* `score_of_abs_le`: inside the window box the score is axial RoPE's: locality is kept exactly;
* `score_eq_sum`: the score is a row score plus a column score, each one-dimensional ReRoPE with the
  two-sided clamp on the planes of its axis. `score_eq_sum_merge`: with the key above and to the
  left of the query, each is Su's merged score of his two attention matrices;
* `exists_score_eq`: on a grid of any size every score is the axial RoPE score of two patches of the
  grid `S₀ × S₁` when the windows are below `S₀` and `S₁`. A model trained with axial RoPE on that
  grid has seen every relative position 2D ReRoPE uses;
* `single_lt`: the patches on the line of one axis through the corner lie in the grid, where the
  no-go theorems place their positions.
-/

@[expose] public section

open scoped BigOperators
open Matrix Transformer.RoPE.RoFormer
open Transformer.RoPE.ReRoPE (rectifiedPosition merge_rectified)

namespace Transformer.RoPE.ReRoPE2D

variable {p : ℕ}

/-- The two-sided clamp of the offset `d` at the window `w`: `d` inside `[−w, w]`, `±w` outside. -/
noncomputable def clampPosition (w d : ℝ) : ℝ :=
  max (-w) (min d w)

/-- Whatever the offset, the clamped position lies in `[−w, w]`. -/
theorem clampPosition_mem_Icc {w : ℝ} (hw : 0 ≤ w) (d : ℝ) :
    clampPosition w d ∈ Set.Icc (-w) w :=
  ⟨le_max_left _ _, max_le (by linarith) (min_le_right _ _)⟩

/-- The hypothesis of `clampPosition_mem_Icc` holds for the window `256`. -/
example := clampPosition_mem_Icc (w := 256) (by norm_num) (-1000)

/-- Inside the window the offset is kept: locality is exact. -/
theorem clampPosition_of_mem_Icc {w d : ℝ} (h : d ∈ Set.Icc (-w) w) : clampPosition w d = d := by
  rw [clampPosition, min_eq_left h.2, max_eq_right h.1]

/-- The hypothesis of `clampPosition_of_mem_Icc` holds for the offset `−3` and the window `8`. -/
example := clampPosition_of_mem_Icc (w := 8) (d := -3) ⟨by norm_num, by norm_num⟩

/-- From `−w` on the clamp is Su's matrix (rerope), `min(d, w)`. -/
theorem clampPosition_eq_rectifiedPosition {w d : ℝ} (hw : 0 ≤ w) (hd : -w ≤ d) :
    clampPosition w d = rectifiedPosition w d := by
  rw [clampPosition, rectifiedPosition, max_eq_right (le_min hd (by linarith))]

/-- The hypotheses of `clampPosition_eq_rectifiedPosition` hold for the offset `1000` and the
window `256`. -/
example := clampPosition_eq_rectifiedPosition (w := 256) (d := 1000) (by norm_num) (by norm_num)

/-- The clamp is odd: a key `d` patches after the query is clamped as one `d` patches before. -/
theorem clampPosition_neg {w : ℝ} (hw : 0 ≤ w) (d : ℝ) :
    clampPosition w (-d) = -clampPosition w d := by
  simp only [clampPosition, max_def, min_def]
  split_ifs <;> linarith

/-- The hypothesis of `clampPosition_neg` holds for the window `256`. -/
example := clampPosition_neg (w := 256) (by norm_num) 1000

/-- Su's matrix (rerope) does not bound a negative offset: `min(d, w)` takes every value below
`w`, so in a model attending both ways the keys after the query keep RoPE's positions. -/
theorem exists_rectifiedPosition_lt (w M : ℝ) : ∃ d, rectifiedPosition w d < M := by
  refine ⟨min M w - 1, ?_⟩
  rw [rectifiedPosition, min_eq_left (by linarith [min_le_right M w])]
  linarith [min_le_left M w]

/-- 2D ReRoPE's relative position of a query at `x` and a key at `y`: the offset on each axis,
clamped on both sides at that axis's window. -/
noncomputable def clampOffset (w x y : Fin 2 → ℝ) : Fin 2 → ℝ :=
  fun a => clampPosition (w a) (x a - y a)

/-- **2D ReRoPE's score** `qᵀ R^{−c(x−y)} k` of a query at `x` and a key at `y`, where `c` clamps
the offset of each axis at its window. -/
noncomputable def score (θ : Fin p → ℝ) (ax : Fin p → Fin 2) (w x y : Fin 2 → ℝ)
    (q k : Fin 2 × Fin p → ℝ) : ℝ :=
  q ⬝ᵥ (axialRotary θ ax (-clampOffset w x y) *ᵥ k)

/-- **Locality**: inside the window box, `|x a − y a| ≤ w a` on both axes, 2D ReRoPE's score is
axial RoPE's. -/
theorem score_of_abs_le (θ : Fin p → ℝ) (ax : Fin p → Fin 2) {w x y : Fin 2 → ℝ}
    (h : ∀ a, |x a - y a| ≤ w a) (q k : Fin 2 × Fin p → ℝ) :
    score θ ax w x y q k = (axialRotary θ ax x *ᵥ q) ⬝ᵥ (axialRotary θ ax y *ᵥ k) := by
  rw [axialRotary_dotProduct, score]
  congr 3
  funext a
  rw [Pi.neg_apply, clampOffset, clampPosition_of_mem_Icc (abs_le.mp (h a)), Pi.sub_apply,
    neg_sub]

/-- The hypothesis of `score_of_abs_le` holds in MoonViT's layout for patches `(3, 4)` apart and
the windows `8`. -/
example (q k : Fin 2 × Fin 32 → ℝ) := score_of_abs_le (kimiFreq 10000) kimiAxis
  (w := fun _ => 8) (x := ![10, 20]) (y := ![7, 16])
  (fun a => by fin_cases a <;> norm_num [abs_le]) q k

/-- **A row score plus a column score**: on the planes of the axis `a` the score is
one-dimensional ReRoPE, RoFormer's `rotary` at the offset `x a − y a` clamped on both sides. -/
theorem score_eq_sum (θ : Fin p → ℝ) (ax : Fin p → Fin 2) (w x y : Fin 2 → ℝ)
    (q k : Fin 2 × Fin p → ℝ) :
    score θ ax w x y q k =
      ∑ a, axisPart ax a q ⬝ᵥ (rotary θ (-clampPosition (w a) (x a - y a)) *ᵥ k) :=
  dotProduct_axialRotary_eq_sum θ ax _ q k

/-- **Su's two matrices on each axis**: with the key above and to the left of the query,
`y a ≤ x a`, the part of each axis is Su's merged score of kexue.fm/archives/9708, 计算
(Computation): `(R^{x a} q_a)ᵀ(R^{y a} k)` inside the window, `(R^{w a} q_a)ᵀ k` outside. -/
theorem score_eq_sum_merge (θ : Fin p → ℝ) (ax : Fin p → Fin 2) {w x y : Fin 2 → ℝ}
    (hw : ∀ a, 0 ≤ w a) (h : ∀ a, y a ≤ x a) (q k : Fin 2 × Fin p → ℝ) :
    score θ ax w x y q k = ∑ a, if x a - y a < w a then
        (rotary θ (x a) *ᵥ axisPart ax a q) ⬝ᵥ (rotary θ (y a) *ᵥ k)
      else (rotary θ (w a) *ᵥ axisPart ax a q) ⬝ᵥ k := by
  rw [score_eq_sum]
  refine Finset.sum_congr rfl fun a _ => ?_
  rw [merge_rectified, clampPosition_eq_rectifiedPosition (hw a) (by linarith [hw a, h a])]

/-- The hypotheses of `score_eq_sum_merge` hold in GLM-5.3-Flash's layout for a key `(2, 30)`
patches above and to the left, with the windows `16`. -/
example (q k : Fin 2 × Fin 32 → ℝ) := score_eq_sum_merge (glmFreq 10000) glmAxis
  (w := fun _ => 16) (x := ![12, 40]) (y := ![10, 10]) (fun _ => by norm_num)
  (fun a => by fin_cases a <;> norm_num) q k

/-- A clamped integer offset is the offset `a − b` of two positions below `S` when `w < S`. -/
theorem exists_clampPosition_eq {w S : ℕ} (hwS : w < S) (d : ℤ) :
    ∃ a b : ℕ, a < S ∧ b < S ∧ clampPosition w d = (a : ℝ) - b := by
  have hcast : clampPosition w d = ((max (-(w : ℤ)) (min d w) : ℤ) : ℝ) := by
    simp [clampPosition]
  have h1 : -(w : ℤ) ≤ max (-(w : ℤ)) (min d w) := le_max_left _ _
  have h2 : max (-(w : ℤ)) (min d w) ≤ w := max_le (by omega) (min_le_right _ _)
  rcases le_total 0 (max (-(w : ℤ)) (min d w)) with h0 | h0
  · obtain ⟨n, hn⟩ := Int.eq_ofNat_of_zero_le h0
    refine ⟨n, 0, by omega, by omega, ?_⟩
    rw [hcast, hn]
    simp
  · obtain ⟨n, hn⟩ := Int.eq_ofNat_of_zero_le (neg_nonneg.mpr h0)
    refine ⟨0, n, by omega, by omega, ?_⟩
    rw [hcast, show max (-(w : ℤ)) (min d w) = -(n : ℤ) by omega]
    simp

/-- The hypothesis of `exists_clampPosition_eq` holds for the window `256` below `512`. -/
example := exists_clampPosition_eq (w := 256) (S := 512) (by norm_num) (-1000)

/-- **Every score occurs on the training grid**: with the windows `w a < S a`, the 2D ReRoPE
score of patches at any integer positions `X`, `Y`, on a grid of any size, is the axial RoPE
score of two patches of the grid `S₀ × S₁`. -/
theorem exists_score_eq (θ : Fin p → ℝ) (ax : Fin p → Fin 2) {w S : Fin 2 → ℕ}
    (hwS : ∀ a, w a < S a) (X Y : Fin 2 → ℤ) (q k : Fin 2 × Fin p → ℝ) :
    ∃ x y : Fin 2 → ℕ, (∀ a, x a < S a ∧ y a < S a) ∧
      score θ ax (fun a => (w a : ℝ)) (fun a => (X a : ℝ)) (fun a => (Y a : ℝ)) q k =
        (axialRotary θ ax (fun a => (x a : ℝ)) *ᵥ q) ⬝ᵥ
          (axialRotary θ ax (fun a => (y a : ℝ)) *ᵥ k) := by
  choose x y hx hy hxy using fun a => exists_clampPosition_eq (hwS a) (X a - Y a)
  refine ⟨x, y, fun a => ⟨hx a, hy a⟩, ?_⟩
  rw [axialRotary_dotProduct, score]
  congr 3
  funext a
  have := hxy a
  push_cast at this
  simp only [Pi.neg_apply, Pi.sub_apply, clampOffset, this]
  ring

/-- The hypothesis of `exists_score_eq` holds in MoonViT's layout for the windows `256` and
its table of `512 × 512` positions. -/
example (X Y : Fin 2 → ℤ) (q k : Fin 2 × Fin 32 → ℝ) := exists_score_eq (kimiFreq 10000)
  kimiAxis (w := fun _ => 256) (S := fun _ => 512) (fun _ => by norm_num) X Y q k

/-- The patch at `c` on the line of the axis `b` through the corner lies in the grid `S₀ × S₁`
if `c < S_b`. -/
theorem single_lt {S : Fin 2 → ℕ} (hS0 : ∀ a, 0 < S a) {b : Fin 2} {c : ℕ} (hc : c < S b)
    (a : Fin 2) : (Pi.single b c : Fin 2 → ℕ) a < S a := by
  by_cases ha : a = b
  · subst ha
    rwa [Pi.single_eq_same]
  · rw [Pi.single_eq_of_ne ha]
    exact hS0 a

/-- The hypotheses of `single_lt` hold for the patch `(0, 2)` of the grid `3 × 3`. -/
example := single_lt (S := fun _ => 3) (fun _ => by norm_num) (b := 1) (c := 2) (by norm_num) 0

end Transformer.RoPE.ReRoPE2D
