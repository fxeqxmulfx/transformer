/-
# Block scaling: the dead zone becomes relative

Microscaled formats store a block of numbers as a shared scale `σ` times
elements of a small format: in NVFP4, blocks of `16` elements in E2M1 with an
E4M3 scale chosen so that the largest element `a` of the block maps to the top
`6` of E2M1 (NVIDIA, "Pretraining Large Language Models with NVFP4",
arXiv:2509.25149, §2).  The dead zone of the element format then scales with
the block: an element is stored as `0` once it is below `θ / K` of the block's
maximum, `K` the top of the format and `[0, θ)` its dead zone
(`blockScale_eq_zero`).  For NVFP4 that is `a / 24`, a key `ln 24 ≈ 3.2` below
the top of its block.

Unlike `tail_eq`, this does not grow with the context.  Every block keeps its
own maximum, so the dropped elements of a block are at most `B - 1` elements
below `θ c` times its maximum, `c` the ratio of the scale to that maximum; the
mass they carry is at most `(B - 1) θ c` of the total, whatever `n`
(`blockScale_dropped_le`).  For NVFP4 with the scale rounded to nearest in
E4M3, `c ≤ (1 + 2^{-4}) / 6` and the bound is `255/384 < 2/3`.

The statements are this repository's own.
-/

import Transformer.Precision.Float

open scoped BigOperators

namespace Transformer
namespace Precision

/-- **A number rounded to `0` lies in the dead zone.**  If `g > 0` is a point
of the grid, anything nonnegative rounded to `0` is at most `g / 2`. -/
theorem IsNearest.le_of_eq_zero {G : Set ℝ} {Q : ℝ → ℝ} (hQ : IsNearest G Q) {g : ℝ} (hg : g ∈ G)
    (hg0 : 0 < g) {y : ℝ} (hy : 0 ≤ y) (h : Q y = 0) : y ≤ g / 2 := by
  have := (hQ y).2 g hg
  rw [h, zero_sub, abs_neg, abs_of_nonneg hy] at this
  rcases abs_cases (g - y) with ⟨h1, _⟩ | ⟨h1, _⟩ <;> linarith

/-- The hypotheses of `IsNearest.le_of_eq_zero` are satisfiable: `1` in `{0, 1}`. -/
example : ∃ Q, IsNearest {0, 1} Q := exists_isNearest (Set.toFinite _) ⟨0, by simp⟩

/-- **The dead zone of a scaled block is relative to its maximum.**  With the
scale `σ` large enough that the block's maximum `a` fits under the top `K` of
the format, every element below `θ / K · a` is stored as `0`. -/
theorem blockScale_eq_zero {Q : ℝ → ℝ} {θ K σ a x : ℝ} (hQ0 : ∀ y, 0 ≤ y → y < θ → Q y = 0)
    (hθ : 0 ≤ θ) (hK : 0 < K) (hσ : 0 < σ) (ha : a ≤ K * σ) (hx0 : 0 ≤ x) (hx : x < θ / K * a) :
    σ * Q (x / σ) = 0 := by
  rw [hQ0 _ (div_nonneg hx0 hσ.le) ?_, mul_zero]
  rw [div_lt_iff₀ hσ]
  calc x < θ / K * a := hx
    _ ≤ θ / K * (K * σ) := by gcongr
    _ = θ * σ := by field_simp

/-- The hypotheses of `blockScale_eq_zero` are satisfiable: the dead zone
`[0, 1)` of `Q = ⌊·⌋`, `K = σ = a = 1`, `x = 0`. -/
example : (∀ y : ℝ, 0 ≤ y → y < 1 → (⌊y⌋ : ℝ) = 0) ∧ (0 : ℝ) < 1 / 1 * 1 :=
  ⟨fun y h0 h1 => by rw [Int.floor_eq_zero_iff.2 ⟨h0, h1⟩, Int.cast_zero], by norm_num⟩

variable {n : ℕ} {β : Type*} [DecidableEq β]

/-- **Block scaling bounds the dropped mass independently of the length.**
Nonnegative `x` in blocks `b` of at most `B` elements, each block `b j` with a
scale `σ (b j) > 0` at most `c` times its element `t (b j)`; if rounding to `0`
puts an element under `θ` and `θ c < 1`, the elements stored as `0` carry at
most `(B - 1) θ c` of the total mass. -/
theorem blockScale_dropped_le {Q : ℝ → ℝ} {θ c : ℝ} {B : ℕ}
    (hQz : ∀ y, 0 ≤ y → Q y = 0 → y ≤ θ) (hθ : 0 ≤ θ) (hθc : θ * c < 1)
    {x : Idx n → ℝ} (hx : ∀ j, 0 ≤ x j) {b : Idx n → β} {σ : β → ℝ} (hσ : ∀ k, 0 < σ k)
    {t : β → Idx n} (hbt : ∀ j, b (t (b j)) = b j) (hσt : ∀ j, σ (b j) ≤ c * x (t (b j)))
    (hB : ∀ j, (Finset.univ.filter fun i => b i = b j).card ≤ B) :
    ∑ j ∈ Finset.univ.filter (fun j => Q (x j / σ (b j)) = 0), x j ≤
      (B - 1) * (θ * c) * ∑ j, x j := by
  set Dr := Finset.univ.filter fun j => Q (x j / σ (b j)) = 0
  have hlow : ∀ j ∈ Dr, x j ≤ θ * σ (b j) := fun j hj => by
    have := hQz _ (div_nonneg (hx j) (hσ _).le) (Finset.mem_filter.1 hj).2
    rw [div_le_iff₀ (hσ _)] at this
    linarith
  have htop : ∀ j, t (b j) ∉ Dr := fun j ht => by
    have h1 := hlow _ ht
    rw [hbt] at h1
    have hpos : 0 < x (t (b j)) := by
      rcases (hx (t (b j))).lt_or_eq with h | h
      · exact h
      · have := hσt j; rw [← h, mul_zero] at this; linarith [hσ (b j)]
    nlinarith [hσt j]
  -- every dropped element is below `θ c` times the maximum of its block
  have h1 : ∑ j ∈ Dr, x j ≤ θ * c * ∑ j ∈ Dr, x (t (b j)) := by
    rw [Finset.mul_sum]
    exact Finset.sum_le_sum fun j hj => by
      nlinarith [hlow j hj, mul_le_mul_of_nonneg_left (hσt j) hθ]
  -- a block has at most `B - 1` dropped elements, its maximum being kept
  have hmem : ∀ j, t (b j) ∈ Finset.univ.filter fun i => b i = b j := fun j =>
    Finset.mem_filter.2 ⟨Finset.mem_univ _, hbt j⟩
  have hcard : ∀ k ∈ Dr.image b, ((Dr.filter fun j => b j = k).card : ℝ) ≤ B - 1 := by
    intro k hk
    obtain ⟨j, -, rfl⟩ := Finset.mem_image.1 hk
    have hsub : (Dr.filter fun i => b i = b j) ⊆
        (Finset.univ.filter fun i => b i = b j).erase (t (b j)) := fun i hi => by
      obtain ⟨hiD, hib⟩ := Finset.mem_filter.1 hi
      exact Finset.mem_erase.2 ⟨fun h => htop j (h ▸ hiD),
        Finset.mem_filter.2 ⟨Finset.mem_univ _, hib⟩⟩
    have := Finset.card_le_card hsub
    rw [Finset.card_erase_of_mem (hmem j)] at this
    have hB' := hB j
    have h1 : 1 ≤ (Finset.univ.filter fun i => b i = b j).card :=
      Finset.card_pos.2 ⟨_, hmem j⟩
    have : (Dr.filter fun i => b i = b j).card + 1 ≤ B := by omega
    have : ((Dr.filter fun i => b i = b j).card : ℝ) + 1 ≤ B := by exact_mod_cast this
    linarith
  have h2 : ∑ j ∈ Dr, x (t (b j)) ≤ (B - 1) * ∑ k ∈ Dr.image b, x (t k) := by
    rw [← Finset.sum_fiberwise_of_maps_to (fun j hj => Finset.mem_image_of_mem b hj),
      Finset.mul_sum]
    refine Finset.sum_le_sum fun k hk => ?_
    rw [Finset.sum_congr rfl fun j hj => by rw [(Finset.mem_filter.1 hj).2], Finset.sum_const,
      nsmul_eq_mul]
    exact mul_le_mul_of_nonneg_right (hcard k hk) (hx _)
  -- the maxima of distinct blocks are distinct elements
  have h3 : ∑ k ∈ Dr.image b, x (t k) ≤ ∑ j, x j := by
    rw [← Finset.sum_image (f := x) fun k₁ hk₁ k₂ hk₂ h => by
      obtain ⟨j₁, -, rfl⟩ := Finset.mem_image.1 hk₁
      obtain ⟨j₂, -, rfl⟩ := Finset.mem_image.1 hk₂
      rw [← hbt j₁, ← hbt j₂, h]]
    exact Finset.sum_le_univ_sum_of_nonneg hx
  rcases isEmpty_or_nonempty (Idx n) with hE | ⟨⟨j₀⟩⟩
  · simp [Dr, Finset.univ_eq_empty]
  have hB1 : (1 : ℝ) ≤ B := by
    exact_mod_cast (Finset.card_pos.2 ⟨_, hmem j₀⟩).trans_le (hB j₀)
  have hc : 0 < c := by
    by_contra! h
    nlinarith [hσt j₀, hσ (b j₀), hx (t (b j₀))]
  calc ∑ j ∈ Dr, x j ≤ θ * c * ∑ j ∈ Dr, x (t (b j)) := h1
    _ ≤ θ * c * ((B - 1) * ∑ k ∈ Dr.image b, x (t k)) := by gcongr
    _ ≤ θ * c * ((B - 1) * ∑ j, x j) := by gcongr
    _ = _ := by ring

/-- The hypotheses of `blockScale_dropped_le` are satisfiable: one block of
one element `1` with the scale `1/6`, and the dead zone `[0, 1/4]` of E2M1. -/
example : (∃ Q, IsNearest (grid 2 1) Q) ∧ ∃ (x : Idx 1 → ℝ) (b : Idx 1 → Unit) (σ : Unit → ℝ)
    (t : Unit → Idx 1), (∀ j, 0 ≤ x j) ∧ (∀ k, 0 < σ k) ∧ (∀ j, b (t (b j)) = b j) ∧
      (∀ j, σ (b j) ≤ 17 / 96 * x (t (b j))) ∧
      ∀ j, (Finset.univ.filter fun i => b i = b j).card ≤ 16 :=
  ⟨exists_isNearest_grid (by norm_num) 1, fun _ => 1, fun _ => (), fun _ => 1 / 6, fun _ => 0,
    fun _ => by norm_num, fun _ => by norm_num, fun _ => rfl, fun _ => by norm_num,
    fun j => by revert j; decide⟩

variable {Q : ℝ → ℝ}

/-- **NVFP4 drops a key `ln 24` below the top of its block.**  E2M1 elements
under a scale that fits the block's maximum `a` under `6`: every element below
`a / 24` is stored as `0`. -/
theorem nvfp4_eq_zero (hQ : IsNearest (grid 2 1) Q) {σ a x : ℝ} (hσ : 0 < σ) (ha : a ≤ 6 * σ)
    (hx0 : 0 ≤ x) (hx : x < a / 24) : σ * Q (x / σ) = 0 :=
  blockScale_eq_zero (θ := 1 / 4) (K := 6)
    (fun _ h0 h => ieee_eq_zero (by norm_num) hQ h0 (by rw [minSub_e2m1]; norm_num; linarith))
    (by norm_num) (by norm_num) hσ ha hx0 (by linarith)

/-- The hypotheses of `nvfp4_eq_zero` are satisfiable: `σ = 1`, `a = 6`, `x = 0`. -/
example : (∃ Q, IsNearest (grid 2 1) Q) ∧ (0 : ℝ) < 1 ∧ (6 : ℝ) ≤ 6 * 1 ∧ (0 : ℝ) < 6 / 24 :=
  ⟨exists_isNearest_grid (by norm_num) 1, by norm_num, by norm_num, by norm_num⟩

/-- **NVFP4 drops less than two thirds of the mass, at any length.**  Blocks of
at most `16` E2M1 elements, each with a scale at most `(1 + 2^{-4}) / 6` of the
block's maximum, as rounding `a / 6` to nearest in E4M3 gives: the elements
stored as `0` carry at most `255/384` of the total mass. -/
theorem nvfp4_dropped_le (hQ : IsNearest (grid 2 1) Q) {n : ℕ} {x : Idx n → ℝ} (hx : ∀ j, 0 ≤ x j)
    {b : Idx n → β} {σ : β → ℝ} (hσ : ∀ k, 0 < σ k) {t : β → Idx n}
    (hbt : ∀ j, b (t (b j)) = b j) (hσt : ∀ j, σ (b j) ≤ 17 / 96 * x (t (b j)))
    (hB : ∀ j, (Finset.univ.filter fun i => b i = b j).card ≤ 16) :
    ∑ j ∈ Finset.univ.filter (fun j => Q (x j / σ (b j)) = 0), x j ≤ 255 / 384 * ∑ j, x j :=
  (blockScale_dropped_le (θ := 1 / 4) (fun _ hy h => by
      have := hQ.le_of_eq_zero (g := 1 / 2) ⟨1, by norm_num, by norm_num [ieee, bias]⟩
        (by norm_num) hy h
      linarith) (by norm_num) (by norm_num) hx hσ hbt hσt hB).trans_eq (by norm_num)

end Precision
end Transformer
