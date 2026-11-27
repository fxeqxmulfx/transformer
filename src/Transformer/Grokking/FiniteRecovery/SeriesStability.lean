module

public import Transformer.Grokking.FiniteRecovery.SeriesCodec
public import Mathlib.Order.Interval.Set.Infinite

/-! Stability of a fixed first-difference grid. The relevant distance is
to the nearest rounding threshold in the scaled coordinates. At a threshold
a positive uniform radius is impossible for the quantized levels. -/

@[expose] public section

namespace Transformer.Grokking.FiniteRecovery.FirstDifferenceGrid

/-- Distance to the nearest rounding boundary, in grid units. -/
noncomputable def margin (grid : FirstDifferenceGrid) (n : ℕ) (x : ℝ) : ℝ :=
  let u := Int.fract (grid.scale * x - grid.fraction n + 1 / 2)
  min u (1 - u)

/-- A perturbation smaller than the margin stays in the same grid cell. -/
theorem level_eq_of_scaled_error_lt_margin (grid : FirstDifferenceGrid) (n : ℕ) (x y : ℝ)
    (h : |grid.scale * (y - x)| < grid.margin n x) :
    grid.level n y = grid.level n x := by
  let u := grid.scale * x - grid.fraction n + 1 / 2
  let v := grid.scale * y - grid.fraction n + 1 / 2
  have hsub : v - u = grid.scale * (y - x) := by dsimp [u, v]; ring
  change |grid.scale * (y - x)| < min (Int.fract u) (1 - Int.fract u) at h
  rw [← hsub, lt_min_iff] at h
  have hlo := (abs_lt.mp h.1).1
  have hhi := (abs_lt.mp h.2).2
  have hu := Int.floor_add_fract u
  change ⌊v⌋ = ⌊u⌋
  rw [Int.floor_eq_iff]
  constructor <;> linarith

/-- If every past value stays within its cell, the entire token context
is unchanged. This includes the clipped vocabulary representation. -/
theorem contextWord_eq_of_scaled_error_lt_margin {T vocab : ℕ} (grid : FirstDifferenceGrid)
    (hv : 0 < vocab) (x y : Fin (T + 1) → ℝ)
    (h : ∀ j, |grid.scale * (y j - x j)| < grid.margin j.val (x j)) :
    grid.contextWord hv y = grid.contextWord hv x := by
  funext j
  have h₀ := grid.level_eq_of_scaled_error_lt_margin j.val (x j.castSucc) (y j.castSucc) (h j.castSucc)
  have h₁ := grid.level_eq_of_scaled_error_lt_margin (j.val + 1) (x j.succ) (y j.succ) (h j.succ)
  simp only [contextWord, step, h₀, h₁]

/-- A fixed grid of positive scale has a rounding boundary. Arbitrarily
small perturbations can change the quantized level there. -/
theorem exists_level_change (grid : FirstDifferenceGrid) (n : ℕ) (hscale : 0 < grid.scale)
    (ε : ℝ) (hε : 0 < ε) :
    ∃ x y, |x - y| < ε ∧ grid.level n x ≠ grid.level n y := by
  let x := (grid.fraction n - 1 / 2) / grid.scale
  let y := x - ε / 2
  have hx : grid.scale * x - grid.fraction n + 1 / 2 = 0 := by
    dsimp [x]
    field_simp
    ring
  have hy : grid.scale * y - grid.fraction n + 1 / 2 = -(grid.scale * (ε / 2)) := by
    dsimp [y]
    linarith only [hx, show grid.scale * (x - ε / 2) =
      grid.scale * x - grid.scale * (ε / 2) by ring]
  refine ⟨x, y, ?_, ?_⟩
  · have hd : x - y = ε / 2 := by dsimp [y]; ring
    rw [hd, abs_of_nonneg (by positivity)]
    linarith
  · rw [level, hx, Int.floor_zero, level, hy]
    have hfloor := Int.floor_le (-(grid.scale * (ε / 2)))
    have hprod := mul_pos hscale (half_pos hε)
    have hneg : (⌊-(grid.scale * (ε / 2))⌋ : ℝ) < 0 := by linarith
    exact ne_of_gt (by exact_mod_cast hneg)

/-- If the constant MASH phase may be calibrated, a fixed phase and a
uniform positive margin exist on any finite set of calibration values. -/
theorem exists_fixed_phase_margin {I : Type*} [Finite I] (s : ℝ) (zero : ℤ) (sample : I → ℝ) :
    ∃ seed ∈ Set.Ioo (0 : ℝ) 1, ∃ margin₀ > 0, ∀ i n,
      margin₀ ≤ (FirstDifferenceGrid.mk .mash s zero seed).margin n (sample i) := by
  let forbidden := Set.range fun i => Int.fract (s * sample i + 1 / 2)
  obtain ⟨seed, hseed, hnot⟩ :
      (Set.Ioo (0 : ℝ) 1 \ forbidden).Nonempty :=
    ((Set.Ioo_infinite zero_lt_one).sdiff (Set.finite_range _)).nonempty
  let grid := FirstDifferenceGrid.mk .mash s zero seed
  have hfrac : Int.fract seed = seed := Int.fract_eq_self.mpr ⟨hseed.1.le, hseed.2⟩
  have hmargin (i : I) (n : ℕ) : 0 < grid.margin n (sample i) := by
    let u := s * sample i - seed + 1 / 2
    have hne : Int.fract u ≠ 0 := by
      intro hzero
      have hu := Int.floor_add_fract u
      rw [hzero, add_zero] at hu
      have heq : s * sample i + 1 / 2 = seed + (⌊u⌋ : ℝ) := by
        dsimp [u] at hu ⊢
        linarith
      apply hnot
      refine ⟨i, ?_⟩
      change Int.fract (s * sample i + 1 / 2) = seed
      rw [heq, Int.fract_add_intCast, hfrac]
    simp only [margin, grid, fraction, hfrac]
    exact lt_min ((Int.fract_nonneg u).lt_of_ne' hne) (sub_pos.mpr (Int.fract_lt_one u))
  refine ⟨seed, hseed, ?_⟩
  by_cases hi : Nonempty I
  · obtain ⟨i, _, hmin⟩ := Set.exists_min_image Set.univ
      (fun i => grid.margin 0 (sample i)) (Set.toFinite _) Set.univ_nonempty
    refine ⟨grid.margin 0 (sample i), hmargin i 0, fun j n => ?_⟩
    simpa only [margin, grid, fraction] using hmin j (Set.mem_univ _)
  · exact ⟨1, one_pos, fun i _ => (hi ⟨i⟩).elim⟩

/-- The corrected steps telescope. A decoder that starts at the correct
grid level and receives exact, unclipped tokens carries no accumulated
rounding error in the grid levels. -/
theorem sum_corrected_steps (grid : FirstDifferenceGrid) (series : ℕ → ℝ) (N : ℕ) :
    (∑ n ∈ Finset.range N, (grid.step n (series n) (series (n + 1)) - grid.offset n)) =
      grid.level N (series N) - grid.level 0 (series 0) := by
  induction N with
  | zero => simp
  | succ N ih =>
    rw [Finset.sum_range_succ, ih]
    unfold step
    ring

private def unitGrid : FirstDifferenceGrid := ⟨.lgt, 1, 128, 0⟩

example (ε : ℝ) (hε : 0 < ε) :
    ∃ x y, |x - y| < ε ∧ unitGrid.level 0 x ≠ unitGrid.level 0 y :=
  unitGrid.exists_level_change 0 (by norm_num [unitGrid]) ε hε

example (x y : Fin 4 → ℝ)
    (h : ∀ j, |y j - x j| < unitGrid.margin j.val (x j)) :
    unitGrid.contextWord (vocab := 256) (by decide) y =
      unitGrid.contextWord (vocab := 256) (by decide) x :=
  unitGrid.contextWord_eq_of_scaled_error_lt_margin (by decide) x y
    (fun j => by simpa [unitGrid] using h j)

example := exists_fixed_phase_margin 1 128 (fun i : Fin 128 => (i.val : ℝ))

/-- The more stable phase depends on where the data fall. With unit scale,
LGT has margin one half at an integer, whereas MASH-half with seed zero
has margin one half at a half-integer at the second position. -/
example :
    let halfGrid := FirstDifferenceGrid.mk .mashHalf 1 128 0
    unitGrid.margin 1 0 = 1 / 2 ∧ halfGrid.margin 1 0 = 0 ∧
      unitGrid.margin 1 (1 / 2) = 0 ∧ halfGrid.margin 1 (1 / 2) = 1 / 2 := by
  have hhalf : Int.fract (1 / 2 : ℝ) = 1 / 2 :=
    Int.fract_eq_self.mpr (by norm_num)
  norm_num [margin, unitGrid, fraction, hhalf]

end Transformer.Grokking.FiniteRecovery.FirstDifferenceGrid
