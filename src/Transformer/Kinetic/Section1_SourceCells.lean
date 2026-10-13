/-
# Finite source cells and the claimed unique retrieval minimum

ArXiv:2605.09213v1, §1, `eq:Acc-def`, and §1.1,
the final sentence of `thm:U-shape`. The source claims a unique interior
minimum of the actual finite-token soft accuracy for all large `N`.

The actual source index is the integer `floor(σN)`, converted to the
original one-based token and clamped in `sourceIdx`. At every interior
position there is a nonempty interval immediately to its right on which
that actual index stays unchanged, including at exact grid boundaries.
Every scalar observable that depends on position through this index is
therefore constant on that interval. If its value is minimal, the whole
interval consists of minimizers. A unique interior minimum is impossible.
An interior minimum is attained because the actual range is finite
and nonempty; that separate fact is also proved below.

The argument retains the actual floor index and its clamps. It requires
no assumption on token trajectories, probability law or interaction
parameter, so applies in particular under all of the source's conditions.
The limiting correction's separate uniqueness claim is not addressed here.
-/

import Transformer.Kinetic.Accuracy

open Real Set

namespace Transformer.Kinetic

/-- The actual finite source index is constant on a right interval at
every interior position. Source: arXiv:2605.09213v1, §1,
`eq:Acc-def`, and §1.1, the finite-accuracy clause of `thm:U-shape`.
The open right endpoint lies at or before the next grid boundary and one;
the interval includes the original position even when it is a boundary. -/
theorem exists_right_interval_sourceIdx (N : ℕ) (hN : 0 < N)
    (s : ℝ) (hs : s ∈ Ioo (0 : ℝ) 1) :
    ∃ b : ℝ, s < b ∧ b ≤ 1 ∧
      ∀ τ ∈ Ico s b, sourceIdx N hN τ = sourceIdx N hN s := by
  have hNR : (0 : ℝ) < N := by exact_mod_cast hN
  let k : ℕ := ⌊s * N⌋₊
  let b : ℝ := min 1 (((k : ℝ) + 1) / N)
  have hsN : 0 ≤ s * (N : ℝ) := mul_nonneg hs.1.le hNR.le
  have hlow : (k : ℝ) ≤ s * N := Nat.floor_le hsN
  have hhigh : s * (N : ℝ) < (k : ℝ) + 1 := Nat.lt_floor_add_one _
  have hsb : s < b := lt_min hs.2 ((lt_div_iff₀ hNR).mpr hhigh)
  refine ⟨b, hsb, min_le_left _ _, fun τ hτ => ?_⟩
  have hτ0 : 0 ≤ τ := hs.1.le.trans hτ.1
  have hfloor : ⌊τ * N⌋₊ = k := by
    apply (Nat.floor_eq_iff (mul_nonneg hτ0 hNR.le)).mpr
    constructor
    · exact hlow.trans (mul_le_mul_of_nonneg_right hτ.1 hNR.le)
    · have ht : τ < ((k : ℝ) + 1) / N := hτ.2.trans_le (min_le_right _ _)
      exact (lt_div_iff₀ hNR).mp ht
  apply Fin.ext
  change min (⌊τ * N⌋₊ - 1) (N - 1) = min (k - 1) (N - 1)
  rw [hfloor]

/-- There is a distinct larger interior position with the same actual
source token. Source: arXiv:2605.09213v1, §1, `eq:Acc-def`,
and §1.1, the alleged unique finite-accuracy minimum in `thm:U-shape`.
The midpoint is chosen inside the proved right interval. -/
theorem exists_larger_same_sourceIdx (N : ℕ) (hN : 0 < N)
    (s : ℝ) (hs : s ∈ Ioo (0 : ℝ) 1) :
    ∃ τ ∈ Ioo (0 : ℝ) 1, s < τ ∧ sourceIdx N hN τ = sourceIdx N hN s := by
  obtain ⟨b, hsb, hb, hcell⟩ := exists_right_interval_sourceIdx N hN s hs
  refine ⟨(s + b) / 2, ⟨by linarith [hs.1], by linarith⟩, by linarith, ?_⟩
  exact hcell _ ⟨by linarith, by linarith⟩

/-- Infinitely many interior positions have the same actual finite
source index as any given interior position. Source:
arXiv:2605.09213v1, §1, `eq:Acc-def`, and §1.1, `thm:U-shape`.
This is an interval fiber of the actual index, not an assumed partition. -/
theorem sourceIdx_interior_fiber_infinite (N : ℕ) (hN : 0 < N)
    (s : ℝ) (hs : s ∈ Ioo (0 : ℝ) 1) :
    {τ : ℝ | τ ∈ Ioo (0 : ℝ) 1 ∧ sourceIdx N hN τ = sourceIdx N hN s}.Infinite := by
  obtain ⟨b, hsb, hb, hcell⟩ := exists_right_interval_sourceIdx N hN s hs
  apply (Set.Ioo_infinite hsb).mono
  intro τ hτ
  exact ⟨⟨hs.1.trans hτ.1, hτ.2.trans_le hb⟩, hcell τ ⟨hτ.1.le, hτ.2⟩⟩

/-- Every attained interior minimum of an actual finite-source observable
has infinitely many minimizers. Source: arXiv:2605.09213v1,
§1.1, the finite-accuracy clause of `thm:U-shape`.
The original values at every token are retained as the arbitrary map `F`. -/
theorem sourceIdx_minimizers_infinite (N : ℕ) (hN : 0 < N) (F : Idx N → ℝ)
    (s : ℝ) (hs : s ∈ Ioo (0 : ℝ) 1)
    (hmin : ∀ u ∈ Ioo (0 : ℝ) 1, F (sourceIdx N hN s) ≤ F (sourceIdx N hN u)) :
    {τ : ℝ | τ ∈ Ioo (0 : ℝ) 1 ∧
      ∀ u ∈ Ioo (0 : ℝ) 1, F (sourceIdx N hN τ) ≤ F (sourceIdx N hN u)}.Infinite := by
  apply (sourceIdx_interior_fiber_infinite N hN s hs).mono
  intro τ hτ
  exact ⟨hτ.1, fun u hu => by rw [hτ.2]; exact hmin u hu⟩

/-- Every actual finite-source observable does attain an interior global
minimum, since its nonempty range there is finite. Source:
arXiv:2605.09213v1, §1.1, the finite-accuracy clause of `thm:U-shape`.
Together with the interval-fiber theorem, attainment never means uniqueness;
no monotonicity near the endpoints is assumed or inferred. -/
theorem exists_minimum_of_sourceIdx (N : ℕ) (hN : 0 < N) (F : Idx N → ℝ) :
    ∃ s ∈ Ioo (0 : ℝ) 1,
      ∀ u ∈ Ioo (0 : ℝ) 1, F (sourceIdx N hN s) ≤ F (sourceIdx N hN u) := by
  let S : Set ℝ := (fun s => F (sourceIdx N hN s)) '' Ioo (0 : ℝ) 1
  have hfin : S.Finite := (finite_range F).subset (by
    rintro v ⟨s, -, rfl⟩
    exact mem_range_self _)
  have hne : S.Nonempty := ⟨F (sourceIdx N hN (1 / 2)),
    mem_image_of_mem _ (by norm_num : (1 / 2 : ℝ) ∈ Ioo (0 : ℝ) 1)⟩
  obtain ⟨v, hv, hmin⟩ := Set.exists_min_image S (fun x => x) hfin hne
  obtain ⟨s, hs, rfl⟩ := hv
  exact ⟨s, hs, fun u hu => hmin _ (mem_image_of_mem _ hu)⟩

/-- The actual finite-source observable never has a unique interior
global minimum. Source: arXiv:2605.09213v1, §1.1,
the final sentence of `thm:U-shape`. This holds for every positive prompt
size and every actual token observable, regardless of its dynamics. -/
theorem not_unique_minimum_of_sourceIdx (N : ℕ) (hN : 0 < N) (F : Idx N → ℝ) :
    ¬ ∃! s : ℝ, s ∈ Ioo (0 : ℝ) 1 ∧
      ∀ u ∈ Ioo (0 : ℝ) 1, F (sourceIdx N hN s) ≤ F (sourceIdx N hN u) := by
  rintro ⟨s, ⟨hs, hmin⟩, huniq⟩
  obtain ⟨τ, hτ, hst, heq⟩ := exists_larger_same_sourceIdx N hN s hs
  have hminτ : ∀ u ∈ Ioo (0 : ℝ) 1,
      F (sourceIdx N hN τ) ≤ F (sourceIdx N hN u) := by
    rw [heq]
    exact hmin
  exact hst.ne' (huniq τ ⟨hτ, hminτ⟩)

/-- Restricting to valid positive source positions cannot restore a
unique minimum. Source: arXiv:2605.09213v1, §1, `eq:Acc-def`,
and §1.1, `thm:U-shape`. In particular the lower threshold can be `1/N`,
where the paper's original floor index is a valid one-based token. -/
theorem not_unique_minimum_of_sourceIdx_above (N : ℕ) (hN : 0 < N)
    (F : Idx N → ℝ) (l : ℝ) (hl : 0 ≤ l) :
    ¬ ∃! s : ℝ, s ∈ Ioo l 1 ∧
      ∀ u ∈ Ioo l 1, F (sourceIdx N hN s) ≤ F (sourceIdx N hN u) := by
  rintro ⟨s, ⟨hs, hmin⟩, huniq⟩
  obtain ⟨τ, hτ, hst, heq⟩ := exists_larger_same_sourceIdx N hN s ⟨hl.trans_lt hs.1, hs.2⟩
  have hminτ : ∀ u ∈ Ioo l 1, F (sourceIdx N hN τ) ≤ F (sourceIdx N hN u) := by
    rw [heq]
    exact hmin
  exact hst.ne' (huniq τ ⟨⟨hs.1.trans hst, hτ.2⟩, hminτ⟩)

/-- Dimension-free source-cell hypotheses hold jointly at an exact
grid boundary with four actual tokens. Two distinct interior positions
then select token one. Source: arXiv:2605.09213v1, §1,
`eq:Acc-def`, and §1.1, `thm:U-shape`. -/
example : (0 : ℕ) < 4 ∧ (1 / 2 : ℝ) ∈ Ioo (0 : ℝ) 1 ∧
    (0 : ℝ) ≤ 1 / 4 ∧
    sourceIdx 4 (by norm_num) (1 / 2) = sourceIdx 4 (by norm_num) (5 / 8) := by
  have hf : ⌊(5 / 2 : ℝ)⌋₊ = 2 := (Nat.floor_eq_iff (by norm_num)).mpr (by norm_num)
  norm_num [sourceIdx, hf]

/-- The minimizer hypothesis is realized by the actual nonconstant
observable `F(j)=j` on four tokens. The selected token at `s=1/4` is zero,
the minimum, whereas `s=1/2` selects token one.
Source: arXiv:2605.09213v1, §1, `eq:Acc-def`, and §1.1, `thm:U-shape`. -/
example : (0 : ℕ) < 4 ∧ (1 / 4 : ℝ) ∈ Ioo (0 : ℝ) 1 ∧
    (∀ u ∈ Ioo (0 : ℝ) 1,
      (sourceIdx 4 (by norm_num) (1 / 4) : ℝ) ≤ (sourceIdx 4 (by norm_num) u : ℝ)) ∧
    (sourceIdx 4 (by norm_num) (1 / 4) : ℝ) ≠
      (sourceIdx 4 (by norm_num) (1 / 2) : ℝ) := by
  have hz : (sourceIdx 4 (by norm_num) (1 / 4) : ℝ) = 0 := by norm_num [sourceIdx]
  refine ⟨by norm_num, by norm_num, fun u _ => ?_, ?_⟩
  · rw [hz]
    exact Nat.cast_nonneg _
  · norm_num [sourceIdx]

end Transformer.Kinetic
