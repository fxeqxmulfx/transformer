/-
# Sphere packing at the R'enyi scale `δ = c β^{-1/2}`

The separation of the R'enyi centers of §5 of 2411.04990v2 is `c β^{-1/2}`, so
the packing bounds of `Packing.Count` become powers of `β`: with
`b = β^{1/2} ≥ 1`,

  `(4/δ)^{d-1} = (4/c)^{d-1} b^{d-1}`  and  `b^{d-1} = β^{(d-1)/2}`.

Two regimes have to be kept apart, because `δ = c β^{-1/2}` is not small unless
`β` is large: for `δ > 1/2` the covering bound says nothing, and the lower
bound is carried instead by a single unit vector, which suffices because
`β^{(d-1)/2} < (2c)^{d-1}` there.  That is why the constants below are minima
and maxima of two expressions rather than one.
-/

import Transformer.Causal.Packing.Count

open Metric

namespace Transformer
namespace Causal
namespace Packing

/-- **The packing bound at the R'enyi scale, upper.**  A `c β^{-1/2}`-separated
set of unit vectors of `EucSpace d` has `O(β^{(d-1)/2})` elements, with a
constant depending only on `d` and `c`.

Auxiliary (not from the paper): the `β`-form of `card_le_of_separatedOnSphere`,
quoted by `Causal.renyi_count`. -/
theorem card_le_at_renyi_scale (d : ℕ) (hd : 1 ≤ d) (c : ℝ) (hc : 0 < c) (β : ℝ) (hβ : 1 ≤ β)
    (S : Finset (EucSpace d)) (hS : SeparatedOnSphere d S (c * β ^ (-(1 / 2 : ℝ)))) :
    (S.card : ℝ) ≤ 2 * d * max (4 / c) 2 ^ (d - 1) * β ^ (((d : ℝ) - 1) / 2) := by
  have hβ0 : (0 : ℝ) < β := by linarith
  set b := β ^ ((1 : ℝ) / 2) with hbdef
  have hb1 : 1 ≤ b := Real.one_le_rpow hβ (by norm_num)
  have hb0 : 0 < b := by linarith
  have hb_pow : b ^ (d - 1) = β ^ (((d : ℝ) - 1) / 2) := by
    rw [hbdef, ← Real.rpow_natCast (β ^ ((1 : ℝ) / 2)) (d - 1), ← Real.rpow_mul hβ0.le]
    congr 1
    rw [Nat.cast_sub hd]
    push_cast
    ring
  set δ := c * β ^ (-(1 / 2 : ℝ)) with hδdef
  have hδeq : δ = c / b := by rw [hδdef, Real.rpow_neg hβ0.le, hbdef]; ring
  have hδpos : 0 < δ := by rw [hδeq]; positivity
  set δ₀ := min δ 2 with hδ0def
  have hδ0pos : 0 < δ₀ := lt_min hδpos (by norm_num)
  have hS0 : SeparatedOnSphere d S δ₀ := ⟨hS.1, fun x hx y hy hxy =>
    lt_of_le_of_lt (min_le_left _ _) (hS.2 x hx y hy hxy)⟩
  have hcard := card_le_of_separatedOnSphere d hd δ₀ hδ0pos (min_le_right _ _) S hS0
  have hbound : 4 / δ₀ ≤ max (4 / c) 2 * b := by
    rcases le_or_gt δ 2 with h | h
    · rw [hδ0def, min_eq_left h, hδeq, div_div_eq_mul_div,
        show (4 : ℝ) * b / c = 4 / c * b by ring]
      exact mul_le_mul_of_nonneg_right (le_max_left _ _) hb0.le
    · rw [hδ0def, min_eq_right h.le]
      nlinarith [le_max_right (4 / c) 2, hb1]
  have hpow : (4 / δ₀) ^ (d - 1) ≤ max (4 / c) 2 ^ (d - 1) * b ^ (d - 1) := by
    rw [← mul_pow]
    exact pow_le_pow_left₀ (by positivity) hbound _
  calc (S.card : ℝ) ≤ 2 * d * (4 / δ₀) ^ (d - 1) := hcard
    _ ≤ 2 * d * (max (4 / c) 2 ^ (d - 1) * b ^ (d - 1)) :=
        mul_le_mul_of_nonneg_left hpow (by positivity)
    _ = 2 * d * max (4 / c) 2 ^ (d - 1) * β ^ (((d : ℝ) - 1) / 2) := by rw [hb_pow]; ring

/-- The hypotheses of `card_le_at_renyi_scale` are satisfiable: `d = 1`,
`c = 1`, `β = 1`, `S = ∅`. -/
example : SeparatedOnSphere 1 (∅ : Finset (EucSpace 1)) (1 * (1 : ℝ) ^ (-(1 / 2 : ℝ))) :=
  ⟨fun _ hx => absurd hx (Finset.notMem_empty _), fun _ hx => absurd hx (Finset.notMem_empty _)⟩

/-- **The packing bound at the R'enyi scale, lower.**  There is a
`c β^{-1/2}`-separated set of unit vectors of `EucSpace d` with
`Ω(β^{(d-1)/2})` elements, the constant depending only on `d` and `c`.

For `β` large enough that `c β^{-1/2} ≤ 1/2` this is a maximal separated set
and `card_ge_of_maximalSeparated`; for smaller `β` a single unit vector already
exceeds the claimed bound.

Auxiliary (not from the paper): the `β`-form of `card_ge_of_maximalSeparated`,
quoted by `Causal.renyi_count`. -/
theorem exists_card_ge_at_renyi_scale (d : ℕ) (hd : 1 ≤ d) (c : ℝ) (hc : 0 < c)
    (β : ℝ) (hβ : 1 ≤ β) :
    ∃ S : Finset (EucSpace d), SeparatedOnSphere d S (c * β ^ (-(1 / 2 : ℝ))) ∧
      min ((d : ℝ) * (1 / (4 * c)) ^ (d - 1)) ((1 / (2 * c)) ^ (d - 1))
          * β ^ (((d : ℝ) - 1) / 2) ≤ (S.card : ℝ) := by
  have hβ0 : (0 : ℝ) < β := by linarith
  set b := β ^ ((1 : ℝ) / 2) with hbdef
  have hb1 : 1 ≤ b := Real.one_le_rpow hβ (by norm_num)
  have hb0 : 0 < b := by linarith
  have hb_pow : b ^ (d - 1) = β ^ (((d : ℝ) - 1) / 2) := by
    rw [hbdef, ← Real.rpow_natCast (β ^ ((1 : ℝ) / 2)) (d - 1), ← Real.rpow_mul hβ0.le]
    congr 1
    rw [Nat.cast_sub hd]
    push_cast
    ring
  set δ := c * β ^ (-(1 / 2 : ℝ)) with hδdef
  have hδeq : δ = c / b := by rw [hδdef, Real.rpow_neg hβ0.le, hbdef]; ring
  have hδpos : 0 < δ := by rw [hδeq]; positivity
  have hrpow : (0 : ℝ) ≤ β ^ (((d : ℝ) - 1) / 2) := (Real.rpow_nonneg hβ0.le _)
  rcases le_or_gt δ (1 / 2) with hcase | hcase
  · obtain ⟨S, hSmax⟩ := exists_maximalSeparated d hd δ hδpos (by linarith)
    refine ⟨S, hSmax.1, ?_⟩
    have hcard := card_ge_of_maximalSeparated d hd δ hδpos hcase S hSmax
    have hid : (1 / (4 * δ)) ^ (d - 1) = (1 / (4 * c)) ^ (d - 1) * b ^ (d - 1) := by
      rw [← mul_pow]
      congr 1
      rw [hδeq]
      field_simp
    calc min ((d : ℝ) * (1 / (4 * c)) ^ (d - 1)) ((1 / (2 * c)) ^ (d - 1))
          * β ^ (((d : ℝ) - 1) / 2)
        ≤ (d : ℝ) * (1 / (4 * c)) ^ (d - 1) * β ^ (((d : ℝ) - 1) / 2) :=
          mul_le_mul_of_nonneg_right (min_le_left _ _) hrpow
      _ = (d : ℝ) * (1 / (4 * δ)) ^ (d - 1) := by rw [hid, ← hb_pow]; ring
      _ ≤ (S.card : ℝ) := hcard
  · have hb2c : b < 2 * c := by
      rw [hδeq, lt_div_iff₀ hb0] at hcase
      linarith
    have hu : ‖(EuclideanSpace.single (⟨0, by omega⟩ : Fin d) (1 : ℝ))‖ = 1 := by simp
    refine ⟨{EuclideanSpace.single (⟨0, by omega⟩ : Fin d) (1 : ℝ)}, ⟨?_, ?_⟩, ?_⟩
    · intro x hx
      rw [Finset.mem_singleton.mp hx]
      exact hu
    · intro x hx y hy hxy
      rw [Finset.mem_singleton.mp hx, Finset.mem_singleton.mp hy] at hxy
      exact absurd rfl hxy
    · rw [Finset.card_singleton, Nat.cast_one, ← hb_pow]
      calc min ((d : ℝ) * (1 / (4 * c)) ^ (d - 1)) ((1 / (2 * c)) ^ (d - 1)) * b ^ (d - 1)
          ≤ (1 / (2 * c)) ^ (d - 1) * b ^ (d - 1) :=
            mul_le_mul_of_nonneg_right (min_le_right _ _) (pow_nonneg hb0.le _)
        _ ≤ (1 / (2 * c)) ^ (d - 1) * (2 * c) ^ (d - 1) :=
            mul_le_mul_of_nonneg_left (pow_le_pow_left₀ hb0.le hb2c.le _) (by positivity)
        _ = 1 := by rw [← mul_pow, show 1 / (2 * c) * (2 * c) = 1 by field_simp, one_pow]

/-- The hypotheses of `exists_card_ge_at_renyi_scale` are satisfiable: `d = 1`,
`c = 1`, `β = 1`. -/
example : (1 : ℕ) ≤ 1 ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 1 := ⟨le_rfl, one_pos, le_rfl⟩

end Packing
end Causal
end Transformer
