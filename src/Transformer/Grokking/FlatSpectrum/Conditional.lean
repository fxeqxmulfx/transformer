module

public import Mathlib.Analysis.Complex.Basic
public import Mathlib.Analysis.Real.Sqrt

/-!
# Conditional means on a finite uniform space

Source: `review/flat_spectrum.md`, Theorem B, where `E[· | f]` and `‖·‖` are taken over uniform
`(a, b)`. On a finite type with the uniform distribution, `E[g | F](x)` is the mean of `g` over
the fiber of `F` through `x`, and `‖h‖² = E |h|²`. Grouping by fibers,
`∑_x |E[g | F](x)|² = ∑_c |∑_{F = c} g|² / #{F = c}`. The tower rule
`E[conj φ(F) g] = E[conj φ(F) E[g | F]]` and Cauchy–Schwarz give
`|E[conj φ(F) g]| ≤ ‖φ(F)‖ ‖E[g | F]‖`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Finset

variable {α β : Type*} [Fintype α] [DecidableEq β]

/-- The mean `E g = (∑_x g x) / #α` on a finite uniform space. -/
noncomputable def mean (g : α → ℂ) : ℂ := (∑ x, g x) / Fintype.card α

/-- The mean square `‖g‖² = E |g|²` on a finite uniform space, `review/flat_spectrum.md`,
Theorem B. -/
noncomputable def meanSq (g : α → ℂ) : ℝ := (∑ x, ‖g x‖ ^ 2) / Fintype.card α

/-- The conditional mean `E[g | F]` on a finite uniform space: the mean of `g` over the fiber
of `F` through `x`, `review/flat_spectrum.md`, Theorem B. -/
noncomputable def condMean (F : α → β) (g : α → ℂ) (x : α) : ℂ :=
  (∑ y with F y = F x, g y) / #{y | F y = F x}

lemma condMean_of_eq (F : α → β) (g : α → ℂ) {x : α} {c : β} (hx : F x = c) :
    condMean F g x = (∑ y with F y = c, g y) / #{y | F y = c} := by
  simp only [condMean, hx]

example := condMean_of_eq (fun x : Fin 2 => x) (fun _ => 1) (x := 0) (c := 0) rfl

/-- `∑_x |E[g | F](x)|² = ∑_c |∑_{F = c} g|² / #{F = c}`. -/
lemma sum_norm_condMean_sq [Fintype β] (F : α → β) (g : α → ℂ) :
    ∑ x, ‖condMean F g x‖ ^ 2 = ∑ c, ‖∑ y with F y = c, g y‖ ^ 2 / #{y | F y = c} := by
  rw [← Finset.sum_fiberwise univ F]
  refine Finset.sum_congr rfl fun c _ => ?_
  have h (x : α) (hx : x ∈ ({y | F y = c} : Finset α)) : ‖condMean F g x‖ ^ 2 =
      ‖∑ y with F y = c, g y‖ ^ 2 / (#{y | F y = c} : ℝ) ^ 2 := by
    rw [condMean_of_eq F g (Finset.mem_filter.mp hx).2, norm_div, Complex.norm_natCast, div_pow]
  rw [Finset.sum_congr rfl h, Finset.sum_const, nsmul_eq_mul]
  rcases eq_or_ne (#{y | F y = c} : ℝ) 0 with h0 | h0
  · rw [h0]
    simp
  · field_simp

/-- The tower rule: `∑_x conj φ(F x) E[g | F](x) = ∑_x conj φ(F x) g(x)`. -/
lemma sum_star_mul_condMean [Fintype β] (F : α → β) (g : α → ℂ) (φ : β → ℂ) :
    ∑ x, star (φ (F x)) * condMean F g x = ∑ x, star (φ (F x)) * g x := by
  rw [← Finset.sum_fiberwise univ F, ← Finset.sum_fiberwise univ F fun x => star (φ (F x)) * g x]
  refine Finset.sum_congr rfl fun c _ => ?_
  have h (x : α) (hx : x ∈ ({y | F y = c} : Finset α)) : star (φ (F x)) * condMean F g x =
      star (φ c) * ((∑ y with F y = c, g y) / #{y | F y = c}) := by
    rw [condMean_of_eq F g (Finset.mem_filter.mp hx).2, (Finset.mem_filter.mp hx).2]
  have h' (x : α) (hx : x ∈ ({y | F y = c} : Finset α)) :
      star (φ (F x)) * g x = star (φ c) * g x := by
    rw [(Finset.mem_filter.mp hx).2]
  rw [Finset.sum_congr rfl h, Finset.sum_congr rfl h', Finset.sum_const, nsmul_eq_mul,
    ← Finset.mul_sum]
  rcases eq_or_ne #{y | F y = c} 0 with h0 | h0
  · rw [Finset.card_eq_zero.mp h0]
    simp
  · have : (#{y | F y = c} : ℂ) ≠ 0 := Nat.cast_ne_zero.mpr h0
    field_simp

/-- Cauchy–Schwarz through the tower rule: `|E[conj φ(F) g]| ≤ ‖φ(F)‖ ‖E[g | F]‖`. -/
theorem norm_mean_star_mul_le [Fintype β] (F : α → β) (g : α → ℂ) (φ : β → ℂ) :
    ‖mean fun x => star (φ (F x)) * g x‖ ≤
      √(meanSq fun x => φ (F x)) * √(meanSq (condMean F g)) := by
  have hcs : ‖∑ x, star (φ (F x)) * condMean F g x‖ ≤
      √(∑ x, ‖φ (F x)‖ ^ 2) * √(∑ x, ‖condMean F g x‖ ^ 2) :=
    (norm_sum_le _ _).trans (by
      simpa only [norm_mul, norm_star] using Real.sum_mul_le_sqrt_mul_sqrt univ
        (fun x => ‖φ (F x)‖) fun x => ‖condMean F g x‖)
  rw [mean, meanSq, meanSq, ← sum_star_mul_condMean, norm_div, Complex.norm_natCast,
    Real.sqrt_div (Finset.sum_nonneg fun _ _ => sq_nonneg _),
    Real.sqrt_div (Finset.sum_nonneg fun _ _ => sq_nonneg _), div_mul_div_comm,
    Real.mul_self_sqrt (Nat.cast_nonneg _)]
  exact div_le_div_of_nonneg_right hcs (Nat.cast_nonneg _)

example := norm_mean_star_mul_le (fun x : Fin 2 => x) (fun _ => 1) fun _ => 1

end Transformer.Grokking.FlatSpectrum
