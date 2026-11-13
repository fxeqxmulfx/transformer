module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityBounds.Basic

/-!
# Quantitative diagonalization in the structural Fourier basis

Source: `review/flat_spectrum.md`, CM22. These estimates hold for arbitrary
coefficient matrices; PSD enters later to interpret diagonal entries as weights.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityBounds

open Finset Matrix DensityExtraction FourierExtraction

variable {q : ℕ} [NeZero q] {ι X : Type*} [Fintype ι]

/-- A separated nonzero difference isolates an entry from actual diagonal
errors, CM22. The normalization is the unnormalized character basis of CM21. -/
theorem error_coefficient {k : ι → ZMod q} (hk : Separated k)
    (M : Matrix ι ι ℂ) {i j : ι} (hij : i ≠ j) :
    (q : ℂ) * M i j = ∑ x, star (ZMod.stdAddChar (x * (k i - k j))) *
      (reading (basis id k) M x x - 1) := by
  classical
  have hc := diagonal_coefficient k M (k i - k j)
  have heq (u v : ι) : k u - k v = k i - k j ↔ u = i ∧ v = j := by
    constructor
    · intro hv
      obtain ⟨hv, hu⟩ := hk j i hij.symm v u hv.symm
      exact ⟨hu.symm, hv.symm⟩
    · rintro ⟨rfl, rfl⟩
      rfl
  simp_rw [heq] at hc
  have hr : (∑ u, ∑ v, if u = i ∧ v = j then (q : ℂ) * M u v else 0) =
      (q : ℂ) * M i j := by
    rw [Fintype.sum_eq_single i]
    · simp
    · intro u hu
      simp [hu]
  rw [hr] at hc
  have hex (z a : ℂ) : z * (a - 1) = z * a - z := by ring
  simp_rw [hex, sum_sub_distrib]
  rw [hc, ← star_sum, sum_stdAddChar_mul]
  simp [hk.difference_ne_zero hij.symm]

/-- Uniform diagonal error controls each off-diagonal hidden entry, CM22.
No rank assumption or recovered mode is used. -/
theorem off_diagonal_le (r : X ≃ ZMod q) {k : ι → ZMod q} (hk : Separated k)
    (M : Matrix ι ι ℂ) (ε : ℝ) (he : ∀ x, ‖reading (basis r k) M x x - 1‖ ≤ ε)
    {i j : ι} (hij : i ≠ j) : ‖M i j‖ ≤ ε := by
  have herror (x : ZMod q) : ‖reading (basis id k) M x x - 1‖ ≤ ε := by
    simpa [reading_eq, basis] using he (r.symm x)
  have hb : (q : ℝ) * ‖M i j‖ ≤ (q : ℝ) * ε := by
    calc
      _ = ‖(q : ℂ) * M i j‖ := by simp
      _ = ‖∑ x, star (ZMod.stdAddChar (x * (k i - k j))) *
          (reading (basis id k) M x x - 1)‖ := by rw [error_coefficient hk M hij]
      _ ≤ ∑ x, ‖star (ZMod.stdAddChar (x * (k i - k j))) *
          (reading (basis id k) M x x - 1)‖ := norm_sum_le _ _
      _ ≤ ∑ x : ZMod q, ε := by
        apply sum_le_sum
        intro x hx
        simpa only [norm_mul, norm_star, ZMod.stdAddChar_apply, Circle.norm_coe,
          one_mul] using herror x
      _ = _ := by simp
  exact le_of_mul_le_mul_left hb (Nat.cast_pos.mpr (Nat.pos_of_ne_zero (NeZero.ne q)))

/-- Replacing small off-diagonal coefficients by zero changes every Fourier
kernel entry by at most `d² ε`, CM22. The dimension factor is explicit and
deliberately conservative; this is not a dimension-free bound. -/
theorem diagonal_reading_error [DecidableEq ι] (r : X → ZMod q) (k : ι → ZMod q)
    (M : Matrix ι ι ℂ) {ε : ℝ} (hε : 0 ≤ ε)
    (he : ∀ i j, i ≠ j → ‖M i j‖ ≤ ε) (x y : X) :
    ‖reading (basis r k) M x y -
      reading (basis r k) (diagonal fun i => M i i) x y‖ ≤ (Fintype.card ι : ℝ)^2 * ε := by
  rw [reading_eq, reading_eq, ← sum_sub_distrib]
  calc
    _ ≤ ∑ i, ‖(∑ j, M i j * (basis r k x i * star (basis r k y j))) -
        ∑ j, (diagonal fun u => M u u) i j * (basis r k x i * star (basis r k y j))‖ :=
      norm_sum_le _ _
    _ ≤ ∑ i : ι, ∑ j : ι, ε := by
      apply sum_le_sum
      intro i hi
      rw [← sum_sub_distrib]
      apply norm_sum_le_of_le
      intro j hj
      by_cases hij : i = j
      · subst j
        simpa using hε
      · simpa only [diagonal_apply_ne _ hij, zero_mul, sub_zero, norm_mul,
          norm_star, basis, ZMod.stdAddChar_apply, Circle.norm_coe, mul_one] using he i j hij
    _ = _ := by simp [pow_two, mul_assoc]

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityBounds
