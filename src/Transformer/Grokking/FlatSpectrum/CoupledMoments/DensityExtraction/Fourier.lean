module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityExtraction.Basic

/-!
# Constant diagonal forces a diagonal Fourier density

Source: `review/flat_spectrum.md`, CM21. Orthogonality reads the entries of an
arbitrary matrix, rather than products from an assumed rank-one factorization.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityExtraction

open Finset Matrix FourierExtraction
open scoped ComplexOrder

variable {q : ℕ} [NeZero q] {ι X : Type*} [Fintype ι]

/-- The actual lifted diagonal has frequencies `k_i-k_j`, CM21. -/
theorem fourier_diagonal (k : ι → ZMod q) (M : Matrix ι ι ℂ) (x : ZMod q) :
    reading (basis id k) M x x =
      ∑ i, ∑ j, M i j * ZMod.stdAddChar (x * (k i - k j)) := by
  rw [reading_eq]
  apply sum_congr rfl
  intro i hi
  apply sum_congr rfl
  intro j hj
  simp only [basis, id_eq]
  rw [mul_comm (ZMod.stdAddChar (x * k i)), star_stdAddChar_mul, ← mul_sub]

/-- Orthogonality computes the diagonal reading's Fourier coefficient even
when frequency differences collide, CM21. -/
theorem diagonal_coefficient (k : ι → ZMod q) (M : Matrix ι ι ℂ) (t : ZMod q) :
    (∑ x, star (ZMod.stdAddChar (x * t)) * reading (basis id k) M x x) =
      ∑ i, ∑ j, if k i - k j = t then (q : ℂ) * M i j else 0 := by
  classical
  simp only [fourier_diagonal, mul_sum]
  rw [sum_comm]
  apply sum_congr rfl
  intro i hi
  rw [sum_comm]
  apply sum_congr rfl
  intro j hj
  have ht (x : ZMod q) :
      star (ZMod.stdAddChar (x * t)) * (M i j * ZMod.stdAddChar (x * (k i - k j))) =
        M i j * ZMod.stdAddChar (x * (k i - k j - t)) := by
    calc
      _ = M i j * (star (ZMod.stdAddChar (x * t)) *
          ZMod.stdAddChar (x * (k i - k j))) := by ring
      _ = _ := by rw [star_stdAddChar_mul, ← mul_sub]
  simp_rw [ht]
  rw [← mul_sum, sum_stdAddChar_mul]
  by_cases h : k i - k j = t <;> simp [h, sub_eq_zero, mul_comm]

/-- Separation and constant readings kill every off-diagonal matrix entry,
CM21. Positivity and rank are not needed for this implication. -/
theorem flat_off_diagonal {k : ι → ZMod q} (hk : Separated k)
    (M : Matrix ι ι ℂ) (hf : ∀ x, reading (basis id k) M x x = 1)
    {i j : ι} (hij : i ≠ j) : M i j = 0 := by
  classical
  have hc := diagonal_coefficient k M (k i - k j)
  have heq (u v : ι) : k u - k v = k i - k j ↔ u = i ∧ v = j := by
    constructor
    · intro hv
      obtain ⟨hv, hu⟩ := hk j i hij.symm v u hv.symm
      exact ⟨hu.symm, hv.symm⟩
    · rintro ⟨rfl, rfl⟩
      rfl
  simp_rw [hf, heq] at hc
  have hr : (∑ u, ∑ v, if u = i ∧ v = j then (q : ℂ) * M u v else 0) =
      (q : ℂ) * M i j := by
    rw [Fintype.sum_eq_single i]
    · simp
    · intro u hu
      simp [hu]
  rw [hr] at hc
  simp only [mul_one, ← star_sum, sum_stdAddChar_mul,
    ite_eq_right (hk.difference_ne_zero hij.symm), star_zero] at hc
  exact (mul_eq_zero.mp hc.symm).resolve_left (Nat.cast_ne_zero.mpr (NeZero.ne q))

/-- Every diagonal probability mixture is feasible, including genuine mixed
densities, CM21. Frequency separation is unnecessary in this direction. -/
theorem diagonal_feasible [DecidableEq ι] (r : X → ZMod q) (k : ι → ZMod q)
    {w : ι → ℝ} (hw : Weights w) :
    Feasible (basis r k) (diagonal fun i => (w i : ℂ)) := by
  refine ⟨(diagonal_density w).mpr hw, fun x => ?_⟩
  rw [reading_diagonal]
  have ht (i : ι) : basis r k x i * star (basis r k x i) = 1 := by
    simp only [basis]
    rw [mul_comm, star_stdAddChar_mul, sub_self, AddChar.map_zero_eq_one]
  simp_rw [ht, mul_one]
  simpa using congrArg (fun x : ℝ => (x : ℂ)) hw.2

/-- Under separated differences the feasible matrices are exactly the real
diagonal probability mixtures, CM21. The row permutation is not learner input. -/
theorem feasible_iff_diagonal [DecidableEq ι] (r : X ≃ ZMod q)
    {k : ι → ZMod q} (hk : Separated k) (M : Matrix ι ι ℂ) :
    Feasible (basis r k) M ↔
      ∃ w : ι → ℝ, Weights w ∧ M = diagonal (fun i => (w i : ℂ)) := by
  constructor
  · intro hM
    have hf (x : ZMod q) : reading (basis id k) M x x = 1 := by
      simpa [reading_eq, basis] using hM.2 (r.symm x)
    have hd : M = diagonal (fun i => ((M i i).re : ℂ)) := by
      ext i j
      by_cases hij : i = j
      · subst j
        simp only [diagonal_apply_eq]
        apply Complex.ext
        · rfl
        · exact (Complex.nonneg_iff.mp (hM.1.1.diag_nonneg (i := i))).2.symm
      · rw [diagonal_apply_ne _ hij]
        exact flat_off_diagonal hk M hf hij
    refine ⟨fun i => (M i i).re, ?_, hd⟩
    exact (diagonal_density _).mp (hd ▸ hM.1)
  · rintro ⟨w, hw, rfl⟩
    exact diagonal_feasible r k hw

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityExtraction
