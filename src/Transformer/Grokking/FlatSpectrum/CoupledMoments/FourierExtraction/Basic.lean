module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.FourierExtraction.Defs

/-!
# Isolating cross terms in the squared amplitude

Source: `review/flat_spectrum.md`, CM20, Fourier coefficient identity. Distinct
off-diagonal frequency differences isolate each pair of mixture coefficients.
Constant amplitude annihilates every such nonconstant coefficient.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.FourierExtraction

open Finset

variable {q : ℕ} [NeZero q] {ι : Type*} [Fintype ι]

/-- Expand the actual squared amplitude into its pairwise Fourier readings,
CM20. This is valid even when several differences coincide. -/
theorem wave_power (k : ι → ZMod q) (a : ι → ℂ) (x : ZMod q) :
    (Complex.normSq (wave k a x) : ℂ) =
      ∑ i, ∑ j, star (a i) * a j * ZMod.stdAddChar (x * (k j - k i)) := by
  rw [Complex.normSq_eq_conj_mul_self]
  change star (wave k a x) * wave k a x = _
  simp only [wave, star_sum, star_mul', sum_mul, mul_sum]
  rw [sum_comm]
  apply sum_congr rfl
  intro i hi
  apply sum_congr rfl
  intro j hj
  calc
    _ = star (a i) * a j *
        (star (ZMod.stdAddChar (x * k i)) * ZMod.stdAddChar (x * k j)) := by ring
    _ = _ := by rw [star_stdAddChar_mul, ← mul_sub]

/-- Character orthogonality computes the Fourier coefficient of squared
amplitude as the sum of every pair with that difference, CM20. -/
theorem powerCoefficient_eq (k : ι → ZMod q) (a : ι → ℂ) (t : ZMod q) :
    powerCoefficient k a t =
      ∑ i, ∑ j, if k j - k i = t then (q : ℂ) * (star (a i) * a j) else 0 := by
  classical
  simp only [powerCoefficient, wave_power, mul_sum]
  rw [sum_comm]
  apply sum_congr rfl
  intro i hi
  rw [sum_comm]
  apply sum_congr rfl
  intro j hj
  have hterm (x : ZMod q) :
      star (ZMod.stdAddChar (x * t)) *
          (star (a i) * a j * ZMod.stdAddChar (x * (k j - k i))) =
        (star (a i) * a j) * ZMod.stdAddChar (x * (k j - k i - t)) := by
    calc
      _ = (star (a i) * a j) * (star (ZMod.stdAddChar (x * t)) *
          ZMod.stdAddChar (x * (k j - k i))) := by ring
      _ = _ := by rw [star_stdAddChar_mul, ← mul_sub]
  simp_rw [hterm]
  rw [← mul_sum, sum_stdAddChar_mul]
  by_cases h : k j - k i = t <;> simp [h, sub_eq_zero, mul_comm]

/-- A constant squared amplitude has no nonzero Fourier coefficient, CM20. -/
theorem powerCoefficient_const (k : ι → ZMod q) (a : ι → ℂ) {ρ : ℝ}
    (hflat : ∀ x, Complex.normSq (wave k a x) = ρ) {t : ZMod q} (ht : t ≠ 0) :
    powerCoefficient k a t = 0 := by
  simp only [powerCoefficient, hflat]
  rw [← sum_mul, ← star_sum, sum_stdAddChar_mul]
  simp [ht]

omit [Fintype ι] [NeZero q] in
/-- The separation condition excludes zero off-diagonal differences, CM20. -/
theorem Separated.difference_ne_zero {k : ι → ZMod q} (h : Separated k)
    {i j : ι} (hij : i ≠ j) : k j - k i ≠ 0 := by
  intro hz
  have hp := h i j hij i i (by simpa using hz)
  exact hij hp.2.symm

/-- A separated nonzero difference reads one coefficient product exactly,
CM20. No statement about a leading singular vector is assumed here. -/
theorem Separated.coefficient {k : ι → ZMod q} (h : Separated k)
    (a : ι → ℂ) {i j : ι} (hij : i ≠ j) :
    powerCoefficient k a (k j - k i) = (q : ℂ) * (star (a i) * a j) := by
  classical
  rw [powerCoefficient_eq]
  have heq (u v : ι) : k v - k u = k j - k i ↔ u = i ∧ v = j := by
    constructor
    · intro hv
      obtain ⟨hu, hw⟩ := h i j hij u v hv.symm
      exact ⟨hu.symm, hw.symm⟩
    · rintro ⟨rfl, rfl⟩
      rfl
  simp_rw [heq]
  rw [Fintype.sum_eq_single i]
  · simp
  · intro u hu
    simp [hu]

/-- Constant amplitude and separated differences forbid two nonzero mixture
coefficients, CM20. The modulus may be composite. -/
theorem flat_pair_zero {k : ι → ZMod q} (h : Separated k) (a : ι → ℂ) {ρ : ℝ}
    (hflat : ∀ x, Complex.normSq (wave k a x) = ρ) {i j : ι} (hij : i ≠ j) :
    star (a i) * a j = 0 := by
  have hc := powerCoefficient_const k a hflat (h.difference_ne_zero hij)
  rw [h.coefficient a hij] at hc
  exact (mul_eq_zero.mp hc).resolve_left (Nat.cast_ne_zero.mpr (NeZero.ne q))

end Transformer.Grokking.FlatSpectrum.CoupledMoments.FourierExtraction
