module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.FourierExtraction.Basic
public import Mathlib.Analysis.Convex.Basic

/-!
# Constant amplitude selects a single Fourier direction

Source: `review/flat_spectrum.md`, CM20, extraction and centering. The theorem
assumes an exact subspace with separated frequency differences, not a discovered
character. It does not assert that an estimated singular subspace is exact or
that an iteration reaches constant amplitude.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.FourierExtraction

open Finset

variable {q : ℕ} [NeZero q] {ι : Type*} [Fintype ι]

/-- A coefficient vector supported at one mode evaluates to that character,
CM20. This implication does not require frequency separation. -/
theorem wave_of_support (k : ι → ZMod q) (a : ι → ℂ) (i : ι)
    (hs : ∀ j, j ≠ i → a j = 0) (x : ZMod q) :
    wave k a x = a i * ZMod.stdAddChar (x * k i) := by
  apply Fintype.sum_eq_single i
  intro j hj
  rw [hs j hj, zero_mul]

/-- Extract an actual single mode from constant amplitude in a separated
Fourier subspace, CM20. Neither the mode nor its coefficient is assumed known. -/
theorem flat_extract {k : ι → ZMod q} (h : Separated k) (a : ι → ℂ)
    (ha : ∃ i, a i ≠ 0) {ρ : ℝ} (hflat : ∀ x, Complex.normSq (wave k a x) = ρ) :
    ∃ i, a i ≠ 0 ∧ (∀ j, j ≠ i → a j = 0) ∧
      ∀ x, wave k a x = a i * ZMod.stdAddChar (x * k i) := by
  obtain ⟨i, hi⟩ := ha
  have hs (j : ι) (hj : j ≠ i) : a j = 0 := by
    have hz := flat_pair_zero h a hflat hj.symm
    exact (mul_eq_zero.mp hz).resolve_left (fun hz => hi (star_eq_zero.mp hz))
  exact ⟨i, hi, hs, wave_of_support k a i hs⟩

/-- Centering excludes extraction of the constant mode, CM20. It does not
require the modulus to be prime. -/
theorem centered_extract {k : ι → ZMod q} (h : Separated k) (a : ι → ℂ)
    (ha : ∃ i, a i ≠ 0) {ρ : ℝ} (hflat : ∀ x, Complex.normSq (wave k a x) = ρ)
    (hmean : ∑ x, wave k a x = 0) :
    ∃ i, k i ≠ 0 ∧ a i ≠ 0 ∧ ∀ x, wave k a x = a i * ZMod.stdAddChar (x * k i) := by
  obtain ⟨i, hi, hs, hw⟩ := flat_extract h a ha hflat
  refine ⟨i, ?_, hi, hw⟩
  intro hk
  have hz : (q : ℂ) * a i = 0 := by
    simpa [hw, hk] using hmean
  exact (mul_ne_zero (Nat.cast_ne_zero.mpr (NeZero.ne q)) hi) hz

/-- For a prime modulus, a nonzero centered constant-amplitude vector in the
separated subspace gives distinct phases at all latent states, CM20. -/
theorem centered_injective [Fact q.Prime] {k : ι → ZMod q} (h : Separated k)
    (a : ι → ℂ) (ha : ∃ i, a i ≠ 0) {ρ : ℝ}
    (hflat : ∀ x, Complex.normSq (wave k a x) = ρ)
    (hmean : ∑ x, wave k a x = 0) : Function.Injective (wave k a) := by
  obtain ⟨i, hk, hi, hw⟩ := centered_extract h a ha hflat hmean
  intro x y hxy
  rw [hw, hw] at hxy
  exact mul_right_cancel₀ hk (ZMod.injective_stdAddChar (mul_left_cancel₀ hi hxy))

/-- An unknown permutation of categorical rows does not change extraction,
CM20. The permutation is a structural hypothesis, not a learner input. -/
theorem permuted_extract {n : Type*} (r : n ≃ ZMod q)
    {k : ι → ZMod q} (h : Separated k) (a : ι → ℂ) (ha : ∃ i, a i ≠ 0) {ρ : ℝ}
    (hflat : ∀ x, Complex.normSq (wave k a (r x)) = ρ) :
    ∃ i, a i ≠ 0 ∧ ∀ x, wave k a (r x) = a i * ZMod.stdAddChar (r x * k i) := by
  have hf (x : ZMod q) : Complex.normSq (wave k a x) = ρ := by
    simpa using hflat (r.symm x)
  obtain ⟨i, hi, hs, hw⟩ := flat_extract h a ha hf
  exact ⟨i, hi, fun x => hw (r x)⟩

/-- A pure Fourier mode really has constant amplitude, CM20. -/
theorem support_flat (k : ι → ZMod q) (a : ι → ℂ) (i : ι)
    (hs : ∀ j, j ≠ i → a j = 0) (x : ZMod q) :
    Complex.normSq (wave k a x) = Complex.normSq (a i) := by
  rw [wave_of_support k a i hs, map_mul, ZMod.stdAddChar_apply, Circle.normSq_coe,
    mul_one]

/-- The unit-amplitude coefficient set is not convex, CM20. The two opposite
pure modes belong to it, but their midpoint does not. This condition cannot be
added directly as a convex constraint on the coefficient vector. -/
theorem amplitude_not_convex (k : ι → ZMod q) (i : ι) :
    ¬ Convex ℝ {a : ι → ℂ | ∀ x, Complex.normSq (wave k a x) = 1} := by
  classical
  let a : ι → ℂ := Pi.single i 1
  have ha : ∀ x, Complex.normSq (wave k a x) = 1 := by
    intro x
    simpa [a] using support_flat k a i (fun j hj => by simp [a, hj]) x
  have hneg (x : ZMod q) : wave k (-a) x = -wave k a x := by
    simp [wave]
  have hb : ∀ x, Complex.normSq (wave k (-a) x) = 1 := by
    intro x
    simpa [hneg] using ha x
  intro hc
  have hm := hc.midpoint_mem hb ha
  rw [midpoint_neg_self] at hm
  have hz := hm (0 : ZMod q)
  simp [wave] at hz

end Transformer.Grokking.FlatSpectrum.CoupledMoments.FourierExtraction
