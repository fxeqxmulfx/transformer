module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityExtraction.Rotated

/-!
# Observable centering excludes the constant mode

Source: `review/flat_spectrum.md`, CM21. Centering is checked on the supplied
feature columns, not on a latent frequency selected by the proof.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityExtraction

open Finset Matrix FourierExtraction

variable {ι X : Type*} [Fintype ι] [Fintype X]

/-- Changing column coordinates preserves zero column sums, CM21. -/
theorem centered_mul (B : Matrix X ι ℂ) (Q : Matrix ι ι ℂ)
    (hB : ∀ j, ∑ x, B x j = 0) (i : ι) : ∑ x, (B * Q) x i = 0 := by
  simp only [Matrix.mul_apply]
  rw [sum_comm]
  simp_rw [← sum_mul, hB, zero_mul]
  exact sum_const_zero

/-- Observable zero means rule out every constant Fourier column under the
exact unitary-subspace hypothesis, CM21. No chosen frequency is presumed. -/
theorem centered_nonzero {q : ℕ} [NeZero q] [DecidableEq ι] (r : X ≃ ZMod q)
    (k : ι → ZMod q) (Q : Matrix ι ι ℂ) (hQ' : Q * Q.conjTranspose = 1)
    (hc : ∀ j, ∑ x, (basis r k * Q) x j = 0) : ∀ i, k i ≠ 0 := by
  intro i hi
  have hs := centered_mul (basis r k * Q) Q.conjTranspose hc i
  rw [Matrix.mul_assoc, hQ', Matrix.mul_one] at hs
  have he : (∑ x, basis r k x i) = (q : ℂ) := by
    simp only [basis]
    rw [r.sum_comp (fun x => ZMod.stdAddChar (x * k i)), sum_stdAddChar_mul]
    simp [hi]
  rw [he] at hs
  exact Nat.cast_ne_zero.mpr (NeZero.ne q) hs

/-- The supplied-basis version uses only observable centering and a declared
exact-subspace assumption, CM21. Its optimizer gives an injective phase column;
it makes no claim about any unseen decoded answer. -/
theorem supplied_extraction {q : ℕ} [NeZero q] [Fact q.Prime] [Nonempty ι]
    [DecidableEq ι] (B : Matrix X ι ℂ) (r : X ≃ ZMod q)
    {k : ι → ZMod q} (hk : Separated k) (Q : Matrix ι ι ℂ)
    (hQ : Q.conjTranspose * Q = 1) (hQ' : Q * Q.conjTranspose = 1)
    (hB : B = basis r k * Q) (hc : ∀ j, ∑ x, B x j = 0)
    {x y : X} (hxy : x ≠ y) :
    ∃ M, LexMaximum B x y M ∧ ∀ a, Function.Injective (fun z => reading B M z a) := by
  subst B
  obtain ⟨i, hi⟩ := exists_rotated_max r hk Q hQ hQ' x y
  refine ⟨Q.conjTranspose * projector i * Q, hi, fun a => ?_⟩
  exact extracted_column_injective r hk (centered_nonzero r k Q hQ' hc) Q hQ hQ'
    hxy _ hi a

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityExtraction
