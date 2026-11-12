module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityExtraction.Selection

/-!
# Observable two-stage selection exposes a pure Fourier mode

Source: `review/flat_spectrum.md`, CM21. Select any two distinct categorical
rows, maximize the real kernel reading, and then its imaginary part within
the first optimum face. No latent ordering is an input to these objectives.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityExtraction

open Finset Matrix FourierExtraction

variable {q : ℕ} [NeZero q] {ι X : Type*} [Fintype ι] [DecidableEq ι]

omit [Fintype ι] in
/-- Distinct rows give distinct mode scores at a prime modulus, CM21.
Separation is a sufficient condition for the frequencies to be distinct. -/
theorem score_injective [Fact q.Prime] (r : X ≃ ZMod q)
    {k : ι → ZMod q} (hk : Separated k) {x y : X} (hxy : x ≠ y) :
    Function.Injective (fun i => ZMod.stdAddChar ((r x - r y) * k i)) := by
  have hd : r x - r y ≠ 0 := sub_ne_zero.mpr (fun he => hxy (r.injective he))
  intro i j he
  have hf := mul_left_cancel₀ hd (ZMod.injective_stdAddChar he)
  by_contra hij
  exact hk.difference_ne_zero hij (sub_eq_zero.mpr hf.symm)

/-- The two linear objectives always have a feasible optimizer in the exact
subspace, CM21. Its existence is proved, not assumed as a solver guarantee. -/
theorem exists_fourier_max [Nonempty ι] (r : X ≃ ZMod q)
    {k : ι → ZMod q} (hk : Separated k) (x y : X) :
    ∃ i, LexMaximum (basis r k) x y (projector i) := by
  let s := fun i => ZMod.stdAddChar ((r x - r y) * k i)
  obtain ⟨i, hr, hi⟩ := exists_lex_vertex s
  refine ⟨i, projector_feasible r k i, ?_, ?_⟩
  · intro N hN
    obtain ⟨w, hw, rfl⟩ := (feasible_iff_diagonal r hk N).mp hN
    simp only [diagonal_reading, projector_reading, Complex.re_sum, Complex.mul_re,
      Complex.ofReal_re, Complex.ofReal_im, zero_mul, sub_zero]
    exact weighted_bound hw (fun j => (s j).re) (s i).re (fun j hj => hr j)
  · intro N hN he
    obtain ⟨w, hw, rfl⟩ := (feasible_iff_diagonal r hk N).mp hN
    simp only [diagonal_reading, projector_reading, Complex.re_sum, Complex.mul_re,
      Complex.ofReal_re, Complex.ofReal_im, zero_mul, sub_zero] at he
    have hs := weighted_support hw (fun j => (s j).re) (s i).re (fun j hj => hr j) he
    simp only [diagonal_reading, projector_reading, Complex.im_sum, Complex.mul_im,
      Complex.ofReal_re, Complex.ofReal_im, zero_mul, add_zero]
    exact weighted_bound hw (fun j => (s j).im) (s i).im (fun j hj => hi j (hs j hj))

/-- Every exact two-stage optimizer is pure at a prime modulus, CM21. No
rank-one condition occurs in either the domain or the optimization predicate. -/
theorem fourier_max_pure [Nonempty ι] [Fact q.Prime] (r : X ≃ ZMod q)
    {k : ι → ZMod q} (hk : Separated k) {x y : X} (hxy : x ≠ y)
    (M : Matrix ι ι ℂ) (hM : LexMaximum (basis r k) x y M) :
    ∃ i, M = projector i := by
  let s := fun i => ZMod.stdAddChar ((r x - r y) * k i)
  obtain ⟨i, hr, hi⟩ := exists_lex_vertex s
  obtain ⟨w, hw, rfl⟩ := (feasible_iff_diagonal r hk M).mp hM.1
  have her : ∑ j, w j * (s j).re = (s i).re := by
    apply le_antisymm (weighted_bound hw _ _ (fun j hj => hr j))
    have h := hM.2.1 (projector i) (projector_feasible r k i)
    simpa [diagonal_reading, projector_reading, s] using h
  have hs := weighted_support hw (fun j => (s j).re) (s i).re (fun j hj => hr j) her
  have hei : ∑ j, w j * (s j).im = (s i).im := by
    apply le_antisymm (weighted_bound hw _ _ (fun j hj => hi j (hs j hj)))
    have h := hM.2.2 (projector i) (projector_feasible r k i)
      (by simpa [diagonal_reading, projector_reading, s] using her.symm)
    simpa [diagonal_reading, projector_reading, s] using h
  have hwone := lex_weights_pure hw s (score_injective r hk hxy) i hr hi her hei
  refine ⟨i, ?_⟩
  rw [hwone]
  unfold projector
  congr 1
  funext j
  by_cases hj : j = i <;> simp [hj]

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityExtraction
