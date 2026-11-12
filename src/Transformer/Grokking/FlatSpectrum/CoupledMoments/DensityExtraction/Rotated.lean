module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityExtraction.Exposure

/-!
# Extraction through observable linear objectives in the supplied basis

Source: `review/flat_spectrum.md`, CM21. Unitary coordinate changes preserve
the entire optimization problem. Exact subspace geometry gives a pure
optimizer and, for nonzero prime frequencies, distinct phases at every row.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityExtraction

open Finset Matrix FourierExtraction

variable {ι X : Type*} [Fintype ι] [DecidableEq ι]

/-- Both optimization stages are unchanged by a unitary basis change, CM21.
Every comparison ranges over the full convex feasible set. -/
theorem lex_change (B : Matrix X ι ℂ) (Q M : Matrix ι ι ℂ)
    (hQ : Q.conjTranspose * Q = 1) (hQ' : Q * Q.conjTranspose = 1) (x y : X) :
    LexMaximum (B * Q) x y M ↔ LexMaximum B x y (Q * M * Q.conjTranspose) := by
  have undo (N : Matrix ι ι ℂ) :
      Q * (Q.conjTranspose * N * Q) * Q.conjTranspose = N := by
    simpa using undo_conjugate Q.conjTranspose N (by simpa using hQ')
  have hf (N : Matrix ι ι ℂ) (hN : Feasible B N) :
      Feasible (B * Q) (Q.conjTranspose * N * Q) := by
    apply (feasible_change B Q _ hQ hQ').mpr
    rwa [undo]
  constructor
  · intro hM
    refine ⟨(feasible_change B Q M hQ hQ').mp hM.1, ?_, ?_⟩
    · intro N hN
      have h := hM.2.1 _ (hf N hN)
      simpa only [reading_change, undo] using h
    · intro N hN he
      have he' : (reading (B * Q) (Q.conjTranspose * N * Q) x y).re =
          (reading (B * Q) M x y).re := by
        simpa only [reading_change, undo] using he
      have h := hM.2.2 _ (hf N hN) he'
      simpa only [reading_change, undo] using h
  · intro hM
    refine ⟨(feasible_change B Q M hQ hQ').mpr hM.1, ?_, ?_⟩
    · intro N hN
      simpa only [reading_change] using
        hM.2.1 _ ((feasible_change B Q N hQ hQ').mp hN)
    · intro N hN he
      have he' := he
      simp only [reading_change] at he'
      simpa only [reading_change] using
        hM.2.2 _ ((feasible_change B Q N hQ hQ').mp hN) he'

variable {q : ℕ} [NeZero q]

/-- A pure rotated optimizer exists without supplying the selected mode to
the objectives, CM21. This is exact mathematical existence, not iteration cost. -/
theorem exists_rotated_max [Nonempty ι] (r : X ≃ ZMod q)
    {k : ι → ZMod q} (hk : Separated k) (Q : Matrix ι ι ℂ)
    (hQ : Q.conjTranspose * Q = 1) (hQ' : Q * Q.conjTranspose = 1) (x y : X) :
    ∃ i, LexMaximum (basis r k * Q) x y (Q.conjTranspose * projector i * Q) := by
  obtain ⟨i, hi⟩ := exists_fourier_max r hk x y
  refine ⟨i, (lex_change _ Q _ hQ hQ' x y).mpr ?_⟩
  have he : Q * (Q.conjTranspose * projector i * Q) * Q.conjTranspose = projector i := by
    simpa using undo_conjugate Q.conjTranspose (projector i) (by simpa using hQ')
  rwa [he]

/-- Every optimizer in an exact rotated separated Fourier subspace is a pure
mode projector, CM21. The optimization itself sees only the supplied basis. -/
theorem rotated_max_pure [Nonempty ι] [Fact q.Prime] (r : X ≃ ZMod q)
    {k : ι → ZMod q} (hk : Separated k) (Q : Matrix ι ι ℂ)
    (hQ : Q.conjTranspose * Q = 1) (hQ' : Q * Q.conjTranspose = 1)
    {x y : X} (hxy : x ≠ y) (M : Matrix ι ι ℂ)
    (hM : LexMaximum (basis r k * Q) x y M) :
    ∃ i, M = Q.conjTranspose * projector i * Q := by
  obtain ⟨i, hi⟩ := fourier_max_pure r hk hxy _ ((lex_change _ Q M hQ hQ' x y).mp hM)
  refine ⟨i, ?_⟩
  rw [← undo_conjugate Q M hQ, hi]

/-- The extracted kernel column distinguishes all categorical rows if the
exact subspace has no constant mode, CM21. This identifies phases, not labels. -/
theorem extracted_column_injective [Nonempty ι] [Fact q.Prime] (r : X ≃ ZMod q)
    {k : ι → ZMod q} (hk : Separated k) (hk0 : ∀ i, k i ≠ 0) (Q : Matrix ι ι ℂ)
    (hQ : Q.conjTranspose * Q = 1) (hQ' : Q * Q.conjTranspose = 1)
    {x y : X} (hxy : x ≠ y) (M : Matrix ι ι ℂ)
    (hM : LexMaximum (basis r k * Q) x y M) (a : X) :
    Function.Injective (fun z => reading (basis r k * Q) M z a) := by
  obtain ⟨i, hi⟩ := rotated_max_pure r hk Q hQ hQ' hxy M hM
  have hm : Q * M * Q.conjTranspose = projector i := by
    rw [hi]
    simpa using undo_conjugate Q.conjTranspose (projector i) (by simpa using hQ')
  intro u v he
  simp only [reading_change, hm, projector_reading] at he
  exact r.injective (sub_left_inj.mp (mul_right_cancel₀ (hk0 i)
    (ZMod.injective_stdAddChar he)))

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DensityExtraction
