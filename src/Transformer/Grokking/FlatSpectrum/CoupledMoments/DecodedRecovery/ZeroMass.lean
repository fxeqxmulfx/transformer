module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Schatten.PairedLimits

/-!
# Symmetric latent tables on three states

Source: `review/flat_spectrum.md`, CM18. On three latent states an even probability row is one
number, its mass at zero. For the decoder that merges the two nonzero states, the decoded row
and both nonconstant Fourier moments of such a table are affine images of that number
(`decode_zeroMass`, `moment_zeroMass`). These are the computations behind the wrong minimum of
`Obstruction.lean`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery

open Matrix Finset Schatten

variable {n : Type*}

/-- A symmetric latent table on three states: the mass `A i j` at zero, the rest shared
equally by the two opposite states. -/
noncomputable def zeroMass (A : n → n → ℝ) : Table n n (ZMod 3) :=
  fun i j z => (1 - A i j) / 2 + if z = 0 then (3 * A i j - 1) / 2 else 0

/-- Opposite states have equal probabilities (CM18). -/
theorem zeroMass_even (A : n → n → ℝ) : EvenLatent (zeroMass A) := by
  intro i j z
  simp only [zeroMass, neg_eq_zero]

/-- A mass at zero between `0` and `1` gives probability rows (CM18). -/
theorem rows_zeroMass {A : n → n → ℝ} (hA : ∀ i j, 0 ≤ A i j ∧ A i j ≤ 1) :
    Rows (zeroMass A) := by
  constructor
  · intro i j z
    simp only [zeroMass]
    split_ifs
    · linarith [(hA i j).1]
    · linarith [(hA i j).2]
  · intro i j
    simp only [zeroMass, sum_add_distrib, sum_const, card_univ, ZMod.card, sum_ite_eq',
      mem_univ, ↓reduceIte, nsmul_eq_mul]
    push_cast
    ring

/-- The decoder that merges the two nonzero states reads the mass at zero. -/
theorem decode_zeroMass {A : n → n → ℝ} (hA : ∀ i j, 0 ≤ A i j ∧ A i j ≤ 1) (i j : n) :
    decode pairedBoolDecoder (zeroMass A) i j = fun b => if b then 1 - A i j else A i j := by
  have hf : decode pairedBoolDecoder (zeroMass A) i j false = A i j := by
    change (∑ z : ZMod 3, if pairedBoolDecoder z = false then zeroMass A i j z else 0) = A i j
    rw [sum_eq_single 0]
    · simp only [pairedBoolDecoder, zeroMass]
      norm_num
      ring
    · intro z _ hz
      simp [pairedBoolDecoder, hz]
    · simp
  have hs := (rows_decode pairedBoolDecoder (rows_zeroMass hA)).2 i j
  rw [Fintype.sum_bool] at hs
  funext b
  cases b
  · simpa using hf
  · simp only [↓reduceIte]
    linarith

/-- A nonconstant Fourier moment of the table is an affine image of the mass at zero. -/
theorem moment_zeroMass (A : n → n → ℝ) {f : ZMod 3} (hf : f ≠ 0) :
    moment (phaseCode (Equiv.refl (ZMod 3)) f) (zeroMass A) =
      fun i j => (((3 * A i j - 1) / 2 : ℝ) : ℂ) := by
  have hs : ∑ z : ZMod 3, ZMod.stdAddChar (f * z) = 0 := by
    have h := sum_stdAddChar_mul 3 f
    rw [ite_eq_right hf] at h
    exact (sum_congr rfl fun z _ => by rw [mul_comm]).trans h
  ext i j
  have hrow (z : ZMod 3) :
      (zeroMass A i j z : ℂ) * phaseCode (Equiv.refl (ZMod 3)) f z =
        (((1 - A i j) / 2 : ℝ) : ℂ) * ZMod.stdAddChar (f * z) +
          if z = 0 then (((3 * A i j - 1) / 2 : ℝ) : ℂ) else 0 := by
    change (((1 - A i j) / 2 + if z = 0 then (3 * A i j - 1) / 2 else 0 : ℝ) : ℂ) *
      ZMod.stdAddChar (f * z) = _
    split_ifs with hz
    · rw [hz, mul_zero, AddChar.map_zero_eq_one]
      push_cast
      ring
    · simp
  change (∑ z, (zeroMass A i j z : ℂ) * phaseCode (Equiv.refl (ZMod 3)) f z) = _
  simp only [hrow, sum_add_distrib, ← mul_sum, hs, mul_zero, zero_add, sum_ite_eq', mem_univ,
    ↓reduceIte]

/-! A constant table with half of the mass at zero. -/

private lemma half_range (i j : Fin 2) :
    0 ≤ (fun _ _ : Fin 2 => (1 / 2 : ℝ)) i j ∧ (fun _ _ : Fin 2 => (1 / 2 : ℝ)) i j ≤ 1 := by
  norm_num

example := zeroMass_even (fun _ _ : Fin 2 => (1 / 2 : ℝ))
example := rows_zeroMass half_range
example := decode_zeroMass half_range 0 1
example := moment_zeroMass (fun _ _ : Fin 2 => (1 / 2 : ℝ)) (f := 1) (by decide)

end Transformer.Grokking.FlatSpectrum.CoupledMoments.DecodedRecovery
