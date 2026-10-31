module

public import Transformer.Rotate90.Defs
public import Transformer.Grokking.FlatSpectrum.ReadoutChoice

/-!
# Character geometry and the exact branch norm

Source: `review/flat_spectrum.md`, The readout decides, and the user-requested
existence theorem for `grokking_abmod/docs/rotate90-swiglu-model.md`, Model.
A faithful character puts nonzero residues on the circle; an extra coordinate
represents zero. Every code has unit length and uniquely maximizes its own
inner-product score. The shared embedding has four adjacent pairs:
`(1,0, Re χ,0, Im χ,0, [a=0],0)`.

Writing C = χ(a/b) and A = χ(a), the branch pairs are `(0,0)`,
`(Re C-Re A, Re C+Re A)`, `(Im C-Im A, Im C+Im A)`, `(0,2[a=0])`.
Both raw vectors have squared length four. Their RMSNorm scales therefore
agree exactly. Odd coordinates preserve that equality and are invisible to
the tied readout, but are included in every normalization denominator.
-/

@[expose] public section

open scoped BigOperators

namespace Transformer.Rotate90

open Landscape.Neural Grokking.FlatSpectrum

variable {p : ℕ} [Fact p.Prime] {K : Type*} [Fintype K]

/-- The zero-numerator coordinate; it is part of the shared embedding. -/
def zeroFlag (a : ZMod p) : ℝ := if a = 0 then 1 else 0

/-- The inner product of the real three-coordinate character codes. -/
noncomputable def characterScore (χ : MulChar (ZMod p) ℂ) (a q : ZMod p) : ℝ :=
  (χ a).re * (χ q).re + (χ a).im * (χ q).im + zeroFlag a * zeroFlag q

/-- The code has exactly unit length, both on the circle and at zero. -/
theorem character_sq (χ : MulChar (ZMod p) ℂ) (a : ZMod p) :
    (χ a).re ^ 2 + (χ a).im ^ 2 + zeroFlag a ^ 2 = 1 := by
  by_cases ha : a = 0
  · simp [ha, zeroFlag, MulChar.map_zero]
  · have hn := norm_mulChar_eq_one χ ha
    have hs := Complex.normSq_eq_norm_sq (χ a)
    rw [hn, Complex.normSq_apply] at hs
    simp only [zeroFlag, ha, ↓reduceIte]
    nlinarith

/-- Multiplication by the conjugate is division in the faithful code.
This identity even respects the totalized inverse at zero; classification
will explicitly restrict the denominator to nonzero residues. -/
theorem character_div (χ : MulChar (ZMod p) ℂ) (a b : ZMod p) :
    χ (a / b) = χ a * star (χ b) := by
  rw [div_eq_mul_inv, map_mul, ← MulChar.inv_apply', ← MulChar.star_apply']

/-- Division by a nonzero residue preserves exactly the zero numerator. -/
theorem zeroFlag_div (a : ZMod p) {b : ZMod p} (hb : b ≠ 0) :
    zeroFlag (a / b) = zeroFlag a := by
  simp only [zeroFlag, div_eq_zero_iff, hb, or_false]

/-- The code's self-score is one. -/
theorem characterScore_self (χ : MulChar (ZMod p) ℂ) (a : ZMod p) :
    characterScore χ a a = 1 := by
  simpa only [characterScore, pow_two] using character_sq χ a

/-- Different classes have strictly smaller score for an injective character.
The proof uses the squared distance of the two unit codes. -/
theorem characterScore_lt_one (χ : MulChar (ZMod p) ℂ) (hχ : Function.Injective χ)
    {a q : ZMod p} (haq : a ≠ q) : characterScore χ a q < 1 := by
  have ha := character_sq χ a
  have hq := character_sq χ q
  by_contra h
  have hle : 1 ≤ characterScore χ a q := le_of_not_gt h
  unfold characterScore at hle
  have hr := sq_nonneg ((χ a).re - (χ q).re)
  have hi := sq_nonneg ((χ a).im - (χ q).im)
  have hz := sq_nonneg (zeroFlag a - zeroFlag q)
  have he : (χ a).re = (χ q).re := by nlinarith
  have hf : (χ a).im = (χ q).im := by nlinarith
  exact haq (hχ (Complex.ext he hf))

/-- Every prime, including two, has the required injective character. -/
theorem exists_division_code : ∃ χ : MulChar (ZMod p) ℂ,
    Function.Injective χ ∧ ∀ a q : ZMod p, a ≠ q → characterScore χ a q < 1 := by
  obtain ⟨χ, hχ⟩ := exists_injective_mulChar (p := p)
  exact ⟨χ, hχ, fun _ _ h => characterScore_lt_one χ hχ h⟩

example := character_sq (p := 3) 1 0
example := character_sq (p := 3) 1 1
example := character_div (p := 3) 1 2 2
example := zeroFlag_div (p := 3) 0 (b := 2) (by decide)
example := characterScore_self (p := 3) 1 0
example := exists_division_code (p := 2)

example : ∃ χ : MulChar (ZMod 3) ℂ, Function.Injective χ ∧ characterScore χ 1 2 < 1 := by
  obtain ⟨χ, hχ⟩ := exists_injective_mulChar (p := 3)
  exact ⟨χ, hχ, characterScore_lt_one χ hχ (by decide)⟩

/-- The actual shared embedding, with four active pairs and optional padding. -/
noncomputable def embedding (χ : MulChar (ZMod p) ℂ) (a : ZMod p) :
    Channels (PairIndex K) → ℝ :=
  pack ![![1, 0], ![(χ a).re, 0], ![(χ a).im, 0], ![zeroFlag a, 0]]

/-- The raw rotated sum, before its first RMSNorm. -/
noncomputable def raw (χ : MulChar (ZMod p) ℂ) (a b : ZMod p) :
    Channels (PairIndex K) → ℝ :=
  pack ![![1, 1], ![(χ a).re, (χ b).re], ![(χ a).im, (χ b).im],
    ![zeroFlag a, zeroFlag b]]

/-- The target branch vector. `Products` will derive it from the actual fixed
SwiGLU matrices, without placing division in the forward definition. -/
noncomputable def branchVector (χ : MulChar (ZMod p) ℂ) (a b : ZMod p) :
    Channels (PairIndex K) → ℝ :=
  pack ![![0, 0], ![(χ (a / b)).re - (χ a).re, (χ (a / b)).re + (χ a).re],
    ![(χ (a / b)).im - (χ a).im, (χ (a / b)).im + (χ a).im], ![0, 2 * zeroFlag a]]

omit [Fintype K] in
/-- The actual fixed quarter-turn gives the asserted raw coordinates. -/
theorem tiedInput_embedding (χ : MulChar (ZMod p) ℂ) (a b : ZMod p) :
    tiedInput (embedding (K := K) χ) a b = raw χ a b := by
  funext ⟨k, j⟩
  rcases k with k | k
  · fin_cases k <;> fin_cases j <;> simp [tiedInput, embedding, rotate90, raw, pack]
  · fin_cases j <;> simp [tiedInput, embedding, rotate90, raw, pack]

/-- The original rotated input has squared length four, including zero inputs. -/
theorem raw_sq (χ : MulChar (ZMod p) ℂ) (a b : ZMod p) :
    ∑ i, (raw (K := K) χ a b i) ^ 2 = 4 := by
  rw [raw, sum_sq_pack]
  norm_num [Fin.sum_univ_succ]
  have ha := character_sq χ a
  have hb := character_sq χ b
  nlinarith

/-- The odd branch coordinates make its squared length exactly four too. -/
theorem branchVector_sq (χ : MulChar (ZMod p) ℂ) (a : ZMod p) {b : ZMod p}
    (hb : b ≠ 0) : ∑ i, (branchVector (K := K) χ a b i) ^ 2 = 4 := by
  rw [branchVector, sum_sq_pack]
  norm_num [Fin.sum_univ_succ]
  have ha := character_sq χ a
  have hc := character_sq χ (a / b)
  rw [zeroFlag_div a hb] at hc
  nlinarith

/-- After the cancellation, the tied readout sees the quotient code exactly.
The odd residual coordinates still contribute to the final RMSNorm. -/
theorem readout_raw_add_branchVector (χ : MulChar (ZMod p) ℂ) (a : ZMod p)
    {b : ZMod p} (hb : b ≠ 0) (q : ZMod p) :
    tiedReadout (embedding (K := K) χ)
      (fun i => raw χ a b i + branchVector χ a b i) q =
        1 + characterScore χ (a / b) q := by
  simp only [tiedReadout, Fintype.sum_prod_type, Fintype.sum_sum_type, embedding,
    raw, branchVector, pack, zero_mul, Finset.sum_const_zero, add_zero]
  norm_num [Fin.sum_univ_succ, characterScore, zeroFlag_div a hb]
  ring

example := tiedInput_embedding (K := Fin 0) (p := 3) 1 0 1
example := raw_sq (K := Fin 4) (p := 3) 1 0 2
example := branchVector_sq (K := Fin 4) (p := 3) 1 0 (b := 2) (by decide)
example := readout_raw_add_branchVector (K := Fin 0) (p := 3) 1 0 (b := 2) (by decide) 0

end Transformer.Rotate90
