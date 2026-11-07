module

public import Transformer.Grokking.FlatSpectrum.FeatureCodes.FiberCodes
public import Transformer.Grokking.FlatSpectrum.AddReadoutSpectrum
public import Mathlib.Data.ZMod.Basic
public import Mathlib.Tactic.NormNum.Prime

/-!
# The cubic task only needs to distinguish answer-equivalence classes

Source: `review/flat_spectrum.md`, *Answer equivalence*, QE4.
In ZMod 97, `(s-t)(s-35t)(s-61t)=s^3-t^3`. Thus the three multiples of a
nonzero sum have the same cube. Isometric feature columns for sums 1 and 35
are different but the answer-fiber readout gives identical complete score
vectors. The result pinpoints information that full additive-character recovery
would unnecessarily require. It does not identify the learned representation,
construct tied source weights, or establish optimizer convergence.
-/

@[expose] public section
noncomputable section

namespace Transformer.Grokking.FlatSpectrum.FeatureCodes

open Matrix

private instance : Fact (Nat.Prime 97) := ⟨by norm_num⟩

/-- The cubic roots-of-unity factorization at the experiment's modulus, QE4. -/
theorem cubic97_factor (s t : ZMod 97) :
    (s - t) * (s - 35 * t) * (s - 61 * t) = s ^ 3 - t ^ 3 := by
  ring_nf
  rw [show (2231 : ZMod 97) = 0 by decide, show (97 : ZMod 97) = 0 by decide,
    show (2135 : ZMod 97) = 1 by decide]
  ring

/-- **Exact answer equivalence, including sum zero**, QE4. -/
theorem cubic97_eq_iff (s t : ZMod 97) :
    s ^ 3 = t ^ 3 ↔ s = t ∨ s = 35 * t ∨ s = 61 * t := by
  rw [← sub_eq_zero, ← cubic97_factor, mul_eq_zero, mul_eq_zero]
  simp only [sub_eq_zero]
  tauto

/-- The same equivalence for sums of the original raw operands, QE4. -/
theorem cubic97_answers_eq_iff (a b c d : ZMod 97) :
    (a + b) ^ 3 = (c + d) ^ 3 ↔
      a + b = c + d ∨ a + b = 35 * (c + d) ∨ a + b = 61 * (c + d) :=
  cubic97_eq_iff (a + b) (c + d)

/-- **Different full features, identical cubic classification scores**, QE4.
This holds for every isometric basis, so it does not depend on picking a
particular Fourier coordinate convention. -/
theorem cubic97_distinct_features_same_readouts {F : Matrix (ZMod 97) (ZMod 97) ℂ}
    (hF : Fᴴ * F = 1) :
    F *ᵥ Pi.single 1 1 ≠ F *ᵥ Pi.single 35 1 ∧
    ∀ a : ZMod 97,
      fiberReadout F (fun s => s ^ 3) a (F *ᵥ Pi.single 1 1) =
      fiberReadout F (fun s => s ^ 3) a (F *ᵥ Pi.single 35 1) := by
  refine ⟨feature_columns_ne hF (by decide), ?_⟩
  exact fiber_readouts_eq_of_same_answer hF (fun s : ZMod 97 => s ^ 3)
    (s := 1) (t := 35) (by decide)

/-- The actual additive-character basis, normalized to be an isometry, QE4. -/
def cubicFourier97 : Matrix (ZMod 97) (ZMod 97) ℂ :=
  (((Real.sqrt 97)⁻¹ : ℝ) : ℂ) • fourierMatrix 97

/-- Normalizing the additive Fourier basis gives the required isometry, QE4. -/
theorem cubicFourier97_isometry : cubicFourier97ᴴ * cubicFourier97 = 1 := by
  have hs : (Real.sqrt 97) ^ 2 = 97 := Real.sq_sqrt (by norm_num)
  have hn : Real.sqrt 97 ≠ 0 := by positivity
  have hr : (Real.sqrt 97)⁻¹ * (Real.sqrt 97)⁻¹ * 97 = 1 := by
    field_simp
    nlinarith
  have hc : (((Real.sqrt 97)⁻¹ : ℝ) : ℂ) *
      (((Real.sqrt 97)⁻¹ : ℝ) : ℂ) * 97 = 1 := by exact_mod_cast hr
  rw [cubicFourier97, conjTranspose_smul, Matrix.smul_mul, Matrix.mul_smul,
    conjTranspose_fourierMatrix_mul, smul_smul, smul_smul]
  simp only [Complex.star_def, Complex.conj_ofReal]
  norm_num only [Nat.cast_ofNat]
  rw [hc, one_smul]

/-- Full additive-character features can differ while all cubic answer scores
agree, in the concrete Fourier basis of the spectral theory, QE4. -/
theorem cubic97_fourier_features_same_readouts :
    cubicFourier97 *ᵥ Pi.single 1 1 ≠ cubicFourier97 *ᵥ Pi.single 35 1 ∧
    ∀ a : ZMod 97,
      fiberReadout cubicFourier97 (fun s => s ^ 3) a (cubicFourier97 *ᵥ Pi.single 1 1) =
      fiberReadout cubicFourier97 (fun s => s ^ 3) a (cubicFourier97 *ᵥ Pi.single 35 1) :=
  cubic97_distinct_features_same_readouts cubicFourier97_isometry

example := cubic97_factor 1 35
example := cubic97_eq_iff 0 0
example := cubic97_answers_eq_iff 1 0 35 0
example := cubic97_distinct_features_same_readouts
  (F := (1 : Matrix (ZMod 97) (ZMod 97) ℂ)) (by simp)

end Transformer.Grokking.FlatSpectrum.FeatureCodes
