module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.ComplexHeadBudget
public import Transformer.Grokking.FlatSpectrum.AddReadoutSpectrum
public import Mathlib.Data.Matrix.Block

/-!
# Flat categorical codes for arbitrary state counts and query blocks

Source: `review/flat_spectrum.md`, CFt5. Normalize the existing character
Fourier matrix, prove both Gram identities and use finite block diagonals for
the native per-query state dictionaries. Flatness describes this fixed code
basis. The initialized or trained head and empirical task need not be flat.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth

open scoped BigOperators
open Matrix

/-- Native categorical state code, CFt5: the Fourier basis normalized by the
square root of its state count. State indices carry no target arithmetic. -/
noncomputable def categoricalFourier (N : ℕ) [NeZero N] : Matrix (ZMod N) (ZMod N) ℂ :=
  (((Real.sqrt (N : ℝ))⁻¹ : ℝ) : ℂ) • fourierMatrix N

/-- Every nonempty categorical Fourier code is exactly unitary, CFt5.
This derives the native SVD target from the existing character identities. -/
theorem categoricalFourier_unitary (N : ℕ) [NeZero N] :
    categoricalFourier N ∈ Matrix.unitaryGroup (ZMod N) ℂ := by
  have hn : 0 < (N : ℝ) := Nat.cast_pos.mpr (NeZero.pos N)
  have hs : Real.sqrt (N : ℝ) ≠ 0 := ne_of_gt (Real.sqrt_pos.mpr hn)
  have hr : (Real.sqrt (N : ℝ))⁻¹ * (Real.sqrt (N : ℝ))⁻¹ * (N : ℝ) = 1 := by
    field_simp [hs]
    nlinarith [Real.sq_sqrt hn.le]
  have hc : (((Real.sqrt (N : ℝ))⁻¹ : ℝ) : ℂ) *
      (((Real.sqrt (N : ℝ))⁻¹ : ℝ) : ℂ) * (N : ℂ) = 1 := by
    exact_mod_cast hr
  constructor
  · change (categoricalFourier N).conjTranspose * categoricalFourier N = 1
    simp only [categoricalFourier, Matrix.conjTranspose_smul, Matrix.smul_mul,
      Matrix.mul_smul, Complex.star_def, Complex.conj_ofReal, smul_smul]
    rw [conjTranspose_fourierMatrix_mul, smul_smul, hc, one_smul]
  · change categoricalFourier N * (categoricalFourier N).conjTranspose = 1
    simp only [categoricalFourier, Matrix.conjTranspose_smul, Matrix.smul_mul,
      Matrix.mul_smul, Complex.star_def, Complex.conj_ofReal, smul_smul]
    rw [fourierMatrix_mul_conjTranspose, smul_smul, hc, one_smul]

/-- The existing flat-spectrum predicate holds at squared singular value
one for the actual normalized basis, CFt5. -/
theorem categoricalFourier_isFlat (N : ℕ) [NeZero N] : IsFlat (categoricalFourier N) 1 :=
  unitary_isFlat _ (categoricalFourier_unitary N)

/-- Every actual categorical-code singular value equals one, CFt5. -/
theorem categoricalFourier_singularValues (N : ℕ) [NeZero N] (i : ZMod N) :
    singularValues (categoricalFourier N) i = 1 :=
  unitary_singularValues _ (categoricalFourier_unitary N) i

/-- Exact code-entry magnitudes match native normalized phases, CFt5.
Only the categorical state count controls this feature scale. -/
theorem categoricalFourier_entry_norm (N : ℕ) [NeZero N] (s k : ZMod N) :
    ‖categoricalFourier N s k‖ = (Real.sqrt (N : ℝ))⁻¹ := by
  simp [categoricalFourier, fourierMatrix, Matrix.smul_apply, smul_eq_mul,
    ZMod.stdAddChar_apply]

/-- Exact nuclear norm audited by the native basis SVD, CFt5. -/
theorem categoricalFourier_nuclear (N : ℕ) [NeZero N] :
    nuclearNorm (categoricalFourier N) = (N : ℝ) := by
  simp [nuclearNorm, categoricalFourier_singularValues]

/-- The fixed code's squared Frobenius norm equals its dimension, CFt5. -/
theorem categoricalFourier_frobenius (N : ℕ) [NeZero N] :
    frobeniusSq (categoricalFourier N) = (N : ℝ) := by
  have hg : (categoricalFourier N).conjTranspose * categoricalFourier N = 1 :=
    Unitary.star_mul_self_of_mem (categoricalFourier_unitary N)
  rw [frobeniusSq_eq_re_trace, hg]
  simp

variable {Q : Type*} [Fintype Q] [DecidableEq Q]
  {J : Q → Type*} [∀ q, Fintype (J q)] [∀ q, DecidableEq (J q)]

/-- Query dictionaries may have different finite state counts. Their block
diagonal code is still unitary, with no cross-query interaction (CFt5). -/
theorem categorical_blocks_unitary (B : ∀ q, Matrix (J q) (J q) ℂ)
    (hB : ∀ q, B q ∈ Matrix.unitaryGroup (J q) ℂ) :
    Matrix.blockDiagonal' B ∈ Matrix.unitaryGroup (Sigma J) ℂ := by
  have hleft : (fun q => (B q).conjTranspose * B q) = (1 : ∀ q, Matrix (J q) (J q) ℂ) := by
    funext q
    exact Unitary.star_mul_self_of_mem (hB q)
  have hright : (fun q => B q * (B q).conjTranspose) = (1 : ∀ q, Matrix (J q) (J q) ℂ) := by
    funext q
    exact Unitary.mul_star_self_of_mem (hB q)
  constructor
  · change (Matrix.blockDiagonal' B).conjTranspose * Matrix.blockDiagonal' B = 1
    rw [Matrix.blockDiagonal'_conjTranspose, ← Matrix.blockDiagonal'_mul, hleft,
      Matrix.blockDiagonal'_one]
  · change Matrix.blockDiagonal' B * (Matrix.blockDiagonal' B).conjTranspose = 1
    rw [Matrix.blockDiagonal'_conjTranspose, ← Matrix.blockDiagonal'_mul, hright,
      Matrix.blockDiagonal'_one]

/-- The actual family of independently sized categorical Fourier query
blocks, CFt5. Features are zero outside the active query's block. -/
noncomputable def categoricalFourierFamily (n : Q → ℕ) [∀ q, NeZero (n q)] :
    Matrix (Σ q, ZMod (n q)) (Σ q, ZMod (n q)) ℂ :=
  Matrix.blockDiagonal' (fun q => categoricalFourier (n q))

/-- The full query-block code has unitary Gram matrix for arbitrary nonzero
state counts; no target function enters its construction (CFt5). -/
theorem categoricalFourierFamily_unitary (n : Q → ℕ) [∀ q, NeZero (n q)] :
    categoricalFourierFamily n ∈ Matrix.unitaryGroup (Σ q, ZMod (n q)) ℂ :=
  categorical_blocks_unitary _ (fun q => categoricalFourier_unitary (n q))

/-- The entire block code, as well as each component, has unit singular
values. This is the fixed basis to which the flat capacity bound applies. -/
theorem categoricalFourierFamily_singularValues (n : Q → ℕ) [∀ q, NeZero (n q)]
    (i : Σ q, ZMod (n q)) : singularValues (categoricalFourierFamily n) i = 1 :=
  unitary_singularValues _ (categoricalFourierFamily_unitary n) i

/-- Native margin24 fits radius64 times the total actual categorical state
count, including each query's unknown state, CFt5. -/
theorem categoricalFourierFamily_native_budget {C : Type*} [Fintype C] [DecidableEq C]
    (n : Q → ℕ) [∀ q, NeZero (n q)] (labels : (Σ q, ZMod (n q)) → C) :
    complexCoefficients (complexInitializer (categoricalFourierFamily n) labels 24) ∈
      headBudget (64 * (∑ q, (n q : ℝ))) := by
  have h := complexInitializer_native_budget _ (categoricalFourierFamily_unitary n) labels
  simpa only [Fintype.card_sigma, ZMod.card, Nat.cast_sum] using h

example := categoricalFourier_unitary 3

example := categoricalFourier_isFlat 3

example := categoricalFourier_singularValues 3 1

example := categorical_blocks_unitary (fun _ : Bool => fourier4) (fun _ => fourier4_unitary)

example := categoricalFourierFamily_unitary (fun _ : Bool => 3)

example := categoricalFourierFamily_native_budget (fun _ : Bool => 3) (fun _ => true)

example : nuclearNorm (categoricalFourier 5) = 5 := categoricalFourier_nuclear 5

example : frobeniusSq (categoricalFourier 5) = 5 := categoricalFourier_frobenius 5

example : ‖categoricalFourier 4 1 2‖ = (Real.sqrt 4)⁻¹ :=
  categoricalFourier_entry_norm 4 1 2

private def witnessStateCounts (q : Bool) : ℕ := if q then 3 else 5

private instance (q : Bool) : NeZero (witnessStateCounts q) := by
  cases q <;> exact ⟨by decide⟩

example := categoricalFourierFamily_unitary witnessStateCounts

example := categoricalFourierFamily_singularValues witnessStateCounts ⟨true, 1⟩

example := categoricalFourierFamily_native_budget witnessStateCounts (fun s => s.1)

end Transformer.Grokking.FlatSpectrum.ConvexDepth
