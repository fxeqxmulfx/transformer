module

public import Mathlib.Analysis.Matrix.HermitianFunctionalCalculus
public import Mathlib.Analysis.Matrix.PosDef
public import Mathlib.Analysis.SpecialFunctions.Complex.CircleAddChar

/-!
# Nuclear norm and task kernels: definitions

Source: `review/flat_spectrum.md`, "Setting". The nuclear norm `‖X‖_S1` is the sum of
the singular values, the square roots of the eigenvalues of `Xᴴ X`; `‖X‖_S2²` is the
sum of the squared entry moduli. A task `f` with readout `φ` has the kernel
`K_φ(a, b) = φ (f a b)`; `ψ` is `ZMod.stdAddChar`, `ψ x = exp (2πi x / p)`.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

open Matrix
open scoped ComplexOrder

variable {m n : Type*} [Fintype m] [Fintype n] [DecidableEq n]

/-- Squared Frobenius norm `‖X‖_S2² = ∑ |X_ij|²`, `review/flat_spectrum.md`, Setting. -/
noncomputable def frobeniusSq (X : Matrix m n ℂ) : ℝ := ∑ i, ∑ j, ‖X i j‖ ^ 2

/-- Operator norm at most one, written `Zᴴ Z ≤ 1`: the unit ball dual to `‖·‖_S1`. -/
def IsContraction (Z : Matrix m n ℂ) : Prop := (1 - Zᴴ * Z).PosSemidef

/-- Singular values: square roots of the eigenvalues of `Xᴴ X`,
`review/flat_spectrum.md`, Setting. -/
noncomputable def singularValues (X : Matrix m n ℂ) (i : n) : ℝ :=
  √((isHermitian_conjTranspose_mul_self X).eigenvalues i)

/-- The nuclear norm `‖X‖_S1`, the sum of the singular values,
`review/flat_spectrum.md`, Setting. -/
noncomputable def nuclearNorm (X : Matrix m n ℂ) : ℝ := ∑ i, singularValues X i

/-- The effective rank `r_eff = (‖K‖_S1 / ‖K‖_S2)²`, `review/flat_spectrum.md`, Setting. -/
noncomputable def effectiveRank (X : Matrix m n ℂ) : ℝ :=
  (nuclearNorm X / √(frobeniusSq X)) ^ 2

/-- The polar factor `W = K (Kᴴ K)^{-1/2}`, `review/flat_spectrum.md`, Theorem A.
The inverse square root is the continuous functional calculus of `x ↦ (√x)⁻¹`. -/
noncomputable def polarFactor (K : Matrix m n ℂ) : Matrix m n ℂ :=
  K * cfc (fun x : ℝ => (√x)⁻¹) (Kᴴ * K)

/-- The task kernel `K_φ(a, b) = φ (f a b)`, `review/flat_spectrum.md`, Setting. -/
def taskKernel {G : Type*} (f : G → G → G) (φ : G → ℂ) : Matrix G G ℂ :=
  Matrix.of fun a b => φ (f a b)

/-- The kernel of `a + b` with readout `ψ`, `review/flat_spectrum.md`, Proposition 2. -/
noncomputable def addKernel (p : ℕ) [NeZero p] : Matrix (ZMod p) (ZMod p) ℂ :=
  taskKernel (· + ·) ZMod.stdAddChar

/-- The kernel of `a³ + a b` with readout `ψ`, `review/flat_spectrum.md`, Proposition 2. -/
noncomputable def cubicKernel (p : ℕ) [NeZero p] : Matrix (ZMod p) (ZMod p) ℂ :=
  taskKernel (fun a b => a ^ 3 + a * b) ZMod.stdAddChar

end Transformer.Grokking.FlatSpectrum
