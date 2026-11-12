module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.Stability
public import Transformer.Grokking.FlatSpectrum.FlatKernel
public import Mathlib.Analysis.CStarAlgebra.Matrix
public import Mathlib.Analysis.InnerProductSpace.Adjoint

/-!
# A four-stream flat-spectrum complex transport

Source: `review/convex_depth.md`, CD3. A normalized four-point Fourier matrix replaces
the stochastic skip map. Phase conjugation follows DiffGPT `fe1d528`,
`src/native_transformer/src/phase_mhc.rs`, and its `ComplexPhaseRouting.lean` proof.
This is a new connection: complex coefficients may cancel, and the fixed Fourier map
is unitary rather than doubly stochastic. Phases and transport remain fixed in CD1.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth

open scoped BigOperators
open ComplexConjugate

/-- Normalized four-point Fourier transport (CD3); all streams participate. -/
noncomputable def fourier4 : Matrix (Fin 4) (Fin 4) ℂ :=
  (1 / 2 : ℂ) • !![1, 1, 1, 1; 1, Complex.I, -1, -Complex.I;
    1, -1, 1, -1; 1, -Complex.I, -1, Complex.I]

/-- The Fourier skip has Gram identity, hence unit singular values (CD3). -/
theorem fourier4_unitary : fourier4 ∈ Matrix.unitaryGroup (Fin 4) ℂ := by
  constructor <;> ext i j <;> fin_cases i <;> fin_cases j <;>
    norm_num [fourier4, Matrix.mul_apply, Fin.sum_univ_succ, Matrix.star_apply,
      Matrix.one_apply, Complex.ext_iff]

/-- Flat entry magnitudes give nontrivial mixing of every pair of streams (CD3). -/
theorem fourier4_entry_norm (i j : Fin 4) : ‖fourier4 i j‖ = 1 / 2 := by
  fin_cases i <;> fin_cases j <;> norm_num [fourier4, norm_mul]

/-- The exact real phase formula used by DiffGPT `fe1d528` (CD3). The native
kernel uses a rescaled evaluation of the same expression to avoid overflow. -/
noncomputable def phase (a : ℝ) : ℂ :=
  ⟨1 / Real.sqrt (1 + a ^ 2), a / Real.sqrt (1 + a ^ 2)⟩

/-- The DiffGPT phase has unit modulus. Proof adapted with attribution from
`research/main_model/proofs/ComplexPhaseRouting.lean` at `fe1d528` (CD3). -/
theorem phase_norm (a : ℝ) : ‖phase a‖ = 1 := by
  have hp : 0 < 1 + a ^ 2 := by positivity
  have hs := Real.sq_sqrt hp.le
  have hn : Real.sqrt (1 + a ^ 2) ≠ 0 := ne_of_gt (Real.sqrt_pos.2 hp)
  have hsq : Complex.normSq (phase a) = 1 := by
    dsimp [phase]
    field_simp [hn]
    nlinarith
  rw [Complex.norm_def, hsq, Real.sqrt_one]

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- A diagonal unit-phase matrix is itself unitary (CD3). -/
theorem phase_diagonal_unitary (r : ι → ℂ) (hr : ∀ i, ‖r i‖ = 1) :
    Matrix.diagonal r ∈ Matrix.unitaryGroup ι ℂ := by
  constructor <;> simp only [Matrix.star_eq_conjTranspose, Matrix.diagonal_conjTranspose,
    Matrix.diagonal_mul_diagonal]
  · have he : (fun i => star (r i) * r i) = (fun _ => (1 : ℂ)) := by
      funext i
      simp [Complex.conj_mul', hr]
    rw [show (fun i => star r i * r i) = (fun _ => (1 : ℂ)) from he]
    exact Matrix.diagonal_one
  · have he : (fun i => r i * star (r i)) = (fun _ => (1 : ℂ)) := by
      funext i
      simp [Complex.mul_conj', hr]
    rw [show (fun i => r i * star r i) = (fun _ => (1 : ℂ)) from he]
    exact Matrix.diagonal_one

/-- The same `D U D*` structure as phase-mHC, with a fixed unitary base U (CD3). -/
noncomputable def phaseConjugate (U : Matrix ι ι ℂ) (r : ι → ℂ) : Matrix ι ι ℂ :=
  Matrix.diagonal r * U * star (Matrix.diagonal r)

/-- Unlike stochastic averaging, the new skip preserves the entire spectrum at
one under phase conjugation (CD3), not merely an upper norm bound. -/
theorem phaseConjugate_unitary (U : Matrix ι ι ℂ) (hU : U ∈ Matrix.unitaryGroup ι ℂ)
    (r : ι → ℂ) (hr : ∀ i, ‖r i‖ = 1) :
    phaseConjugate U r ∈ Matrix.unitaryGroup ι ℂ := by
  have hd := phase_diagonal_unitary r hr
  exact (unitary _).mul_mem ((unitary _).mul_mem hd hU) (Unitary.star_mem hd)

/-- The skip satisfies the existing flat-kernel predicate at squared singular value
one (CD3), directly linking the new connection to the repository's spectrum theory. -/
theorem unitary_isFlat (U : Matrix ι ι ℂ) (hU : U ∈ Matrix.unitaryGroup ι ℂ) :
    IsFlat U 1 := by
  left
  simpa only [Matrix.star_eq_conjTranspose, Complex.ofReal_one, one_smul] using
    Unitary.star_mul_self_of_mem hU

/-- Every singular value, in the existing spectral definition, is exactly one
(CD3). The theorem concerns the fixed transport, not the learned injections. -/
theorem unitary_singularValues (U : Matrix ι ι ℂ) (hU : U ∈ Matrix.unitaryGroup ι ℂ)
    (i : ι) : singularValues U i = 1 := by
  have hg : U.conjTranspose * U = (1 : ℂ) • 1 := by
    simpa only [Matrix.star_eq_conjTranspose, one_smul] using Unitary.star_mul_self_of_mem hU
  simpa using singularValues_eq_of_conjTranspose_mul_self (c := 1) hg i

/-- Complex stream mixing viewed as a real-linear transport for the jointly real
convex objective; real and imaginary components can both be trained in injections. -/
noncomputable def transport (U : Matrix ι ι ℂ) :
    EuclideanSpace ℂ ι →L[ℝ] EuclideanSpace ℂ ι :=
  (Matrix.toEuclideanCLM (𝕜 := ℂ) (n := ι) U).restrictScalars ℝ

/-- The concrete transport preserves the L2 norm of the actual complex state (CD3). -/
theorem transport_norm (U : Matrix ι ι ℂ) (hU : U ∈ Matrix.unitaryGroup ι ℂ)
    (x : EuclideanSpace ℂ ι) : ‖transport U x‖ = ‖x‖ := by
  have hm : Matrix.toEuclideanCLM (𝕜 := ℂ) (n := ι) U ∈
      unitary (EuclideanSpace ℂ ι →L[ℂ] EuclideanSpace ℂ ι) := by
    constructor
    · simpa only [map_mul, map_star, map_one] using
        congrArg (Matrix.toEuclideanCLM (𝕜 := ℂ) (n := ι)) (Unitary.star_mul_self_of_mem hU)
    · simpa only [map_mul, map_star, map_one] using
        congrArg (Matrix.toEuclideanCLM (𝕜 := ℂ) (n := ι)) (Unitary.mul_star_self_of_mem hU)
  exact ContinuousLinearMap.norm_map_of_mem_unitary hm x

/-- Three fixed coordinates and the native stream-zero phase gauge (CD3). -/
noncomputable def phaseVector (b : Fin 3 → ℝ) : Fin 4 → ℂ :=
  Fin.cases 1 (fun i => phase (b i))

/-- Every coordinate of the actual three-parameter phase vector is unit modulus (CD3). -/
theorem phaseVector_norm (b : Fin 3 → ℝ) (i : Fin 4) : ‖phaseVector b i‖ = 1 := by
  refine Fin.cases ?_ ?_ i
  · norm_num [phaseVector]
  · intro j
    exact phase_norm (b j)

/-- The proposed four-stream connection satisfies the stability premise of CD2
at arbitrary, fixed DiffGPT-form phases (CD3). -/
theorem fourier_phase_transport_norm (b : Fin 3 → ℝ) (x : EuclideanSpace ℂ (Fin 4)) :
    ‖transport (phaseConjugate fourier4 (phaseVector b)) x‖ = ‖x‖ := by
  apply transport_norm
  exact phaseConjugate_unitary _ fourier4_unitary _ (phaseVector_norm b)

example := phase_diagonal_unitary (fun i : Fin 4 => phase (i : ℝ))
  (fun i => phase_norm (i : ℝ))

example := phaseConjugate_unitary fourier4 fourier4_unitary
  (fun i : Fin 4 => phase (i : ℝ)) (fun i => phase_norm (i : ℝ))

example := transport_norm fourier4 fourier4_unitary
  (WithLp.toLp 2 (fun _ : Fin 4 => (1 : ℂ)))

example := unitary_isFlat fourier4 fourier4_unitary
example (i : Fin 4) := unitary_singularValues fourier4 fourier4_unitary i

end Transformer.Grokking.FlatSpectrum.ConvexDepth
