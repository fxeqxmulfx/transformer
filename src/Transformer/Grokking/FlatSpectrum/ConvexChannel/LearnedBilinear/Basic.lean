module

public import Transformer.Grokking.FlatSpectrum.Factorization
public import Transformer.Grokking.FlatSpectrum.OneCell
public import Transformer.Grokking.FlatSpectrum.OnesCertificate
public import Mathlib.Analysis.Matrix.Normed

/-!
# Exact learned rectangular factors and their complete complex interaction

Source: `review/convex/moments.md`, CM86, using Lemma 1 of the same manuscript.
A rectangular joint matrix recovers two learned representations by a balanced
spectral factorization through its row dimension. The learned nuclear-norm
readout multiplies their actual complex interaction. The joint bilinear identity
is proved, including zero and rank-deficient matrices. This does not preserve
a smaller fixed-rank bottleneck or provide a native spectral implementation.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear

open scoped BigOperators Matrix ComplexOrder Matrix.Norms.Elementwise

variable {Q D H : Type*} [Fintype Q] [Fintype D] [DecidableEq Q] [DecidableEq D]

/-- The eigenbasis of M M*, using the query-sized square matrix, CM86. -/
noncomputable def basis (M : Matrix Q D ℂ) : Matrix Q Q ℂ :=
  (Matrix.isHermitian_conjTranspose_mul_self M.conjTranspose).eigenvectorUnitary

/-- The complete query-side singular-value list, including its zeros, CM86. -/
noncomputable def values (M : Matrix Q D ℂ) : Q → ℝ := singularValues M.conjTranspose

/-- Learned query factor of the exact balanced decomposition, CM86. -/
noncomputable def leftFactor (M : Matrix Q D ℂ) : Matrix Q Q ℂ :=
  basis M * Matrix.diagonal (fun q => (Real.sqrt (values M q) : ℂ))

/-- Learned context factor, keeping every input coordinate and singular
direction, CM86. Inverse zero is zero, as in the exact spectral proof. -/
noncomputable def rightFactor (M : Matrix Q D ℂ) : Matrix D Q ℂ :=
  M.conjTranspose * basis M * Matrix.diagonal (fun q => (((Real.sqrt (values M q))⁻¹ : ℝ) : ℂ))

omit [DecidableEq D] in
/-- Both actual recovered factors multiply to the entire joint matrix,
CM86. Their width is |Q|, regardless of a possibly larger context dimension. -/
theorem factorization (M : Matrix Q D ℂ) :
    leftFactor M * (rightFactor M).conjTranspose = M := by
  have h := (factor_of_diagonalization (X := M.conjTranspose)
    (fun q => Real.sqrt_nonneg _) (eigenvectorUnitary_conjTranspose_mul _)
    (gram_diagonalization M.conjTranspose)).1
  change M.conjTranspose = rightFactor M * (leftFactor M).conjTranspose at h
  have ht := congrArg Matrix.conjTranspose h
  simpa only [Matrix.conjTranspose_mul,
    Matrix.conjTranspose_conjTranspose] using ht.symm

/-- The two learned factors have the same squared Frobenius norm, equal
to the joint nuclear norm, CM86. This is Lemma 1's exact balanced cost. -/
theorem factor_cost (M : Matrix Q D ℂ) :
    frobeniusSq (leftFactor M) = nuclearNorm M ∧
      frobeniusSq (rightFactor M) = nuclearNorm M := by
  have h := (factor_of_diagonalization (X := M.conjTranspose)
    (fun q => Real.sqrt_nonneg _) (eigenvectorUnitary_conjTranspose_mul _)
    (gram_diagonalization M.conjTranspose)).2
  have hn : (∑ q, values M q) = nuclearNorm M := by
    change nuclearNorm M.conjTranspose = nuclearNorm M
    exact nuclearNorm_conjTranspose M
  exact ⟨h.2.trans hn, h.1.trans hn⟩

/-- The learned scalar readout, with a fixed complex phase supplied later,
CM86. Both representations are normalized by its square root. -/
noncomputable def readout (M : Matrix Q D ℂ) : ℝ := nuclearNorm M

/-- Actual learned query representation, CM86. -/
noncomputable def leftRepresentation (M : Matrix Q D ℂ) : Matrix Q Q ℂ :=
  (Real.sqrt (readout M))⁻¹ • leftFactor M

/-- Actual learned context representation, CM86. -/
noncomputable def rightRepresentation (M : Matrix Q D ℂ) : Matrix D Q ℂ :=
  (Real.sqrt (readout M))⁻¹ • rightFactor M

variable [Fintype H]

/-- Actual interaction after both learned linear encoders, CM86. -/
noncomputable def interaction (L : Matrix Q H ℂ) (R : Matrix D H ℂ)
    (x : Q → ℂ) (y : D → ℂ) : ℂ :=
  star (L.conjTranspose *ᵥ x) ⬝ᵥ (R.conjTranspose *ᵥ y)

omit [DecidableEq Q] [DecidableEq D] in
/-- The physical two-encoder computation equals its complete product,
CM86. This is valid at arbitrary physical hidden width and complex inputs. -/
theorem interaction_product (L : Matrix Q H ℂ) (R : Matrix D H ℂ)
    (x : Q → ℂ) (y : D → ℂ) :
    interaction L R x y = star x ⬝ᵥ ((L * R.conjTranspose) *ᵥ y) := by
  unfold interaction
  calc
    _ = (star x ᵥ* L) ⬝ᵥ (R.conjTranspose *ᵥ y) := by
      congr 1
      symm
      simpa only [Matrix.conjTranspose_conjTranspose, star_star] using
        Matrix.vecMul_conjTranspose L.conjTranspose (star x)
    _ = star x ⬝ᵥ (L *ᵥ (R.conjTranspose *ᵥ y)) :=
      (Matrix.dotProduct_mulVec _ _ _).symm
    _ = _ := by rw [Matrix.mulVec_mulVec]

omit [DecidableEq Q] [DecidableEq D] in
/-- Rescaling both learned encoders gives their actual degree-two scale,
CM86. The scalar is real, so its conjugate is unchanged. -/
theorem interaction_smul (L : Matrix Q H ℂ) (R : Matrix D H ℂ)
    (x : Q → ℂ) (y : D → ℂ) (a : ℝ) :
    interaction (a • L) (a • R) x y = (a ^ 2 : ℝ) * interaction L R x y := by
  simp only [interaction, Matrix.conjTranspose_smul, star_trivial, Matrix.smul_mulVec,
    star_smul, smul_dotProduct, dotProduct_smul, Complex.real_smul, Complex.ofReal_pow]
  ring

/-- Actual trainable encoders, interaction and complex readout, CM86. -/
noncomputable def forward (M : Matrix Q D ℂ) (x : Q → ℂ) (y : D → ℂ) : ℂ :=
  (readout M : ℂ) * (1 + Complex.I) *
    interaction (leftRepresentation M) (rightRepresentation M) x y

/-- Exact equivalence of the learned path to its joint rectangular reading,
CM86. No internal label, factor-free inference definition, or rank assumption. -/
theorem forward_eq (M : Matrix Q D ℂ) (x : Q → ℂ) (y : D → ℂ) :
    forward M x y = (1 + Complex.I) * (star x ⬝ᵥ (M *ᵥ y)) := by
  rw [forward, leftRepresentation, rightRepresentation, interaction_smul,
    interaction_product, factorization]
  by_cases hz : readout M = 0
  · have hM : M = 0 := eq_zero_of_nuclearNorm_eq_zero hz
    rw [hz, hM]
    simp
  · have hn : 0 ≤ readout M := nuclearNorm_nonneg M
    have hc : (readout M : ℂ) * (((Real.sqrt (readout M))⁻¹ ^ 2 : ℝ) : ℂ) = 1 := by
      rw [← Complex.ofReal_mul]
      norm_cast
      rw [inv_pow, Real.sq_sqrt hn, mul_inv_cancel₀ hz]
    calc
      _ = ((readout M : ℂ) * (((Real.sqrt (readout M))⁻¹ ^ 2 : ℝ) : ℂ)) *
          ((1 + Complex.I) * (star x ⬝ᵥ (M *ᵥ y))) := by ring
      _ = _ := by rw [hc, one_mul]

/-- Every finite physical two-encoder interaction is exactly covered,
CM86. Recovery may widen a smaller imposed physical bottleneck. -/
theorem physical_coverage (L : Matrix Q H ℂ) (R : Matrix D H ℂ)
    (x : Q → ℂ) (y : D → ℂ) :
    forward (L * R.conjTranspose) x y = (1 + Complex.I) * interaction L R x y := by
  rw [forward_eq, interaction_product]

end Transformer.Grokking.FlatSpectrum.ConvexChannel.LearnedBilinear
