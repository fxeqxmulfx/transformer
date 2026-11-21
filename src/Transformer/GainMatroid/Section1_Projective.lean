module

public import Mathlib.LinearAlgebra.Basis.VectorSpace
public import Mathlib.LinearAlgebra.Determinant
public import Mathlib.LinearAlgebra.Dimension.Free
public import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas
public import Mathlib.LinearAlgebra.Matrix.ToLin
public import Mathlib.LinearAlgebra.Projection

/-!
# Projective equivalence of matrices

Funk, Pivotto and Slilaty, *Matrix representations of frame and lifted-graphic matroids correspond
to gain functions* (arXiv:1609.05574v5), §1.1.

The source calls two matrices with the same labelled columns *projectively equivalent* when one
comes from the other by a sequence of elementary row and column operations. `ProjEquiv A B` is the
working form of that relation: some nonzero scaling of the columns of `A` has the row space of `B`.
The two matrices may have different row types, as canonical frame and lift matrices do.
`Section1_Elementary` defines the source's operations and proves that their
sequences are exactly `ProjEquiv`.

`projEquiv_iff_exists_mul_mul_diagonal` is the linear algebra behind Proposition 1.1.
-/

@[expose] public section

open Matrix

namespace Transformer.GainMatroid

variable {R R' R'' E F : Type*} [Field F]

/-- The row space of `A`: the span of its rows in `E → F`. -/
def rowSpace (A : Matrix R E F) : Submodule F (E → F) :=
  Submodule.span F (Set.range A.row)

/-- `A` with column `j` multiplied by the nonzero scalar `s j`. -/
def scaleCols (s : E → Fˣ) (A : Matrix R E F) : Matrix R E F :=
  Matrix.of fun i j => A i j * s j

/-- Multiplying each coordinate by a nonzero scalar, as a linear automorphism of `E → F`. -/
def scaleEquiv (s : E → Fˣ) : (E → F) ≃ₗ[F] (E → F) where
  toFun v j := v j * s j
  invFun v j := v j * ((s j)⁻¹ : Fˣ)
  map_add' v w := by ext j; simp [add_mul]
  map_smul' c v := by ext j; simp [mul_assoc]
  left_inv v := by ext j; simp
  right_inv v := by ext j; simp

theorem rowSpace_scaleCols (s : E → Fˣ) (A : Matrix R E F) :
    rowSpace (scaleCols s A) = (rowSpace A).map (scaleEquiv s).toLinearMap := by
  unfold rowSpace
  rw [Submodule.map_span, ← Set.range_comp]
  rfl

/-- **Projective equivalence** (§1.1), in row-space form: some nonzero scaling of the columns of `A`
has the row space of `B`. `elemOps_iff_projEquiv` identifies it with the source's sequences of
elementary operations. -/
def ProjEquiv (A : Matrix R E F) (B : Matrix R' E F) : Prop :=
  ∃ s : E → Fˣ, rowSpace (scaleCols s A) = rowSpace B

theorem scaleCols_one (A : Matrix R E F) : scaleCols 1 A = A := by
  ext i j
  simp [scaleCols]

theorem scaleCols_scaleCols (s t : E → Fˣ) (A : Matrix R E F) :
    scaleCols t (scaleCols s A) = scaleCols (s * t) A := by
  ext i j
  simp [scaleCols, mul_assoc]

theorem ProjEquiv.refl (A : Matrix R E F) : ProjEquiv A A :=
  ⟨1, by rw [scaleCols_one]⟩

theorem ProjEquiv.symm {A : Matrix R E F} {B : Matrix R' E F} (h : ProjEquiv A B) :
    ProjEquiv B A := by
  obtain ⟨s, hs⟩ := h
  refine ⟨s⁻¹, ?_⟩
  rw [rowSpace_scaleCols, ← hs, ← rowSpace_scaleCols, scaleCols_scaleCols, mul_inv_cancel,
    scaleCols_one]

theorem ProjEquiv.trans {A : Matrix R E F} {B : Matrix R' E F} {C : Matrix R'' E F}
    (h : ProjEquiv A B) (h' : ProjEquiv B C) : ProjEquiv A C := by
  obtain ⟨s, hs⟩ := h
  obtain ⟨t, ht⟩ := h'
  refine ⟨s * t, ?_⟩
  rw [← scaleCols_scaleCols, rowSpace_scaleCols, hs, ← rowSpace_scaleCols, ht]

/-- Two linear maps out of a finite-dimensional space with the same range differ by an automorphism
of the source. -/
theorem exists_linearEquiv_comp_eq {V W : Type*} [AddCommGroup V] [Module F V]
    [FiniteDimensional F V] [AddCommGroup W] [Module F W] (f g : V →ₗ[F] W)
    (h : LinearMap.range f = LinearMap.range g) :
    ∃ τ : V ≃ₗ[F] V, f ∘ₗ τ.toLinearMap = g := by
  obtain ⟨Cf, hf⟩ := (LinearMap.ker f).exists_isCompl
  obtain ⟨Cg, hg⟩ := (LinearMap.ker g).exists_isCompl
  let ef : Cf ≃ₗ[F] LinearMap.range f :=
    (Submodule.quotientEquivOfIsCompl _ _ hf).symm.trans f.quotKerEquivRange
  let eg : Cg ≃ₗ[F] LinearMap.range g :=
    (Submodule.quotientEquivOfIsCompl _ _ hg).symm.trans g.quotKerEquivRange
  have hef : ∀ x : Cf, (ef x : W) = f x := fun x => by
    simp [ef, LinearMap.quotKerEquivRange]
  have heg : ∀ x : Cg, (eg x : W) = g x := fun x => by
    simp [eg, LinearMap.quotKerEquivRange]
  have hk : Module.finrank F (LinearMap.ker g) = Module.finrank F (LinearMap.ker f) := by
    have h₁ := f.finrank_range_add_finrank_ker
    have h₂ := g.finrank_range_add_finrank_ker
    rw [h] at h₁
    omega
  let ek : LinearMap.ker g ≃ₗ[F] LinearMap.ker f := LinearEquiv.ofFinrankEq _ _ hk
  let σ : Cg ≃ₗ[F] Cf := eg.trans ((LinearEquiv.ofEq _ _ h.symm).trans ef.symm)
  have hσ : ∀ x : Cg, f (σ x) = g x := fun x => by
    rw [← hef, ← heg]
    simp [σ]
  refine ⟨(Submodule.prodEquivOfIsCompl _ _ hg).symm.trans
    ((ek.prodCongr σ).trans (Submodule.prodEquivOfIsCompl _ _ hf)), ?_⟩
  ext v
  obtain ⟨⟨k, c⟩, rfl⟩ := (Submodule.prodEquivOfIsCompl _ _ hg).surjective v
  have hk₀ : g k = 0 := k.2
  have hk₁ : f (ek k) = 0 := (ek k).2
  simp [hk₀, hk₁, hσ]

/-- Matrices with the same rows and the same row space differ by a nonsingular matrix on the left. -/
theorem exists_isUnit_mul_eq_of_rowSpace_eq [Fintype R] [DecidableEq R] (M N : Matrix R E F)
    (h : rowSpace M = rowSpace N) : ∃ T : Matrix R R F, IsUnit T ∧ T * M = N := by
  have hr : LinearMap.range Mᵀ.mulVecLin = LinearMap.range Nᵀ.mulVecLin := by
    simpa [range_vecMulLinear, rowSpace] using h
  obtain ⟨τ, hτ⟩ := exists_linearEquiv_comp_eq _ _ hr
  have hunit : IsUnit (LinearMap.toMatrix' τ.toLinearMap) := by
    rw [Matrix.isUnit_iff_isUnit_det, LinearMap.det_toMatrix']
    exact τ.isUnit_det'
  refine ⟨(LinearMap.toMatrix' τ.toLinearMap)ᵀ, ?_, ?_⟩
  · rw [Matrix.isUnit_iff_isUnit_det, Matrix.det_transpose]
    exact (Matrix.isUnit_iff_isUnit_det _).1 hunit
  · have hmul : Mᵀ * LinearMap.toMatrix' τ.toLinearMap = Nᵀ := by
      ext i j
      have := congrArg (fun φ => φ (Pi.single j 1) i) hτ
      simpa [Matrix.vecMul, dotProduct, LinearMap.toMatrix'_apply, Matrix.mul_apply, mul_comm,
        Pi.single_apply] using this
    have := congrArg Matrix.transpose hmul
    simpa [Matrix.transpose_mul] using this

/-- Multiplying on the left by a nonsingular matrix does not change the row space. -/
theorem rowSpace_mul_of_isUnit [Fintype R] [DecidableEq R] {T : Matrix R R F} (hT : IsUnit T)
    (M : Matrix R E F) : rowSpace (T * M) = rowSpace M := by
  have key : ∀ (U : Matrix R R F) (N : Matrix R E F), rowSpace (U * N) ≤ rowSpace N := by
    intro U N
    refine Submodule.span_le.2 ?_
    rintro _ ⟨i, rfl⟩
    have : (U * N).row i = ∑ j, U i j • N.row j := by
      ext k
      simp [Matrix.mul_apply, Matrix.row]
    rw [this]
    exact Submodule.sum_mem _ fun j _ =>
      Submodule.smul_mem _ _ (Submodule.subset_span ⟨j, rfl⟩)
  refine le_antisymm (key T M) ?_
  obtain ⟨U, rfl⟩ := hT
  have : M = (U⁻¹ : (Matrix R R F)ˣ).1 * ((U : Matrix R R F) * M) := by
    rw [← Matrix.mul_assoc, Units.inv_mul, Matrix.one_mul]
  conv_lhs => rw [this]
  exact key _ _

/-- `A` scaled by `s` is `A` times the diagonal matrix of `s`. -/
theorem scaleCols_eq_mul_diagonal [Fintype E] [DecidableEq E] (s : E → Fˣ) (A : Matrix R E F) :
    scaleCols s A = A * Matrix.diagonal fun j => (s j : F) := by
  ext i j
  simp [scaleCols, Matrix.mul_diagonal]

/-- Matrices with the same rows satisfy `ProjEquiv` exactly when `T A S = B` for a nonsingular `T`
and a diagonal `S` with nonzero entries. Proposition 1.1 is `elemOps_iff_exists_mul_mul`. -/
theorem projEquiv_iff_exists_mul_mul_diagonal [Fintype R] [DecidableEq R] [Fintype E]
    [DecidableEq E] (A B : Matrix R E F) :
    ProjEquiv A B ↔ ∃ (T : Matrix R R F) (s : E → Fˣ), IsUnit T ∧
      T * A * Matrix.diagonal (fun j => (s j : F)) = B := by
  constructor
  · rintro ⟨s, hs⟩
    obtain ⟨T, hT, rfl⟩ := exists_isUnit_mul_eq_of_rowSpace_eq _ _ hs
    exact ⟨T, s, hT, by rw [scaleCols_eq_mul_diagonal, Matrix.mul_assoc]⟩
  · rintro ⟨T, s, hT, rfl⟩
    refine ⟨s, ?_⟩
    rw [Matrix.mul_assoc, rowSpace_mul_of_isUnit hT, scaleCols_eq_mul_diagonal]

example : ProjEquiv (1 : Matrix (Fin 2) (Fin 2) ℚ) (1 : Matrix (Fin 2) (Fin 2) ℚ) :=
  (ProjEquiv.refl _).symm

example : ProjEquiv (1 : Matrix (Fin 2) (Fin 2) ℚ) (1 : Matrix (Fin 2) (Fin 2) ℚ) :=
  (ProjEquiv.refl _).trans (ProjEquiv.refl _)

example : ∃ τ : ℚ ≃ₗ[ℚ] ℚ, (LinearMap.id : ℚ →ₗ[ℚ] ℚ) ∘ₗ τ.toLinearMap = LinearMap.id :=
  exists_linearEquiv_comp_eq _ _ rfl

example : ∃ T : Matrix (Fin 2) (Fin 2) ℚ, IsUnit T ∧
    T * (1 : Matrix (Fin 2) (Fin 2) ℚ) = (1 : Matrix (Fin 2) (Fin 2) ℚ) :=
  exists_isUnit_mul_eq_of_rowSpace_eq _ _ rfl

example : rowSpace ((1 : Matrix (Fin 2) (Fin 2) ℚ) * (1 : Matrix (Fin 2) (Fin 2) ℚ)) =
    rowSpace (1 : Matrix (Fin 2) (Fin 2) ℚ) :=
  rowSpace_mul_of_isUnit isUnit_one _

end Transformer.GainMatroid
