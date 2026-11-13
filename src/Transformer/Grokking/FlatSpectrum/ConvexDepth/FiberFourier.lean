module

public import Transformer.Grokking.FlatSpectrum.ConvexDepth.QuotientHead

/-!
# Fourier blocks indexed by latent fibers of the refined input states

Source: `review/flat_spectrum.md`, CM32. Group categorical states by a fixed
latent map and construct one normalized Fourier block in each nonempty fiber.
Reindex the direct sum back to the original state names. The actual matrix
is unitary with zero entries between fibers. Native storage needs the sum
of squared block sizes, while the head still pays for every refined state.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.ConvexDepth

open Matrix

variable {I Z : Type*} [Fintype I] [DecidableEq I] [Fintype Z] [DecidableEq Z]

/-- A block Fourier code using only categorical state identities and a
fixed block assignment, CM32. The reindexing contains no labels. -/
noncomputable def fiberFourier (block : I → Z) [∀ z, Nonempty {s // block s = z}] :
    Matrix I I ℂ :=
  let e := (Equiv.sigmaFiberEquiv block).symm
  (Matrix.blockDiagonal' (fun z => categoricalFourierOn {s // block s = z})).submatrix e e

/-- The actual reindexed block matrix is unitary, CM32. This uses both
Gram identities of every independently sized Fourier block. -/
theorem fiberFourier_unitary (block : I → Z) [∀ z, Nonempty {s // block s = z}] :
    fiberFourier block ∈ Matrix.unitaryGroup I ℂ := by
  let B := Matrix.blockDiagonal' (fun z => categoricalFourierOn {s // block s = z})
  let e := (Equiv.sigmaFiberEquiv block).symm
  have hB : B ∈ Matrix.unitaryGroup (Σ z, {s // block s = z}) ℂ :=
    categorical_blocks_unitary _ (fun z => categoricalFourierOn_unitary _)
  have hl : B.conjTranspose * B = 1 := Unitary.star_mul_self_of_mem hB
  have hr : B * B.conjTranspose = 1 := Unitary.mul_star_self_of_mem hB
  constructor
  · change (B.submatrix e e).conjTranspose * B.submatrix e e = 1
    rw [Matrix.conjTranspose_submatrix, Matrix.submatrix_mul_equiv, hl,
      Matrix.submatrix_one_equiv]
  · change B.submatrix e e * (B.submatrix e e).conjTranspose = 1
    rw [Matrix.conjTranspose_submatrix, Matrix.submatrix_mul_equiv, hr,
      Matrix.submatrix_one_equiv]

/-- Every singular value of the full code is one despite different fiber
sizes, CM32. Flatness is a property of this fixed basis. -/
theorem fiberFourier_singularValues (block : I → Z)
    [∀ z, Nonempty {s // block s = z}] (i : I) :
    singularValues (fiberFourier block) i = 1 :=
  unitary_singularValues _ (fiberFourier_unitary block) i

omit [DecidableEq I] [Fintype Z] in
/-- Different fibers do not interact in the actual matrix; storing those
off-block zeros is unnecessary, CM32. -/
theorem fiberFourier_cross (block : I → Z) [∀ z, Nonempty {s // block s = z}]
    {i j : I} (hij : block i ≠ block j) : fiberFourier block i j = 0 := by
  simp [fiberFourier, Matrix.blockDiagonal'_apply, Equiv.sigmaFiberEquiv, hij]

omit [DecidableEq I] in
/-- Fiber state counts sum to the full state count. This charges all
observed identities and default states after refinement, CM32. -/
theorem fiberFourier_state_count (block : I → Z) :
    (∑ z, Fintype.card {s // block s = z}) = Fintype.card I := by
  simpa only [Fintype.card_sigma] using Fintype.card_congr (Equiv.sigmaFiberEquiv block)

omit [DecidableEq I] in
/-- The finite collection of stored within-block phase entries has exactly
the sum of squared fiber sizes, CM32. The theorem does not allocate a matrix. -/
theorem fiberFourier_phase_count (block : I → Z) :
    Fintype.card (Σ z, {s // block s = z} × {s // block s = z}) =
      ∑ z, (Fintype.card {s // block s = z})^2 := by
  simp only [Fintype.card_sigma, Fintype.card_prod, pow_two]

omit [DecidableEq I] in
/-- Block storage is no larger than dense storage; balanced fibers make
it smaller, without a claim that arbitrary learned fibers are balanced. -/
theorem fiberFourier_phase_le_dense (block : I → Z) :
    (∑ z, (Fintype.card {s // block s = z})^2) ≤ (Fintype.card I)^2 := by
  have h := Finset.sum_sq_le_sq_sum_of_nonneg
    (s := Finset.univ) (f := fun z => Fintype.card {s // block s = z}) (by intros; omega)
  rw [fiberFourier_state_count block] at h
  exact h

omit [DecidableEq I] in
/-- A known maximum fiber size gives a linear storage bound `B * N`,
CM32. Balancedness helps only when this explicit size premise holds. -/
theorem fiberFourier_phase_bound (block : I → Z) (B : ℕ)
    (hB : ∀ z, Fintype.card {s // block s = z} ≤ B) :
    (∑ z, (Fintype.card {s // block s = z})^2) ≤ B * Fintype.card I := by
  calc
    _ ≤ ∑ z, B * Fintype.card {s // block s = z} := by
      apply Finset.sum_le_sum
      intro z hz
      rw [pow_two]
      exact mul_le_mul_of_nonneg_right (hB z) (Nat.zero_le _)
    _ = _ := by rw [← Finset.mul_sum, fiberFourier_state_count]

variable {X : Type*} [DecidableEq X]

/-- Group the CM31 states by their existing latent code. Each fiber gets
all its observed identities plus its single default, CM32. -/
def refinedBlock (observed : Finset X) (latent : X → Z) : ↥observed ⊕ Z → Z :=
  Sum.elim (fun s => latent s) id

instance refinedBlock_fiber_nonempty (observed : Finset X) (latent : X → Z) (z : Z) :
    Nonempty {s // refinedBlock observed latent s = z} := ⟨⟨Sum.inr z, rfl⟩⟩

omit [Fintype I] [DecidableEq I] [Fintype Z] [DecidableEq Z] in
/-- Reblocking changes no input's latent code, CM32. The feature extractor
is still CM31's observed-identity refinement. -/
theorem refinedBlock_state (observed : Finset X) (latent : X → Z) (x : X) :
    refinedBlock observed latent (refinedState observed latent x) = latent x := by
  exact refinedState_forget observed latent x

omit [Fintype I] [DecidableEq I] in
/-- Every observed-input refinement has a valid block Fourier basis,
including latent fibers containing no observations, CM32. -/
theorem refinedBlock_unitary (observed : Finset X) (latent : X → Z) :
    fiberFourier (refinedBlock observed latent) ∈ Matrix.unitaryGroup (↥observed ⊕ Z) ℂ :=
  fiberFourier_unitary _

omit [Fintype I] [DecidableEq I] [DecidableEq X] in
/-- All block states still total observed inputs plus latent codes. There
is one charged default per code, not one uncharged global fallback, CM32. -/
theorem refinedBlock_state_count (observed : Finset X) (latent : X → Z) :
    (∑ z, Fintype.card {s // refinedBlock observed latent s = z}) =
      observed.card + Fintype.card Z := by
  rw [fiberFourier_state_count, Fintype.card_sum, Fintype.card_coe]

private def exampleBlock (i : Fin 4) : Bool := decide (i.val < 2)

private instance (z : Bool) : Nonempty {s // exampleBlock s = z} := by
  cases z
  · exact ⟨⟨2, by decide⟩⟩
  · exact ⟨⟨0, by decide⟩⟩

example := fiberFourier_unitary exampleBlock

example := fiberFourier_singularValues exampleBlock 2

example := fiberFourier_cross exampleBlock (i := 0) (j := 2) (by decide)

example : (∑ z, (Fintype.card {s // exampleBlock s = z})^2) = 8 := by decide

example := fiberFourier_phase_le_dense exampleBlock

example := fiberFourier_phase_bound exampleBlock 2 (by intro z; cases z <;> decide)

example := refinedBlock_unitary ({false} : Finset Bool) (id : Bool → Bool)

example : (∑ z, Fintype.card {s //
    refinedBlock ({false} : Finset Bool) (id : Bool → Bool) s = z}) = 3 := by
  rw [refinedBlock_state_count]
  decide

end Transformer.Grokking.FlatSpectrum.ConvexDepth
