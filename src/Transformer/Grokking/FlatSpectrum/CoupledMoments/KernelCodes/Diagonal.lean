module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.KernelCodes.Basic
public import Mathlib.LinearAlgebra.StdBasis
public import Mathlib.LinearAlgebra.Dimension.StrongRankCondition

/-!
# A genuinely two-dimensional observed kernel

Source: `review/flat_spectrum.md`, CM34. Two diagonal observations and the
two declared gauges leave two independent coefficients. The equivalence
below proves that these parameters exhaust the actual observed kernel.
Its standard and sheared bases give different vectors but the same decisions.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments.KernelCodes

/-- Actual diagonal observations with their two original raw labels, CM34. -/
def diagonalKernel : Submodule ℚ (Coordinates Bool Bool Bool → ℚ) :=
  observedKernel (fun b : Bool => (b, b)) id false false

/-- Two free coefficients, with the observed true class equal to their sum,
CM34. All zero coordinates are consequences of the two gauges and observations. -/
def diagonalVector (x : Bool → ℚ) : Coordinates Bool Bool Bool → ℚ :=
  pack (fun i => if i then x false else 0) (fun j => if j then x true else 0)
    (fun c => if c then x false + x true else 0)

/-- Every parameter pair satisfies the actual original equations, CM34. -/
theorem diagonalVector_mem (x : Bool → ℚ) : diagonalVector x ∈ diagonalKernel := by
  refine ⟨?_, rfl, rfl⟩
  intro b
  cases b <;> simp [queryAction, diagonalVector, pack]

/-- The two free coordinates recover every observed-kernel vector, CM34.
This proves spanning rather than assuming a convenient answer subspace. -/
theorem diagonal_reconstruct (w : diagonalKernel) :
    diagonalVector (fun b => if b then w.val (.inr (.inl true)) else w.val (.inl true)) =
      w.val := by
  have h00 := w.property.1 false
  have h11 := w.property.1 true
  have hr := w.property.2.1
  have he := w.property.2.2
  change w.val (.inl false) + w.val (.inr (.inl false)) -
    w.val (.inr (.inr false)) = 0 at h00
  change w.val (.inl true) + w.val (.inr (.inl true)) -
    w.val (.inr (.inr true)) = 0 at h11
  funext v
  rcases v with i | (j | c)
  · cases i <;> simp [diagonalVector, pack]; linarith
  · cases j <;> simp [diagonalVector, pack]; linarith
  · cases c <;> simp [diagonalVector, pack] <;> linarith

/-- An actual linear equivalence onto the whole observed kernel, CM34. -/
def diagonalEquiv : (Bool → ℚ) ≃ₗ[ℚ] diagonalKernel where
  toFun x := ⟨diagonalVector x, diagonalVector_mem x⟩
  invFun w b := if b then w.val (.inr (.inl true)) else w.val (.inl true)
  left_inv x := by funext b; cases b <;> rfl
  right_inv w := Subtype.ext (diagonal_reconstruct w)
  map_add' x y := by
    apply Subtype.ext
    funext v
    rcases v with i | (j | c)
    · cases i <;> simp [diagonalVector, pack]
    · cases j <;> simp [diagonalVector, pack]
    · cases c <;> simp [diagonalVector, pack]; ring
  map_smul' a x := by
    apply Subtype.ext
    funext v
    rcases v with i | (j | c)
    · cases i <;> simp [diagonalVector, pack]
    · cases j <;> simp [diagonalVector, pack]
    · cases c <;> simp [diagonalVector, pack]; ring

/-- Read both independent standard directions, CM34. -/
noncomputable def diagonalBasis : Module.Basis Bool ℚ diagonalKernel :=
  (Pi.basisFun ℚ Bool).map diagonalEquiv

/-- This is a dimension-two kernel after both gauges, CM34. -/
theorem diagonal_dimension : Module.finrank ℚ diagonalKernel = 2 := by
  simpa using Module.finrank_eq_card_basis diagonalBasis

/-- A nontrivial invertible change of the two parameter directions, CM34. -/
def shear : (Bool → ℚ) ≃ₗ[ℚ] (Bool → ℚ) where
  toFun x b := if b then x true else x false + x true
  invFun x b := if b then x true else x false - x true
  left_inv x := by funext b; cases b <;> simp
  right_inv x := by funext b; cases b <;> simp
  map_add' x y := by funext b; cases b <;> simp; ring
  map_smul' a x := by funext b; cases b <;> simp; ring

/-- A second complete basis of exactly the same actual kernel, CM34. -/
noncomputable def shearedBasis : Module.Basis Bool ℚ diagonalKernel :=
  (Pi.basisFun ℚ Bool).map (shear.trans diagonalEquiv)

/-- Both class vectors and both unobserved query vectors are explicit,
CM34. Their two coordinates retain different information. -/
theorem diagonal_codes :
    classCode diagonalBasis false = (fun _ => 0) ∧
    classCode diagonalBasis true = (fun _ => 1) ∧
    queryCode diagonalBasis true false = (fun k => if k then 0 else 1) ∧
    queryCode diagonalBasis false true = (fun k => if k then 1 else 0) := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> funext k <;> cases k <;>
    norm_num [classCode, queryCode, diagonalBasis, Module.Basis.map_apply,
      Pi.basisFun_apply, diagonalEquiv, diagonalVector, pack, Pi.single_apply]

/-- The sheared class coordinates really change, CM34. Basis invariance
therefore cannot be explained by identical stored vectors. -/
theorem diagonal_basis_changes : classCode diagonalBasis true ≠ classCode shearedBasis true := by
  intro h
  have ht := congrFun h true
  norm_num [classCode, diagonalBasis, shearedBasis, Module.Basis.map_apply,
    Pi.basisFun_apply, diagonalEquiv, diagonalVector, pack, Pi.single_apply, shear] at ht

/-- A single direction accepts a false class match, while neither class
matches the complete query vector, CM34. Thus the search must be allowed to refuse. -/
theorem diagonal_one_direction_loses :
    queryCode diagonalBasis true false true = classCode diagonalBasis false true ∧
    queryCode diagonalBasis true false ≠ classCode diagonalBasis false ∧
    queryCode diagonalBasis true false ≠ classCode diagonalBasis true := by
  rw [diagonal_codes.1, diagonal_codes.2.1, diagonal_codes.2.2.1]
  refine ⟨rfl, ?_, ?_⟩
  · intro h
    have hf := congrFun h false
    norm_num at hf
  · intro h
    have ht := congrFun h true
    norm_num at ht

/-- Retaining both directions refuses this query rather than claiming
an unseen answer, CM34. The observed diagonal still matches exactly. -/
theorem diagonal_refuses :
    chooseClass diagonalBasis (queryCode diagonalBasis true false) [false, true] = none := by
  classical
  simp only [chooseClass, diagonal_one_direction_loses.2.1,
    diagonal_one_direction_loses.2.2, ite_false]

example : diagonalVector (fun b => if b then 3 else 2) ∈ diagonalKernel :=
  diagonalVector_mem _

example := diagonal_reconstruct (diagonalEquiv (fun b => if b then 3 else 2))

/-- Refusal off the diagonal retains both original observed matches,
CM34. No observed row is discarded to obtain the counterexample. -/
theorem diagonal_seen_matches (b : Bool) :
    queryCode diagonalBasis b b = classCode diagonalBasis b := by
  exact observed_code_match (fun t : Bool => (t, t)) id false false diagonalBasis b

example := diagonal_seen_matches true

end Transformer.Grokking.FlatSpectrum.CoupledMoments.KernelCodes
