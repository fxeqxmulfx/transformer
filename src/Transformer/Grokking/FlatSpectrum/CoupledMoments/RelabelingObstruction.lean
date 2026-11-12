module

public import Transformer.Grokking.FlatSpectrum.CoupledMoments.Relabeling
public import Transformer.Grokking.FlatSpectrum.ConvexDepth.Transport

/-!+# A fixed complex code can exclude an output relabeling

Source: `review/flat_spectrum.md`, CM12. A four-class point-mass table has a
perfectly flat moment of operator norm two. Swapping two class names preserves
the probability rows and every renamed-target CE, but increases that same
code's operator norm. The invariant convex hull retains the renamed predictor.
This is a one-code counterexample; it does not assert feasibility for all
nonconstant Fourier characters, or diagnose the prime-vocabulary experiments.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.CoupledMoments

open Matrix
open scoped BigOperators Matrix.Norms.L2Operator

/-- Four fixed unit class phases, in the original output order (CM12). -/
noncomputable def labelCode : Fin 4 → ℂ := ![1, Complex.I, -1, -Complex.I]

/-- Concrete labels whose coded moment is the unnormalized Fourier matrix
(CM12). The values specify this witness, not features for an experiment. -/
def labelTable : Fin 4 → Fin 4 → Fin 4 :=
  !![0, 0, 0, 0; 0, 1, 2, 3; 0, 2, 0, 2; 0, 3, 2, 1]

/-- The original full probability table (CM12). -/
def codedWitness : Table (Fin 4) (Fin 4) (Fin 4) := pointMass labelTable

/-- A rename with no alteration to input rows or columns (CM12). -/
def labelSwap : Equiv.Perm (Fin 4) := Equiv.swap 1 2

/-- The original family constrains just this fixed code at radius two (CM12). -/
def codedFamily : Set (Table (Fin 4) (Fin 4) (Fin 4)) :=
  {P | Feasible (fun _ : Unit => labelCode) (fun _ => 2) P}

/-- The witness uses genuine unit-modulus complex class values (CM12). -/
theorem labelCode_unit (c : Fin 4) : ‖labelCode c‖ = 1 := by
  fin_cases c <;> norm_num [labelCode]

/-- The code is centered, so uniform class mass has zero moment (CM12). -/
theorem labelCode_centered : ∑ c, labelCode c = 0 := by
  norm_num [labelCode, Fin.sum_univ_succ]

/-- This actual point-mass moment equals twice the unitary Fourier transport
(CM12), rather than merely satisfying an entrywise bound. -/
theorem codedWitness_moment : moment labelCode codedWitness = (2 : ℂ) • ConvexDepth.fourier4 := by
  rw [codedWitness, moment_pointMass]
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [labelCode, labelTable, ConvexDepth.fourier4, Complex.ext_iff]

/-- The Gram matrix is exactly four times identity: all singular values are
two (CM12). This is flatness of the original coded label matrix. -/
theorem codedWitness_gram :
    (moment labelCode codedWitness).conjTranspose * moment labelCode codedWitness =
      (4 : ℂ) • (1 : Matrix (Fin 4) (Fin 4) ℂ) := by
  rw [codedWitness_moment]
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [ConvexDepth.fourier4, Matrix.mul_apply, Matrix.conjTranspose_apply,
      Matrix.one_apply, Fin.sum_univ_succ, Complex.ext_iff]

/-- The exact original spectral radius is two (CM12). -/
theorem codedWitness_norm : ‖moment labelCode codedWitness‖ = 2 := by
  have hu : ‖ConvexDepth.fourier4‖ = 1 :=
    CStarRing.norm_coe_unitary ⟨ConvexDepth.fourier4, ConvexDepth.fourier4_unitary⟩
  rw [codedWitness_moment, norm_smul, hu]
  norm_num

/-- The original nonzero point-mass witness satisfies the complete one-code
coupled probability and spectral domain (CM12). -/
theorem codedWitness_feasible : codedWitness ∈ codedFamily := by
  refine ⟨rows_pointMass labelTable, ?_⟩
  intro f
  exact codedWitness_norm.le

/-- The renamed table is still an ordinary class distribution (CM12). -/
theorem swappedWitness_rows : Rows (relabel labelSwap codedWitness) :=
  rows_relabel labelSwap (rows_pointMass labelTable)

/-- On all sixteen input cells, the renamed actual targets have exactly the
same ordinary CE sum at any fixed scale (CM12). The change in spectral
feasibility therefore cannot be explained by a changed loss. -/
theorem swappedWitness_loss_eq (scale : ℝ) :
    scaledLoss scale (fun s : Fin 4 × Fin 4 => s)
      (fun s => labelSwap (labelTable s.1 s.2)) (relabel labelSwap codedWitness) =
    scaledLoss scale (fun s : Fin 4 × Fin 4 => s)
      (fun s => labelTable s.1 s.2) codedWitness :=
  scaledLoss_relabel labelSwap scale (fun s => s) (fun s => labelTable s.1 s.2) codedWitness

/-- The exact matrix after renaming shows why this is not an input row or
column permutation of the old moment (CM12). -/
theorem swappedWitness_moment : moment labelCode (relabel labelSwap codedWitness) =
    !![(1 : ℂ), 1, 1, 1; 1, -1, Complex.I, -Complex.I;
       1, Complex.I, 1, Complex.I; 1, -Complex.I, Complex.I, -1] := by
  rw [moment_relabel, codedWitness, moment_pointMass]
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [Function.comp_def, labelSwap, labelCode, labelTable, Equiv.swap_apply_def]

/-- A concrete input of squared norm four gains squared norm twenty-four
under the renamed moment (CM12). -/
theorem swappedWitness_energy :
    ∑ i, ‖(moment labelCode (relabel labelSwap codedWitness) *ᵥ
      (fun _ : Fin 4 => (1 : ℂ))) i‖ ^ 2 = 24 := by
  rw [swappedWitness_moment]
  norm_num [Matrix.mulVec, dotProduct, Fin.sum_univ_succ,
    ← Complex.normSq_eq_norm_sq, Complex.normSq_apply]

/-- The old operator bound fails after output renaming: it would imply
24 ≤ 2²·4 = 16 (CM12). The contradiction uses an exact energy witness. -/
theorem swappedWitness_not_bounded :
    ¬ ‖moment labelCode (relabel labelSwap codedWitness)‖ ≤ 2 := by
  intro h
  have he := SpectralBall.energy_le h (fun _ : Fin 4 => (1 : ℂ))
  rw [swappedWitness_energy] at he
  norm_num [Fin.sum_univ_succ] at he

/-- Thus the fixed-code coupled family is not invariant under output class
renaming, even though rows and renamed-target CE are preserved (CM12). -/
theorem swappedWitness_not_feasible : relabel labelSwap codedWitness ∉ codedFamily := by
  intro h
  exact swappedWitness_not_bounded (h.2 ())

/-- The proposed convex extension contains this previously excluded predictor
(CM12), keeping its original bound in the corresponding renamed coding. -/
theorem swappedWitness_in_hull : relabel labelSwap codedWitness ∈ relabelHull codedFamily :=
  relabel_mem_hull labelSwap codedWitness_feasible

/-- The invariant convex hull is strictly more expressive in direct prediction
coordinates for this concrete family (CM12). -/
theorem relabelHull_strict_witness :
    ∃ P ∈ relabelHull codedFamily, P ∉ codedFamily :=
  ⟨relabel labelSwap codedWitness, swappedWitness_in_hull, swappedWitness_not_feasible⟩

example := moment_norm_le_one (fun c => (labelCode_unit c).le) (rows_pointMass labelTable)
example := relabel_mem_hull labelSwap codedWitness_feasible
example := relabelHull_rows codedFamily (fun _ hP => hP.1) swappedWitness_in_hull
example := crossEntropy_relabel labelSwap
  (fun c => 24 * codedWitness 1 2 c) (labelTable 1 2)
example := swappedWitness_loss_eq 24
example := scaledLoss_relabel labelSwap 24 (fun s : Fin 4 × Fin 4 => s)
  (fun s => labelTable s.1 s.2) codedWitness

end Transformer.Grokking.FlatSpectrum.CoupledMoments
