module

public import Transformer.Grokking.FlatSpectrum.FeatureCodes.NonuniqueRecovery
public import Transformer.Grokking.FlatSpectrum.DualCertificate
public import Transformer.Grokking.FlatSpectrum.TwoDiagonals

/-!
# Two different optimal feature matrices with the same correct labels

Source: `review/flat_spectrum.md`, *Classification without unique recovery*, item 3.
Only the diagonal of `diag(1,-1)` is observed. Both that matrix and
`!![1,1; -1,-1]` minimize nuclear norm, with value two. The two inputs use their
diagonal entry as a discriminating feature and entry `(0,1)` as a common score
offset. The latter is unobserved and changes between the two optima; both class
scores change, but every optimum classifies both inputs correctly.

This is a concrete separation of classification from feature uniqueness, not a
claim about discovering unseen modular labels or convergence of a neural learner.
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum.FeatureCodes

open Matrix

/-- Target feature matrix for source item 3; its diagonal gives two different labels. -/
def ambiguousTarget : Matrix (Fin 2) (Fin 2) ℂ := !![1, 0; 0, -1]

/-- A different optimal completion in source item 3. -/
def ambiguousCompletion : Matrix (Fin 2) (Fin 2) ℂ := !![1, 1; -1, -1]

/-- Exactly the diagonal entries are observed in source item 3. -/
def diagonalObservations : Set (Fin 2 × Fin 2) := {ij | ij.1 = ij.2}

/-- Fixed linear readout, source item 3. The unobserved feature changes both scores. -/
def ambiguousScore (q a : Fin 2) : Matrix (Fin 2) (Fin 2) ℂ →ₗ[ℝ] ℝ where
  toFun X := (if a = 0 then 1 else -1) * (X q q).re + (X 0 1).re
  map_add' X Y := by simp only [Matrix.add_apply, Complex.add_re]; ring
  map_smul' s X := by
    simp only [Matrix.smul_apply, Complex.smul_re, RingHom.id_apply, smul_eq_mul]
    ring

private theorem ambiguous_unitary : ambiguousTargetᴴ * ambiguousTarget = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [ambiguousTarget, Matrix.mul_apply, Fin.sum_univ_two, conjTranspose_apply]

private theorem ambiguous_svd :
    ambiguousTarget * ambiguousTargetᴴ * ambiguousTarget = ambiguousTarget ∧
    (ambiguousTargetᴴ * ambiguousTarget).trace = nuclearNorm ambiguousTarget ∧
    nuclearNorm ambiguousTarget = 2 := by
  simpa using svd_hypotheses ambiguous_unitary (V := (1 : Matrix (Fin 2) (Fin 2) ℂ))
    (by simp) (s := fun _ => 1) (fun _ => zero_le_one)

private theorem ambiguous_supported : ∀ i j, (i, j) ∉ diagonalObservations →
    (ambiguousTarget * (1 : Matrix (Fin 2) (Fin 2) ℂ)ᴴ + 0) i j = 0 := by
  intro i j hij
  fin_cases i <;> fin_cases j <;> norm_num [diagonalObservations, ambiguousTarget] at *

example := ambiguous_supported 0 1 (by norm_num [diagonalObservations])

/-- The target is an optimum of the partially observed problem, source item 3. -/
theorem ambiguous_target_minimum : IsMinOn nuclearNorm
    {X | ∀ ij ∈ diagonalObservations, X ij.1 ij.2 = ambiguousTarget ij.1 ij.2}
    ambiguousTarget := by
  apply isMinOn_iff_exists_isContraction.mpr
  refine ⟨ambiguousTarget, ?_, ?_, ambiguous_svd.2.1⟩
  · simpa using ambiguous_supported
  · simpa using isContraction_mul_conjTranspose ambiguous_unitary
      (V := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (by simp)

/-- The alternative also fits and minimizes the objective, source item 3. -/
theorem ambiguous_completion_minimum :
    (∀ ij ∈ diagonalObservations,
      ambiguousCompletion ij.1 ij.2 = ambiguousTarget ij.1 ij.2) ∧
    IsMinOn nuclearNorm
      {X | ∀ ij ∈ diagonalObservations, X ij.1 ij.2 = ambiguousTarget ij.1 ij.2}
      ambiguousCompletion := by
  have hnorm : nuclearNorm ambiguousCompletion = 2 := by
    have h := nuclearNorm_vecMulVec_of_norm_eq_one (a := ![(1 : ℂ), -1]) (b := ![1, 1])
      (fun i => by fin_cases i <;> norm_num) (fun j => by fin_cases j <;> norm_num)
    have heq : vecMulVec ![(1 : ℂ), -1] ![1, 1] = ambiguousCompletion := by
      ext i j
      fin_cases i <;> fin_cases j <;> norm_num [vecMulVec_apply, ambiguousCompletion]
    simpa [heq] using h
  refine ⟨?_, isMinOn_iff.mpr fun X hX => ?_⟩
  · rintro ⟨i, j⟩ hij
    have h : i = j := hij
    subst j
    fin_cases i <;> norm_num [ambiguousCompletion, ambiguousTarget]
  · rw [hnorm, ← ambiguous_svd.2.2]
    exact (isMinOn_iff.mp ambiguous_target_minimum) X hX

/-- Every global optimum gives both correct labels although the completion is not
unique, source item 3. This instantiates the general classification theorem. -/
theorem ambiguous_all_minima_classify : ∀ X : Matrix (Fin 2) (Fin 2) ℂ,
    (∀ ij ∈ diagonalObservations, X ij.1 ij.2 = ambiguousTarget ij.1 ij.2) →
    IsMinOn nuclearNorm
      {Y | ∀ ij ∈ diagonalObservations, Y ij.1 ij.2 = ambiguousTarget ij.1 ij.2} X →
    ∀ q a, a ≠ q → ambiguousScore q a X < ambiguousScore q q X := by
  intro X hfit hmin
  apply classification_of_minimum_without_unique_recovery ambiguous_unitary
    (V := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (by simp) ambiguous_svd.1
    (by simpa using ambiguous_svd.2.1) ambiguous_supported
    (W := 0) (by simp) (by simp) (c := 2) one_lt_two
    (by rw [smul_zero]; exact isContraction_zero) ambiguousScore id ?_ ?_ hfit hmin
  · intro q a ha
    fin_cases q <;> fin_cases a <;> norm_num [ambiguousScore, ambiguousTarget] at *
  · intro P B h q a
    have hdiag := h (q, q) rfl
    simp only [LinearMap.sub_apply, ambiguousScore, LinearMap.coe_mk, AddHom.coe_mk, id_eq]
    rw [hdiag]
    simp

/-- The matrix, and even the absolute scores, are nonunique; correct labels hold
for every optimum. Source item 3 gives a strict weakening of feature recovery. -/
theorem nonunique_recovery_correct_classification :
    ambiguousTarget ≠ ambiguousCompletion ∧
    (∀ ij ∈ diagonalObservations,
      ambiguousCompletion ij.1 ij.2 = ambiguousTarget ij.1 ij.2) ∧
    ambiguousScore 0 0 ambiguousTarget ≠ ambiguousScore 0 0 ambiguousCompletion ∧
    IsMinOn nuclearNorm
      {X | ∀ ij ∈ diagonalObservations, X ij.1 ij.2 = ambiguousTarget ij.1 ij.2}
      ambiguousTarget ∧
    IsMinOn nuclearNorm
      {X | ∀ ij ∈ diagonalObservations, X ij.1 ij.2 = ambiguousTarget ij.1 ij.2}
      ambiguousCompletion ∧
    ∀ X : Matrix (Fin 2) (Fin 2) ℂ,
      (∀ ij ∈ diagonalObservations, X ij.1 ij.2 = ambiguousTarget ij.1 ij.2) →
      IsMinOn nuclearNorm
        {Y | ∀ ij ∈ diagonalObservations, Y ij.1 ij.2 = ambiguousTarget ij.1 ij.2} X →
      ∀ q a, a ≠ q → ambiguousScore q a X < ambiguousScore q q X := by
  refine ⟨?_, ambiguous_completion_minimum.1, ?_, ambiguous_target_minimum,
    ambiguous_completion_minimum.2,
    ambiguous_all_minima_classify⟩
  · intro h
    have h01 := congrFun (congrFun h 0) 1
    norm_num [ambiguousTarget, ambiguousCompletion] at h01
  · norm_num [ambiguousScore, ambiguousTarget, ambiguousCompletion]

example := ambiguous_all_minima_classify ambiguousCompletion
  ambiguous_completion_minimum.1 ambiguous_completion_minimum.2

/-- A nonunique example of the linear-readout theorem itself: the correct-versus-wrong
contrast is unchanged even though the two absolute scores differ. -/
example : (ambiguousScore 0 0 - ambiguousScore 0 1) ambiguousCompletion =
    (ambiguousScore 0 0 - ambiguousScore 0 1) ambiguousTarget := by
  apply linear_readout_eq_of_certificate ambiguous_unitary
    (V := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (by simp) ambiguous_svd.1
    (by simpa using ambiguous_svd.2.1) ambiguous_supported
    (W := 0) (by simp) (by simp) (c := 2) one_lt_two
    (by rw [smul_zero]; exact isContraction_zero)
    (ambiguousScore 0 0 - ambiguousScore 0 1) ?_
    ambiguous_completion_minimum.1 ambiguous_completion_minimum.2
  intro P B h
  have h00 := h (0, 0) rfl
  simp only [LinearMap.sub_apply, ambiguousScore, LinearMap.coe_mk, AddHom.coe_mk]
  rw [h00]
  norm_num

end Transformer.Grokking.FlatSpectrum.FeatureCodes
