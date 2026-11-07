module

public import Transformer.Rotate90.ComplexProduct.Recovery
public import Transformer.Rotate90.ComplexProduct.SourceWitness
public import Transformer.Grokking.FlatSpectrum.FeatureCodes.NonuniqueWitness

/-!
# An actual classifier aligned with a nonunique spectral optimum

Source: `review/flat_spectrum.md`, *Complex-product Rotate90 bridge*, CP5.
The two distinct nuclear-norm optima from the earlier nonunique-recovery witness
align with one explicitly evaluated source classifier. The two inputs have
different correct labels. This proves the bridge's premises are jointly
satisfiable; it does not certify learned cubic-task weights or generalization.
-/

@[expose] public section
noncomputable section

namespace Transformer.Rotate90.ComplexProduct

open Matrix Grokking.FlatSpectrum Grokking.FlatSpectrum.FeatureCodes

/-- Fixed spectral score aligned to the source model, source CP5. -/
def copyScore (q v : Fin 2) : Matrix (Fin 2) (Fin 2) ℂ →ₗ[ℝ] ℝ where
  toFun X := (if v = 0 then 1 else -1) * (X q q).re / Real.sqrt 2
  map_add' X Y := by simp only [Matrix.add_apply, Complex.add_re]; ring
  map_smul' s X := by
    simp only [Matrix.smul_apply, Complex.smul_re, RingHom.id_apply, smul_eq_mul]
    ring

/-- The alignment premise is derived from the actual source state in CP5. -/
theorem copy_alignment {X : Matrix (Fin 2) (Fin 2) ℂ}
    (hfit : ∀ ij ∈ diagonalObservations, X ij.1 ij.2 = ambiguousTarget ij.1 ij.2)
    (q v : Fin 2) :
    readout copyWeights.embedding (state 1 copyWeights q 0) v = copyScore q v X := by
  rw [copy_readout]
  change _ = (if v = 0 then 1 else -1) * (X q q).re / Real.sqrt 2
  rw [hfit (q, q) rfl]
  fin_cases q <;> norm_num [ambiguousTarget]

private theorem target_unitary : ambiguousTargetᴴ * ambiguousTarget = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [ambiguousTarget, Matrix.mul_apply, Fin.sum_univ_two, conjTranspose_apply]

private theorem target_svd :
    ambiguousTarget * ambiguousTargetᴴ * ambiguousTarget = ambiguousTarget ∧
    (ambiguousTargetᴴ * ambiguousTarget).trace = nuclearNorm ambiguousTarget := by
  have h := svd_hypotheses target_unitary (V := (1 : Matrix (Fin 2) (Fin 2) ℂ))
    (by simp) (s := fun _ => 1) (fun _ => zero_le_one)
  simpa using And.intro h.1 h.2.1

/-- **The full spectral-to-source bridge has nonunique, nonconstant instances** (CP5).
Every minimum aligns with the same explicitly computed source classifier. -/
theorem copy_spectral_classifies {X : Matrix (Fin 2) (Fin 2) ℂ}
    (hfit : ∀ ij ∈ diagonalObservations, X ij.1 ij.2 = ambiguousTarget ij.1 ij.2)
    (hmin : IsMinOn nuclearNorm
      {Y | ∀ ij ∈ diagonalObservations, Y ij.1 ij.2 = ambiguousTarget ij.1 ij.2} X) :
    ∀ q v : Fin 2, v ≠ q →
      realLogit 1 copyWeights q 0 v < realLogit 1 copyWeights q 0 q := by
  apply executed_correct_of_nonunique_minimum target_unitary
    (V₀ := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (by simp) target_svd.1
    (by simpa using target_svd.2) (W := 0) ?_ (by simp) (by simp)
    (c := 2) one_lt_two (by rw [smul_zero]; exact isContraction_zero)
    copyScore id ?_ hfit hmin (eps := 1) (by norm_num) copyWeights (fun q => (q, 0))
    (fun q v => by rw [copy_alignment hfit, copy_alignment hfit])
    (fun q => realLogit 1 copyWeights q 0)
    (fun _ => 2 / Real.sqrt 2) (fun _ => 0) ?_ ?_ ?_
  · intro i j hij
    fin_cases i <;> fin_cases j <;> norm_num [diagonalObservations, ambiguousTarget] at *
  · intro P B hz q v
    have hdiag := hz (q, q) rfl
    simp only [LinearMap.sub_apply, copyScore, LinearMap.coe_mk, AddHom.coe_mk]
    rw [hdiag]
    simp
  · intro q v hv
    rw [← copy_alignment (X := ambiguousTarget) (fun _ _ => rfl),
      ← copy_alignment (X := ambiguousTarget) (fun _ _ => rfl)]
    exact le_of_eq (copy_gap q v hv).symm
  · intro q v
    simp
  · intro q
    simp only [mul_zero, zero_mul]
    positivity

/-- Two genuinely distinct spectral optima satisfy the actual-source bridge (CP5). -/
theorem copy_nonunique_bridge :
    ambiguousTarget ≠ ambiguousCompletion ∧
    IsMinOn nuclearNorm
      {X | ∀ ij ∈ diagonalObservations, X ij.1 ij.2 = ambiguousTarget ij.1 ij.2}
      ambiguousTarget ∧
    IsMinOn nuclearNorm
      {X | ∀ ij ∈ diagonalObservations, X ij.1 ij.2 = ambiguousTarget ij.1 ij.2}
      ambiguousCompletion ∧
    (∀ q v, readout copyWeights.embedding (state 1 copyWeights q 0) v =
      copyScore q v ambiguousCompletion) ∧
    ∀ q v : Fin 2, v ≠ q →
      realLogit 1 copyWeights q 0 v < realLogit 1 copyWeights q 0 q := by
  exact ⟨nonunique_recovery_correct_classification.1, ambiguous_target_minimum,
    ambiguous_completion_minimum.2, copy_alignment ambiguous_completion_minimum.1,
    copy_spectral_classifies ambiguous_completion_minimum.1 ambiguous_completion_minimum.2⟩

example := copy_alignment ambiguous_completion_minimum.1 0 1
example := copy_spectral_classifies (X := ambiguousTarget)
  (fun _ _ => rfl) ambiguous_target_minimum
example := copy_spectral_classifies ambiguous_completion_minimum.1 ambiguous_completion_minimum.2

/-- A direct instance of the numerical-margin theorem, with exact execution (CP3, CP5). -/
example : ∀ q v : Fin 2, v ≠ q →
    realLogit 1 copyWeights q 0 v < realLogit 1 copyWeights q 0 q := by
  intro q
  apply executed_correct_of_raw_margin (eps := 1) (by norm_num) copyWeights q 0 q
    (realLogit 1 copyWeights q 0) (δ := 2 / Real.sqrt 2) (η := 0) ?_ ?_ ?_
  · intro v hv
    rw [copy_gap q v hv]
  · intro v
    simp
  · simp only [mul_zero, zero_mul]
    positivity

end Transformer.Rotate90.ComplexProduct
