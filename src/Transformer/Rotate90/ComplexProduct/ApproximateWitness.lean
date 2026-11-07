module

public import Transformer.Rotate90.ComplexProduct.ApproximateRecovery
public import Transformer.Rotate90.ComplexProduct.RecoveryWitness

/-!
# A noisy nonunique instance of quantitative source recovery

Source: `review/flat_spectrum.md`, *Approximate nonunique recovery*, QR5.
Perturb the earlier alternative optimum by any matrix of nuclear norm at most
1/8. The fitting and objective errors are then at most 1/8. The two diagonal
contrasts have hidden conditioning zero and global bound 2/√2. Alignment to
the actual two-label source has error at most (2/√2)/8. The combined bound is
strictly below its raw gap 2/√2. A nonzero perturbation is exhibited explicitly.
This remains a consistency witness, not a modular-learning certificate.
-/

@[expose] public section
noncomputable section

namespace Transformer.Rotate90.ComplexProduct

open Matrix Grokking.FlatSpectrum Grokking.FlatSpectrum.FeatureCodes

private theorem target_unitary : ambiguousTargetᴴ * ambiguousTarget = 1 := by
  ext i j
  fin_cases i <;> fin_cases j <;>
    norm_num [ambiguousTarget, Matrix.mul_apply, Fin.sum_univ_two, conjTranspose_apply]

private theorem target_svd :
    ambiguousTarget * ambiguousTargetᴴ * ambiguousTarget = ambiguousTarget ∧
    (ambiguousTargetᴴ * ambiguousTarget).trace = nuclearNorm ambiguousTarget ∧
    nuclearNorm ambiguousTarget = 2 := by
  simpa using svd_hypotheses target_unitary (V := (1 : Matrix (Fin 2) (Fin 2) ℂ))
    (by simp) (s := fun _ => 1) (fun _ => zero_le_one)

private theorem target_support : ∀ i j, (i, j) ∉ diagonalObservations →
    (ambiguousTarget * (1 : Matrix (Fin 2) (Fin 2) ℂ)ᴴ + 0) i j = 0 := by
  intro i j hij
  fin_cases i <;> fin_cases j <;> norm_num [diagonalObservations, ambiguousTarget] at *

private theorem copy_bound (q v : Fin 2) (M : Matrix (Fin 2) (Fin 2) ℂ) :
    |(copyScore q q - copyScore q v) M| ≤ (2 / Real.sqrt 2) * nuclearNorm M := by
  have h := (Complex.abs_re_le_norm (M q q)).trans (norm_apply_le_nuclearNorm M q q)
  have hs : 0 < Real.sqrt 2 := by positivity
  have hcoeff : |(if q = 0 then (1 : ℝ) else -1) -
      (if v = 0 then 1 else -1)| ≤ 2 := by split_ifs <;> norm_num
  change |(if q = 0 then (1 : ℝ) else -1) * (M q q).re / Real.sqrt 2 -
    (if v = 0 then 1 else -1) * (M q q).re / Real.sqrt 2| ≤ _
  rw [← sub_div, ← sub_mul, abs_div, abs_mul, abs_of_pos hs]
  calc
    _ ≤ (2 * nuclearNorm M) / Real.sqrt 2 :=
      (div_le_div_iff_of_pos_right hs).mpr
        (mul_le_mul hcoeff h (abs_nonneg _) (by norm_num))
    _ = _ := by ring

private theorem copy_hidden (q v : Fin 2) (M : Matrix (Fin 2) (Fin 2) ℂ)
    (hM : ∀ ij ∈ diagonalObservations, M ij.1 ij.2 = 0) :
    (copyScore q q - copyScore q v) M = 0 := by
  simp only [LinearMap.sub_apply, copyScore, LinearMap.coe_mk, AddHom.coe_mk, hM (q, q) rfl,
    Complex.zero_re, mul_zero, zero_div, sub_self]

private theorem perturbed_objective {E : Matrix (Fin 2) (Fin 2) ℂ} {ρ : ℝ}
    (hE : nuclearNorm E ≤ ρ) :
    nuclearNorm (ambiguousCompletion + E) ≤ nuclearNorm ambiguousTarget + ρ := by
  have hmin := (isMinOn_iff.mp ambiguous_completion_minimum.2)
    ambiguousTarget (fun _ _ => rfl)
  exact (nuclearNorm_add_le _ _).trans (add_le_add hmin hE)

example := copy_bound 0 1 ambiguousCompletion
example := copy_hidden 0 1 (0 : Matrix (Fin 2) (Fin 2) ℂ) (fun _ _ => rfl)
example := perturbed_objective (E := (0 : Matrix (Fin 2) (Fin 2) ℂ)) (ρ := 0)
  (by rw [nuclearNorm_zero])
example := target_support 0 1 (by norm_num [diagonalObservations])

/-- An explicit nonzero fitting perturbation, QR5. -/
def witnessNoise : Matrix (Fin 2) (Fin 2) ℂ := (1 / 16 : ℂ) • ambiguousTarget

/-- The nonzero perturbation has exactly the allowed noise norm, QR5. -/
theorem witnessNoise_norm : nuclearNorm witnessNoise = 1 / 8 := by
  rw [witnessNoise, nuclearNorm_smul, target_svd.2.2]
  norm_num

/-- The noise witness does change the observed entries, QR5. -/
theorem witnessNoise_ne_zero : witnessNoise ≠ 0 := by
  intro h
  have h00 := congrFun (congrFun h 0) 0
  norm_num [witnessNoise, ambiguousTarget] at h00

example := exists_readout_conditioning (U := ambiguousTarget)
  (V := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (Ω := diagonalObservations)
  (copyScore 0 0 - copyScore 0 1) fun _ _ h => copy_hidden 0 1 _ h

example := normal_error_le_of_approximate_minimum target_unitary
  (V := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (by simp) target_svd.1
  (by simpa using target_svd.2.1) target_support (W := 0) (by simp) (by simp)
  (c := 2) one_lt_two (by rw [smul_zero]; exact isContraction_zero)
  (X := ambiguousCompletion + witnessNoise) (E := witnessNoise)
  (by simpa using ambiguous_completion_minimum.1)
  (perturbed_objective witnessNoise_norm.le) witnessNoise_norm.le

example := linear_readout_error_le_of_approximate_minimum target_unitary
  (V := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (by simp) target_svd.1
  (by simpa using target_svd.2.1) target_support (W := 0) (by simp) (by simp)
  (c := 2) one_lt_two (by rw [smul_zero]; exact isContraction_zero)
  (copyScore 0 0 - copyScore 0 1) (κ := 0) (ℓ := 2 / Real.sqrt 2)
  le_rfl (by positivity) (fun M h => by rw [copy_hidden 0 1 M h, abs_zero, zero_mul])
  (copy_bound 0 1) (X := ambiguousCompletion + witnessNoise) (E := witnessNoise)
  (by simpa using ambiguous_completion_minimum.1)
  (perturbed_objective witnessNoise_norm.le) witnessNoise_norm.le

/-- The quantitative recovery theorem has a concrete noisy, nonunique instance,
QR5. Both inputs retain distinct correct answers through the actual source. -/
theorem copy_approximate_spectral_classifies {E : Matrix (Fin 2) (Fin 2) ℂ}
    (hE : nuclearNorm E ≤ 1 / 8) : ∀ q v : Fin 2, v ≠ q →
      realLogit 1 copyWeights q 0 v < realLogit 1 copyWeights q 0 q := by
  have hfit : ∀ ij ∈ diagonalObservations,
      ((ambiguousCompletion + E) - E) ij.1 ij.2 = ambiguousTarget ij.1 ij.2 := by
    simpa using ambiguous_completion_minimum.1
  have hs : 0 < 2 / Real.sqrt 2 := by positivity
  apply executed_correct_of_approximate_recovery target_unitary
    (V₀ := (1 : Matrix (Fin 2) (Fin 2) ℂ)) (by simp) target_svd.1
    (by simpa using target_svd.2.1) target_support (W := 0) (by simp) (by simp)
    (c := 2) one_lt_two (by rw [smul_zero]; exact isContraction_zero)
    copyScore id (fun _ _ => 0) (fun _ _ => 2 / Real.sqrt 2)
    (fun _ _ => le_rfl) (fun _ _ => hs.le) ?_ copy_bound
    hfit (perturbed_objective hE) hE (eps := 1) (by norm_num) copyWeights
    (fun q => (q, 0)) (fun _ _ => (2 / Real.sqrt 2) * (1 / 8)) ?_
    (fun q => realLogit 1 copyWeights q 0) (fun _ => 2 / Real.sqrt 2)
    (fun _ => 0) ?_ (fun _ _ => by simp) ?_
  · intro q v M hM
    simp only [id_eq]
    rw [copy_hidden q v M hM, abs_zero, zero_mul]
  · intro q v
    simp only [id_eq]
    rw [copy_alignment ambiguous_completion_minimum.1,
      copy_alignment ambiguous_completion_minimum.1, map_add, map_add]
    have hb := (copy_bound q v E).trans (mul_le_mul_of_nonneg_left hE hs.le)
    convert hb using 1
    simp only [LinearMap.sub_apply]
    rw [show copyScore q q ambiguousCompletion - copyScore q v ambiguousCompletion -
      (copyScore q q ambiguousCompletion + copyScore q q E -
        (copyScore q v ambiguousCompletion + copyScore q v E)) =
        -(copyScore q q E - copyScore q v E) by ring, abs_neg]
  · intro q v hv
    simp only [id_eq] at hv ⊢
    rw [← copy_alignment (X := ambiguousTarget) (fun _ _ => rfl),
      ← copy_alignment (X := ambiguousTarget) (fun _ _ => rfl), copy_gap q v hv]
  · intro q v hv
    simp only [zero_mul, mul_zero, add_zero, zero_add]
    linarith

example := copy_approximate_spectral_classifies (E := witnessNoise) witnessNoise_norm.le

/-- Direct contrast-error specialization with positive error allowance, QR4–QR5. -/
example (q : Fin 2) : ∀ v : Fin 2, v ≠ q →
    realLogit 1 copyWeights q 0 v < realLogit 1 copyWeights q 0 q := by
  apply executed_correct_of_contrast_error (eps := 1) (by norm_num) copyWeights q 0 q
    (fun v => readout copyWeights.embedding (state 1 copyWeights q 0) v)
    (realLogit 1 copyWeights q 0) (δ := 2 / Real.sqrt 2) (B := 1 / 8) (η := 0)
    (fun v hv => (copy_gap q v hv).symm.le) (fun _ _ => by norm_num)
    (fun _ => by simp) ?_
  simp only [mul_zero, zero_mul, add_zero]
  have htwo : Real.sqrt 2 < 2 := by nlinarith [Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 2)]
  apply (lt_div_iff₀ (by positivity : 0 < Real.sqrt 2)).mpr
  linarith

end Transformer.Rotate90.ComplexProduct
