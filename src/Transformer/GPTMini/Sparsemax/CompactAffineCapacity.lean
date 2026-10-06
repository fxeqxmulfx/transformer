import Transformer.GPTMini.Sparsemax.CompactAffineMinimum
import Mathlib.LinearAlgebra.Dimension.Constructions

/-!
# Capacity cost and genuine learned witnesses for compact values

New architectural tradeoff following arXiv:1602.02068v2, §2.5.
A coefficient-affine decoder fitting every arbitrary scalar answer table
needs at least as many scalar coefficients as observed slots. Constant
learned value storage therefore requires a restricted response family in
this affine-forward framework. The two-feature construction is exact on
its family and has ordinary squared error at least 2/3 on answers (0,1,0).
The unique best compact point attains that positive error with genuine
nonidentity sparsemax and generated original common values.

Both physical Q and K change between the earlier two nonconstant compact
fits; their actual sparse supports also change. Common values are genuinely
inverse-adjusted. This capacity example states the loss of arbitrary
memorization explicitly, rather than disguising a table in a fixed basis.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- A fixed linear value-response decoder memorizing all scalar answers needs at least P coefficients.
Source: finite-dimensional capacity obstruction in the affine-forward setting after sparsemax §2.5. -/
theorem compactLinearResponse_capacity {P K : ℕ} (f : (Fin K → ℝ) →ₗ[ℝ] (Fin P → ℝ))
    (hf : Function.Surjective f) : P ≤ K := by
  have h := LinearMap.finrank_le_finrank_of_surjective hf
  rw [Module.finrank_pi, Module.finrank_pi, Fintype.card_fin, Fintype.card_fin] at h
  exact h

/-- A genuine complete three-coefficient decoder inhabits the universal-fitting premise. -/
example : (3 : ℕ) ≤ 3 :=
  compactLinearResponse_capacity (LinearMap.id : (Fin 3 → ℝ) →ₗ[ℝ] (Fin 3 → ℝ)) (fun y => ⟨y, rfl⟩)

/-- Adding a fixed output offset does not evade the coefficient-count obstruction.
Source: the same finite affine-forward capacity argument following arXiv:1602.02068v2, §2.5. -/
theorem compactAffineResponse_capacity {P K : ℕ} (offset : Fin P → ℝ)
    (f : (Fin K → ℝ) →ₗ[ℝ] (Fin P → ℝ)) (hf : Function.Surjective (fun x => offset + f x)) : P ≤ K := by
  apply compactLinearResponse_capacity f
  intro y
  obtain ⟨x, hx⟩ := hf (offset + y)
  exact ⟨x, add_left_cancel hx⟩

/-- A nonconstant fixed offset and an identity decoder inhabit the affine-capacity premises. -/
example : (3 : ℕ) ≤ 3 := by
  apply compactAffineResponse_capacity (fun i : Fin 3 => (i.val : ℝ)) LinearMap.id
  intro y
  refine ⟨y - (fun i : Fin 3 => (i.val : ℝ)), ?_⟩
  ext i
  simp only [LinearMap.id_apply, Pi.add_apply, Pi.sub_apply]
  ring

/-- Fewer coefficients cannot retain arbitrary exact answer fitting through any affine decoder.
Source: the derived fixed-parameter capacity obstruction after sparsemax §2.5. -/
theorem compactAffineResponse_not_surjective {P K : ℕ} (offset : Fin P → ℝ)
    (f : (Fin K → ℝ) →ₗ[ℝ] (Fin P → ℝ)) (hK : K < P) :
    ¬ Function.Surjective (fun x => offset + f x) := by
  intro hf
  have h := compactAffineResponse_capacity offset f hf
  omega

/-- Every two-coefficient affine decoder fails universal fitting on three slots. -/
example (offset : Fin 3 → ℝ) (f : (Fin 2 → ℝ) →ₗ[ℝ] (Fin 3 → ℝ)) :
    ¬ Function.Surjective (fun x => offset + f x) :=
  compactAffineResponse_not_surjective offset f (by norm_num)

/-- An explicit nonlinear three-slot answer table leaves a sharp positive error in the generated family.
Source: a capacity counterexample to arbitrary compact fitting after sparsemax §2.5. -/
theorem compactAffine_three_loss_floor (W : Matrix (Fin 2) (Fin 1) ℝ) :
    (2 / 3 : ℝ) ≤ matrixOutputError !![0; 1; 0] (affineValueOutput 1 W) := by
  norm_num [matrixOutputError, affineValueOutput, affineValuePosition, Fin.sum_univ_succ]
  nlinarith [sq_nonneg (W 0 0 - 1 / 3), sq_nonneg (W 1 0)]

/-- A concrete positive-error compact point satisfies the complete learned domain.
Source: the attained finite capacity counterexample after arXiv:1602.02068v2, §2.5. -/
theorem compactAffine_nonfit_mem :
    (!![1 / 3; 0] : Matrix (Fin 2) (Fin 1) ℝ) ∈
      compactAffineMemoryDomain 1 1 (3 / 4) (1 / 16) (1 / 16) 0 := by
  constructor
  · intro k d
    fin_cases k <;> fin_cases d <;> norm_num
  · norm_num [compactAffineMemoryScalar]

/-- The best compact point genuinely attains the positive ordinary error, with no route labels.
Source: exact compact sparsemax/common-value forward after Eq. (1) and the capacity counterexample. -/
theorem compactAffine_nonfit_loss :
    compactAffineSquaredError (1 / 16) (1 / 16) 0 !![0; 1; 0] !![1 / 3; 0] = 2 / 3 := by
  rw [compactAffineSquaredError_eq 1 (3 / 4) (1 / 16) (1 / 16) 0 _ _ (by norm_num) compactAffine_nonfit_mem]
  norm_num [matrixOutputError, affineValueOutput, affineValuePosition, Fin.sum_univ_succ]

/-- The positive-error compact witness is an actual attained ordinary minimum.
Source: the sharp generated-family loss floor following sparsemax §2.5. -/
theorem compactAffine_nonfit_min :
    IsMinOn (compactAffineSquaredError (1 / 16) (1 / 16) 0 !![0; 1; 0])
      (compactAffineMemoryDomain 1 1 (3 / 4) (1 / 16) (1 / 16) 0) !![1 / 3; 0] := by
  intro W hW
  change compactAffineSquaredError (1 / 16) (1 / 16) 0 !![0; 1; 0] !![1 / 3; 0] ≤
    compactAffineSquaredError (1 / 16) (1 / 16) 0 !![0; 1; 0] W
  rw [compactAffine_nonfit_loss,
    compactAffineSquaredError_eq 1 (3 / 4) (1 / 16) (1 / 16) 0 _ W (by norm_num) hW]
  exact compactAffine_three_loss_floor W

/-- Opposite learned slopes really change physical width-three queries, rather than freezing Q.
Source: genuine compact learned embeddings before arXiv:1602.02068v2, Eq. (1). -/
theorem compactAffine_query_changes :
    periodicMemoryQuery (affineValueEdges 2 (affineValueNormalizedMix 2
      (compactAffineMemoryScalar (1 / 16) (1 / 16) 0 (compactAffineExampleCoefficients 1)))) (0 : Fin 4) ≠
    periodicMemoryQuery (affineValueEdges 2 (affineValueNormalizedMix 2
      (compactAffineMemoryScalar (1 / 16) (1 / 16) 0 (compactAffineExampleCoefficients 0)))) (0 : Fin 4) := by
  intro h
  have he := congrFun h 1
  norm_num [periodicMemoryQuery, localMemoryNeighbours, localSlotClamp, periodicMemoryClass,
    periodicMemoryScale, localIncidentWeight, affineValueEdges, affineValueNormalizedMix,
    compactAffineMemoryScalar, compactAffineExampleCoefficients, Fin.sum_univ_succ,
    localMemoryCore_scores_apply, Fintype.sum_option, localMemoryWeights,
    localMemoryPermutation, Equiv.swap_apply_def] at he

/-- The same compact learned slope change alters physical keys too.
Source: generated incident-mass key scales before sparsemax Eq. (1). -/
theorem compactAffine_key_changes :
    periodicMemoryKey (affineValueEdges 2 (affineValueNormalizedMix 2
      (compactAffineMemoryScalar (1 / 16) (1 / 16) 0 (compactAffineExampleCoefficients 1)))) (0 : Fin 4) ≠
    periodicMemoryKey (affineValueEdges 2 (affineValueNormalizedMix 2
      (compactAffineMemoryScalar (1 / 16) (1 / 16) 0 (compactAffineExampleCoefficients 0)))) (0 : Fin 4) := by
  intro h
  have he := congrFun h 0
  norm_num [periodicMemoryKey, periodicMemoryScale, periodicMemoryClass, localIncidentWeight,
    affineValueEdges, affineValueNormalizedMix, compactAffineMemoryScalar,
    compactAffineExampleCoefficients, Fin.sum_univ_succ] at he

/-- The compact learned points change actual sparsemax support, not just raw QK coordinates.
Source: the genuine probability-score projection following arXiv:1602.02068v2, Eq. (1). -/
theorem compactAffine_support_changes :
    periodicMemoryAttention (affineValueEdges 2 (affineValueNormalizedMix 2
      (compactAffineMemoryScalar (1 / 16) (1 / 16) 0 (compactAffineExampleCoefficients 1)))) (0 : Fin 4) 1 = 3 / 128 ∧
    periodicMemoryAttention (affineValueEdges 2 (affineValueNormalizedMix 2
      (compactAffineMemoryScalar (1 / 16) (1 / 16) 0 (compactAffineExampleCoefficients 0)))) (0 : Fin 4) 1 = 0 := by
  have ht (edge : Fin 2) := compactAffineMemoryEdges_mem 2 1 (3 / 4) (1 / 16) (1 / 16) 0
    (compactAffineExampleCoefficients edge) (compactAffineExample_mem edge)
  constructor
  · rw [periodicMemoryAttention_normalized (3 / 4) _ (by norm_num) (ht 1)]
    norm_num [localMemoryCore_scores_apply, Fintype.sum_option, localMemoryWeights,
      localMemoryPermutation, Equiv.swap_apply_def, affineValueEdges, affineValueNormalizedMix,
      compactAffineMemoryScalar, compactAffineExampleCoefficients, Fin.sum_univ_succ]
  · rw [periodicMemoryAttention_normalized (3 / 4) _ (by norm_num) (ht 0)]
    norm_num [localMemoryCore_scores_apply, Fintype.sum_option, localMemoryWeights,
      localMemoryPermutation, Equiv.swap_apply_def, affineValueEdges, affineValueNormalizedMix,
      compactAffineMemoryScalar, compactAffineExampleCoefficients, Fin.sum_univ_succ]

/-- Original compact values are nonconstant and genuinely differ from the learned output coefficients.
Source: the proved compact nonsingular decoder following sparsemax Eq. (1). -/
example : affineValueOutput 2 (affineValueCoefficients (1 / 128) (compactAffineExampleCoefficients 1)) 0 0 = -64 / 63 ∧
    affineValueOutput 2 (affineValueCoefficients (1 / 128) (compactAffineExampleCoefficients 1)) 3 0 = 64 / 63 := by
  norm_num [affineValueOutput, affineValuePosition, affineValueCoefficients, compactAffineExampleCoefficients]

end Transformer.GPTMini.Sparsemax
