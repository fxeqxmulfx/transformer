/-
# DASH — repaired behavior on the earlier scaling counterexamples

arXiv:2602.02016v2, §3.2–3.5 and Appendix A. These results exercise
the actual guard on zero input, the failed CN unit endpoint and the
PI pool trapped in the smaller eigenspace of `diag(100,1)`.
-/

import Transformer.DASH.Section4_GuardedCn
import Transformer.DASH.Section4_GuardedNdb
import Transformer.DASH.Section4_GuardedChebyshev

open scoped BigOperators Matrix Topology
open Filter

noncomputable section

namespace Transformer.DASH

/-- The guard has no division by a zero scale on zero input and estimate.
Source: arXiv:2602.02016v2, §3.4, repaired normalization at zero. -/
theorem guardedScale_zero (n : ℕ) :
    guardedScale (0 : Matrix (Fin n) (Fin n) ℝ) 0 = 1 := by
  simp [guardedScale, certifiedSpectralBound, frobeniusNorm, Muon.squaredFrobenius,
    absoluteRowBound]

/-- Zero input is preserved by the actual normalization procedure.
This does not assert that its unregularized inverse root exists.
Source: arXiv:2602.02016v2, §3.4, corrected zero-input preprocessing. -/
theorem guardedNormalize_zero (n : ℕ) :
    guardedNormalize (0 : Matrix (Fin n) (Fin n) ℝ) 0 = 0 := by
  rw [guardedNormalize, guardedScale_zero]
  simp

/-- The inexpensive certificate is exactly 100 on the failed PI example.
Source: arXiv:2602.02016v2, §3.4–3.5, repair of the factor-two counterexample. -/
theorem unsafePowerMatrix_certifiedBound : certifiedSpectralBound unsafePowerMatrix = 100 := by
  have hrows : (fun i : Fin 2 => ∑ j : Fin 2, |unsafePowerMatrix i j|) =
      (![100, 1] : Fin 2 → ℝ) := by
    funext i
    fin_cases i <;> norm_num [unsafePowerMatrix, Fin.sum_univ_two]
  have hr : absoluteRowBound unsafePowerMatrix = 100 := by
    rw [absoluteRowBound, hrows]
    apply le_antisymm
    · apply (pi_norm_le_iff_of_nonneg (by norm_num : (0 : ℝ) ≤ 100)).mpr
      intro i
      fin_cases i <;> norm_num [Real.norm_eq_abs]
    · simpa only [Matrix.cons_val_zero, Real.norm_eq_abs,
        abs_of_nonneg (by norm_num : (0 : ℝ) ≤ 100)]
        using (norm_le_pi_norm (![100, 1] : Fin 2 → ℝ) 0)
  have hf : (100 : ℝ) ≤ frobeniusNorm unsafePowerMatrix := by
    apply (Real.le_sqrt (by norm_num) (Muon.squaredFrobenius_nonneg _)).mpr
    norm_num [Muon.squaredFrobenius, unsafePowerMatrix, Fin.sum_univ_two]
  rw [certifiedSpectralBound, hr]
  exact min_eq_right hf

/-- Even a permanently trapped PI pool now selects scale 200 instead of
the unsafe scale 2. This tests the complete pool/guard procedure at every PI
budget, not a supplied safe estimate.
Source: arXiv:2602.02016v2, §3.4–3.5, corrected deterministic safety guarantee. -/
theorem guarded_trapped_pool_scale {κ : Type} [Fintype κ] [Nonempty κ] (k : ℕ) :
    pooledGuardedScale unsafePowerMatrix (fun _ : κ => unsafePowerStart) k = 200 := by
  rw [pooledGuardedScale, pooled_power_counterexample, guardedScale_fallback]
  · norm_num [unsafePowerMatrix_certifiedBound]
  · norm_num [unsafePowerMatrix_certifiedBound]
  · norm_num [unsafePowerMatrix_certifiedBound]

/-- The unsafe PI example's normalized matrix is now `diag(1/2,1/200)`.
Both eigenvalues are strictly inside the common CN/NDB unit domain.
Source: arXiv:2602.02016v2, §3.4–3.5, actual corrected preprocessing. -/
theorem guarded_trapped_pool_matrix {κ : Type} [Fintype κ] [Nonempty κ] (k : ℕ) :
    guardedNormalize unsafePowerMatrix
      (pooledRayleigh unsafePowerMatrix (fun _ : κ => unsafePowerStart) k) =
        !![(1 / 2 : ℝ), 0; 0, (1 / 200 : ℝ)] := by
  have hscale := guarded_trapped_pool_scale (κ := κ) k
  change guardedScale unsafePowerMatrix
    (pooledRayleigh unsafePowerMatrix (fun _ : κ => unsafePowerStart) k) = 200 at hscale
  rw [guardedNormalize, hscale]
  ext i j
  fin_cases i <;> fin_cases j <;> norm_num [unsafePowerMatrix]

/-- The identity formerly collapsed to zero with the printed CN
comparison scale. The actual guarded CN algorithm instead converges to
its correct inverse root, for every positive order and every estimate.
Source: arXiv:2602.02016v2, §3.2–3.4, repaired unit-endpoint counterexample. -/
theorem guarded_cn_unit_convergence (p : ℕ) (r : ℝ) (hp : 0 < p) :
    Tendsto (guardedCnIterate p (1 : Matrix (Fin 1) (Fin 1) ℝ) r) atTop (𝓝 1) := by
  have hQ : Orthogonal (1 : Matrix (Fin 1) (Fin 1) ℝ) := by
    simp [Orthogonal, Muon.OrthonormalColumns]
  simpa only [spectralPower, Real.one_rpow, spectralMatrix_one _ hQ] using
    guardedCnIterate_convergence p 1 (fun _ : Fin 1 => 1) r hQ (by intro i; norm_num) hp

/-- The repaired unit CN example has an admissible order,
arXiv:2602.02016v2, §3.2–3.4. -/
example : 0 < (4 : ℕ) := by norm_num

/-- A zero requested sample budget is repaired to `d+1`, so even this
extreme input avoids the previously proved degree-`N` cosine alias.
Source: arXiv:2602.02016v2, Appendix A, corrected fitting recipe. -/
theorem guarded_zero_sample_budget (d : ℕ) : guardedSampleCount 0 d = d + 1 := by
  simp [guardedSampleCount]

end Transformer.DASH
