/-
# DASH — fitted coefficients and convergence of the batched power solver

arXiv:2602.02016v2, §4 and Appendix A. Each block has its own
normalization, fitted coefficient vector and output rescaling.
For a finite bucket, one sufficiently large degree works for every
block and every sample count greater than that degree.
-/

import Transformer.DASH.Section4_BatchedClenshaw
import Transformer.DASH.SectionA_MatrixFitConvergence
import Mathlib.Order.Filter.Finite

open scoped Matrix Topology
open Filter

noncomputable section

namespace Transformer.DASH

variable {ι : Type} {n : ℕ}

/-- Fit the common number of coefficient slots at each block's corrected
regularization `ε/c`. Values may differ across the bucket.
Source: arXiv:2602.02016v2, §4 and Appendix A, fitting and normalized output;
the scaling correction is documented in the single-matrix modules. -/
def batchPowerFit (c : ι → ℝ) (ε exponent : ℝ) (N d : ℕ) : List (ι → ℝ) :=
  List.ofFn (fun k : Fin (d + 1) => fun j =>
    chebFitCoefficient (fun x : ℝ => x ^ exponent) (ε / c j) (1 + ε / c j) N k.val)

/-- A batch entry's coefficient list is exactly the actual cosine fit,
not an abstract list satisfying a presumed error bound.
Source: arXiv:2602.02016v2, §4 and Appendix A, coefficient fitting. -/
theorem batchPowerFit_map (c : ι → ℝ) (ε exponent : ℝ) (N d : ℕ) (j : ι) :
    (batchPowerFit c ε exponent N d).map (fun a => a j) =
      chebFit (fun x : ℝ => x ^ exponent) (ε / c j) (1 + ε / c j) N d := by
  simp [batchPowerFit, chebFit, List.map_ofFn, Function.comp_def]

/-- Fit, normalize, evaluate by counted `bmm`, and undo input scaling.
This targets `(A+εI)^exponent`; the printed algorithm omits the adjustment
of `ε` and the final scaling needed for that original-scale target.
Source: arXiv:2602.02016v2, §4 and Appendix A, corrected matrix Clenshaw. -/
def countedBatchFittedPower (A : ι → Matrix (Fin n) (Fin n) ℝ) (c : ι → ℝ)
    (ε exponent : ℝ) (N d : ℕ) : StateM ℕ (ι → Matrix (Fin n) (Fin n) ℝ) := do
  let root ← countedBatchClenshaw (fun j => chebMatrixArgument (A j) (c j))
    (batchPowerFit c ε exponent N d)
  pure (fun j => c j ^ exponent • root j)

/-- The complete fitted batch solver equals the corrected single-matrix
algorithm for each input and uses `max(d-1,0)` batched products. Fitting,
argument construction and scalar operations are outside the source's count.
Source: arXiv:2602.02016v2, §4 and Appendix A, optimized product count. -/
theorem countedBatchFittedPower_correct (A : ι → Matrix (Fin n) (Fin n) ℝ)
    (c : ι → ℝ) (ε exponent : ℝ) (N d count : ℕ) :
    (countedBatchFittedPower A c ε exponent N d).run count =
      (fun j => c j ^ exponent •
        (optimizedClenshaw (chebMatrixArgument (A j) (c j))
          (chebFit (fun x : ℝ => x ^ exponent) (ε / c j) (1 + ε / c j) N d)).1,
        count + (d - 1)) := by
  unfold countedBatchFittedPower
  rw [StateT.run_bind, countedBatchClenshaw_correct]
  change (fun j => c j ^ exponent •
    (optimizedClenshaw (chebMatrixArgument (A j) (c j))
      ((batchPowerFit c ε exponent N d).map (fun a => a j))).1,
    count + ((batchPowerFit c ε exponent N d).length - 2)) = _
  apply Prod.ext
  · funext j
    dsimp only
    rw [batchPowerFit_map]
  · have hlength : (batchPowerFit c ε exponent N d).length = d + 1 := by
      simp [batchPowerFit]
    rw [hlength]
    omega

/-- The actual fitted batched solver approximates every original-scale
regularized matrix power to any positive tolerance with one common degree
threshold for a finite bucket. No accuracy hypothesis about its coefficients
or intermediate Clenshaw states is assumed.
Source: arXiv:2602.02016v2, §4 and Appendix A, stacked fitted powers. -/
theorem countedBatchFittedPower_uniform_accuracy [Finite ι]
    (Q : ι → Matrix (Fin n) (Fin n) ℝ) (s : ι → Fin n → ℝ) (c : ι → ℝ)
    (ε exponent δ : ℝ) (hQ : ∀ j, Orthogonal (Q j)) (hc : ∀ j, 0 < c j)
    (hε : 0 < ε) (hs : ∀ j i, 0 ≤ s j i) (hupper : ∀ j i, s j i ≤ c j)
    (hδ : 0 < δ) :
    ∃ D : ℕ, ∀ d : ℕ, D ≤ d → ∀ N : ℕ, d < N → ∀ j : ι, frobeniusNorm
      (((countedBatchFittedPower (fun j => spectralMatrix (Q j) (s j)) c
          ε exponent N d).run 0).1 j -
        spectralPower (Q j) (fun i => s j i + ε) exponent) < δ := by
  have hper : ∀ j, ∀ᶠ d : ℕ in atTop, ∀ N : ℕ, d < N → frobeniusNorm
      (c j ^ exponent •
        (optimizedClenshaw (chebMatrixArgument (spectralMatrix (Q j) (s j)) (c j))
          (chebFit (fun x : ℝ => x ^ exponent) (ε / c j) (1 + ε / c j) N d)).1 -
        spectralPower (Q j) (fun i => s j i + ε) exponent) < δ := by
    intro j
    obtain ⟨D, hD⟩ := rescaledFittedClenshaw_regularized_uniform_accuracy
      (Q j) (s j) (c j) ε exponent δ (hQ j) (hc j) hε (hs j) (hupper j) hδ
    exact eventually_atTop.mpr ⟨D, hD⟩
  obtain ⟨D, hD⟩ := eventually_atTop.mp (Filter.eventually_all.mpr hper)
  refine ⟨D, ?_⟩
  intro d hd N hN j
  simpa only [countedBatchFittedPower_correct] using hD d hd j N hN

/-- A two-block bucket satisfies the uniform fitted-accuracy assumptions,
arXiv:2602.02016v2, §4 and Appendix A. -/
example : (∀ j : Fin 2, Orthogonal ((fun _ : Fin 2 =>
    (1 : Matrix (Fin 1) (Fin 1) ℝ)) j)) ∧
    (∀ j : Fin 2, (0 : ℝ) < (fun _ : Fin 2 => (2 : ℝ)) j) ∧ (0 : ℝ) < 1 ∧
    (∀ j : Fin 2, ∀ i : Fin 1, (0 : ℝ) ≤ (fun (_ : Fin 2) (_ : Fin 1) => (1 : ℝ)) j i) ∧
    (∀ j : Fin 2, ∀ i : Fin 1,
      (fun (_ : Fin 2) (_ : Fin 1) => (1 : ℝ)) j i ≤ (fun _ : Fin 2 => (2 : ℝ)) j) ∧
    (0 : ℝ) < 1 := by
  norm_num [Orthogonal, Muon.OrthonormalColumns]

end Transformer.DASH
