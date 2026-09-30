/-
# DASH — automatic domain and sample repair for batched Chebyshev powers

arXiv:2602.02016v2, §3.4, §4 and Appendix A. The complete solver
chooses a positive spectral upper bound, fits at the corrected `ε/c`,
rescales by `c^exponent`, and raises an undersized sample count above
the requested degree. Accuracy is proved for the actual fitted array.
-/

import Transformer.DASH.Section4_BatchedFitConvergence
import Transformer.DASH.Section3_GuardedScaling

open scoped Matrix Topology
open Filter

noncomputable section

namespace Transformer.DASH

variable {ι : Type} {n : ℕ}

/-- Repair the printed fixed-sample recipe when its requested degree
is too large. Source: arXiv:2602.02016v2, Appendix A, corrected cosine fitting. -/
def guardedSampleCount (N d : ℕ) : ℕ := max N (d + 1)

/-- The actual sample choice always avoids the degree-`N` alias.
Source: arXiv:2602.02016v2, Appendix A, repaired fitting sample count. -/
theorem degree_lt_guardedSampleCount (N d : ℕ) : d < guardedSampleCount N d := by
  unfold guardedSampleCount
  omega

/-- Corrected end-to-end fitted power algorithm for an entire bucket:
automatic scaling and sample repair surround the already corrected fit
and optimized Clenshaw evaluation.
Source: arXiv:2602.02016v2, §3.4, §4 and Appendix A, repaired power pipeline. -/
def countedBatchGuardedPower (A : ι → Matrix (Fin n) (Fin n) ℝ) (r : ι → ℝ)
    (ε exponent : ℝ) (N d : ℕ) : StateM ℕ (ι → Matrix (Fin n) (Fin n) ℝ) :=
  countedBatchFittedPower A (fun j => guardedScale (A j) (r j)) ε exponent
    (guardedSampleCount N d) d

/-- Each actual output is its corrected scalar fit evaluated on the
normalized matrix, with original-scale rescaling. Sample repair does not
change the `max(d-1,0)` batched Clenshaw product count. Preprocessing and
coefficient fitting are outside this count, as in the manuscript.
Source: arXiv:2602.02016v2, §4 and Appendix A, optimized fitted powers. -/
theorem countedBatchGuardedPower_correct (A : ι → Matrix (Fin n) (Fin n) ℝ)
    (r : ι → ℝ) (ε exponent : ℝ) (N d count : ℕ) :
    (countedBatchGuardedPower A r ε exponent N d).run count =
      (fun j => guardedScale (A j) (r j) ^ exponent •
        (optimizedClenshaw (chebMatrixArgument (A j) (guardedScale (A j) (r j)))
          (chebFit (fun x : ℝ => x ^ exponent) (ε / guardedScale (A j) (r j))
            (1 + ε / guardedScale (A j) (r j)) (guardedSampleCount N d) d)).1,
        count + (d - 1)) :=
  countedBatchFittedPower_correct A (fun j => guardedScale (A j) (r j)) ε exponent
    (guardedSampleCount N d) d count

/-- For every finite positive-semidefinite bucket, the complete algorithm
approximates the original regularized powers uniformly to arbitrary positive
tolerance. No external scale, PI accuracy, coefficient-error assumption or
sample-count restriction is imposed: both guards are part of the algorithm.
Source: arXiv:2602.02016v2, §3.4, §4 and Appendix A, corrected fitted accuracy. -/
theorem countedBatchGuardedPower_uniform_accuracy [Finite ι]
    (Q : ι → Matrix (Fin n) (Fin n) ℝ) (s : ι → Fin n → ℝ) (r : ι → ℝ)
    (ε exponent δ : ℝ) (hQ : ∀ j, Orthogonal (Q j))
    (hε : 0 < ε) (hs : ∀ j i, 0 ≤ s j i) (hδ : 0 < δ) :
    ∃ D : ℕ, ∀ d : ℕ, D ≤ d → ∀ N : ℕ, ∀ j : ι, frobeniusNorm
      (((countedBatchGuardedPower (fun j => spectralMatrix (Q j) (s j)) r
          ε exponent N d).run 0).1 j -
        spectralPower (Q j) (fun i => s j i + ε) exponent) < δ := by
  obtain ⟨D, hD⟩ := countedBatchFittedPower_uniform_accuracy Q s
    (fun j => guardedScale (spectralMatrix (Q j) (s j)) (r j)) ε exponent δ hQ
    (fun j => guardedScale_pos _ _) hε hs
    (fun j i => (eigenvalue_lt_guardedScale (Q j) (s j) (r j) (hQ j) i).le) hδ
  refine ⟨D, ?_⟩
  intro d hd N j
  exact hD d hd (guardedSampleCount N d) (degree_lt_guardedSampleCount N d) j

/-- The automatic fitted solver has admissible nonempty buckets,
arXiv:2602.02016v2, §3.4, §4 and Appendix A. -/
example : (∀ j : Fin 2, Orthogonal ((fun _ : Fin 2 =>
    (1 : Matrix (Fin 1) (Fin 1) ℝ)) j)) ∧ (0 : ℝ) < 1 ∧
    (∀ j : Fin 2, ∀ i : Fin 1, (0 : ℝ) ≤ (fun (_ : Fin 2) (_ : Fin 1) => (1 : ℝ)) j i) ∧
    (0 : ℝ) < 1 := by
  norm_num [Orthogonal, Muon.OrthonormalColumns]

/-- Complete multi-PI, guarded fitting and batched Clenshaw power solver.
Source: arXiv:2602.02016v2, §3.4–3.5, §4 and Appendix A, repaired PI scaling. -/
def countedBatchPiPower {κ : Type} [Fintype κ] [Nonempty κ]
    (A : ι → Matrix (Fin n) (Fin n) ℝ) (starts : ι → κ → Fin n → ℝ)
    (piSteps : ℕ) (ε exponent : ℝ) (N d : ℕ) : StateM ℕ (ι → Matrix (Fin n) (Fin n) ℝ) :=
  countedBatchGuardedPower A (fun j => pooledRayleigh (A j) (starts j) piSteps)
    ε exponent N d

end Transformer.DASH
