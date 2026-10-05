import Transformer.GPTMini.Sparsemax.SupportSegment

/-!
# A shared matrix segment with exact affine QKNorm scores

Derived upstream restriction for arXiv:1602.02068v2, §2.2 and §2.5,
and QKNorm and shared projections at `73f8a0b`. Keep projected endpoint
keys within the epsilon ball. Their convex segment remains there, where
the actual normalization `x / max (norm x) epsilon` is linear. Queries,
inputs, values and gain remain fixed. The same matrix segment applies to
every row and every example, without independent row parameters.

This explicitly changes the earlier unit-key path restriction to the
epsilon-clipped key regime. Unit keys satisfy it at epsilon one, but
ordinary epsilon `1e-6` requires keys within that much smaller ball.
Unrestricted normalized unit-key segments need not have affine scores.
The condition is proved along the segment from endpoint norm bounds.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex

/-- Convex combinations stay in a closed norm ball.
Source: the derived clipped-key condition for §2.5 of arXiv:1602.02068v2,
used before normalization in the matrix path at `73f8a0b`. -/
theorem norm_segment_le {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]
    (start stop : V) (bound t : ℝ) (hs : ‖start‖ ≤ bound) (he : ‖stop‖ ≤ bound)
    (ht : 0 ≤ t) (hu : t ≤ 1) : ‖(1 - t) • start + t • stop‖ ≤ bound := by
  have hc : 0 ≤ 1 - t := by linarith
  have hn := norm_add_le ((1 - t) • start) (t • stop)
  rw [norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs,
    abs_of_nonneg hc, abs_of_nonneg ht] at hn
  have hl := mul_le_mul_of_nonneg_left hs hc
  have hr := mul_le_mul_of_nonneg_left he ht
  nlinarith

/-- A genuine orthogonal pair inhabits the closed-ball segment hypotheses.
Source context: the derived clipped-key path at `73f8a0b`. -/
example : ‖(1 - (1 / 2 : ℝ)) • qkQueryExample + (1 / 2 : ℝ) • qkTransverseExample‖ ≤ 1 :=
  norm_segment_le _ _ _ _ (le_of_eq qkFrame_example.1) (le_of_eq qkFrame_example.2.1)
    (by norm_num) (by norm_num)

/-- The actual normalization is linear inside its epsilon ball.
Source: `QKNormScores.unit` at `73f8a0b`, with its maximum denominator;
the norm-ball restriction is an explicit new upstream constraint. -/
theorem normL2_of_norm_le {h : ℕ} (eps : ℝ) (key : EucSpace h) (hk : ‖key‖ ≤ eps) :
    normL2 eps key = (1 / eps) • key := by
  rw [normL2, max_eq_right hk]

/-- A unit key satisfies the clipping premise at epsilon one.
Source context: actual QKNorm at `73f8a0b`, using the existing unit frame. -/
example : normL2 1 qkQueryExample = (1 / (1 : ℝ)) • qkQueryExample :=
  normL2_of_norm_le _ _ (le_of_eq qkFrame_example.1)

/-- Actual normalized scores are affine on a clipped key segment.
Source: `QKNormScores.forward` at `73f8a0b` and the derived closed-ball
restriction preceding §2.5 of arXiv:1602.02068v2. -/
theorem score_key_segment {h : ℕ} (alpha eps : ℝ) (query start stop : EucSpace h) (t : ℝ)
    (hs : ‖start‖ ≤ eps) (he : ‖stop‖ ≤ eps) (ht : 0 ≤ t) (hu : t ≤ 1) :
    score alpha eps query ((1 - t) • start + t • stop) =
      (1 - t) * score alpha eps query start + t * score alpha eps query stop := by
  have hm := norm_segment_le start stop eps t hs he ht hu
  simp only [score, normL2_of_norm_le eps _ hm, normL2_of_norm_le eps _ hs,
    normL2_of_norm_le eps _ he, smul_add, smul_smul, inner_add_right, inner_smul_right]
  ring

/-- Orthogonal unit endpoints inhabit the exact normalized-score segment premises.
Source context: QKNorm at `73f8a0b`, epsilon one and fixed gain. -/
example : score 0 1 qkQueryExample
    ((1 - (1 / 2 : ℝ)) • qkQueryExample + (1 / 2 : ℝ) • qkTransverseExample) =
      (1 - (1 / 2 : ℝ)) * score 0 1 qkQueryExample qkQueryExample +
        (1 / 2 : ℝ) * score 0 1 qkQueryExample qkTransverseExample :=
  score_key_segment _ _ _ _ _ _ (le_of_eq qkFrame_example.1) (le_of_eq qkFrame_example.2.1)
    (by norm_num) (by norm_num)

/-- One actual shared key-matrix segment, with the query matrix fixed.
Source: the shared Q/K matrices at `73f8a0b`; no row-specific parameters
are introduced in the derived support-preserving path for §2.5. -/
def keyMatrixSegment {F h : ℕ} (queries start stop : Fin F → EucSpace h) (t : ℝ) :
    (Fin F → EucSpace h) × (Fin F → EucSpace h) :=
  (queries, (1 - t) • start + t • stop)

/-- The matrix segment starts at the given joint parameter point.
Source: the actual affine shared-matrix path preceding §2.5. -/
theorem keyMatrixSegment_zero {F h : ℕ} (queries start stop : Fin F → EucSpace h) :
    keyMatrixSegment queries start stop 0 = (queries, start) := by
  simp only [keyMatrixSegment, sub_zero, one_smul, zero_smul, add_zero]

/-- The matrix segment ends at the given shared correction matrix.
Source: the same actual affine path preceding §2.5. -/
theorem keyMatrixSegment_one {F h : ℕ} (queries start stop : Fin F → EucSpace h) :
    keyMatrixSegment queries start stop 1 = (queries, stop) := by
  simp only [keyMatrixSegment, sub_self, zero_smul, one_smul, zero_add]

/-- Projected keys also follow this same affine segment.
Source: the finite linear projections at `73f8a0b`, before QKNorm. -/
theorem projectionEvaluation_key_segment {F T h : ℕ} (inputs : Fin T → (Fin F → ℝ))
    (start stop : Fin F → EucSpace h) (t : ℝ) :
    projectionEvaluation inputs ((1 - t) • start + t • stop) =
      (1 - t) • projectionEvaluation inputs start + t • projectionEvaluation inputs stop := by
  rw [map_add, map_smul, map_smul]

/-- Actual shared projected scores are affine when endpoint projected keys are clipped.
Source: shared projections and QKNorm at `73f8a0b`, through the derived
closed-ball restriction for §2.2 and §2.5 of arXiv:1602.02068v2. -/
theorem projectedQKScores_key_segment {F T h : ℕ} (alpha eps : ℝ)
    (queries start stop : Fin F → EucSpace h) (inputs : Fin T → (Fin F → ℝ)) (i : Fin T) (t : ℝ)
    (hs : ∀ n, ‖projectionEvaluation inputs start n‖ ≤ eps)
    (he : ∀ n, ‖projectionEvaluation inputs stop n‖ ≤ eps) (ht : 0 ≤ t) (hu : t ≤ 1) :
    projectedQKScores alpha eps queries ((1 - t) • start + t • stop) inputs i =
      (1 - t) • projectedQKScores alpha eps queries start inputs i +
        t • projectedQKScores alpha eps queries stop inputs i := by
  rw [projectedQKScores_evaluation, projectionEvaluation_key_segment,
    projectedQKScores_evaluation, projectedQKScores_evaluation]
  funext n
  simp only [Pi.add_apply, Pi.smul_apply, smul_eq_mul]
  exact score_key_segment alpha eps _ _ _ t (hs n) (he n) ht hu

/-- Shared projections on repeated inputs inhabit both endpoint norm bounds.
Source context: the derived §2.5 clipped-key path; independence is unnecessary. -/
example : projectedQKScores 0 1 (fun _ : Fin 1 => qkQueryExample)
    ((1 - (1 / 2 : ℝ)) • (fun _ : Fin 1 => qkQueryExample) +
      (1 / 2 : ℝ) • (fun _ : Fin 1 => qkTransverseExample))
    (fun _ : Fin 2 => basis (0 : Fin 1)) 1 =
    (1 - (1 / 2 : ℝ)) • projectedQKScores 0 1 (fun _ : Fin 1 => qkQueryExample)
      (fun _ : Fin 1 => qkQueryExample) (fun _ : Fin 2 => basis (0 : Fin 1)) 1 +
    (1 / 2 : ℝ) • projectedQKScores 0 1 (fun _ : Fin 1 => qkQueryExample)
      (fun _ : Fin 1 => qkTransverseExample) (fun _ : Fin 2 => basis (0 : Fin 1)) 1 := by
  apply projectedQKScores_key_segment
  · intro n
    rw [projectionEvaluation_apply, projection_basis_column]
    exact le_of_eq qkFrame_example.1
  · intro n
    rw [projectionEvaluation_apply, projection_basis_column]
    exact le_of_eq qkFrame_example.2.1
  · norm_num
  · norm_num

/-- The shared matrix path is differentiable at every real parameter.
Source: the actual affine interpolation of matrices at `73f8a0b`;
no differentiability of sparsemax is asserted here. -/
theorem keyMatrixSegment_differentiableAt {F h : ℕ}
    (queries start stop : Fin F → EucSpace h) (t : ℝ) :
    DifferentiableAt ℝ (keyMatrixSegment queries start stop) t := by
  unfold keyMatrixSegment
  fun_prop

end Transformer.GPTMini.Sparsemax
