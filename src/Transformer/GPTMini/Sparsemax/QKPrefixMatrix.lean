import Transformer.GPTMini.Sparsemax.LongContextQK

/-!
# Accessible Q/K anchor charts without full input independence

Derived from arXiv:1602.02068v2, §2.5, and actual shared projections
and QKNorm at `73f8a0b`. A partial decoder separates anchor inputs and
kills ordinary inputs. It lifts the varying anchor keys into a smooth
path through the actual current matrix, preserving every ordinary key.
The full projected key row exactly equals the anchored unit-key family
along the path, even when ordinary inputs repeat or outnumber input width.

The decoder conditions are explicit and have a proved implementation in
`PrefixInputs`, with arbitrary ordinary embeddings. The current projected
keys must belong to the stated family; this remains an architectural
restriction. Queries and ordinary keys are fixed on this anchor path.
The statements concern a single unrotated readout before XSA and output
projection. They do not assume independent keys for all ordinary tokens.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.GPTMini.Convex

/-- Ordinary keys do not depend on the learned anchor coordinates.
Source: the derived QKNorm realization of §2.5 of arXiv:1602.02068v2;
the gain depends on ordinary scores, which are fixed on this path. -/
theorem qkAnchoredKeys_ordinary_parameters_eq {h A N : ℕ} (query transverse : EucSpace h)
    (cap : ℝ) (initial parameters : Fin A → ℝ) (ordinary : Fin N → ℝ) (n : Fin N) :
    qkAnchoredKeys query transverse cap initial ordinary (Fin.natAdd A n) =
      qkAnchoredKeys query transverse cap parameters ordinary (Fin.natAdd A n) := by
  simp only [qkAnchoredKeys, anchoredScores, Fin.addCases_right]

/-- Lift only desired anchor keys as an affine change from the actual current matrix.
Source: the derived partial accessibility construction for §2.5 of
arXiv:1602.02068v2, using the shared projection at `73f8a0b`. -/
def qkPrefixMatrix {F h A N : ℕ} (inputs : Fin (A + N) → (Fin F → ℝ))
    (decoder : (Fin F → ℝ) →L[ℝ] (Fin A → ℝ)) (columns : Fin F → EucSpace h)
    (query transverse : EucSpace h) (cap : ℝ) (parameters : Fin A → ℝ) (ordinary : Fin N → ℝ) :
    Fin F → EucSpace h :=
  inputProjectionUpdate (fun a : Fin A => inputs (Fin.castAdd N a)) decoder columns
    (fun a => qkAnchoredKeys query transverse cap parameters ordinary (Fin.castAdd N a))

/-- At the current anchor assignment the path starts at the actual matrix.
Source: the derived partial matrix lift for §2.5 of arXiv:1602.02068v2;
no arbitrary component of the current matrix is discarded. -/
theorem qkPrefixMatrix_center {F h A N : ℕ} (inputs : Fin (A + N) → (Fin F → ℝ))
    (decoder : (Fin F → ℝ) →L[ℝ] (Fin A → ℝ)) (columns : Fin F → EucSpace h)
    (query transverse : EucSpace h) (cap : ℝ) (parameters : Fin A → ℝ) (ordinary : Fin N → ℝ)
    (hkeys : projectionEvaluation inputs columns = qkAnchoredKeys query transverse cap parameters ordinary) :
    qkPrefixMatrix inputs decoder columns query transverse cap parameters ordinary = columns := by
  have he : (fun a => qkAnchoredKeys query transverse cap parameters ordinary (Fin.castAdd N a)) =
      projectionEvaluation (fun a : Fin A => inputs (Fin.castAdd N a)) columns := by
    funext a
    have hk := congrFun hkeys (Fin.castAdd N a)
    rw [projectionEvaluation_apply] at hk
    rw [projectionEvaluation_apply]
    exact hk.symm
  rw [qkPrefixMatrix, he, inputProjectionUpdate_center]

/-- The center premise holds on a dependent context with one hundred ordinary tokens.
Source context: arXiv:1602.02068v2, §2.5, partial matrix accessibility. -/
example : qkPrefixMatrix (longQKInputs 100) (prefixInputDecoder 2 1) (longQKKeys 100 (fun _ => 0))
    qkQueryExample qkTransverseExample (1 / 8) (fun _ => 0) (fun _ : Fin 100 => 0) =
      longQKKeys 100 (fun _ => 0) :=
  qkPrefixMatrix_center _ _ _ _ _ _ _ _ (longQKKeys_row _ _)

/-- The lifted path realizes the whole key family while changing only accessible anchors.
Source: the derived partial decoder restriction for §2.5 of arXiv:1602.02068v2,
with actual shared key projection at `73f8a0b`. -/
theorem qkPrefixMatrix_row {F h A N : ℕ} (inputs : Fin (A + N) → (Fin F → ℝ))
    (decoder : (Fin F → ℝ) →L[ℝ] (Fin A → ℝ)) (columns : Fin F → EucSpace h)
    (query transverse : EucSpace h) (cap : ℝ) (initial parameters : Fin A → ℝ) (ordinary : Fin N → ℝ)
    (hd : ∀ a, decoder (inputs (Fin.castAdd N a)) = Transformer.ConvexRecall.basis a)
    (hzero : ∀ n, decoder (inputs (Fin.natAdd A n)) = 0)
    (hkeys : projectionEvaluation inputs columns = qkAnchoredKeys query transverse cap initial ordinary) :
    projectionEvaluation inputs (qkPrefixMatrix inputs decoder columns query transverse cap parameters ordinary) =
      qkAnchoredKeys query transverse cap parameters ordinary := by
  funext j
  refine Fin.addCases ?_ ?_ j
  · intro a
    have hr := congrFun (inputProjectionUpdate_realizes
      (fun a : Fin A => inputs (Fin.castAdd N a)) decoder hd columns
      (fun a => qkAnchoredKeys query transverse cap parameters ordinary (Fin.castAdd N a))) a
    rw [projectionEvaluation_apply] at hr
    rw [projectionEvaluation_apply, qkPrefixMatrix]
    exact hr
  · intro n
    rw [projectionEvaluation_apply, qkPrefixMatrix,
      inputProjectionUpdate_preserves_kernel _ _ _ _ _ (hzero n)]
    have hk := congrFun hkeys (Fin.natAdd A n)
    rw [projectionEvaluation_apply] at hk
    rw [hk]
    exact qkAnchoredKeys_ordinary_parameters_eq query transverse cap initial parameters ordinary n

/-- All partial-decoder and current-key premises are inhabited beyond input width.
Source context: arXiv:1602.02068v2, §2.5, dependent ordinary inputs. -/
example (parameters : Fin 2 → ℝ) : projectionEvaluation (longQKInputs 100)
    (qkPrefixMatrix (longQKInputs 100) (prefixInputDecoder 2 1) (longQKKeys 100 (fun _ => 0))
      qkQueryExample qkTransverseExample (1 / 8) parameters (fun _ : Fin 100 => 0)) =
      qkAnchoredKeys qkQueryExample qkTransverseExample (1 / 8) parameters (fun _ : Fin 100 => 0) :=
  qkPrefixMatrix_row _ _ _ _ _ _ (fun _ => 0) _ _
    (longQKInput_anchor 100) (longQKInput_ordinary 100) (longQKKeys_row 100 _)

/-- The lifted shared matrix is differentiable at every finite anchor assignment.
Source: the genuine key chart derived from §2.5 of arXiv:1602.02068v2,
through a continuous linear lift and an affine matrix update at `73f8a0b`. -/
theorem qkPrefixMatrix_differentiableAt {F h A N : ℕ} (inputs : Fin (A + N) → (Fin F → ℝ))
    (decoder : (Fin F → ℝ) →L[ℝ] (Fin A → ℝ)) (columns : Fin F → EucSpace h)
    (query transverse : EucSpace h) (cap : ℝ) (parameters : Fin A → ℝ) (ordinary : Fin N → ℝ)
    (hc : 0 < cap) :
    DifferentiableAt ℝ (fun p => qkPrefixMatrix inputs decoder columns query transverse cap p ordinary)
      parameters := by
  have hp : DifferentiableAt ℝ
      (fun p => fun a : Fin A => qkAnchoredKeys query transverse cap p ordinary (Fin.castAdd N a))
      parameters := by
    apply differentiableAt_pi.mpr
    intro a
    let coordinate : (Fin (A + N) → EucSpace h) →L[ℝ] EucSpace h :=
      ContinuousLinearMap.proj (Fin.castAdd N a)
    simpa only [Function.comp_def, coordinate, ContinuousLinearMap.proj_apply] using
      coordinate.differentiableAt.comp parameters
        (qkAnchoredKeys_differentiableAt query transverse cap parameters ordinary hc)
  unfold qkPrefixMatrix
  simpa only [Function.comp_def] using (inputProjectionUpdate_differentiableAt
    (fun a : Fin A => inputs (Fin.castAdd N a)) decoder columns _).comp parameters hp

/-- The differentiability premise has a finite, dependent long-context instance.
Source context: the derived §2.5 matrix path, with positive cap one eighth. -/
example : DifferentiableAt ℝ (fun p : Fin 2 → ℝ => qkPrefixMatrix (longQKInputs 100)
    (prefixInputDecoder 2 1) (longQKKeys 100 (fun _ => 0)) qkQueryExample qkTransverseExample
    (1 / 8) p (fun _ : Fin 100 => 0)) (fun _ => 0) :=
  qkPrefixMatrix_differentiableAt _ _ _ _ _ _ _ _ (by norm_num)

/-- Actual QKNorm scores along the matrix path equal the desired anchored Q/K scores.
Source: shared projections and normalization at `73f8a0b`, with the
derived partial input accessibility restriction for §2.5 of arXiv:1602.02068v2. -/
theorem projectedQKScores_qkPrefixMatrix {F h A N : ℕ} (inputs : Fin (A + N) → (Fin F → ℝ))
    (decoder : (Fin F → ℝ) →L[ℝ] (Fin A → ℝ)) (queries columns : Fin F → EucSpace h)
    (transverse : EucSpace h) (eps cap : ℝ) (initial parameters : Fin A → ℝ)
    (ordinary : Fin N → ℝ) (i : Fin (A + N))
    (hd : ∀ a, decoder (inputs (Fin.castAdd N a)) = Transformer.ConvexRecall.basis a)
    (hzero : ∀ n, decoder (inputs (Fin.natAdd A n)) = 0)
    (hkeys : projectionEvaluation inputs columns = qkAnchoredKeys
      (projectionEvaluation inputs queries i) transverse cap initial ordinary) :
    projectedQKScores (Real.log (qkAnchorGain cap ordinary)) eps queries
      (qkPrefixMatrix inputs decoder columns (projectionEvaluation inputs queries i)
        transverse cap parameters ordinary) inputs i =
      qkAnchoredScores (projectionEvaluation inputs queries i) transverse eps cap parameters ordinary := by
  rw [projectedQKScores_evaluation,
    qkPrefixMatrix_row inputs decoder columns _ transverse cap initial parameters ordinary hd hzero hkeys]
  rfl

/-- Actual score-path hypotheses hold on a context longer than its matrix width.
Source context: arXiv:1602.02068v2, §2.5, partial matrix accessibility. -/
example (parameters : Fin 2 → ℝ) : projectedQKScores
    (Real.log (qkAnchorGain (1 / 8) (fun _ : Fin 100 => 0))) (1 / 1000000) longQKQueries
    (qkPrefixMatrix (longQKInputs 100) (prefixInputDecoder 2 1) (longQKKeys 100 (fun _ => 0))
      (projectionEvaluation (longQKInputs 100) longQKQueries (Fin.natAdd 2 0))
      qkTransverseExample (1 / 8) parameters (fun _ : Fin 100 => 0)) (longQKInputs 100) (Fin.natAdd 2 0) =
      qkAnchoredScores (projectionEvaluation (longQKInputs 100) longQKQueries (Fin.natAdd 2 0))
        qkTransverseExample (1 / 1000000) (1 / 8) parameters (fun _ : Fin 100 => 0) := by
  apply projectedQKScores_qkPrefixMatrix _ _ _ _ _ _ _ (fun _ => 0) _ _ _
    (longQKInput_anchor 100) (longQKInput_ordinary 100)
  rw [longQKQueries_row]
  exact longQKKeys_row 100 _

end Transformer.GPTMini.Sparsemax
