import Transformer.GPTMini.Sparsemax.QKTaskDirections

/-!
# Task directions reach shared query/key projection matrices

Derived from arXiv:1602.02068v2, §2.5, and `Attention.forward` at
`73f8a0b`: query and key vectors are linear projections of the inputs,
then normalized before their dot product. A matrix is represented by
its columns; the existing linear weighted sum evaluates its action.
On standard-basis inputs, each input selects one column of the same
shared matrix. Thus the certified key-vector directions reach actual
projection parameters, even when Q and K are trained jointly.
This row is unrotated and is read before XSA and the output projection.

The standard-basis inputs are an explicit architectural restriction,
requiring independent positional coordinates for the considered row.
This is not asserted for arbitrary learned embeddings or for a shared
multi-row objective. The unit-key and active-value prefix restrictions
remain part of the construction; normalization alone enforces neither.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex
open scoped BigOperators

/-- Normalized scores after shared linear query and key projections.
Source: `Attention.forward` at `73f8a0b`, with matrices represented by
columns; §2.5 of arXiv:1602.02068v2 supplies the subsequent score path. -/
def projectedQKScores {F T h : ℕ} (alpha eps : ℝ)
    (queries keys : Fin F → EucSpace h) (inputs : Fin T → (Fin F → ℝ)) (i : Fin T) :
    Fin T → ℝ := fun n => score alpha eps
      (frozenValueReadout queries (inputs i)) (frozenValueReadout keys (inputs n))

/-- Evaluating a matrix on a basis input selects its actual column.
Source: linear Q/K projections in `Attention.forward` at `73f8a0b`,
represented by the finite weighted sum already used for values. -/
theorem projection_basis_column {T h : ℕ} (columns : Fin T → EucSpace h) (i : Fin T) :
    frozenValueReadout columns (basis i) = columns i := by
  classical
  rw [frozenValueReadout_apply]
  simp [basis]

/-- Basis inputs make the shared projected scores the corresponding
column dot products. Source: `Attention.forward` at `73f8a0b`, under
the derived input restriction for §2.5 of arXiv:1602.02068v2. -/
theorem projectedQKScores_basis {T h : ℕ} (alpha eps : ℝ)
    (queries keys : Fin T → EucSpace h) (i : Fin T) :
    projectedQKScores alpha eps queries keys basis i =
      (fun n => score alpha eps (queries i) (keys n)) := by
  funext n
  rw [projectedQKScores, projection_basis_column, projection_basis_column]

/-- Equal input vectors force equal scores under every shared projection.
Source: the linear projections and normalized dot product in
`Attention.forward` at `73f8a0b`. This obstruction explains why the
derived §2.5 construction needs independent input coordinates. -/
theorem projectedQKScores_equal_inputs {F T h : ℕ} (alpha eps : ℝ)
    (queries keys : Fin F → EucSpace h) (inputs : Fin T → (Fin F → ℝ)) (i j k : Fin T)
    (he : inputs j = inputs k) :
    projectedQKScores alpha eps queries keys inputs i j =
      projectedQKScores alpha eps queries keys inputs i k := by
  unfold projectedQKScores
  rw [he]

/-- The equal-input premise is inhabited by two nonzero repeated vectors.
Source context: `Attention.forward` at `73f8a0b`, shared projections. -/
example : projectedQKScores 0 (1 / 1000000) (fun _ : Fin 1 => qkQueryExample)
    (fun _ : Fin 1 => qkTransverseExample) (fun _ : Fin 2 => fun _ : Fin 1 => 1) 1 0 =
    projectedQKScores 0 (1 / 1000000) (fun _ : Fin 1 => qkQueryExample)
      (fun _ : Fin 1 => qkTransverseExample) (fun _ : Fin 2 => fun _ : Fin 1 => 1) 1 1 :=
  projectedQKScores_equal_inputs _ _ _ _ _ _ _ _ rfl

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Nonzero ordinary output derivatives survive in jointly trained
Q/K projection matrices on the basis-encoded input row. Source: the
derived prefix and unit-key restrictions for §2.5 of arXiv:1602.02068v2,
composed with the actual linear projections and QKNorm at `73f8a0b`.
Only the query column used by this row must satisfy the unit-frame premises. -/
theorem qkAnchoredTaskLoss_no_zero_projection_derivative {h d N : ℕ}
    (queries : Fin (d + 1 + N) → EucSpace h) (transverse : EucSpace h)
    (frame : Module.Basis (Fin d) ℝ E) (base : E) (scales : Fin d → ℝ)
    (ordinaryValues : Fin N → E) (eps cap : ℝ) (parameters : Fin (d + 1) → ℝ)
    (ordinary : Fin N → ℝ) (i : Fin (d + 1 + N)) (loss : E → ℝ) (gradient : E →L[ℝ] ℝ)
    (hq : ‖queries i‖ = 1) (ht : ‖transverse‖ = 1)
    (ho : inner (𝕜 := ℝ) (queries i) transverse = 0) (he : eps ≤ 1) (hc : 0 < cap)
    (hcap : 2 * ((d + 1 : ℕ) : ℝ) * cap < 1)
    (hvisible : ∀ a : Fin (d + 1), Fin.castAdd N a ≤ i) (hg : gradient ≠ 0)
    (hl : HasFDerivAt (𝕜 := ℝ) loss gradient
      (frozenValueReadout (anchoredValues frame base scales ordinaryValues)
        (sparseWeights (qkAnchoredScores (queries i) transverse eps cap parameters ordinary) i))) :
    ¬ HasFDerivAt (𝕜 := ℝ)
      (fun projections : (Fin (d + 1 + N) → EucSpace h) × (Fin (d + 1 + N) → EucSpace h) =>
        loss (frozenValueReadout (anchoredValues frame base scales ordinaryValues)
          (sparseWeights (projectedQKScores (Real.log (qkAnchorGain cap ordinary)) eps
            projections.1 projections.2 basis i) i))) 0
      (queries, qkAnchoredKeys (queries i) transverse cap parameters ordinary) := by
  intro hz
  let keys := qkAnchoredKeys (queries i) transverse cap parameters ordinary
  have hf : (fun projections :
      (Fin (d + 1 + N) → EucSpace h) × (Fin (d + 1 + N) → EucSpace h) =>
      loss (frozenValueReadout (anchoredValues frame base scales ordinaryValues)
        (sparseWeights (projectedQKScores (Real.log (qkAnchorGain cap ordinary)) eps
          projections.1 projections.2 basis i) i))) =
      (fun projections => loss (frozenValueReadout (anchoredValues frame base scales ordinaryValues)
        (sparseWeights (fun n => score (Real.log (qkAnchorGain cap ordinary)) eps
          (projections.1 i) (projections.2 n)) i))) := by
    funext projections
    rw [projectedQKScores_basis]
  rw [hf] at hz
  have hi := (hasFDerivAt_const (𝕜 := ℝ) queries keys).prodMk
    (hasFDerivAt_id (𝕜 := ℝ) keys)
  have hzero : HasFDerivAt (𝕜 := ℝ)
      (fun k : Fin (d + 1 + N) → EucSpace h => loss
        (frozenValueReadout (anchoredValues frame base scales ordinaryValues)
          (sparseWeights (fun n => score (Real.log (qkAnchorGain cap ordinary)) eps (queries i)
            (k n)) i))) 0 keys := by
    simpa only [Function.comp_def, id_eq, ContinuousLinearMap.zero_comp] using hz.comp keys hi
  exact qkAnchoredTaskLoss_no_zero_key_derivative (queries i) transverse frame base scales
    ordinaryValues eps cap parameters ordinary i loss gradient hq ht ho he hc hcap
    hvisible hg hl hzero

/-- All projection, support, value-span and task hypotheses hold on a
sparse three-position row with standard-basis inputs. Source context:
arXiv:1602.02068v2, §2.5, derived shared projection at `73f8a0b`. -/
example : ¬ HasFDerivAt (𝕜 := ℝ)
    (fun projections : (Fin 3 → EucSpace 2) × (Fin 3 → EucSpace 2) => frozenValueReadout
      (anchoredValues (Module.Basis.singleton (Fin 1) ℝ) 0 (fun _ => 0) (fun _ : Fin 1 => 7))
      (sparseWeights (projectedQKScores (Real.log (qkAnchorGain (1 / 8) (fun _ : Fin 1 => 0)))
        (1 / 1000000) projections.1 projections.2 basis 2) 2)) 0
    ((fun _ => qkQueryExample), qkAnchoredKeys qkQueryExample qkTransverseExample (1 / 8)
      (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0)) := by
  apply qkAnchoredTaskLoss_no_zero_projection_derivative (fun _ => qkQueryExample)
    qkTransverseExample (Module.Basis.singleton (Fin 1) ℝ) 0 (fun _ => 0)
    (fun _ : Fin 1 => 7) (1 / 1000000) (1 / 8) (fun _ : Fin 2 => 0)
    (fun _ : Fin 1 => 0) 2 (fun x : ℝ => x) (ContinuousLinearMap.id ℝ ℝ)
    qkFrame_example.1 qkFrame_example.2.1 qkFrame_example.2.2 (by norm_num) (by norm_num)
    (by norm_num) (fun a => Fin.le_last (Fin.castAdd 1 a))
  · intro hz
    have hn := congrArg (fun f : ℝ →L[ℝ] ℝ => f 1) hz
    norm_num at hn
  · exact (ContinuousLinearMap.id ℝ ℝ).hasFDerivAt

/-- One shared key matrix retains the visible zero after projection,
normalization and sparsemax. Source: §2.2's derived prefix rule in
arXiv:1602.02068v2, composed with `Attention.forward` at `73f8a0b`. -/
theorem projectedQKScores_sparse_example : sparseWeights
    (projectedQKScores (Real.log (qkAnchorGain (1 / 8) (fun _ : Fin 1 => 0))) (1 / 1000000)
      (fun _ : Fin 3 => qkQueryExample) (qkAnchoredKeys qkQueryExample qkTransverseExample
        (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0)) basis 2) 2 =
      (fun n : Fin 3 => if n = 2 then 0 else 1 / 2) := by
  rw [projectedQKScores_basis]
  exact qkAnchoredScores_sparse_example

end Transformer.GPTMini.Sparsemax
