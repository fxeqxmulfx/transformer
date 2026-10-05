import Transformer.GPTMini.Sparsemax.QKProjection
import Transformer.GPTMini.Sparsemax.QKSquaredError

/-!
# Squared-error correction in joint Q/K projection parameters

Derived from arXiv:1602.02068v2, §2.2 and §2.5, and the shared linear
projections, normalized score and value sum at `73f8a0b`. On the explicit
basis-input, anchored unit-key construction, an incorrect output has a
nonzero ordinary loss direction in the jointly learned Q/K matrices and
is not a local minimum of their squared-error loss. A finite change of
the key matrix reaches zero loss in the scalar sparse example.

The input basis and constrained keys are architectural restrictions.
Values, inputs, epsilon and temperature stay fixed in these projection
perturbations. The row is unrotated and is read before XSA and the output
projection. These results do not exclude cancellations in a multi-row
objective or assert global convergence with arbitrary embeddings.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Ordinary squared output error after the actual shared Q/K projections.
Source: `Attention.forward` at `73f8a0b`, before XSA and the output
projection; §2.5 of arXiv:1602.02068v2 supplies the sparsemax direction. -/
def projectedSquaredLoss {F T h : ℕ} (values : Fin T → E) (alpha eps : ℝ)
    (inputs : Fin T → (Fin F → ℝ)) (i : Fin T) (target : E)
    (projections : (Fin F → EucSpace h) × (Fin F → EucSpace h)) : ℝ :=
  squaredReadoutLoss target (frozenValueReadout values
    (sparseWeights (projectedQKScores alpha eps projections.1 projections.2 inputs i) i))

/-- Wrong outputs exclude zero derivatives in all joint projection directions.
Source: the derived unit-key and value-anchor restrictions for §2.5 of
arXiv:1602.02068v2, through the actual Q/K matrices at `73f8a0b`. -/
theorem qkAnchoredSquaredError_no_zero_projection_derivative {h d N : ℕ}
    (queries : Fin (d + 1 + N) → EucSpace h) (transverse : EucSpace h)
    (frame : Module.Basis (Fin d) ℝ E) (base : E) (scales : Fin d → ℝ)
    (ordinaryValues : Fin N → E) (eps cap : ℝ) (parameters : Fin (d + 1) → ℝ)
    (ordinary : Fin N → ℝ) (i : Fin (d + 1 + N)) (target : E)
    (hq : ‖queries i‖ = 1) (ht : ‖transverse‖ = 1)
    (ho : inner (𝕜 := ℝ) (queries i) transverse = 0) (he : eps ≤ 1) (hc : 0 < cap)
    (hcap : 2 * ((d + 1 : ℕ) : ℝ) * cap < 1)
    (hvisible : ∀ a : Fin (d + 1), Fin.castAdd N a ≤ i)
    (hbad : frozenValueReadout (anchoredValues frame base scales ordinaryValues)
      (sparseWeights (qkAnchoredScores (queries i) transverse eps cap parameters ordinary) i) ≠ target) :
    ¬ HasFDerivAt (𝕜 := ℝ)
      (projectedSquaredLoss (anchoredValues frame base scales ordinaryValues)
        (Real.log (qkAnchorGain cap ordinary)) eps basis i target) 0
      (queries, qkAnchoredKeys (queries i) transverse cap parameters ordinary) := by
  let output := frozenValueReadout (anchoredValues frame base scales ordinaryValues)
    (sparseWeights (qkAnchoredScores (queries i) transverse eps cap parameters ordinary) i)
  unfold projectedSquaredLoss
  exact qkAnchoredTaskLoss_no_zero_projection_derivative queries transverse frame base scales
    ordinaryValues eps cap parameters ordinary i (squaredReadoutLoss target)
    (2 • innerSL ℝ (output - target)) hq ht ho he hc hcap hvisible
    (squaredReadoutLoss_gradient_ne_zero target output hbad)
    (squaredReadoutLoss_hasFDerivAt target output)

/-- The no-zero joint-projection premises are inhabited by a sparse wrong row.
Source context: arXiv:1602.02068v2, §2.5, shared Q/K at `73f8a0b`. -/
example : ¬ HasFDerivAt (𝕜 := ℝ)
    (projectedSquaredLoss
      (anchoredValues (Module.Basis.singleton (Fin 1) ℝ) 0 (fun _ => 0) (fun _ : Fin 1 => 7))
      (Real.log (qkAnchorGain (1 / 8) (fun _ : Fin 1 => 0))) (1 / 1000000) basis 2 (0 : ℝ)) 0
    ((fun _ : Fin 3 => qkQueryExample), qkAnchoredKeys qkQueryExample qkTransverseExample
      (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0)) := by
  apply qkAnchoredSquaredError_no_zero_projection_derivative (fun _ => qkQueryExample)
    qkTransverseExample (Module.Basis.singleton (Fin 1) ℝ) 0 (fun _ => 0)
    (fun _ : Fin 1 => 7) (1 / 1000000) (1 / 8) (fun _ : Fin 2 => 0)
    (fun _ : Fin 1 => 0) 2 0 qkFrame_example.1 qkFrame_example.2.1 qkFrame_example.2.2
    (by norm_num) (by norm_num) (by norm_num) (fun a => Fin.le_last (Fin.castAdd 1 a))
  rw [qkAnchoredScores_eq _ _ _ _ _ _ qkFrame_example.1 qkFrame_example.2.1
    qkFrame_example.2.2 (by norm_num) (by norm_num), anchoredScalarReadout_sparse_example]
  norm_num

/-- Wrong outputs cannot be local minima in the jointly trained Q/K matrices.
Source: the derived score and value restrictions for §2.5 of
arXiv:1602.02068v2, with actual shared projections at `73f8a0b`.
Continuously freezing Q transports local minimality to the K-vector loss. -/
theorem qkAnchoredSquaredError_not_isLocalMin_projections {h d N : ℕ}
    (queries : Fin (d + 1 + N) → EucSpace h) (transverse : EucSpace h)
    (frame : Module.Basis (Fin d) ℝ E) (base : E) (scales : Fin d → ℝ)
    (ordinaryValues : Fin N → E) (eps cap : ℝ) (parameters : Fin (d + 1) → ℝ)
    (ordinary : Fin N → ℝ) (i : Fin (d + 1 + N)) (target : E)
    (hq : ‖queries i‖ = 1) (ht : ‖transverse‖ = 1)
    (ho : inner (𝕜 := ℝ) (queries i) transverse = 0) (he : eps ≤ 1) (hc : 0 < cap)
    (hcap : 2 * ((d + 1 : ℕ) : ℝ) * cap < 1)
    (hvisible : ∀ a : Fin (d + 1), Fin.castAdd N a ≤ i)
    (hbad : frozenValueReadout (anchoredValues frame base scales ordinaryValues)
      (sparseWeights (qkAnchoredScores (queries i) transverse eps cap parameters ordinary) i) ≠ target) :
    ¬ IsLocalMin
      (projectedSquaredLoss (anchoredValues frame base scales ordinaryValues)
        (Real.log (qkAnchorGain cap ordinary)) eps basis i target)
      (queries, qkAnchoredKeys (queries i) transverse cap parameters ordinary) := by
  intro hm
  let keys := qkAnchoredKeys (queries i) transverse cap parameters ordinary
  have hi : ContinuousAt (fun k : Fin (d + 1 + N) → EucSpace h => (queries, k)) keys :=
    continuousAt_const.prodMk continuousAt_id
  have hm' := hm.comp_continuous (g := fun k => (queries, k)) (b := keys) hi
  have hf : (projectedSquaredLoss (anchoredValues frame base scales ordinaryValues)
      (Real.log (qkAnchorGain cap ordinary)) eps basis i target ∘ fun k => (queries, k)) =
      (fun k => squaredReadoutLoss target
        (frozenValueReadout (anchoredValues frame base scales ordinaryValues)
          (sparseWeights (fun n => score (Real.log (qkAnchorGain cap ordinary)) eps
            (queries i) (k n)) i))) := by
    funext k
    simp only [Function.comp_def, projectedSquaredLoss]
    rw [projectedQKScores_basis]
  rw [hf] at hm'
  exact qkAnchoredSquaredError_not_isLocalMin (queries i) transverse frame base scales
    ordinaryValues eps cap parameters ordinary i target hq ht ho he hc hcap hvisible hbad hm'

/-- All joint local-minimum exclusion premises have a sparse scalar instance.
Source context: arXiv:1602.02068v2, §2.5, shared Q/K at `73f8a0b`. -/
example : ¬ IsLocalMin
    (projectedSquaredLoss
      (anchoredValues (Module.Basis.singleton (Fin 1) ℝ) 0 (fun _ => 0) (fun _ : Fin 1 => 7))
      (Real.log (qkAnchorGain (1 / 8) (fun _ : Fin 1 => 0))) (1 / 1000000) basis 2 (0 : ℝ))
    ((fun _ : Fin 3 => qkQueryExample), qkAnchoredKeys qkQueryExample qkTransverseExample
      (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0)) := by
  apply qkAnchoredSquaredError_not_isLocalMin_projections (fun _ => qkQueryExample)
    qkTransverseExample (Module.Basis.singleton (Fin 1) ℝ) 0 (fun _ => 0)
    (fun _ : Fin 1 => 7) (1 / 1000000) (1 / 8) (fun _ : Fin 2 => 0)
    (fun _ : Fin 1 => 0) 2 0 qkFrame_example.1 qkFrame_example.2.1 qkFrame_example.2.2
    (by norm_num) (by norm_num) (by norm_num) (fun a => Fin.le_last (Fin.castAdd 1 a))
  rw [qkAnchoredScores_eq _ _ _ _ _ _ qkFrame_example.1 qkFrame_example.2.1
    qkFrame_example.2.2 (by norm_num) (by norm_num), anchoredScalarReadout_sparse_example]
  norm_num

/-- The actual shared projection example starts with positive squared error.
Source: §2.2 and §2.5's derived sparse correction at `73f8a0b`. -/
theorem qkProjectedCorrection_initial_loss : projectedSquaredLoss correctionAnchorValues
    (Real.log (qkAnchorGain (1 / 8) (fun _ : Fin 1 => 0))) (1 / 1000000) basis 2 (1 / 2 : ℝ)
    ((fun _ : Fin 3 => qkQueryExample), qkAnchoredKeys qkQueryExample qkTransverseExample
      (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0)) = 1 / 256 := by
  unfold projectedSquaredLoss
  rw [projectedQKScores_basis]
  exact qkAnchoredCorrection_initial_loss

/-- Changing the actual key matrix reaches zero ordinary output error.
Source: §2.2 and §2.5's derived correction, with actual projections at `73f8a0b`. -/
theorem qkProjectedCorrection_zero_loss : projectedSquaredLoss correctionAnchorValues
    (Real.log (qkAnchorGain (1 / 8) (fun _ : Fin 1 => 0))) (1 / 1000000) basis 2 (1 / 2 : ℝ)
    ((fun _ : Fin 3 => qkQueryExample), qkAnchoredKeys qkQueryExample qkTransverseExample
      (1 / 8) correctionAnchorParameters (fun _ : Fin 1 => 0)) = 0 := by
  unfold projectedSquaredLoss
  rw [projectedQKScores_basis]
  exact qkAnchoredCorrection_zero_loss

end Transformer.GPTMini.Sparsemax
