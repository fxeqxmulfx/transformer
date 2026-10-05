import Transformer.GPTMini.Sparsemax.QKPrefixDirections

/-!
# Incorrect matrix outputs are not minima under partial anchor access

Derived from arXiv:1602.02068v2, §2.2 and §2.5, and shared
Q/K projections and QKNorm at `73f8a0b`. Only anchor inputs need an
independent decoder; ordinary inputs may be dependent and the context
may exceed input width. At the specified unit-key and value-anchor
family, a wrong ordinary output has no zero joint matrix derivative and
cannot be a local minimum of its squared output error.

The proof transports local minimality through the smooth anchor matrix
path, so it does not assume a full sparsemax derivative at support
boundaries. The decoder conditions have a proved channel construction
and a context-size-independent implementation in `PrefixInputs`.
The restricted key family and unit query frame remain assumptions. This
is an unrotated single-row readout before XSA and output projection;
shared-row cancellation, uniform gradient bounds and global convergence
are not established. The target is ordinary output data, not a route.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.ConvexRecall Transformer.GPTMini.Convex

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Wrong output excludes zero joint Q/K matrix derivatives with only
partial anchor accessibility. Source: the derived restriction for
§2.5 of arXiv:1602.02068v2 and actual projections at `73f8a0b`. -/
theorem qkPrefixSquaredError_no_zero_projection_derivative {F h d N : ℕ}
    (inputs : Fin (d + 1 + N) → (Fin F → ℝ))
    (decoder : (Fin F → ℝ) →L[ℝ] (Fin (d + 1) → ℝ))
    (queries columns : Fin F → EucSpace h) (transverse : EucSpace h)
    (frame : Module.Basis (Fin d) ℝ E) (base : E) (scales : Fin d → ℝ)
    (ordinaryValues : Fin N → E) (eps cap : ℝ) (parameters : Fin (d + 1) → ℝ)
    (ordinary : Fin N → ℝ) (i : Fin (d + 1 + N)) (target : E)
    (hd : ∀ a, decoder (inputs (Fin.castAdd N a)) = basis a)
    (hzero : ∀ n, decoder (inputs (Fin.natAdd (d + 1) n)) = 0)
    (hq : ‖projectionEvaluation inputs queries i‖ = 1) (ht : ‖transverse‖ = 1)
    (ho : inner (𝕜 := ℝ) (projectionEvaluation inputs queries i) transverse = 0)
    (he : eps ≤ 1) (hc : 0 < cap) (hcap : 2 * ((d + 1 : ℕ) : ℝ) * cap < 1)
    (hvisible : ∀ a : Fin (d + 1), Fin.castAdd N a ≤ i)
    (hkeys : projectionEvaluation inputs columns = qkAnchoredKeys
      (projectionEvaluation inputs queries i) transverse cap parameters ordinary)
    (hbad : frozenValueReadout (anchoredValues frame base scales ordinaryValues)
      (sparseWeights (projectedQKScores (Real.log (qkAnchorGain cap ordinary))
        eps queries columns inputs i) i) ≠ target) :
    ¬ HasFDerivAt (𝕜 := ℝ) (projectedSquaredLoss (anchoredValues frame base scales ordinaryValues)
      (Real.log (qkAnchorGain cap ordinary)) eps inputs i target) 0 (queries, columns) := by
  let output := frozenValueReadout (anchoredValues frame base scales ordinaryValues)
    (sparseWeights (projectedQKScores (Real.log (qkAnchorGain cap ordinary))
      eps queries columns inputs i) i)
  unfold projectedSquaredLoss
  exact qkPrefixTaskLoss_no_zero_projection_derivative inputs decoder queries columns transverse
    frame base scales ordinaryValues eps cap parameters ordinary i (squaredReadoutLoss target)
    (2 • innerSL ℝ (output - target)) hd hzero hq ht ho he hc hcap hvisible hkeys
    (squaredReadoutLoss_gradient_ne_zero target output hbad)
    (squaredReadoutLoss_hasFDerivAt target output)

/-- A wrong sparse output on one hundred repeated tokens inhabits every premise.
Source context: arXiv:1602.02068v2, §2.5, partial matrix accessibility. -/
example : ¬ HasFDerivAt (𝕜 := ℝ) (projectedSquaredLoss
    (anchoredValues (Module.Basis.singleton (Fin 1) ℝ) 0 (fun _ => 0) (fun _ : Fin 100 => 7))
    (Real.log (qkAnchorGain (1 / 8) (fun _ : Fin 100 => 0))) (1 / 1000000)
    (longQKInputs 100) (Fin.natAdd 2 0) (0 : ℝ)) 0 (longQKQueries, longQKKeys 100 (fun _ => 0)) := by
  apply qkPrefixSquaredError_no_zero_projection_derivative (longQKInputs 100) (prefixInputDecoder 2 1)
    longQKQueries (longQKKeys 100 (fun _ => 0)) qkTransverseExample (Module.Basis.singleton (Fin 1) ℝ)
    0 (fun _ => 0) (fun _ : Fin 100 => 7) (1 / 1000000) (1 / 8) (fun _ : Fin 2 => 0)
    (fun _ : Fin 100 => 0) (Fin.natAdd 2 0) 0 (longQKInput_anchor 100) (longQKInput_ordinary 100)
    (by rw [longQKQueries_row]; exact qkFrame_example.1) qkFrame_example.2.1
    (by rw [longQKQueries_row]; exact qkFrame_example.2.2) (by norm_num) (by norm_num)
    (by norm_num) (longQK_anchor_visible 100 0)
    (by rw [longQKQueries_row]; exact longQKKeys_row 100 _)
  rw [longQKReadout_initial]
  norm_num

/-- A wrong ordinary output is not a local minimum in the actual joint
Q/K matrices under only partial anchor accessibility. Source: the
derived §2.5 matrix path and active-pair direction of arXiv:1602.02068v2,
with actual projections at `73f8a0b`; full input rank is not required. -/
theorem qkPrefixSquaredError_not_isLocalMin_projections {F h d N : ℕ}
    (inputs : Fin (d + 1 + N) → (Fin F → ℝ))
    (decoder : (Fin F → ℝ) →L[ℝ] (Fin (d + 1) → ℝ))
    (queries columns : Fin F → EucSpace h) (transverse : EucSpace h)
    (frame : Module.Basis (Fin d) ℝ E) (base : E) (scales : Fin d → ℝ)
    (ordinaryValues : Fin N → E) (eps cap : ℝ) (parameters : Fin (d + 1) → ℝ)
    (ordinary : Fin N → ℝ) (i : Fin (d + 1 + N)) (target : E)
    (hd : ∀ a, decoder (inputs (Fin.castAdd N a)) = basis a)
    (hzero : ∀ n, decoder (inputs (Fin.natAdd (d + 1) n)) = 0)
    (hq : ‖projectionEvaluation inputs queries i‖ = 1) (ht : ‖transverse‖ = 1)
    (ho : inner (𝕜 := ℝ) (projectionEvaluation inputs queries i) transverse = 0)
    (he : eps ≤ 1) (hc : 0 < cap) (hcap : 2 * ((d + 1 : ℕ) : ℝ) * cap < 1)
    (hvisible : ∀ a : Fin (d + 1), Fin.castAdd N a ≤ i)
    (hkeys : projectionEvaluation inputs columns = qkAnchoredKeys
      (projectionEvaluation inputs queries i) transverse cap parameters ordinary)
    (hbad : frozenValueReadout (anchoredValues frame base scales ordinaryValues)
      (sparseWeights (projectedQKScores (Real.log (qkAnchorGain cap ordinary))
        eps queries columns inputs i) i) ≠ target) :
    ¬ IsLocalMin (projectedSquaredLoss (anchoredValues frame base scales ordinaryValues)
      (Real.log (qkAnchorGain cap ordinary)) eps inputs i target) (queries, columns) := by
  intro hm
  have hm0 : IsLocalMin (projectedSquaredLoss (anchoredValues frame base scales ordinaryValues)
      (Real.log (qkAnchorGain cap ordinary)) eps inputs i target)
      (queries, qkPrefixMatrix inputs decoder columns (projectionEvaluation inputs queries i)
        transverse cap parameters ordinary) := by
    simpa only [qkPrefixMatrix_center inputs decoder columns _ transverse cap parameters ordinary hkeys] using hm
  have hm' := hm0.comp_continuous
    (g := fun p => (queries, qkPrefixMatrix inputs decoder columns (projectionEvaluation inputs queries i)
      transverse cap p ordinary)) (b := parameters)
    (continuousAt_const.prodMk (qkPrefixMatrix_differentiableAt inputs decoder columns
      (projectionEvaluation inputs queries i) transverse cap parameters ordinary hc).continuousAt)
  have hf : (projectedSquaredLoss (anchoredValues frame base scales ordinaryValues)
      (Real.log (qkAnchorGain cap ordinary)) eps inputs i target ∘ fun p =>
        (queries, qkPrefixMatrix inputs decoder columns (projectionEvaluation inputs queries i)
          transverse cap p ordinary)) = (fun p => squaredReadoutLoss target (frozenValueReadout
            (anchoredValues frame base scales ordinaryValues) (sparseWeights (anchoredScores cap p ordinary) i))) := by
    funext p
    simp only [Function.comp_def, projectedSquaredLoss]
    rw [projectedQKScores_qkPrefixMatrix inputs decoder queries columns transverse eps cap
      parameters p ordinary i hd hzero hkeys,
      qkAnchoredScores_eq _ transverse eps cap p ordinary hq ht ho he hc]
  rw [hf] at hm'
  rw [projectedQKScores_evaluation, hkeys] at hbad
  change frozenValueReadout (anchoredValues frame base scales ordinaryValues)
    (sparseWeights (qkAnchoredScores (projectionEvaluation inputs queries i)
      transverse eps cap parameters ordinary) i) ≠ target at hbad
  rw [qkAnchoredScores_eq _ transverse eps cap parameters ordinary hq ht ho he hc] at hbad
  exact anchoredSquaredError_not_isLocalMin frame base scales ordinaryValues cap parameters
    ordinary i target hc hcap hvisible hbad hm'

/-- The local-minimum exclusion premises are inhabited without full input independence.
Source context: arXiv:1602.02068v2, §2.5, long dependent sparse row. -/
example : ¬ IsLocalMin (projectedSquaredLoss
    (anchoredValues (Module.Basis.singleton (Fin 1) ℝ) 0 (fun _ => 0) (fun _ : Fin 100 => 7))
    (Real.log (qkAnchorGain (1 / 8) (fun _ : Fin 100 => 0))) (1 / 1000000)
    (longQKInputs 100) (Fin.natAdd 2 0) (0 : ℝ)) (longQKQueries, longQKKeys 100 (fun _ => 0)) := by
  apply qkPrefixSquaredError_not_isLocalMin_projections (longQKInputs 100) (prefixInputDecoder 2 1)
    longQKQueries (longQKKeys 100 (fun _ => 0)) qkTransverseExample (Module.Basis.singleton (Fin 1) ℝ)
    0 (fun _ => 0) (fun _ : Fin 100 => 7) (1 / 1000000) (1 / 8) (fun _ : Fin 2 => 0)
    (fun _ : Fin 100 => 0) (Fin.natAdd 2 0) 0 (longQKInput_anchor 100) (longQKInput_ordinary 100)
    (by rw [longQKQueries_row]; exact qkFrame_example.1) qkFrame_example.2.1
    (by rw [longQKQueries_row]; exact qkFrame_example.2.2) (by norm_num) (by norm_num)
    (by norm_num) (longQK_anchor_visible 100 0)
    (by rw [longQKQueries_row]; exact longQKKeys_row 100 _)
  rw [longQKReadout_initial]
  norm_num

end Transformer.GPTMini.Sparsemax
