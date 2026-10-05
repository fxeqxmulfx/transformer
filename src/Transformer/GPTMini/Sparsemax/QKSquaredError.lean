import Transformer.GPTMini.Sparsemax.QKTaskDirections
import Transformer.GPTMini.Sparsemax.AnchoredCorrection

/-!
# Wrong QKNorm anchored outputs are not local minima in key vectors

Derived from arXiv:1602.02068v2, §2.2 and §2.5, with the normalized
dot product and value aggregation at `73f8a0b` and ordinary squared
output error. At every finite assignment in the unit-key anchored family,
an incorrect output has a nonzero direction in actual K vectors and
cannot be a local minimum of that key-vector loss. The proof transports
local minimality through the continuous key chart; it does not assume
sparsemax differentiable at an inactive support boundary.
The row is unrotated and is read before XSA and the output projection.

The target is the ordinary output target, not an attention position.
These are single-row statements, with frozen values and query on the
key perturbation. They neither assert a uniform gradient bound nor prove
global convergence of the shared transformer objective.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open Transformer.GPTMini.Convex

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Any wrong output gives a nonzero ordinary squared-loss direction in K.
Source: the derived anchor and unit-key restrictions for §2.2 and §2.5
of arXiv:1602.02068v2 and the actual QKNorm readout at `73f8a0b`. -/
theorem qkAnchoredSquaredError_no_zero_key_derivative {h d N : ℕ}
    (query transverse : EucSpace h) (frame : Module.Basis (Fin d) ℝ E)
    (base : E) (scales : Fin d → ℝ) (ordinaryValues : Fin N → E) (eps cap : ℝ)
    (parameters : Fin (d + 1) → ℝ) (ordinary : Fin N → ℝ) (i : Fin (d + 1 + N)) (target : E)
    (hq : ‖query‖ = 1) (ht : ‖transverse‖ = 1)
    (ho : inner (𝕜 := ℝ) query transverse = 0) (he : eps ≤ 1) (hc : 0 < cap)
    (hcap : 2 * ((d + 1 : ℕ) : ℝ) * cap < 1)
    (hvisible : ∀ a : Fin (d + 1), Fin.castAdd N a ≤ i)
    (hbad : frozenValueReadout (anchoredValues frame base scales ordinaryValues)
      (sparseWeights (qkAnchoredScores query transverse eps cap parameters ordinary) i) ≠ target) :
    ¬ HasFDerivAt (𝕜 := ℝ)
      (fun keys : Fin (d + 1 + N) → EucSpace h => squaredReadoutLoss target
        (frozenValueReadout (anchoredValues frame base scales ordinaryValues)
          (sparseWeights (fun n => score (Real.log (qkAnchorGain cap ordinary)) eps query
            (keys n)) i))) 0 (qkAnchoredKeys query transverse cap parameters ordinary) := by
  let output := frozenValueReadout (anchoredValues frame base scales ordinaryValues)
    (sparseWeights (qkAnchoredScores query transverse eps cap parameters ordinary) i)
  exact qkAnchoredTaskLoss_no_zero_key_derivative query transverse frame base scales ordinaryValues
    eps cap parameters ordinary i (squaredReadoutLoss target) (2 • innerSL ℝ (output - target))
    hq ht ho he hc hcap hvisible (squaredReadoutLoss_gradient_ne_zero target output hbad)
    (squaredReadoutLoss_hasFDerivAt target output)

/-- A positive-error sparse scalar task satisfies every actual-key premise.
Source context: arXiv:1602.02068v2, §2.5, derived Q/K row at `73f8a0b`. -/
example : ¬ HasFDerivAt (𝕜 := ℝ)
    (fun keys : Fin 3 → EucSpace 2 => squaredReadoutLoss (0 : ℝ)
      (frozenValueReadout
        (anchoredValues (Module.Basis.singleton (Fin 1) ℝ) 0 (fun _ => 0) (fun _ : Fin 1 => 7))
        (sparseWeights (fun n => score (Real.log (qkAnchorGain (1 / 8) (fun _ : Fin 1 => 0)))
          (1 / 1000000) qkQueryExample (keys n)) 2))) 0
    (qkAnchoredKeys qkQueryExample qkTransverseExample (1 / 8)
      (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0)) := by
  apply qkAnchoredSquaredError_no_zero_key_derivative qkQueryExample qkTransverseExample
    (Module.Basis.singleton (Fin 1) ℝ) 0 (fun _ => 0) (fun _ : Fin 1 => 7)
    (1 / 1000000) (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0) 2 0
    qkFrame_example.1 qkFrame_example.2.1 qkFrame_example.2.2 (by norm_num) (by norm_num)
    (by norm_num) (fun a => Fin.le_last (Fin.castAdd 1 a))
  rw [qkAnchoredScores_eq _ _ _ _ _ _ qkFrame_example.1 qkFrame_example.2.1
    qkFrame_example.2.2 (by norm_num) (by norm_num), anchoredScalarReadout_sparse_example]
  norm_num

/-- Wrong outputs are not local minima of the actual key-vector loss.
Source: the active-pair direction of §2.5 of arXiv:1602.02068v2,
transported through the derived unit-key chart into the QKNorm operator
at `73f8a0b`; support-boundary differentiability is not required. -/
theorem qkAnchoredSquaredError_not_isLocalMin {h d N : ℕ}
    (query transverse : EucSpace h) (frame : Module.Basis (Fin d) ℝ E)
    (base : E) (scales : Fin d → ℝ) (ordinaryValues : Fin N → E) (eps cap : ℝ)
    (parameters : Fin (d + 1) → ℝ) (ordinary : Fin N → ℝ) (i : Fin (d + 1 + N)) (target : E)
    (hq : ‖query‖ = 1) (ht : ‖transverse‖ = 1)
    (ho : inner (𝕜 := ℝ) query transverse = 0) (he : eps ≤ 1) (hc : 0 < cap)
    (hcap : 2 * ((d + 1 : ℕ) : ℝ) * cap < 1)
    (hvisible : ∀ a : Fin (d + 1), Fin.castAdd N a ≤ i)
    (hbad : frozenValueReadout (anchoredValues frame base scales ordinaryValues)
      (sparseWeights (qkAnchoredScores query transverse eps cap parameters ordinary) i) ≠ target) :
    ¬ IsLocalMin
      (fun keys : Fin (d + 1 + N) → EucSpace h => squaredReadoutLoss target
        (frozenValueReadout (anchoredValues frame base scales ordinaryValues)
          (sparseWeights (fun n => score (Real.log (qkAnchorGain cap ordinary)) eps query
            (keys n)) i))) (qkAnchoredKeys query transverse cap parameters ordinary) := by
  intro hm
  have hrow (p : Fin (d + 1) → ℝ) :=
    qkAnchoredScores_eq query transverse eps cap p ordinary hq ht ho he hc
  have hm' := hm.comp_continuous
    (g := fun p => qkAnchoredKeys query transverse cap p ordinary) (b := parameters)
    (qkAnchoredKeys_differentiableAt query transverse cap parameters ordinary hc).continuousAt
  have hf : (fun p => squaredReadoutLoss target
      (frozenValueReadout (anchoredValues frame base scales ordinaryValues)
        (sparseWeights (fun n => score (Real.log (qkAnchorGain cap ordinary)) eps query
          (qkAnchoredKeys query transverse cap p ordinary n)) i))) =
      (fun p => squaredReadoutLoss target
        (frozenValueReadout (anchoredValues frame base scales ordinaryValues)
          (sparseWeights (anchoredScores cap p ordinary) i))) := by
    funext p
    change squaredReadoutLoss target (frozenValueReadout _ (sparseWeights
      (qkAnchoredScores query transverse eps cap p ordinary) i)) = _
    rw [hrow p]
  simp only [Function.comp_def] at hm'
  rw [hf] at hm'
  rw [hrow parameters] at hbad
  exact anchoredSquaredError_not_isLocalMin frame base scales ordinaryValues cap parameters
    ordinary i target hc hcap hvisible hbad hm'

/-- The local-minimum exclusion hypotheses hold at a sparse wrong output.
Source context: arXiv:1602.02068v2, §2.5, actual normalized scores. -/
example : ¬ IsLocalMin
    (fun keys : Fin 3 → EucSpace 2 => squaredReadoutLoss (0 : ℝ)
      (frozenValueReadout
        (anchoredValues (Module.Basis.singleton (Fin 1) ℝ) 0 (fun _ => 0) (fun _ : Fin 1 => 7))
        (sparseWeights (fun n => score (Real.log (qkAnchorGain (1 / 8) (fun _ : Fin 1 => 0)))
          (1 / 1000000) qkQueryExample (keys n)) 2)))
    (qkAnchoredKeys qkQueryExample qkTransverseExample (1 / 8)
      (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0)) := by
  apply qkAnchoredSquaredError_not_isLocalMin qkQueryExample qkTransverseExample
    (Module.Basis.singleton (Fin 1) ℝ) 0 (fun _ => 0) (fun _ : Fin 1 => 7)
    (1 / 1000000) (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0) 2 0
    qkFrame_example.1 qkFrame_example.2.1 qkFrame_example.2.2 (by norm_num) (by norm_num)
    (by norm_num) (fun a => Fin.le_last (Fin.castAdd 1 a))
  rw [qkAnchoredScores_eq _ _ _ _ _ _ qkFrame_example.1 qkFrame_example.2.1
    qkFrame_example.2.2 (by norm_num) (by norm_num), anchoredScalarReadout_sparse_example]
  norm_num

/-- The actual Q/K correction starts with positive ordinary output error.
Source: the derived construction for §2.2 and §2.5 of arXiv:1602.02068v2,
evaluated through the normalized dot product at `73f8a0b`. -/
theorem qkAnchoredCorrection_initial_loss : squaredReadoutLoss (1 / 2 : ℝ)
    (frozenValueReadout correctionAnchorValues
      (sparseWeights (qkAnchoredScores qkQueryExample qkTransverseExample (1 / 1000000)
        (1 / 8) (fun _ : Fin 2 => 0) (fun _ : Fin 1 => 0)) 2)) = 1 / 256 := by
  rw [qkAnchoredScores_eq _ _ _ _ _ _ qkFrame_example.1 qkFrame_example.2.1
    qkFrame_example.2.2 (by norm_num) (by norm_num)]
  exact correctionAnchorLoss_initial

/-- The earlier finite correction reaches zero loss through actual Q/K scores.
Source: the derived construction for §2.2 and §2.5 of arXiv:1602.02068v2,
with standard epsilon in the QKNorm operator at `73f8a0b`. -/
theorem qkAnchoredCorrection_zero_loss : squaredReadoutLoss (1 / 2 : ℝ)
    (frozenValueReadout correctionAnchorValues
      (sparseWeights (qkAnchoredScores qkQueryExample qkTransverseExample (1 / 1000000)
        (1 / 8) correctionAnchorParameters (fun _ : Fin 1 => 0)) 2)) = 0 := by
  rw [qkAnchoredScores_eq _ _ _ _ _ _ qkFrame_example.1 qkFrame_example.2.1
    qkFrame_example.2.2 (by norm_num) (by norm_num)]
  exact correctionAnchorLoss_final

end Transformer.GPTMini.Sparsemax
