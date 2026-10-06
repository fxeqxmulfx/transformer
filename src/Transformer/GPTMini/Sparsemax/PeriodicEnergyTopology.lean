import Transformer.GPTMini.Sparsemax.OutputTiedMemoryCurvature
import Mathlib.Analysis.Matrix.Order

/-!
# Closed feasible domains and continuous actual ordinary error

New existence prerequisites after arXiv:1602.02068v2, Eq. (1) and §2.5.
The auxiliary score/energy matrix is continuous in the intrinsic learned
coordinates. Its PSD cone, incident bounds and aggregate budget form a
closed domain. The affine output-sharing graph is closed as well.

Actual ordinary squared prediction error is continuous on the proved
inverse domain because the genuine sparsemax/common-value forward equals
the coded output table there. No continuity of an arbitrary inverse at
singular attention, or assumption of an attained optimizer, is used.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

open scoped BigOperators

/-- Every atom coefficient is a continuous scalar function of learned path edges.
Source: the affine auxiliary score chart before arXiv:1602.02068v2, Eq. (1). -/
theorem localMemoryWeights_continuous {N : ℕ} (e : Option (Fin N)) :
    Continuous (fun t : Fin N → ℝ => localMemoryWeights t e) := by
  cases e with
  | none => exact continuous_const.sub (continuous_finsetSum _ (fun i _ => continuous_apply i))
  | some e => exact continuous_apply e

/-- Each auxiliary local score is continuous in the actual stored edges.
Source: the affine path score representation before sparsemax Eq. (1). -/
theorem localMemoryCore_scores_continuous {N : ℕ} (i j : Fin (N + 1)) :
    Continuous (fun t : Fin N → ℝ => memoryGramScores (localMemoryCore t) i j) := by
  simp_rw [localMemoryCore_scores_apply]
  apply continuous_finsetSum
  intro e he
  exact (localMemoryWeights_continuous e).mul continuous_const

/-- The full energy coupling is continuous in all intrinsic edge/output coordinates.
Source: the affine block PSD restriction after arXiv:1602.02068v2, Eq. (1). -/
theorem periodicEnergyMatrix_continuous (N D : ℕ) (energy : ℝ) :
    Continuous (fun x : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ =>
      energyCoupledMemoryMatrix energy ((x.1, 0), x.2)) := by
  simp_rw [energyCoupledMemoryMatrix, localMemoryGram_scores]
  apply continuous_pi
  intro i
  apply continuous_pi
  intro j
  cases i with
  | inl i =>
    cases j with
    | inl j => exact (localMemoryCore_scores_continuous i j).comp continuous_fst
    | inr j => exact continuous_snd.matrix_elem i j
  | inr i =>
    cases j with
    | inl j => exact continuous_snd.matrix_elem j i
    | inr j => exact continuous_const

/-- The genuine joint inverse/energy domain is closed for every finite dictionary size.
Source: closed incident inequalities, budget equality and PSD coupling after sparsemax Eq. (1). -/
theorem periodicEnergyDomain_closed (N D : ℕ) (floor budget energy : ℝ) :
    IsClosed (periodicEnergyDomain N D floor budget energy) := by
  have hp : IsClosed {x : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ |
      (x.1, (0 : Sum (Fin (N + 1)) (Fin (N + 1)) → ℝ)) ∈
        incidentMemoryParameterDomain N 1 floor} :=
    (incidentMemoryParameterDomain_compact N 1 floor).isClosed.preimage
      (continuous_fst.prodMk continuous_const)
  have hb : IsClosed {x : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ |
      (∑ e, x.1 e) = budget} :=
    isClosed_eq (continuous_finsetSum _ (fun e _ => (continuous_apply e).comp continuous_fst))
      continuous_const
  have hs : IsClosed {x : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ |
      (energyCoupledMemoryMatrix energy ((x.1, 0), x.2)).PosSemidef} :=
    Matrix.posSemidef_is_closed.preimage (periodicEnergyMatrix_continuous N D energy)
  exact hp.inter (hb.inter hs)

/-- The fixed adjacent affine edge readout is continuous in every learned output coordinate.
Source: the new parameter-sharing graph before sparsemax Eq. (1). -/
theorem outputTiedMemoryEdges_continuous {N D : ℕ} (offset : Fin N → ℝ) (gain : ℝ)
    (channel : Fin D) : Continuous (outputTiedMemoryEdges offset gain channel) := by
  apply continuous_pi
  intro e
  exact continuous_const.add (continuous_const.mul
    ((continuous_id.matrix_elem e.castSucc channel).add
      (continuous_id.matrix_elem e.succ channel)))

/-- Parameter sharing preserves closedness of the true joint energy domain.
Source: the explicit affine architectural restriction following sparsemax Eq. (1). -/
theorem outputTiedEnergyDomain_closed (N D : ℕ) (floor budget energy : ℝ)
    (offset : Fin N → ℝ) (gain : ℝ) (channel : Fin D) :
    IsClosed (outputTiedEnergyDomain N D floor budget energy offset gain channel) := by
  exact (periodicEnergyDomain_closed N D floor budget energy).inter
    (isClosed_eq continuous_fst
      ((outputTiedMemoryEdges_continuous offset gain channel).comp continuous_snd))

/-- Ordinary finite vector-answer error is continuous in the complete output table.
Source: the ordinary squared criterion derived after arXiv:1602.02068v2, §2.5. -/
theorem matrixOutputError_continuous {R D : ℕ} (target : Matrix (Fin R) (Fin D) ℝ) :
    Continuous (matrixOutputError target) := by
  apply continuous_finsetSum
  intro r hr
  apply continuous_finsetSum
  intro d hd
  exact ((continuous_id.matrix_elem r d).sub continuous_const).pow 2

/-- The actual categorical sparsemax/common-value ordinary error is continuous on its inverse domain.
Source: the proved genuine forward identity after sparsemax Eq. (1) and outer criterion after §2.5. -/
theorem periodicEnergySquaredError_continuousOn {R N D : ℕ} (floor budget energy : ℝ)
    (code : Fin R → Fin (N + 1)) (target : Matrix (Fin R) (Fin D) ℝ) (hf : 1 / 2 < floor) :
    ContinuousOn (periodicEnergySquaredError code target)
      (periodicEnergyDomain N D floor budget energy) := by
  have hc : Continuous (fun x : (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ =>
      Matrix.of (fun r => x.2 (code r))) := by
    apply continuous_pi
    intro r
    apply continuous_pi
    intro d
    exact continuous_snd.matrix_elem (code r) d
  apply ((matrixOutputError_continuous target).comp hc).continuousOn.congr
  intro x hx
  exact periodicEnergySquaredError_eq floor budget energy code target x hf hx

/-- Nonconstant ordinary targets inhabit the actual continuous-error premise. -/
example : ContinuousOn (periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 0))
    (periodicEnergyDomain 2 1 (3 / 4) (1 / 8) 6) :=
  periodicEnergySquaredError_continuousOn _ _ _ _ _ (by norm_num)

/-- The true ordinary error remains continuous on the tied convex domain.
Source: the same actual forward under affine sharing after sparsemax §2.5. -/
theorem outputTiedSquaredError_continuousOn {N D : ℕ} (floor budget energy : ℝ)
    (offset : Fin N → ℝ) (gain : ℝ) (channel : Fin D)
    (target : Matrix (Fin (N + 1)) (Fin D) ℝ) (hf : 1 / 2 < floor) :
    ContinuousOn (periodicEnergySquaredError (fun j : Fin (N + 1) => j) target)
      (outputTiedEnergyDomain N D floor budget energy offset gain channel) :=
  (periodicEnergySquaredError_continuousOn floor budget energy _ target hf).mono (fun _ hx => hx.1)

/-- The nonsingleton nonidentity tied domain inhabits the actual continuity premise. -/
example : ContinuousOn (periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 1))
    (outputTiedEnergyDomain 2 1 (3 / 4) (1 / 8) 6
      (fun _ => (1 / 24 : ℝ)) (1 / 24) 0) :=
  outputTiedSquaredError_continuousOn _ _ _ _ _ _ _ (by norm_num)

/-- The same nonsingleton ordinary-answer domain is a closed parameter set. -/
example : IsClosed (outputTiedEnergyDomain 2 1 (3 / 4) (1 / 8) 6
    (fun _ => (1 / 24 : ℝ)) (1 / 24) 0) :=
  outputTiedEnergyDomain_closed _ _ _ _ _ _ _ _

/-- Larger categorical dictionaries preserve closedness without increasing physical width. -/
example : IsClosed (periodicEnergyDomain 7 2 (3 / 4) 0 1) :=
  periodicEnergyDomain_closed _ _ _ _ _

end Transformer.GPTMini.Sparsemax
