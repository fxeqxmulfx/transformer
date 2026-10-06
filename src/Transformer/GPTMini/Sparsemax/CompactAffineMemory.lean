import Transformer.GPTMini.Sparsemax.AffineValueFeasibility
import Transformer.GPTMini.Sparsemax.PeriodicEnergyTopology

/-!
# Constant-size joint training of generated Q/K and common values

New compact block after arXiv:1602.02068v2, Eq. (1). Two coefficient
rows per channel are the complete learned state. A fixed affine readout
of one slope coefficient generates the single attention scalar. All
edges, bounded width-three Q/K and original common values are generated
from this state and position. No per-prototype learned table is supplied.

The convex domain caps the small coefficient table and bounds the readout
scalar. This replaces the former dictionary-sized global PSD energy
constraint by coefficient bounds and sufficient scalar local budgets;
it is an explicit architectural change. The generated response family is
affine in normalized position. Prototype records and categorical encoding
remain external costs. General arbitrary answer fitting is not asserted.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- A fixed affine readout ties attention geometry to a learned slope coefficient.
Source: compact parameter sharing before sparsemax Eq. (1); no attention targets are supplied. -/
def compactAffineMemoryScalar {D : ℕ} (offset gain : ℝ) (channel : Fin D)
    (W : Matrix (Fin 2) (Fin D) ℝ) : ℝ := offset + gain * W 1 channel

/-- A compact learned coefficient cap and a scalar attention interval; neither depends on prototype count.
Source: the new constant-storage convex domain after arXiv:1602.02068v2, Eq. (1). -/
def compactAffineMemoryDomain (D : ℕ) (cap floor offset gain : ℝ) (channel : Fin D) :
    Set (Matrix (Fin 2) (Fin D) ℝ) :=
  {W | (∀ k d, -cap ≤ W k d ∧ W k d ≤ cap) ∧
    0 ≤ compactAffineMemoryScalar offset gain channel W ∧
      compactAffineMemoryScalar offset gain channel W ≤ 1 - floor}

/-- Opposite nonconstant responses fit with identity and nonidentity attention in one compact domain.
Source: new finite witnesses for the compact sparsemax Eq. (1) architecture. -/
def compactAffineExampleCoefficients (edge : Fin 2) : Matrix (Fin 2) (Fin 1) ℝ :=
  if edge = 0 then !![0; -1] else !![0; 1]

/-- The same size-independent convex domain contains both nonconstant learned witnesses.
Source: explicit compact coefficient and scalar bounds after sparsemax Eq. (1). -/
theorem compactAffineExample_mem (edge : Fin 2) :
    compactAffineExampleCoefficients edge ∈ compactAffineMemoryDomain 1 1 (3 / 4) (1 / 16) (1 / 16) 0 := by
  constructor
  · intro k d
    fin_cases edge <;> fin_cases k <;> fin_cases d <;> norm_num [compactAffineExampleCoefficients]
  · fin_cases edge <;> norm_num [compactAffineMemoryScalar, compactAffineExampleCoefficients]

/-- Affine sharing preserves the attention scalar at every convex interpolation.
Source: the explicit fixed readout before arXiv:1602.02068v2, Eq. (1). -/
theorem compactAffineMemoryScalar_affine {D : ℕ} (offset gain : ℝ) (channel : Fin D)
    (W V : Matrix (Fin 2) (Fin D) ℝ) (a b : ℝ) (hab : a + b = 1) :
    compactAffineMemoryScalar offset gain channel (a • W + b • V) =
      a * compactAffineMemoryScalar offset gain channel W + b * compactAffineMemoryScalar offset gain channel V := by
  change offset + gain * (a * W 1 channel + b * V 1 channel) =
    a * (offset + gain * W 1 channel) + b * (offset + gain * V 1 channel)
  have ha : a = 1 - b := by linarith
  rw [ha]
  ring

/-- Both different nonconstant witnesses inhabit the fixed-readout interpolation premise. -/
example : compactAffineMemoryScalar (1 / 16) (1 / 16) (0 : Fin 1)
    ((1 / 2 : ℝ) • compactAffineExampleCoefficients 0 + (1 / 2 : ℝ) • compactAffineExampleCoefficients 1) =
    (1 / 2 : ℝ) * compactAffineMemoryScalar (1 / 16) (1 / 16) 0 (compactAffineExampleCoefficients 0) +
    (1 / 2 : ℝ) * compactAffineMemoryScalar (1 / 16) (1 / 16) 0 (compactAffineExampleCoefficients 1) :=
  compactAffineMemoryScalar_affine _ _ _ _ _ _ _ (by norm_num)

/-- The complete learned state has a convex domain independent of the number of prototypes.
Source: coefficient intervals and the fixed affine readout following sparsemax Eq. (1). -/
theorem compactAffineMemoryDomain_convex (D : ℕ) (cap floor offset gain : ℝ) (channel : Fin D) :
    Convex ℝ (compactAffineMemoryDomain D cap floor offset gain channel) := by
  intro W hW V hV a b ha hb hab
  constructor
  · intro k d
    have hc : Convex ℝ (Set.Icc (-cap) cap) := convex_Icc _ _
    exact hc (hW.1 k d) (hV.1 k d) ha hb hab
  · rw [compactAffineMemoryScalar_affine offset gain channel W V a b hab]
    have hc : Convex ℝ (Set.Icc 0 (1 - floor)) := convex_Icc _ _
    exact hc hW.2 hV.2 ha hb hab

/-- A zero coefficient point inhabits every compatible cap and attention-scalar interval.
Source: a full compact domain witness following arXiv:1602.02068v2, Eq. (1). -/
theorem zero_mem_compactAffineMemoryDomain (D : ℕ) (cap floor offset gain : ℝ) (channel : Fin D)
    (hc : 0 ≤ cap) (h0 : 0 ≤ offset) (h1 : offset ≤ 1 - floor) :
    (0 : Matrix (Fin 2) (Fin D) ℝ) ∈ compactAffineMemoryDomain D cap floor offset gain channel := by
  constructor
  · intro k d
    change -cap ≤ 0 ∧ 0 ≤ cap
    constructor <;> linarith
  · simpa only [compactAffineMemoryScalar, Matrix.zero_apply, mul_zero, add_zero] using And.intro h0 h1

/-- The compact domain has a zero-output witness with a nonzero attention scalar. -/
example : (0 : Matrix (Fin 2) (Fin 3) ℝ) ∈ compactAffineMemoryDomain 3 1 (3 / 4) (1 / 16) (1 / 16) 0 :=
  zero_mem_compactAffineMemoryDomain _ _ _ _ _ _ (by norm_num) (by norm_num) (by norm_num)

/-- The affine scalar readout is continuous in the full small learned state.
Source: fixed parameter sharing before sparsemax Eq. (1). -/
theorem compactAffineMemoryScalar_continuous {D : ℕ} (offset gain : ℝ) (channel : Fin D) :
    Continuous (compactAffineMemoryScalar offset gain channel : Matrix (Fin 2) (Fin D) ℝ → ℝ) :=
  ((continuous_id.matrix_elem 1 channel).const_mul gain).const_add offset

/-- All compact coefficient and scalar constraints are closed.
Source: the bounded affine learning chart following arXiv:1602.02068v2, Eq. (1). -/
theorem compactAffineMemoryDomain_closed (D : ℕ) (cap floor offset gain : ℝ) (channel : Fin D) :
    IsClosed (compactAffineMemoryDomain D cap floor offset gain channel) := by
  have hc : IsClosed ((Set.Icc (-cap) cap).matrix : Set (Matrix (Fin 2) (Fin D) ℝ)) :=
    (isCompact_Icc : IsCompact (Set.Icc (-cap) cap)).matrix.isClosed
  have hs := compactAffineMemoryScalar_continuous offset gain channel
  exact hc.inter ((isClosed_le continuous_const hs).inter (isClosed_le hs continuous_const))

/-- Finite learned state and its explicit cap give compactness without a supplied optimizer.
Source: the closed subset of a small coefficient box after sparsemax Eq. (1). -/
theorem compactAffineMemoryDomain_compact (D : ℕ) (cap floor offset gain : ℝ) (channel : Fin D) :
    IsCompact (compactAffineMemoryDomain D cap floor offset gain channel) := by
  exact (isCompact_Icc : IsCompact (Set.Icc (-cap) cap)).matrix.of_isClosed_subset
    (compactAffineMemoryDomain_closed D cap floor offset gain channel) (fun W hW => hW.1)

/-- Scalar learning generates genuinely feasible local attention at every prototype count.
Source: the sufficient profile normalization before variational sparsemax Eq. (1). -/
theorem compactAffineMemoryEdges_mem (N : ℕ) {D : ℕ} (cap floor offset gain : ℝ) (channel : Fin D)
    (W : Matrix (Fin 2) (Fin D) ℝ) (hW : W ∈ compactAffineMemoryDomain D cap floor offset gain channel) :
    affineValueEdges N (affineValueNormalizedMix N (compactAffineMemoryScalar offset gain channel W)) ∈
      incidentMemoryWeightDomain (N + 1) floor :=
  affineValueNormalizedEdges_mem N floor _ hW.2.1 hW.2.2

/-- One thousand actual slots are feasible with the same two nonconstant learned coefficient rows. -/
example : affineValueEdges 998 (affineValueNormalizedMix 998
    (compactAffineMemoryScalar (1 / 16) (1 / 16) 0 (compactAffineExampleCoefficients 1))) ∈
      incidentMemoryWeightDomain 999 (3 / 4) :=
  compactAffineMemoryEdges_mem 998 1 (3 / 4) (1 / 16) (1 / 16) 0 _ (compactAffineExample_mem 1)

/-- Actual compact joint sparsemax/common-value prediction, parameterized only by two coefficient rows.
Source: the generated Q/K and original common values following sparsemax Eq. (1). -/
def compactAffineMemoryForward {R N D : ℕ} (code : Fin R → Fin (N + 2)) (offset gain : ℝ) (channel : Fin D)
    (W : Matrix (Fin 2) (Fin D) ℝ) : Matrix (Fin R) (Fin D) ℝ :=
  affineValueForward code (affineValueNormalizedMix N (compactAffineMemoryScalar offset gain channel W)) W

/-- Genuine compact attention and original values give the learned affine positional responses.
Source: the proved invariant-feature inverse following arXiv:1602.02068v2, Eq. (1). -/
theorem compactAffineMemoryForward_eq {R N D : ℕ} (cap floor offset gain : ℝ) (channel : Fin D)
    (code : Fin R → Fin (N + 2)) (W : Matrix (Fin 2) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hW : W ∈ compactAffineMemoryDomain D cap floor offset gain channel) :
    compactAffineMemoryForward code offset gain channel W = Matrix.of (fun r => affineValueOutput N W (code r)) :=
  affineValueForward_eq floor code _ W hf (compactAffineMemoryEdges_mem N cap floor offset gain channel W hW)

/-- A thousand-slot nonconstant true forward inhabits all compact joint learning premises. -/
example : compactAffineMemoryForward (fun j : Fin 1000 => j) (1 / 16) (1 / 16) 0
    (compactAffineExampleCoefficients 1) = affineValueOutput 998 (compactAffineExampleCoefficients 1) :=
  compactAffineMemoryForward_eq 1 (3 / 4) (1 / 16) (1 / 16) 0 _ _ (by norm_num) (compactAffineExample_mem 1)

end Transformer.GPTMini.Sparsemax
