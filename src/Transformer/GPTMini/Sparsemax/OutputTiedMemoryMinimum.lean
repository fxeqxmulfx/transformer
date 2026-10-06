import Transformer.GPTMini.Sparsemax.PeriodicEnergyCompact

/-!
# A unique attained trained point for every ordinary answer table

New guarantee following arXiv:1602.02068v2, §2.5. Any nonempty tied
energy domain attains exactly one ordinary squared-output minimum. The
target need not fit. Compactness, continuity and strict intrinsic curvature
prove existence and uniqueness; an optimizer is not a supplied hypothesis.

The noncomputable selector records this proved mathematical result, without
claiming a numerical solver. Its parameters give the original bounded
width-three Q/K and invertible actual sparsemax. A nonsingleton three-slot
instance has a provably unattainable zero target and an attained positive
minimum, making the distinction between optimality and exact fitting explicit.
-/

noncomputable section

namespace Transformer.GPTMini.Sparsemax

/-- Every ordinary answer table has exactly one trained point in a nonempty tied joint domain.
Source: compactness, actual forward continuity and strict ordinary curvature after sparsemax §2.5. -/
theorem outputTiedSquaredError_existsUnique {N D : ℕ} (floor budget energy : ℝ)
    (offset : Fin N → ℝ) (gain : ℝ) (channel : Fin D)
    (target : Matrix (Fin (N + 1)) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hn : (outputTiedEnergyDomain N D floor budget energy offset gain channel).Nonempty) :
    ∃! x, x ∈ outputTiedEnergyDomain N D floor budget energy offset gain channel ∧
      IsMinOn (periodicEnergySquaredError (fun j : Fin (N + 1) => j) target)
        (outputTiedEnergyDomain N D floor budget energy offset gain channel) x := by
  obtain ⟨x, hx, hm⟩ := (outputTiedEnergyDomain_compact N D floor budget energy offset gain channel
    (by linarith)).exists_isMinOn hn
    (outputTiedSquaredError_continuousOn floor budget energy offset gain channel target hf)
  refine ⟨x, ⟨hx, hm⟩, ?_⟩
  intro y hy
  exact outputTiedSquaredError_unique_min floor budget energy offset gain channel target
    y x hf hy.1 hx hy.2 hm

/-- Ordinary nonconstant targets and a complete feasible instance inhabit every existence premise. -/
example : ∃! x, x ∈ outputTiedEnergyDomain 2 1 (3 / 4) (1 / 8) 6
    (fun _ => (1 / 24 : ℝ)) (1 / 24) (0 : Fin 1) ∧
    IsMinOn (periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 0))
      (outputTiedEnergyDomain 2 1 (3 / 4) (1 / 8) 6
        (fun _ => (1 / 24 : ℝ)) (1 / 24) 0) x :=
  outputTiedSquaredError_existsUnique _ _ _ _ _ _ _ (by norm_num) ⟨_, taskOutputTiedPair_mem 0⟩

/-- Choose the proved unique trained intrinsic point; this is not an assumed optimization oracle.
Source: the attained ordinary-answer minimum after arXiv:1602.02068v2, §2.5. -/
def outputTiedMemoryOptimalPoint {N D : ℕ} (floor budget energy : ℝ)
    (offset : Fin N → ℝ) (gain : ℝ) (channel : Fin D)
    (target : Matrix (Fin (N + 1)) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hn : (outputTiedEnergyDomain N D floor budget energy offset gain channel).Nonempty) :
    (Fin N → ℝ) × Matrix (Fin (N + 1)) (Fin D) ℝ :=
  Classical.choose (outputTiedSquaredError_existsUnique floor budget energy offset gain channel target hf hn).exists

/-- The selected trained point obeys every actual energy and affine sharing constraint.
Source: the proved ordinary-answer optimizer following sparsemax §2.5. -/
theorem outputTiedMemoryOptimalPoint_mem {N D : ℕ} (floor budget energy : ℝ)
    (offset : Fin N → ℝ) (gain : ℝ) (channel : Fin D)
    (target : Matrix (Fin (N + 1)) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hn : (outputTiedEnergyDomain N D floor budget energy offset gain channel).Nonempty) :
    outputTiedMemoryOptimalPoint floor budget energy offset gain channel target hf hn ∈
      outputTiedEnergyDomain N D floor budget energy offset gain channel :=
  (Classical.choose_spec
    (outputTiedSquaredError_existsUnique floor budget energy offset gain channel target hf hn).exists).1

/-- A nonconstant answer pattern inhabits every selected-point feasibility premise. -/
example : outputTiedMemoryOptimalPoint (3 / 4) (1 / 8) 6
    (fun _ : Fin 2 => (1 / 24 : ℝ)) (1 / 24) 0 (taskEnergyTarget 1)
    (by norm_num) ⟨_, taskOutputTiedPair_mem 1⟩ ∈
    outputTiedEnergyDomain 2 1 (3 / 4) (1 / 8) 6 (fun _ => (1 / 24 : ℝ)) (1 / 24) 0 :=
  outputTiedMemoryOptimalPoint_mem _ _ _ _ _ _ _ _ _

/-- The selected point minimizes the unchanged actual ordinary prediction error.
Source: the proved trained point following arXiv:1602.02068v2, §2.5. -/
theorem outputTiedMemoryOptimalPoint_min {N D : ℕ} (floor budget energy : ℝ)
    (offset : Fin N → ℝ) (gain : ℝ) (channel : Fin D)
    (target : Matrix (Fin (N + 1)) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hn : (outputTiedEnergyDomain N D floor budget energy offset gain channel).Nonempty) :
    IsMinOn (periodicEnergySquaredError (fun j : Fin (N + 1) => j) target)
      (outputTiedEnergyDomain N D floor budget energy offset gain channel)
      (outputTiedMemoryOptimalPoint floor budget energy offset gain channel target hf hn) :=
  (Classical.choose_spec
    (outputTiedSquaredError_existsUnique floor budget energy offset gain channel target hf hn).exists).2

/-- The nonidentity attained ordinary fit supplies every selected-minimum premise. -/
example : IsMinOn (periodicEnergySquaredError (fun j : Fin 3 => j) (taskEnergyTarget 0))
    (outputTiedEnergyDomain 2 1 (3 / 4) (1 / 8) 6 (fun _ => (1 / 24 : ℝ)) (1 / 24) 0)
    (outputTiedMemoryOptimalPoint (3 / 4) (1 / 8) 6
      (fun _ => (1 / 24 : ℝ)) (1 / 24) 0 (taskEnergyTarget 0)
      (by norm_num) ⟨_, taskOutputTiedPair_mem 0⟩) :=
  outputTiedMemoryOptimalPoint_min _ _ _ _ _ _ _ _ _

/-- Optimal trained geometry has a genuine invertible width-three sparsemax attention matrix.
Source: the proved physical sparsemax inverse after Eq. (1), applied to the attained optimizer. -/
theorem outputTiedMemoryOptimalPoint_inverse {N D : ℕ} (floor budget energy : ℝ)
    (offset : Fin N → ℝ) (gain : ℝ) (channel : Fin D)
    (target : Matrix (Fin (N + 1)) (Fin D) ℝ) (hf : 1 / 2 < floor)
    (hn : (outputTiedEnergyDomain N D floor budget energy offset gain channel).Nonempty) :
    IsUnit (periodicMemoryAttention
      (outputTiedMemoryOptimalPoint floor budget energy offset gain channel target hf hn).1).det :=
  periodicMemoryAttention_det_unit floor _ hf
    (outputTiedMemoryOptimalPoint_mem floor budget energy offset gain channel target hf hn).1.1.1

/-- A real nonidentity trained answer pattern inhabits the selected-inverse premises. -/
example : IsUnit (periodicMemoryAttention (outputTiedMemoryOptimalPoint (3 / 4) (1 / 8) 6
    (fun _ : Fin 2 => (1 / 24 : ℝ)) (1 / 24) 0 (taskEnergyTarget 0)
    (by norm_num) ⟨_, taskOutputTiedPair_mem 0⟩).1).det :=
  outputTiedMemoryOptimalPoint_inverse _ _ _ _ _ _ _ _ _

/-- The nonsingleton tied regression domain cannot fit zero ordinary answers.
Source: aggregate-budget incompatibility under the new affine sharing restriction after Eq. (1). -/
theorem taskOutputTied_zero_unattainable
    (x : (Fin 2 → ℝ) × Matrix (Fin 3) (Fin 1) ℝ)
    (hx : x ∈ outputTiedEnergyDomain 2 1 (3 / 4) (1 / 8) 6
      (fun _ => (1 / 24 : ℝ)) (1 / 24) 0) :
    periodicEnergyForward (fun j : Fin 3 => j) x ≠ 0 := by
  intro h
  rw [periodicEnergyForward_eq (3 / 4) (1 / 8) 6 _ _ (by norm_num) hx.1] at h
  have hZ : x.2 = 0 := h
  have ht : x.1 = fun _ => (1 / 24 : ℝ) := by
    rw [hx.2, hZ, outputTiedMemoryEdges_zero]
  have hb := hx.1.2.1
  rw [ht] at hb
  norm_num [Fin.sum_univ_two] at hb

/-- A nonidentity/nonconstant fit inhabits the zero-unattainability premise. -/
example : periodicEnergyForward (fun j : Fin 3 => j)
    ((taskEnergyParameters 1).1, taskEnergyTarget 1) ≠ 0 :=
  taskOutputTied_zero_unattainable _ (taskOutputTiedPair_mem 1)

/-- Even an unattainable ordinary target has an attained positive-error minimum.
Source: genuine compact-domain attainment and the explicit incompatibility after sparsemax §2.5. -/
theorem taskOutputTied_positive_minimum_exists :
    ∃ x ∈ outputTiedEnergyDomain 2 1 (3 / 4) (1 / 8) 6
      (fun _ => (1 / 24 : ℝ)) (1 / 24) (0 : Fin 1),
    IsMinOn (periodicEnergySquaredError (fun j : Fin 3 => j) 0)
      (outputTiedEnergyDomain 2 1 (3 / 4) (1 / 8) 6
        (fun _ => (1 / 24 : ℝ)) (1 / 24) 0) x ∧
      0 < periodicEnergySquaredError (fun j : Fin 3 => j) 0 x := by
  obtain ⟨x, hx, hm⟩ := (outputTiedSquaredError_existsUnique (3 / 4) (1 / 8) 6
    (fun _ : Fin 2 => (1 / 24 : ℝ)) (1 / 24) (0 : Fin 1) 0 (by norm_num)
    ⟨_, taskOutputTiedPair_mem 0⟩).exists
  exact ⟨x, hx, hm, matrixOutputError_pos _ _ (taskOutputTied_zero_unattainable x hx)⟩

/-- The same nonsingleton domain has a unique minimum even for its unattainable zero target. -/
example : ∃! x, x ∈ outputTiedEnergyDomain 2 1 (3 / 4) (1 / 8) 6
    (fun _ => (1 / 24 : ℝ)) (1 / 24) (0 : Fin 1) ∧
    IsMinOn (periodicEnergySquaredError (fun j : Fin 3 => j) 0)
      (outputTiedEnergyDomain 2 1 (3 / 4) (1 / 8) 6
        (fun _ => (1 / 24 : ℝ)) (1 / 24) 0) x :=
  outputTiedSquaredError_existsUnique _ _ _ _ _ _ _ (by norm_num) ⟨_, taskOutputTiedPair_mem 0⟩

/-- For both ordinary-answer witnesses the proved optimizer recovers the complete learned point.
Source: strict actual-loss uniqueness after sparsemax §2.5, rather than a supplied selected geometry. -/
theorem taskOutputTied_optimalPoint_eq (edge : Fin 2) :
    outputTiedMemoryOptimalPoint (3 / 4) (1 / 8) 6
      (fun _ : Fin 2 => (1 / 24 : ℝ)) (1 / 24) (0 : Fin 1) (taskEnergyTarget edge)
      (by norm_num) ⟨_, taskOutputTiedPair_mem edge⟩ =
        ((taskEnergyParameters edge).1, taskEnergyTarget edge) :=
  outputTiedSquaredError_unique_min (3 / 4) (1 / 8) 6 _ _ _ _ _ _ (by norm_num)
    (outputTiedMemoryOptimalPoint_mem _ _ _ _ _ _ _ _ _) (taskOutputTiedPair_mem edge)
    (outputTiedMemoryOptimalPoint_min _ _ _ _ _ _ _ _ _) (taskOutputTied_isMinOn edge)

end Transformer.GPTMini.Sparsemax
