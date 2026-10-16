/-
# Exact inverses of finite original autonomous phase sequences

ArXiv:2411.04551v3, §1, invertible solution maps, §3.2,
reverse attention, and §4, restoration after a local transfer.
Reverse the order of the actual phases and reverse each original
clock by scaling both output matrices by minus one. The proved
original weak phase laws are retained under this exact rescaling.

Every such reversed sequence is an actual left and right inverse
on original probability laws. Realizing it at any positive horizon
therefore supplies actual original controls and weak solutions for
an inverse map, with the same phase count and zero scores whenever
the original phases have zero scores.
-/

import Transformer.Interpolation.Section1_PhaseSequenceControls

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Reverse the actual phase order and each original elapsed clock.
Source auxiliary: arXiv:2411.04551v3, §1 and §3.2, inverse solution maps. -/
noncomputable def inversePhaseList (L : List (AutonomousPhase d)) : List (AutonomousPhase d) :=
  L.reverse.map (rescalePhase (-1))

/-- The empty original phase sequence is its own inverse.
Source auxiliary: arXiv:2411.04551v3, §1, the identity map. -/
theorem inversePhaseList_nil : inversePhaseList ([] : List (AutonomousPhase d)) = [] := rfl

/-- The inverse of the first original phase acts after all later
inverse phases. Source auxiliary: arXiv:2411.04551v3, §3.2, restoration. -/
theorem inversePhaseList_cons (P : AutonomousPhase d) (L : List (AutonomousPhase d)) :
    inversePhaseList (P :: L) = inversePhaseList L ++ [rescalePhase (-1) P] := by
  simp only [inversePhaseList, List.reverse_cons, List.map_append, List.map_cons, List.map_nil]

/-- Inversion preserves the finite number of original phases.
Source auxiliary: arXiv:2411.04551v3, §1, inverse map realization. -/
theorem inversePhaseList_length (L : List (AutonomousPhase d)) :
    (inversePhaseList L).length = L.length := by
  rw [inversePhaseList, List.length_map, List.length_reverse]

/-- Reversed phases retain their actual original weak laws and zero
score constraint. Source auxiliary: arXiv:2411.04551v3, §1 and §3.2. -/
theorem inversePhaseList_valid (L : List (AutonomousPhase d))
    (hL : ∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0) :
    ∀ P ∈ inversePhaseList L, IsAutonomousPhase P ∧ P.1.B = 0 := by
  intro P hP
  obtain ⟨Q, hQ, rfl⟩ := List.mem_map.mp hP
  have hQ' : Q ∈ L := List.mem_reverse.mp hQ
  have h := hL Q hQ'
  exact ⟨h.1.rescale Q (-1) (by norm_num), rescalePhase_B_zero Q (-1) h.2⟩

/-- Inverse actual phases cancel the original nonlinear probability
composition on the left. Source auxiliary: arXiv:2411.04551v3,
§1 and §3.2, exact reversal of the solution maps. -/
theorem inversePhaseList_cancel_left (L : List (AutonomousPhase d))
    (hL : ∀ P ∈ L, IsAutonomousPhase P) (ρ : Perspective.ProbSphere d) :
    phaseSequenceLaw (inversePhaseList L) (phaseSequenceLaw L ρ) = ρ := by
  revert hL
  induction L generalizing ρ with
  | nil => intro _; rfl
  | cons P L ih =>
    intro hL
    have hP : IsAutonomousPhase P := hL P List.mem_cons_self
    have htail : ∀ Q ∈ L, IsAutonomousPhase Q := fun Q hQ => hL Q (List.mem_cons_of_mem _ hQ)
    rw [phaseSequenceLaw_cons, inversePhaseList_cons, phaseSequenceLaw_append, ih (P.2 ρ 1) htail]
    change P.2 (P.2 ρ 1) ((-1 : ℝ) * 1) = ρ
    rw [mul_one]
    exact hP.law_inverse P ρ 1

/-- The same actual inverse composition cancels on the right.
Source auxiliary: arXiv:2411.04551v3, §1, invertible original maps. -/
theorem inversePhaseList_cancel_right (L : List (AutonomousPhase d))
    (hL : ∀ P ∈ L, IsAutonomousPhase P) (ρ : Perspective.ProbSphere d) :
    phaseSequenceLaw L (phaseSequenceLaw (inversePhaseList L) ρ) = ρ := by
  obtain ⟨ν, rfl⟩ := (bijective_phaseSequenceLaw L hL).surjective ρ
  rw [inversePhaseList_cancel_left L hL]

/-- An exactly matched original endpoint is sent back to its input
by the actual inverse phases. Source auxiliary: arXiv:2411.04551v3,
§4, the exact restoration step in interpolation. -/
theorem inversePhaseList_of_endpoint (L : List (AutonomousPhase d))
    (hL : ∀ P ∈ L, IsAutonomousPhase P) (ρ σ : Perspective.ProbSphere d)
    (hmatch : phaseSequenceLaw L ρ = σ) : phaseSequenceLaw (inversePhaseList L) σ = ρ := by
  rw [← hmatch, inversePhaseList_cancel_left L hL]

/-- Two original sequences meeting at the same law give exact matching
by running the second sequence backward. Source auxiliary:
arXiv:2411.04551v3, §1 and §4, interpolation through common intermediate data. -/
theorem phaseSequenceLaw_matching_via_common (L M : List (AutonomousPhase d))
    (hM : ∀ P ∈ M, IsAutonomousPhase P) (ρ σ ν : Perspective.ProbSphere d)
    (hρ : phaseSequenceLaw L ρ = ν) (hσ : phaseSequenceLaw M σ = ν) :
    phaseSequenceLaw (L ++ inversePhaseList M) ρ = σ := by
  rw [phaseSequenceLaw_append, hρ, ← hσ, inversePhaseList_cancel_left M hM]

/-- The original reversed map is realized at any positive prescribed
horizon by actual zero-score controls. Every original weak endpoint
maps forward to its starting law. Source auxiliary:
arXiv:2411.04551v3, §1 and §4, inverse map realization. -/
theorem exists_inverse_phase_controls (i : Fin d) (L : List (AutonomousPhase d))
    (hL : ∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0) (T : ℝ) (hT : 0 < T) :
    ∃ θ : TimeParams d, PiecewiseConstant d θ T (L.length + 1) ∧
      (∀ s, (θ s).B = 0) ∧
      (∀ ρ : Perspective.ProbSphere d, ∃ μ : ℝ → Perspective.ProbSphere d,
        μ 0 = ρ ∧ cauchyPB d θ μ) ∧
      ∀ μ : ℝ → Perspective.ProbSphere d, cauchyPB d θ μ →
        phaseSequenceLaw L (μ T) = μ 0 := by
  obtain ⟨θ, hpc, hB, hex, hend⟩ := exists_phase_sequence_realization i (inversePhaseList L)
    (inversePhaseList_valid L hL) T hT
  refine ⟨θ, by simpa only [inversePhaseList_length] using hpc, hB, hex, ?_⟩
  intro μ hμ
  rw [hend μ hμ, inversePhaseList_cancel_right L (fun P hP => (hL P hP).1)]

/-- Original sequences matching a finite family to common intermediate
laws realize exact interpolation at any positive physical time. Source
auxiliary: arXiv:2411.04551v3, §1, the construction of `thm: targets.atoms`. -/
theorem exists_matching_controls_via_common {N : ℕ} (i : Fin d)
    (L M : List (AutonomousPhase d))
    (hL : ∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0)
    (hM : ∀ P ∈ M, IsAutonomousPhase P ∧ P.1.B = 0)
    (ρ σ ν : Idx N → Perspective.ProbSphere d)
    (hρ : ∀ k, phaseSequenceLaw L (ρ k) = ν k)
    (hσ : ∀ k, phaseSequenceLaw M (σ k) = ν k) (T : ℝ) (hT : 0 < T) :
    ∃ θ : TimeParams d, PiecewiseConstant d θ T (L.length + M.length + 1) ∧
      (∀ s, (θ s).B = 0) ∧
      (∀ η : Perspective.ProbSphere d, ∃ μ : ℝ → Perspective.ProbSphere d,
        μ 0 = η ∧ cauchyPB d θ μ) ∧
      ∀ k, ∀ μ : ℝ → Perspective.ProbSphere d, μ 0 = ρ k → cauchyPB d θ μ → μ T = σ k := by
  have hLM : ∀ P ∈ L ++ inversePhaseList M, IsAutonomousPhase P ∧ P.1.B = 0 := by
    intro P hP
    rcases List.mem_append.mp hP with hP | hP
    · exact hL P hP
    · exact inversePhaseList_valid M hM P hP
  obtain ⟨θ, hpc, hB, hex, hend⟩ := exists_phase_sequence_realization i
    (L ++ inversePhaseList M) hLM T hT
  refine ⟨θ, by simpa only [List.length_append, inversePhaseList_length] using hpc, hB, hex, ?_⟩
  intro k μ h0 hμ
  rw [hend μ hμ, h0]
  exact phaseSequenceLaw_matching_via_common L M (fun P hP => (hM P hP).1) _ _ _ (hρ k) (hσ k)

/-- A nonzero actual projection neuron, an off-axis unit atom and its
actual transported endpoint meet every phase-validity and matching
hypothesis with positive physical time. Using this same actual sequence
on both sides witnesses the common-intermediate-law hypotheses too. -/
example : ∃ (L : List (AutonomousPhase 2)) (ρ σ : Perspective.ProbSphere 2),
    (∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0) ∧ (0 : ℝ) < 1 ∧ phaseSequenceLaw L ρ = σ := by
  let p : EucSpace 2 := PiLp.single 2 (0 : Fin 2) 1
  let z : SSphere 2 := ⟨PiLp.single 2 (1 : Fin 2) (1 : ℝ), by simp⟩
  let P := ridgePhase (0 : Fin 2) p 0 1
  let L := [P]
  let ρ := Perspective.diracProb 2 z
  refine ⟨L, ρ, phaseSequenceLaw L ρ, ?_, one_pos, rfl⟩
  intro Q hQ
  have he : Q = P := List.mem_singleton.mp hQ
  subst Q
  exact ⟨isAutonomousPhase_ridge 0 p 0 1, ridgePhase_B_zero 0 p 0 1⟩

end Transformer.Interpolation
