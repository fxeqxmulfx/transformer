/-
# Exact interpolation of arbitrary finite original atom families

ArXiv:2411.04551v3, §1, `thm: targets.atoms`, §3.1,
the quadrant reduction, and §3.2, protected finite transfers.
Reduce both injective original atom families to the positive quadrant,
match those families exactly, and reverse the target reduction.
This constructs at most six phases per atom plus four preprocessing
phases, with zero original score matrices throughout.

The construction requires no support-hole hypothesis for the finite
atoms: a missing point was constructed for each endpoint family.
Every positive physical horizon admits common original controls with
global weak existence and the exact desired endpoint for every weak
solution. For nonempty data, four times dN bounds the piece count.

The exact phase and control statements also permit empty families.
Only the dN reformulation assumes nonempty data: a positive physical
time interval has the terminal constant piece even when there are
no input atoms to match.

This proves the exact finite-atomic full-network special case. It
does not identify the paper's perceptron-only interpolation theorem
with a construction that also uses rank-one attention.
-/

import Transformer.Interpolation.Section4_AtomicQuadrantReduction
import Transformer.Interpolation.Section4_QuadrantAtomicMatching

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d N : ℕ}

/-- Any two distinct original unit-atom families admit simultaneous
exact matching by actual original phases. Source auxiliary:
arXiv:2411.04551v3, §1, the finite-atomic case of `thm: targets.atoms`. -/
theorem exists_atomic_matching_phases (hd : 3 ≤ d) (i : Fin d)
    (X Y : Idx N → SSphere d) (hX : Function.Injective X) (hY : Function.Injective Y) :
    ∃ L : List (AutonomousPhase d), L.length ≤ 6 * N + 4 ∧
      (∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0) ∧
      ∀ k, phaseSequenceLaw L (Perspective.diracProb d (X k)) = Perspective.diracProb d (Y k) := by
  obtain ⟨A, X', hAlen, hA, hXinj, hXQ, hAX⟩ := exists_atomic_quadrant_phases hd i X hX
  obtain ⟨B, Y', hBlen, hB, hYinj, hYQ, hBY⟩ := exists_atomic_quadrant_phases hd i Y hY
  obtain ⟨M, hMlen, hM, hmatch⟩ :=
    exists_quadrant_atomic_matching_phases hd i X' Y' hXinj hYinj hXQ hYQ
  refine ⟨A ++ (M ++ inversePhaseList B), ?_, ?_, ?_⟩
  · rw [List.length_append, List.length_append, inversePhaseList_length, hAlen, hBlen]
    omega
  · intro P hP
    rcases List.mem_append.mp hP with hP | hP
    · exact hA P hP
    · rcases List.mem_append.mp hP with hP | hP
      · exact hM P hP
      · exact inversePhaseList_valid B hB P hP
  · intro k
    rw [phaseSequenceLaw_append, hAX k, phaseSequenceLaw_append, hmatch k]
    exact inversePhaseList_of_endpoint B (fun P hP => (hB P hP).1) _ _ (hBY k)

/-- Exact interpolation of all original atom pairs is realized by
common zero-score controls at any positive time. Every original weak
solution has the prescribed endpoint. Source auxiliary:
arXiv:2411.04551v3, §1, `thm: targets.atoms`, finite atomic inputs. -/
theorem exists_atomic_matching_controls (hd : 3 ≤ d) (i : Fin d)
    (X Y : Idx N → SSphere d) (hX : Function.Injective X) (hY : Function.Injective Y)
    (T : ℝ) (hT : 0 < T) :
    ∃ (θ : TimeParams d) (K : ℕ), K ≤ 6 * N + 5 ∧ PiecewiseConstant d θ T K ∧
      (∀ s, (θ s).B = 0) ∧
      (∀ ρ : Perspective.ProbSphere d, ∃ μ : ℝ → Perspective.ProbSphere d,
        μ 0 = ρ ∧ cauchyPB d θ μ) ∧
      ∀ k, ∀ μ : ℝ → Perspective.ProbSphere d,
        μ 0 = Perspective.diracProb d (X k) → cauchyPB d θ μ →
        μ T = Perspective.diracProb d (Y k) := by
  obtain ⟨L, hlen, hL, hmatch⟩ := exists_atomic_matching_phases hd i X Y hX hY
  obtain ⟨θ, hpc, hB, hex, hend⟩ := exists_phase_sequence_realization i L hL T hT
  refine ⟨θ, L.length + 1, by omega, hpc, hB, hex, ?_⟩
  intro k μ h0 hμ
  rw [hend μ hμ, h0, hmatch k]

/-- The same controls construct a complete family of actual original
weak solutions with exact atomic endpoints. Source auxiliary:
arXiv:2411.04551v3, §1, the atomic specialization of `thm: targets.atoms`. -/
theorem exists_atomic_matching_solutions (hd : 3 ≤ d) (i : Fin d)
    (X Y : Idx N → SSphere d) (hX : Function.Injective X) (hY : Function.Injective Y)
    (T : ℝ) (hT : 0 < T) :
    ∃ (θ : TimeParams d) (K : ℕ) (μ : Idx N → ℝ → Perspective.ProbSphere d),
      K ≤ 6 * N + 5 ∧ PiecewiseConstant d θ T K ∧ (∀ s, (θ s).B = 0) ∧
      (∀ k, μ k 0 = Perspective.diracProb d (X k) ∧ cauchyPB d θ (μ k)) ∧
      ∀ k, μ k T = Perspective.diracProb d (Y k) := by
  obtain ⟨θ, K, hK, hpc, hB, hex, hend⟩ := exists_atomic_matching_controls hd i X Y hX hY T hT
  choose μ h0 hμ using fun k => hex (Perspective.diracProb d (X k))
  exact ⟨θ, K, μ, hK, hpc, hB, (fun k => ⟨h0 k, hμ k⟩), fun k => hend k (μ k) (h0 k) (hμ k)⟩

/-- Nonempty original atomic data admit a uniform four-dN piece
bound, independent of the positions and the horizon. Source
auxiliary: arXiv:2411.04551v3, §1, the switch count in
`thm: targets.atoms`; this is only its exact finite-atomic case. -/
theorem exists_atomic_matching_linear_pieces (hd : 3 ≤ d) (hN : 1 ≤ N)
    (X Y : Idx N → SSphere d) (hX : Function.Injective X) (hY : Function.Injective Y)
    (T : ℝ) (hT : 0 < T) :
    ∃ (θ : TimeParams d) (K : ℕ) (μ : Idx N → ℝ → Perspective.ProbSphere d),
      K ≤ 4 * d * N ∧ PiecewiseConstant d θ T K ∧ (∀ s, (θ s).B = 0) ∧
      (∀ k, μ k 0 = Perspective.diracProb d (X k) ∧ cauchyPB d θ (μ k)) ∧
      ∀ k, μ k T = Perspective.diracProb d (Y k) := by
  let i : Fin d := ⟨0, by omega⟩
  obtain ⟨θ, K, μ, hK, hpc, hB, hμ, hend⟩ := exists_atomic_matching_solutions hd i X Y hX hY T hT
  refine ⟨θ, K, μ, hK.trans ?_, hpc, hB, hμ, hend⟩
  nlinarith

/-- Finite original Dirac inputs satisfy the paper's atomic-target
accuracy conclusion at every positive tolerance, with one uniform
four-dN piece bound. Their constructed error is exactly zero.
Source auxiliary: arXiv:2411.04551v3, §1, `thm: targets.atoms`;
the general-measure and quantitative norm conclusions are separate. -/
theorem exists_atomic_target_accuracy (hd : 3 ≤ d) (hN : 1 ≤ N)
    (X Y : Idx N → SSphere d) (hX : Function.Injective X) (hY : Function.Injective Y)
    (T ε : ℝ) (hT : 0 < T) (hε : 0 < ε) :
    ∃ (θ : TimeParams d) (K : ℕ) (μ : Idx N → ℝ → Perspective.ProbSphere d),
      (K : ℝ) ≤ 4 * (d * N : ℝ) ∧ PiecewiseConstant d θ T K ∧
      (∀ k, μ k 0 = Perspective.diracProb d (X k) ∧ cauchyPB d θ (μ k)) ∧
      ∀ k, W2 d (μ k T : Measure (SSphere d)) (Measure.dirac (Y k)) ≤ ε := by
  obtain ⟨θ, K, μ, hK, hpc, _, hμ, hend⟩ :=
    exists_atomic_matching_linear_pieces hd hN X Y hX hY T hT
  refine ⟨θ, K, μ, ?_, hpc, hμ, ?_⟩
  · rw [← mul_assoc]
    exact_mod_cast hK
  · intro k
    rw [hend k]
    change W2 d (Measure.dirac (Y k)) (Measure.dirac (Y k)) ≤ ε
    rw [W2_self]
    exact hε.le

/-- A genuine swap of two antipodal original atoms has distinct
inputs, distinct targets, positive dimension and horizon, and a
nonempty index family. No original quadrant assumption is needed. -/
example : ∃ X Y : Idx 2 → SSphere 3, 3 ≤ 3 ∧ 1 ≤ 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
    Function.Injective X ∧ Function.Injective Y ∧ X 1 = antipode 3 (X 0) ∧ X 0 ≠ Y 0 := by
  let p := basePoint 2
  let X : Idx 2 → SSphere 3 := ![p, antipode 3 p]
  let Y : Idx 2 → SSphere 3 := fun k => X (Fin.rev k)
  have hinj : Function.Injective X := by
    intro k l he
    fin_cases k <;> fin_cases l
    · rfl
    · exact False.elim (antipode_ne 3 p he.symm)
    · exact False.elim (antipode_ne 3 p he)
    · rfl
  refine ⟨X, Y, le_rfl, by decide, one_pos, one_pos, hinj, hinj.comp Fin.rev_injective, rfl, ?_⟩
  exact fun he => antipode_ne 3 p he.symm

end Transformer.Interpolation
