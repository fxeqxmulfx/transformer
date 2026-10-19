module

public import Transformer.Interpolation.Section4_GenericQuadrantFamily
public import Transformer.Interpolation.Section4_AtomReplacement

/-!
# Matching original quadrant atoms to a generic finite family

ArXiv:2411.04551v3, §1, `thm: targets.atoms`, and §3.2,
the protected interpolation construction. Replace the atoms one at
a time. An atom already processed is a new generic point; an atom
not yet processed is an old point. The two respective plane-exclusion
conditions protect both kinds during every actual replacement.

Induction over the finite set of processed indices constructs an
actual phase list of length at most three times its cardinality.
At the full index set, all original Dirac inputs reach their generic
targets exactly. The same original controls work for the whole family,
and their prescribed-time realization determines all weak endpoints.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d N : ℕ}

/-- An arbitrary finite set of original positive atoms can be replaced
by its generic targets while retaining all other atoms. Source
auxiliary: arXiv:2411.04551v3, §1 and §3.2, finite interpolation. -/
theorem exists_partial_quadrant_atoms_to_generic_phases (hd : 3 ≤ d) (i : Fin d)
    (X Z : Idx N → SSphere d) (hinj : Function.Injective X)
    (hX : ∀ k, X k ∈ positiveQuadrant d) (hZ : ∀ k, Z k ∈ positiveQuadrant d)
    (hplanes : ∀ k j l, ∀ α β : ℝ,
      (Z k : EucSpace d) ≠ α • (X j : EucSpace d) + β • (X l : EucSpace d))
    (hpairs : ∀ k l, k ≠ l → ∀ j, ∀ α β : ℝ,
      (Z k : EucSpace d) ≠ α • (X j : EucSpace d) + β • (Z l : EucSpace d))
    (S : Finset (Idx N)) :
    ∃ L : List (AutonomousPhase d), L.length ≤ 3 * S.card ∧
      (∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0) ∧
      ∀ k, phaseSequenceLaw L (Perspective.diracProb d (X k)) =
        Perspective.diracProb d (if k ∈ S then Z k else X k) := by
  classical
  induction S using Finset.induction_on with
  | empty =>
    refine ⟨[], Nat.zero_le _, ?_, ?_⟩
    · intro P hP
      cases hP
    · intro k
      simp [phaseSequenceLaw_nil]
  | @insert j S hj ih =>
    obtain ⟨L, hlen, hL, hmatch⟩ := ih
    let C (k : Idx N) := if k ∈ S then Z k else X k
    have hCQ (k : Idx N) : C k ∈ positiveQuadrant d := by
      dsimp only [C]
      split_ifs
      · exact hZ k
      · exact hX k
    have hCj : C j = X j := ite_eq_right hj
    have hCplane (k : Idx N) (hkj : k ≠ j) (α β : ℝ) :
        (C k : EucSpace d) ≠ α • (C j : EucSpace d) + β • (Z j : EucSpace d) := by
      rw [hCj]
      by_cases hk : k ∈ S
      · rw [show C k = Z k from ite_eq_left hk]
        exact hpairs k j hkj j α β
      · rw [show C k = X k from ite_eq_right hk]
        exact generic_old_outside_transfer_plane (fun l => (X l : EucSpace d)) Z
          hplanes j j k (quadrant_unit_noncollinear i (X j) (X k) (hX j) (hX k)
            (fun he => hkj (hinj he))) α β
    obtain ⟨R, hRlen, hR, hreplace⟩ :=
      exists_quadrant_atom_replacement_phases hd i C hCQ j (Z j) (hZ j) hCplane
    refine ⟨L ++ R, ?_, ?_, ?_⟩
    · rw [List.length_append, Finset.card_insert_of_notMem hj]
      omega
    · intro P hP
      rcases List.mem_append.mp hP with hP | hP
      · exact hL P hP
      · exact hR P hP
    · intro k
      rw [phaseSequenceLaw_append, hmatch k]
      change phaseSequenceLaw R (Perspective.diracProb d (C k)) = _
      rw [hreplace k]
      apply congrArg (Perspective.diracProb d)
      by_cases hkj : k = j
      · subst k
        simp
      · by_cases hk : k ∈ S
        · simp [hkj, hk, C]
        · simp [hkj, hk, C]

/-- A generic positive target family can be reached exactly from
distinct original quadrant atoms in at most three phases per atom.
Source auxiliary: arXiv:2411.04551v3, §1 and §3.2. -/
theorem exists_quadrant_atoms_to_generic_phases (hd : 3 ≤ d) (i : Fin d)
    (X Z : Idx N → SSphere d) (hinj : Function.Injective X)
    (hX : ∀ k, X k ∈ positiveQuadrant d) (hZ : ∀ k, Z k ∈ positiveQuadrant d)
    (hplanes : ∀ k j l, ∀ α β : ℝ,
      (Z k : EucSpace d) ≠ α • (X j : EucSpace d) + β • (X l : EucSpace d))
    (hpairs : ∀ k l, k ≠ l → ∀ j, ∀ α β : ℝ,
      (Z k : EucSpace d) ≠ α • (X j : EucSpace d) + β • (Z l : EucSpace d)) :
    ∃ L : List (AutonomousPhase d), L.length ≤ 3 * N ∧
      (∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0) ∧
      ∀ k, phaseSequenceLaw L (Perspective.diracProb d (X k)) = Perspective.diracProb d (Z k) := by
  classical
  obtain ⟨L, hlen, hL, hmatch⟩ :=
    exists_partial_quadrant_atoms_to_generic_phases hd i X Z hinj hX hZ hplanes hpairs Finset.univ
  refine ⟨L, ?_, hL, ?_⟩
  · simpa only [Finset.card_univ, Fintype.card_fin] using hlen
  · intro k
    simpa using hmatch k

/-- The same actual zero-score controls realize all generic atomic
matches at any positive prescribed time. Source auxiliary:
arXiv:2411.04551v3, §1, `thm: targets.atoms`, and §3.2. -/
theorem exists_quadrant_atoms_to_generic_controls (hd : 3 ≤ d) (i : Fin d)
    (X Z : Idx N → SSphere d) (hinj : Function.Injective X)
    (hX : ∀ k, X k ∈ positiveQuadrant d) (hZ : ∀ k, Z k ∈ positiveQuadrant d)
    (hplanes : ∀ k j l, ∀ α β : ℝ,
      (Z k : EucSpace d) ≠ α • (X j : EucSpace d) + β • (X l : EucSpace d))
    (hpairs : ∀ k l, k ≠ l → ∀ j, ∀ α β : ℝ,
      (Z k : EucSpace d) ≠ α • (X j : EucSpace d) + β • (Z l : EucSpace d))
    (T : ℝ) (hT : 0 < T) :
    ∃ (θ : TimeParams d) (K : ℕ), K ≤ 3 * N + 1 ∧ PiecewiseConstant d θ T K ∧
      (∀ s, (θ s).B = 0) ∧
      (∀ ρ : Perspective.ProbSphere d, ∃ μ : ℝ → Perspective.ProbSphere d,
        μ 0 = ρ ∧ cauchyPB d θ μ) ∧
      ∀ k, ∀ μ : ℝ → Perspective.ProbSphere d,
        μ 0 = Perspective.diracProb d (X k) → cauchyPB d θ μ →
        μ T = Perspective.diracProb d (Z k) := by
  obtain ⟨L, hlen, hL, hmatch⟩ :=
    exists_quadrant_atoms_to_generic_phases hd i X Z hinj hX hZ hplanes hpairs
  obtain ⟨θ, hpc, hB, hex, hend⟩ := exists_phase_sequence_realization i L hL T hT
  refine ⟨θ, L.length + 1, by omega, hpc, hB, hex, ?_⟩
  intro k μ h0 hμ
  rw [hend μ hμ, h0, hmatch k]

/-- Two distinct genuine positive atoms and a generic positive target
family jointly realize all dimension, injectivity, quadrant and plane
hypotheses, with a nonempty processed set and positive physical time. -/
example : ∃ (X Z : Idx 2 → SSphere 3) (S : Finset (Idx 2)),
    3 ≤ 3 ∧ (0 : ℝ) < 1 ∧ S.Nonempty ∧ Function.Injective X ∧
      (∀ k, X k ∈ positiveQuadrant 3) ∧ (∀ k, Z k ∈ positiveQuadrant 3) ∧
      (∀ k j l, ∀ α β : ℝ,
        (Z k : EucSpace 3) ≠ α • (X j : EucSpace 3) + β • (X l : EucSpace 3)) ∧
      ∀ k l, k ≠ l → ∀ j, ∀ α β : ℝ,
        (Z k : EucSpace 3) ≠ α • (X j : EucSpace 3) + β • (Z l : EucSpace 3) := by
  let X (k : Idx 2) := quadrantThreePoint k.castSucc
  obtain ⟨Z, hZ, hplanes, hpairs⟩ :=
    exists_generic_quadrant_family (d := 3) le_rfl (fun k => (X k : EucSpace 3)) 2
  refine ⟨X, Z, {0}, le_rfl, one_pos, by simp, ?_, ?_, hZ, hplanes, hpairs⟩
  · intro k l he
    exact Fin.castSucc_injective 2 (quadrantThreePoint_injective he)
  · intro k
    exact quadrantThreePoint_mem k.castSucc

end Transformer.Interpolation
