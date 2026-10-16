/-
# Replacing one actual quadrant atom while protecting all others

ArXiv:2411.04551v3, §1, `thm: targets.atoms`, and §3.2,
the protected interpolation construction. A selected positive unit
point can be moved to another positive unit point whenever all other
points lie outside their plane. Three original phases suffice, and
every other original Dirac law is restored exactly.

The family of other atoms can be empty. We adjoin a genuine positive
dummy atom outside the selected plane before applying the common-axis
construction. This supplies a nonempty finite family without adding a
hypothesis on the actual data. If the selected point is already its
target, the empty phase list gives the required replacement instead.

All phases have zero score matrices. Their realization at an arbitrary
positive prescribed horizon constructs original global weak solutions
and determines the endpoint of every original weak solution.
-/

import Transformer.Interpolation.Section4_GenericQuadrantPoints

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d N : ℕ}

/-- Replace one selected original unit atom and restore every other
one, including when there are no other atoms. Source auxiliary:
arXiv:2411.04551v3, §1 and §3.2, protected atomic interpolation. -/
theorem exists_quadrant_atom_replacement_phases (hd : 3 ≤ d) (i : Fin d)
    (X : Idx N → SSphere d) (hX : ∀ k, X k ∈ positiveQuadrant d)
    (j : Idx N) (q : SSphere d) (hq : q ∈ positiveQuadrant d)
    (hplane : ∀ k, k ≠ j → ∀ α β : ℝ,
      (X k : EucSpace d) ≠ α • (X j : EucSpace d) + β • (q : EucSpace d)) :
    ∃ L : List (AutonomousPhase d), L.length ≤ 3 ∧
      (∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0) ∧
      ∀ k, phaseSequenceLaw L (Perspective.diracProb d (X k)) =
        Perspective.diracProb d (if k = j then q else X k) := by
  classical
  by_cases he : X j = q
  · refine ⟨[], Nat.zero_le 3, ?_, ?_⟩
    · intro P hP
      cases hP
    · intro k
      rw [phaseSequenceLaw_nil]
      by_cases hkj : k = j
      · subst k
        rw [ite_eq_left rfl, he]
      · rw [ite_eq_right hkj]
  obtain ⟨r, hrQ, hr⟩ := exists_quadrant_point_avoiding_planes hd
    (fun _ : Fin 1 => (X j : EucSpace d)) (fun _ : Fin 1 => (q : EucSpace d))
  let J := Option {k : Idx N // k ≠ j}
  let D : J → SSphere d := fun k => match k with
    | none => r
    | some k => X k.1
  let η : J → Perspective.ProbSphere d := fun k => Perspective.diracProb d (D k)
  have hDQ (k : J) : D k ∈ positiveQuadrant d := by
    cases k with
    | none => exact hrQ
    | some k => exact hX k.1
  have hηQ (k : J) : (η k : Measure (SSphere d)).support ⊆ positiveQuadrant d := by
    intro z hz
    change z ∈ (Perspective.diracProb d (D k) : Measure (SSphere d)).support at hz
    rw [eq_of_mem_support_dirac hz]
    exact hDQ k
  have hηplane (k : J) (α β : ℝ) :
      barycenter d (η k) ≠ α • (X j : EucSpace d) + β • (q : EucSpace d) := by
    have hb : barycenter d (η k) = (D k : EucSpace d) := by
      simp [η, barycenter, Perspective.diracProb]
    rw [hb]
    cases k with
    | none => exact hr 0 α β
    | some k => exact hplane k.1 k.2 α β
  obtain ⟨L, hlen, hL, hselected, hrest⟩ :=
    exists_quadrant_transfer_phases i (X j) q (hX j) hq he η hηQ hηplane
  refine ⟨L, hlen.le, hL, ?_⟩
  intro k
  by_cases hkj : k = j
  · subst k
    rw [ite_eq_left rfl]
    exact hselected
  · rw [ite_eq_right hkj]
    exact hrest (some ⟨k, hkj⟩)

/-- The exact original atom replacement can be performed in at most
four constant pieces at any positive physical horizon. Every original
weak solution has the prescribed endpoint. Source auxiliary:
arXiv:2411.04551v3, §1, `thm: targets.atoms`, and §3.2. -/
theorem exists_quadrant_atom_replacement_controls (hd : 3 ≤ d) (i : Fin d)
    (X : Idx N → SSphere d) (hX : ∀ k, X k ∈ positiveQuadrant d)
    (j : Idx N) (q : SSphere d) (hq : q ∈ positiveQuadrant d)
    (hplane : ∀ k, k ≠ j → ∀ α β : ℝ,
      (X k : EucSpace d) ≠ α • (X j : EucSpace d) + β • (q : EucSpace d))
    (T : ℝ) (hT : 0 < T) :
    ∃ (θ : TimeParams d) (K : ℕ), K ≤ 4 ∧ PiecewiseConstant d θ T K ∧
      (∀ s, (θ s).B = 0) ∧
      (∀ ρ : Perspective.ProbSphere d, ∃ μ : ℝ → Perspective.ProbSphere d,
        μ 0 = ρ ∧ cauchyPB d θ μ) ∧
      ∀ k, ∀ μ : ℝ → Perspective.ProbSphere d,
        μ 0 = Perspective.diracProb d (X k) → cauchyPB d θ μ →
        μ T = Perspective.diracProb d (if k = j then q else X k) := by
  obtain ⟨L, hlen, hL, hmatch⟩ :=
    exists_quadrant_atom_replacement_phases hd i X hX j q hq hplane
  obtain ⟨θ, hpc, hB, hex, hend⟩ := exists_phase_sequence_realization i L hL T hT
  refine ⟨θ, L.length + 1, by omega, hpc, hB, hex, ?_⟩
  intro k μ h0 hμ
  rw [hend μ hμ, h0, hmatch k]

/-- Any two original positive unit points, equal or distinct, admit
an exact atomic transfer with the same three-phase upper bound.
Source auxiliary: arXiv:2411.04551v3, §1 and §3.2, the singleton
boundary case of protected atomic interpolation. -/
theorem exists_quadrant_singleton_transfer_phases (hd : 3 ≤ d) (i : Fin d)
    (p q : SSphere d) (hp : p ∈ positiveQuadrant d) (hq : q ∈ positiveQuadrant d) :
    ∃ L : List (AutonomousPhase d), L.length ≤ 3 ∧
      (∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0) ∧
      phaseSequenceLaw L (Perspective.diracProb d p) = Perspective.diracProb d q := by
  have hplane : ∀ k : Idx 1, k ≠ 0 → ∀ α β : ℝ,
      (p : EucSpace d) ≠ α • (p : EucSpace d) + β • (q : EucSpace d) := by
    intro k hk
    exact False.elim (hk (Subsingleton.elim _ _))
  obtain ⟨L, hlen, hL, hmatch⟩ := exists_quadrant_atom_replacement_phases hd i
    (fun _ : Idx 1 => p) (fun _ => hp) 0 q hq hplane
  refine ⟨L, hlen, hL, ?_⟩
  simpa using hmatch 0

/-- Genuine two-atom positive-quadrant data and a distinct positive
target satisfy all dimension, time and protection hypotheses. The
other atom lies strictly outside the actual selected-target plane. -/
example : ∃ (X : Idx 2 → SSphere 3) (j : Idx 2) (q : SSphere 3),
    3 ≤ 3 ∧ (0 : ℝ) < 1 ∧ (∀ k, X k ∈ positiveQuadrant 3) ∧
      q ∈ positiveQuadrant 3 ∧ X j ≠ q ∧
      ∀ k, k ≠ j → ∀ α β : ℝ,
        (X k : EucSpace 3) ≠ α • (X j : EucSpace 3) + β • (q : EucSpace 3) := by
  let X : Idx 2 → SSphere 3 := ![quadrantThreePoint 0, quadrantThreePoint 2]
  refine ⟨X, 0, quadrantThreePoint 1, le_rfl, one_pos, ?_, quadrantThreePoint_mem 1, ?_, ?_⟩
  · intro k
    fin_cases k
    · exact quadrantThreePoint_mem 0
    · exact quadrantThreePoint_mem 2
  · exact fun he => (by decide : (0 : Fin 3) ≠ 1) (quadrantThreePoint_injective he)
  · intro k hkj α β
    fin_cases k
    · exact False.elim (hkj rfl)
    · exact quadrantThreePoint_not_plane 0 1 2 (by decide) (by decide) (by decide) α β

/-- A genuine singleton positive atom needs no protection assumption;
the construction still applies to a distinct positive target. -/
example : ∃ (X : Idx 1 → SSphere 3) (j : Idx 1) (q : SSphere 3),
    3 ≤ 3 ∧ (∀ k, X k ∈ positiveQuadrant 3) ∧ q ∈ positiveQuadrant 3 ∧
      X j ≠ q ∧ ∀ k, k ≠ j → ∀ α β : ℝ,
        (X k : EucSpace 3) ≠ α • (X j : EucSpace 3) + β • (q : EucSpace 3) := by
  refine ⟨(fun _ => quadrantThreePoint 0), 0, quadrantThreePoint 1, le_rfl,
    (fun _ => quadrantThreePoint_mem 0), quadrantThreePoint_mem 1, ?_, ?_⟩
  · exact fun he => (by decide : (0 : Fin 3) ≠ 1) (quadrantThreePoint_injective he)
  · intro k hkj
    exact False.elim (hkj (Subsingleton.elim _ _))

end Transformer.Interpolation
