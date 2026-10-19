module

public import Transformer.Interpolation.Section4_AtomsToGeneric
public import Transformer.Interpolation.Section1_PhaseSequenceInverse

/-!
# Exact matching of finite original positive-quadrant atom families

ArXiv:2411.04551v3, §1, `thm: targets.atoms`, and §3.2,
the protected interpolation construction. Choose one generic positive
intermediate family relative to both original endpoint families.
Send each endpoint family to that common family by protected actual
transfers, then reverse the second sequence.

This constructs at most six original phases per atom. They have zero
scores, and their actual original weak laws match the entire finite
family exactly. At every positive prescribed physical horizon the
realization has at most six times the family size plus one constant
pieces, admits global weak solutions from every probability law, and
determines the endpoint of every original weak solution.

This is an auxiliary full-network construction using attention as
well as perceptrons. The perceptron-only claim of §4 is separate.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d N : ℕ}

/-- Any two injective finite families of positive original unit
atoms admit simultaneous exact matching by at most six phases per
atom. Source auxiliary: arXiv:2411.04551v3, §1 and §3.2. -/
theorem exists_quadrant_atomic_matching_phases (hd : 3 ≤ d) (i : Fin d)
    (X Y : Idx N → SSphere d) (hXinj : Function.Injective X) (hYinj : Function.Injective Y)
    (hX : ∀ k, X k ∈ positiveQuadrant d) (hY : ∀ k, Y k ∈ positiveQuadrant d) :
    ∃ L : List (AutonomousPhase d), L.length ≤ 6 * N ∧
      (∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0) ∧
      ∀ k, phaseSequenceLaw L (Perspective.diracProb d (X k)) = Perspective.diracProb d (Y k) := by
  let A : (Idx N ⊕ Idx N) → EucSpace d :=
    Sum.elim (fun k => (X k : EucSpace d)) (fun k => (Y k : EucSpace d))
  obtain ⟨Z, hZ, hplanes, hpairs⟩ := exists_generic_quadrant_family hd A N
  obtain ⟨L, hlen, hL, hLX⟩ := exists_quadrant_atoms_to_generic_phases hd i X Z hXinj hX hZ
    (fun k j l α β => hplanes k (Sum.inl j) (Sum.inl l) α β)
    (fun k l hkl j α β => hpairs k l hkl (Sum.inl j) α β)
  obtain ⟨M, hMlen, hM, hMY⟩ := exists_quadrant_atoms_to_generic_phases hd i Y Z hYinj hY hZ
    (fun k j l α β => hplanes k (Sum.inr j) (Sum.inr l) α β)
    (fun k l hkl j α β => hpairs k l hkl (Sum.inr j) α β)
  refine ⟨L ++ inversePhaseList M, ?_, ?_, ?_⟩
  · rw [List.length_append, inversePhaseList_length]
    omega
  · intro P hP
    rcases List.mem_append.mp hP with hP | hP
    · exact hL P hP
    · exact inversePhaseList_valid M hM P hP
  · intro k
    exact phaseSequenceLaw_matching_via_common L M (fun P hP => (hM P hP).1)
      _ _ _ (hLX k) (hMY k)

/-- Simultaneous exact original atomic matching is realized by common
zero-score controls at every positive horizon. Every weak endpoint
is identified. Source auxiliary: arXiv:2411.04551v3, §1,
`thm: targets.atoms`, and §3.2, prescribed-time realization. -/
theorem exists_quadrant_atomic_matching_controls (hd : 3 ≤ d) (i : Fin d)
    (X Y : Idx N → SSphere d) (hXinj : Function.Injective X) (hYinj : Function.Injective Y)
    (hX : ∀ k, X k ∈ positiveQuadrant d) (hY : ∀ k, Y k ∈ positiveQuadrant d)
    (T : ℝ) (hT : 0 < T) :
    ∃ (θ : TimeParams d) (K : ℕ), K ≤ 6 * N + 1 ∧ PiecewiseConstant d θ T K ∧
      (∀ s, (θ s).B = 0) ∧
      (∀ ρ : Perspective.ProbSphere d, ∃ μ : ℝ → Perspective.ProbSphere d,
        μ 0 = ρ ∧ cauchyPB d θ μ) ∧
      ∀ k, ∀ μ : ℝ → Perspective.ProbSphere d,
        μ 0 = Perspective.diracProb d (X k) → cauchyPB d θ μ →
        μ T = Perspective.diracProb d (Y k) := by
  obtain ⟨L, hlen, hL, hmatch⟩ := exists_quadrant_atomic_matching_phases hd i X Y hXinj hYinj hX hY
  obtain ⟨θ, hpc, hB, hex, hend⟩ := exists_phase_sequence_realization i L hL T hT
  refine ⟨θ, L.length + 1, by omega, hpc, hB, hex, ?_⟩
  intro k μ h0 hμ
  rw [hend μ hμ, h0, hmatch k]

/-- The common original controls supply an actual family of global
weak solutions with exact prescribed atomic endpoints. Source
auxiliary: arXiv:2411.04551v3, §1, the atomic case of
`thm: targets.atoms`; no approximation error is needed for these data. -/
theorem exists_quadrant_atomic_matching_solutions (hd : 3 ≤ d) (i : Fin d)
    (X Y : Idx N → SSphere d) (hXinj : Function.Injective X) (hYinj : Function.Injective Y)
    (hX : ∀ k, X k ∈ positiveQuadrant d) (hY : ∀ k, Y k ∈ positiveQuadrant d)
    (T : ℝ) (hT : 0 < T) :
    ∃ (θ : TimeParams d) (K : ℕ) (μ : Idx N → ℝ → Perspective.ProbSphere d),
      K ≤ 6 * N + 1 ∧ PiecewiseConstant d θ T K ∧ (∀ s, (θ s).B = 0) ∧
      (∀ k, μ k 0 = Perspective.diracProb d (X k) ∧ cauchyPB d θ (μ k)) ∧
      ∀ k, μ k T = Perspective.diracProb d (Y k) := by
  obtain ⟨θ, K, hK, hpc, hB, hex, hend⟩ :=
    exists_quadrant_atomic_matching_controls hd i X Y hXinj hYinj hX hY T hT
  choose μ h0 hμ using fun k => hex (Perspective.diracProb d (X k))
  exact ⟨θ, K, μ, hK, hpc, hB, (fun k => ⟨h0 k, hμ k⟩), fun k => hend k (μ k) (h0 k) (hμ k)⟩

/-- Exact original matching also admits a realized inverse with the
same number of constant pieces. Both controls construct global weak
solutions from every original law. Source auxiliary:
arXiv:2411.04551v3, §1 and §3.2, time-reversible interpolation. -/
theorem exists_bidirectional_quadrant_atomic_controls (hd : 3 ≤ d) (i : Fin d)
    (X Y : Idx N → SSphere d) (hXinj : Function.Injective X) (hYinj : Function.Injective Y)
    (hX : ∀ k, X k ∈ positiveQuadrant d) (hY : ∀ k, Y k ∈ positiveQuadrant d)
    (T : ℝ) (hT : 0 < T) :
    ∃ (θ ψ : TimeParams d) (K : ℕ), K ≤ 6 * N + 1 ∧
      PiecewiseConstant d θ T K ∧ PiecewiseConstant d ψ T K ∧
      (∀ s, (θ s).B = 0 ∧ (ψ s).B = 0) ∧
      (∀ ρ : Perspective.ProbSphere d, ∃ μ ν : ℝ → Perspective.ProbSphere d,
        μ 0 = ρ ∧ cauchyPB d θ μ ∧ ν 0 = ρ ∧ cauchyPB d ψ ν) ∧
      (∀ k, ∀ μ : ℝ → Perspective.ProbSphere d,
        μ 0 = Perspective.diracProb d (X k) → cauchyPB d θ μ →
        μ T = Perspective.diracProb d (Y k)) ∧
      ∀ k, ∀ ν : ℝ → Perspective.ProbSphere d,
        ν 0 = Perspective.diracProb d (Y k) → cauchyPB d ψ ν →
        ν T = Perspective.diracProb d (X k) := by
  obtain ⟨L, hlen, hL, hmatch⟩ := exists_quadrant_atomic_matching_phases hd i X Y hXinj hYinj hX hY
  obtain ⟨θ, hpc, hB, hex, hend⟩ := exists_phase_sequence_realization i L hL T hT
  obtain ⟨ψ, hpc', hB', hex', hend'⟩ :=
    exists_phase_sequence_realization i (inversePhaseList L) (inversePhaseList_valid L hL) T hT
  refine ⟨θ, ψ, L.length + 1, by omega, hpc, ?_, (fun s => ⟨hB s, hB' s⟩), ?_, ?_, ?_⟩
  · simpa only [inversePhaseList_length] using hpc'
  · intro ρ
    obtain ⟨μ, hμ0, hμ⟩ := hex ρ
    obtain ⟨ν, hν0, hν⟩ := hex' ρ
    exact ⟨μ, ν, hμ0, hμ, hν0, hν⟩
  · intro k μ h0 hμ
    rw [hend μ hμ, h0, hmatch k]
  · intro k ν h0 hν
    rw [hend' ν hν, h0]
    exact inversePhaseList_of_endpoint L (fun P hP => (hL P hP).1) _ _ (hmatch k)

/-- A genuine swap of two distinct strictly positive original unit
atoms satisfies every injectivity and quadrant hypothesis, in positive
dimension three and at a positive prescribed physical horizon. -/
example : ∃ X Y : Idx 2 → SSphere 3, 3 ≤ 3 ∧ (0 : ℝ) < 1 ∧
    Function.Injective X ∧ Function.Injective Y ∧
      (∀ k, X k ∈ positiveQuadrant 3) ∧ (∀ k, Y k ∈ positiveQuadrant 3) ∧ X 0 ≠ Y 0 := by
  let X (k : Idx 2) := quadrantThreePoint k.castSucc
  let Y (k : Idx 2) := quadrantThreePoint (Fin.rev k).castSucc
  refine ⟨X, Y, le_rfl, one_pos, ?_, ?_, ?_, ?_, ?_⟩
  · intro k l he
    exact Fin.castSucc_injective 2 (quadrantThreePoint_injective he)
  · intro k l he
    exact Fin.rev_injective (Fin.castSucc_injective 2 (quadrantThreePoint_injective he))
  · intro k
    exact quadrantThreePoint_mem k.castSucc
  · intro k
    exact quadrantThreePoint_mem (Fin.rev k).castSucc
  · intro he
    have hh := quadrantThreePoint_injective he
    change (0 : Fin 3) = 1 at hh
    exact (by decide : (0 : Fin 3) ≠ 1) hh

end Transformer.Interpolation
