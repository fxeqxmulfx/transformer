module

public import Transformer.Interpolation.Section3_QuadrantPhaseSequence
public import Transformer.Interpolation.Section4_GenericQuadrantPoints

/-!
# Reducing arbitrary finite original atoms to the positive quadrant

ArXiv:2411.04551v3, §1, `thm: targets.atoms`, and §3.1,
`lem: first.quadrant`. A finite family of unit atoms always misses
a genuine point of the sphere in dimension at least three. Choose
such a point by avoiding every original one-vector span; the same
point misses the union of all original Dirac supports.

The existing two-projection construction sends the whole finite set
into the positive quadrant. Its actual point map is continuous and
bijective on the entire sphere, hence retains distinctness of the
input atoms. The corresponding two original autonomous phases have
their exact Dirac endpoints for every point, not just the chosen data.

No quadrant, hemisphere or antipodal restriction is imposed on the
original finite family. The new quadrant family is constructed from
the data before any physical horizon is chosen.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d N : ℕ}

/-- A genuine positive unit point misses every point of an arbitrary
original finite atomic family. Source auxiliary: arXiv:2411.04551v3,
§1 and §3.1, the support hole for finite input data. -/
theorem exists_finite_atomic_missing_point (hd : 3 ≤ d) (X : Idx N → SSphere d) :
    ∃ w : SSphere d, w ∈ positiveQuadrant d ∧ w ∉ Set.range X := by
  obtain ⟨w, hwQ, hw⟩ := exists_quadrant_point_avoiding_planes hd
    (fun k => (X k : EucSpace d)) (fun k => (X k : EucSpace d))
  refine ⟨w, hwQ, ?_⟩
  rintro ⟨k, hk⟩
  apply hw k 1 0
  simpa only [one_smul, zero_smul, add_zero] using congrArg Subtype.val hk.symm

/-- The actual union of the original finite Dirac supports is proper,
including for empty data. Source auxiliary: arXiv:2411.04551v3,
§3.1, the hypothesis of `lem: first.quadrant` for atomic data. -/
theorem finite_atomic_support_union_ne_univ (hd : 3 ≤ d) (X : Idx N → SSphere d) :
    (⋃ k, (Perspective.diracProb d (X k) : Measure (SSphere d)).support) ≠ Set.univ := by
  obtain ⟨w, _, hw⟩ := exists_finite_atomic_missing_point hd X
  intro he
  have hm : w ∈ ⋃ k, (Perspective.diracProb d (X k) : Measure (SSphere d)).support :=
    he.symm ▸ Set.mem_univ _
  obtain ⟨k, hk⟩ := Set.mem_iUnion.mp hm
  exact hw ⟨k, (eq_of_mem_support_dirac hk).symm⟩

/-- Two actual original phases supply a continuous bijection of the
whole sphere sending the prescribed arbitrary finite family into
the positive quadrant. Source: arXiv:2411.04551v3, §3.1,
`lem: first.quadrant`, specialized to finite original atoms. -/
theorem exists_finite_atomic_quadrant_map (hd : 3 ≤ d) (i : Fin d)
    (X : Idx N → SSphere d) :
    ∃ (L : List (AutonomousPhase d)) (F : SSphere d → SSphere d), L.length = 2 ∧
      (∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0) ∧
      Continuous F ∧ Function.Bijective F ∧ (∀ k, F (X k) ∈ positiveQuadrant d) ∧
      ∀ x, phaseSequenceLaw L (Perspective.diracProb d x) = Perspective.diracProb d (F x) := by
  obtain ⟨w, _, hw⟩ := exists_finite_atomic_missing_point hd X
  have hne : Set.range X ≠ Set.univ := by
    intro he
    exact hw (he.symm ▸ Set.mem_univ _)
  obtain ⟨w₁, w₂, τ₁, τ₂, _, _, hquad⟩ := exists_two_projection_quadrant
    (by omega : 2 ≤ d) (Set.range X) (Set.finite_range X).isCompact hne
  let F (x : SSphere d) := sphereProjectionFlow w₂ τ₂ (sphereProjectionFlow w₁ τ₁ x)
  have hFc : Continuous F :=
    ((continuous_sphereProjectionFlow w₂).comp (continuous_const.prodMk continuous_id)).comp
      ((continuous_sphereProjectionFlow w₁).comp (continuous_const.prodMk continuous_id))
  have hFsurj : Function.Surjective F := by
    intro x
    refine ⟨sphereProjectionFlow w₁ (-τ₁) (sphereProjectionFlow w₂ (-τ₂) x), ?_⟩
    change sphereProjectionFlow w₂ τ₂ (sphereProjectionFlow w₁ τ₁
      (sphereProjectionFlow w₁ (-τ₁) (sphereProjectionFlow w₂ (-τ₂) x))) = x
    rw [sphereProjectionFlow_neg_right, sphereProjectionFlow_neg_right]
  refine ⟨twoProjectionPhaseList i w₁ w₂ τ₁ τ₂, F, rfl,
    twoProjectionPhaseList_valid i w₁ w₂ τ₁ τ₂, hFc,
    ⟨injective_twoProjection_map w₁ w₂ τ₁ τ₂, hFsurj⟩, ?_, ?_⟩
  · intro k
    exact hquad (X k) ⟨k, rfl⟩
  · intro x
    exact phaseSequenceLaw_twoProjection_dirac i w₁ w₂ τ₁ τ₂ x

/-- Distinct arbitrary original atoms admit two actual phases whose
new atomic endpoints are distinct and strictly positive. Source
auxiliary: arXiv:2411.04551v3, §1 and §3.1, the reduction preceding
protected finite atomic interpolation. -/
theorem exists_atomic_quadrant_phases (hd : 3 ≤ d) (i : Fin d)
    (X : Idx N → SSphere d) (hinj : Function.Injective X) :
    ∃ (L : List (AutonomousPhase d)) (Z : Idx N → SSphere d), L.length = 2 ∧
      (∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0) ∧
      Function.Injective Z ∧ (∀ k, Z k ∈ positiveQuadrant d) ∧
      ∀ k, phaseSequenceLaw L (Perspective.diracProb d (X k)) = Perspective.diracProb d (Z k) := by
  obtain ⟨L, F, hlen, hL, _, hF, hquad, hend⟩ := exists_finite_atomic_quadrant_map hd i X
  refine ⟨L, (fun k => F (X k)), hlen, hL, ?_, hquad, fun k => hend (X k)⟩
  intro k l he
  exact hinj (hF.1 he)

/-- The two actual reversed preprocessing phases recover all original
atoms from a distinct positive-quadrant family. Source auxiliary:
arXiv:2411.04551v3, §1 and §3.1, inverse quadrant reduction. -/
theorem exists_atomic_quadrant_inverse_phases (hd : 3 ≤ d) (i : Fin d)
    (X : Idx N → SSphere d) (hinj : Function.Injective X) :
    ∃ (L : List (AutonomousPhase d)) (Z : Idx N → SSphere d), L.length = 2 ∧
      (∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0) ∧
      Function.Injective Z ∧ (∀ k, Z k ∈ positiveQuadrant d) ∧
      ∀ k, phaseSequenceLaw L (Perspective.diracProb d (Z k)) = Perspective.diracProb d (X k) := by
  obtain ⟨L, Z, hlen, hL, hinjZ, hZ, hend⟩ := exists_atomic_quadrant_phases hd i X hinj
  refine ⟨inversePhaseList L, Z, ?_, inversePhaseList_valid L hL, hinjZ, hZ, ?_⟩
  · rw [inversePhaseList_length, hlen]
  · intro k
    exact inversePhaseList_of_endpoint L (fun P hP => (hL P hP).1) _ _ (hend k)

/-- The same actual continuous bijection sending the finite atoms
into the quadrant is realized at every positive physical horizon.
The controls admit original global weak solutions from every law;
every original weak solution from any unit atom follows the same
endpoint map. Source: arXiv:2411.04551v3, §3.1,
`lem: first.quadrant`, with finite atomic input supports. -/
theorem exists_finite_atomic_quadrant_controls (hd : 3 ≤ d) (i : Fin d)
    (X : Idx N → SSphere d) (T : ℝ) (hT : 0 < T) :
    ∃ (θ : TimeParams d) (F : SSphere d → SSphere d),
      PiecewiseConstant d θ T 3 ∧ (∀ s, (θ s).B = 0) ∧
      Continuous F ∧ Function.Bijective F ∧ (∀ k, F (X k) ∈ positiveQuadrant d) ∧
      (∀ ρ : Perspective.ProbSphere d, ∃ μ : ℝ → Perspective.ProbSphere d,
        μ 0 = ρ ∧ cauchyPB d θ μ) ∧
      ∀ x, ∀ μ : ℝ → Perspective.ProbSphere d,
        μ 0 = Perspective.diracProb d x → cauchyPB d θ μ →
        μ T = Perspective.diracProb d (F x) := by
  obtain ⟨L, F, hlen, hL, hFc, hF, hquad, hmatch⟩ := exists_finite_atomic_quadrant_map hd i X
  obtain ⟨θ, hpc, hB, hex, hend⟩ := exists_phase_sequence_realization i L hL T hT
  refine ⟨θ, F, ?_, hB, hFc, hF, hquad, hex, ?_⟩
  · simpa only [hlen] using hpc
  · intro x μ h0 hμ
    rw [hend μ hμ, h0, hmatch x]

/-- Genuine antipodal input atoms satisfy all finite-reduction
hypotheses. In particular the construction imposes no original
positive-quadrant or non-antipodal condition on the data. -/
example : ∃ X : Idx 2 → SSphere 3, 3 ≤ 3 ∧ (0 : ℝ) < 1 ∧ Function.Injective X ∧
    X 1 = antipode 3 (X 0) := by
  let p := basePoint 2
  let X : Idx 2 → SSphere 3 := ![p, antipode 3 p]
  refine ⟨X, le_rfl, one_pos, ?_, rfl⟩
  intro k l he
  fin_cases k <;> fin_cases l
  · rfl
  · exact False.elim (antipode_ne 3 p he.symm)
  · exact False.elim (antipode_ne 3 p he)
  · rfl

end Transformer.Interpolation
