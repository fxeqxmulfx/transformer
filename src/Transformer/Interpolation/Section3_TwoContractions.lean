/-
# Two projection phases reach the positive quadrant

ArXiv:2411.04551v3, §3.1, proof of `lem: first.quadrant`.
The original compact support misses a point `ω`. The first flow attracts
it to `-ω`, and a positive direction `α ≠ ω` supplies the second attractor.
We choose the first ball using the actual distance between the first
attractor and the second repeller. This replaces the source's fixed
`π/8` ball by a sufficiently small ball, which is all its proof needs.
The conclusion concerns the actual two characteristic maps on the
original supports, before any time rescaling or weak-solution argument.
-/

import Transformer.Interpolation.Section3_QuadrantGeometry

open Real MeasureTheory

namespace Transformer.Interpolation

variable {d N : ℕ}

/-- The first contraction can avoid any point other than its attractor.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`.
The source uses a fixed `π/8` ball; here its radius is half the distance
to the point that the second phase must avoid. -/
theorem exists_projection_time_avoiding (w p : SSphere d) (hwp : w ≠ p)
    (K : Set (SSphere d)) (hK : IsCompact K) (hmiss : antipode d w ∉ K) :
    ∃ t : ℝ, 0 < t ∧ p ∉ sphereProjectionFlow w t '' K := by
  have hδ : 0 < dist w p / 2 := half_pos (dist_pos.mpr hwp)
  obtain ⟨t, ht, hball⟩ :=
    exists_time_sphereProjectionFlow_ball w K hK hmiss (dist w p / 2) hδ
  refine ⟨t, ht, ?_⟩
  rintro ⟨x, hx, hxp⟩
  have h := hball x hx
  rw [hxp, dist_comm p w] at h
  linarith [dist_nonneg (x := w) (y := p)]

/-- A compact set missing the repeller of a positive direction reaches
the positive quadrant at one common positive time.
Source: arXiv:2411.04551v3, §3.1, the second phase in
`lem: first.quadrant`. -/
theorem exists_projection_time_quadrant (α : SSphere d)
    (hα : α ∈ positiveQuadrant d) (K : Set (SSphere d))
    (hK : IsCompact K) (hmiss : antipode d α ∉ K) :
    ∃ t : ℝ, 0 < t ∧ sphereProjectionFlow α t '' K ⊆ positiveQuadrant d := by
  obtain ⟨ε, hε, hball⟩ := exists_ball_positiveQuadrant α hα
  obtain ⟨t, ht, hflow⟩ := exists_time_sphereProjectionFlow_ball α K hK hmiss ε hε
  refine ⟨t, ht, ?_⟩
  rintro y ⟨x, hx, rfl⟩
  exact hball (hflow x hx)

/-- Two actual projection flows carry every point of a proper compact
spherical set into the positive quadrant.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`.
The dimension hypothesis supplies a positive direction different from
the missing point; the first radius is chosen after that direction. -/
theorem exists_two_projection_quadrant (hd : 2 ≤ d) (K : Set (SSphere d))
    (hK : IsCompact K) (hne : K ≠ Set.univ) :
    ∃ w₁ w₂ : SSphere d, ∃ τ₁ τ₂ : ℝ, 0 < τ₁ ∧ 0 < τ₂ ∧
      ∀ x ∈ K, sphereProjectionFlow w₂ τ₂ (sphereProjectionFlow w₁ τ₁ x)
        ∈ positiveQuadrant d := by
  have hmissing : ∃ ω, ω ∉ K := by
    by_contra! h
    exact hne (Set.eq_univ_iff_forall.mpr h)
  obtain ⟨ω, hω⟩ := hmissing
  obtain ⟨α, hα, hαω⟩ := exists_positiveQuadrant_ne hd ω
  have hdistinct : antipode d ω ≠ antipode d α := by
    intro h
    exact hαω (antipode_injective h).symm
  have hrepeller : antipode d (antipode d ω) ∉ K := by
    rw [antipode_antipode]
    exact hω
  obtain ⟨τ₁, hτ₁, hmiss⟩ := exists_projection_time_avoiding
    (antipode d ω) (antipode d α) hdistinct K hK hrepeller
  have himage : IsCompact (sphereProjectionFlow (antipode d ω) τ₁ '' K) :=
    hK.image ((continuous_sphereProjectionFlow _).comp (continuous_const.prodMk continuous_id))
  obtain ⟨τ₂, hτ₂, hquad⟩ := exists_projection_time_quadrant α hα _ himage hmiss
  refine ⟨antipode d ω, α, τ₁, τ₂, hτ₁, hτ₂, fun x hx => ?_⟩
  exact hquad ⟨sphereProjectionFlow (antipode d ω) τ₁ x, ⟨x, hx, rfl⟩, rfl⟩

/-- The original finite family of probability supports has compact union.
Source: arXiv:2411.04551v3, §3.1, the compactness step in
`lem: first.quadrant`. -/
theorem isCompact_support_union (μ₀ : Idx N → Perspective.ProbSphere d) :
    IsCompact (⋃ i : Idx N, (μ₀ i : Measure (SSphere d)).support) := by
  apply isCompact_iUnion
  intro i
  exact Measure.isClosed_support.isCompact

/-- Compactness persists after both actual contraction phases, with no
closure or enlarged support introduced between the phases.
Source: arXiv:2411.04551v3, §3.1, the compact-set argument in
`lem: first.quadrant`. -/
theorem isCompact_two_projection_image (w₁ w₂ : SSphere d) (τ₁ τ₂ : ℝ)
    (K : Set (SSphere d)) (hK : IsCompact K) :
    IsCompact (sphereProjectionFlow w₂ τ₂ '' (sphereProjectionFlow w₁ τ₁ '' K)) := by
  have hfirst : Continuous (sphereProjectionFlow w₁ τ₁) :=
    (continuous_sphereProjectionFlow w₁).comp
      (continuous_const.prodMk continuous_id)
  have hsecond : Continuous (sphereProjectionFlow w₂ τ₂) :=
    (continuous_sphereProjectionFlow w₂).comp
      (continuous_const.prodMk continuous_id)
  exact (hK.image hfirst).image hsecond

/-- Both intrinsic phase times are chosen from the original data,
simultaneously for every original support and independently of the
eventual control horizon.
Source: arXiv:2411.04551v3, §3.1, `lem: first.quadrant`. -/
theorem exists_two_projection_quadrant_supports (hd : 2 ≤ d)
    (μ₀ : Idx N → Perspective.ProbSphere d)
    (hne : (⋃ i : Idx N, (μ₀ i : Measure (SSphere d)).support) ≠ Set.univ) :
    ∃ w₁ w₂ : SSphere d, ∃ τ₁ τ₂ : ℝ, 0 < τ₁ ∧ 0 < τ₂ ∧
      ∀ i : Idx N, ∀ x ∈ (μ₀ i : Measure (SSphere d)).support,
        sphereProjectionFlow w₂ τ₂ (sphereProjectionFlow w₁ τ₁ x)
          ∈ positiveQuadrant d := by
  obtain ⟨w₁, w₂, τ₁, τ₂, hτ₁, hτ₂, hflow⟩ :=
    exists_two_projection_quadrant hd _ (isCompact_support_union μ₀) hne
  exact ⟨w₁, w₂, τ₁, τ₂, hτ₁, hτ₂,
    fun i x hx => hflow x (Set.mem_iUnion.mpr ⟨i, hx⟩)⟩

/-- All compact-set contraction hypotheses hold simultaneously for a
positive circle point and its distinct antipode. -/
example : let α := quadrantPoint (0 : Fin 2) 1 one_pos
    α ∈ positiveQuadrant 2 ∧ α ≠ antipode 2 α ∧
      IsCompact ({α} : Set (SSphere 2)) ∧
      antipode 2 α ∉ ({α} : Set (SSphere 2)) ∧
      ({α} : Set (SSphere 2)) ≠ Set.univ ∧ 2 ≤ 2 := by
  dsimp only
  let α := quadrantPoint (0 : Fin 2) 1 one_pos
  have hmiss : antipode 2 α ∉ ({α} : Set (SSphere 2)) := by
    simpa only [Set.mem_singleton_iff] using antipode_ne 2 α
  refine ⟨quadrantPoint_mem 0 1 one_pos, (antipode_ne 2 α).symm,
    isCompact_singleton, hmiss, ?_, le_rfl⟩
  intro h
  exact hmiss (h.symm ▸ Set.mem_univ _)

/-- The support-family hypothesis has actual probability data on the
circle, rather than an empty compact-set witness. -/
example : 2 ≤ 2 ∧
    (⋃ i : Idx 1, ((fun _ : Idx 1 => Perspective.diracProb 2 (basePoint 1)) i :
      Measure (SSphere 2)).support) ≠ Set.univ := by
  refine ⟨le_rfl, ?_⟩
  intro h
  have hmem : antipode 2 (basePoint 1) ∈
      ⋃ i : Idx 1, ((fun _ : Idx 1 => Perspective.diracProb 2 (basePoint 1)) i :
        Measure (SSphere 2)).support := h.symm ▸ Set.mem_univ _
  obtain ⟨i, hi⟩ := Set.mem_iUnion.mp hmem
  have hi' : antipode 2 (basePoint 1) ∈ (Measure.dirac (basePoint 1)).support := hi
  exact antipode_ne 2 (basePoint 1) (eq_of_mem_support_dirac hi')

end Transformer.Interpolation
