/-
# Short arcs of the circle, and windows of angles

ArXiv:2411.04551v3, §2.2, proof of `prop: compression`, on the circle
(`d = 2`). The point at an angle between two angles closer than `π` lies
in the geodesic hull of their points. Hence a region of the circle that
is open, closed under geodesic hulls and inside an open half-plane is the
arc of an interval of angles; the whole circle less one point is the arc
of a window of length `2π`. Either way one window of angles of length at
most `2π` has its arc inside the region, carrying every law of the region
without atoms and every given point of it.
-/

import Transformer.Interpolation.Section2_CircleChart
import Transformer.Interpolation.AtomlessSphere

open scoped ENNReal
open Real MeasureTheory Set

namespace Transformer.Interpolation

/-- The point at angle `s` is `cos s` times the first axis plus `sin s`
times the second. -/
theorem coe_circlePoint (s : ℝ) : (circlePoint s : EucSpace 2) =
    Real.cos s • (axisX : EucSpace 2) + Real.sin s • (axisY : EucSpace 2) := rfl

/-- **Short arcs lie in geodesic hulls.** The point at an angle between
two angles closer than `π` lies in the geodesic hull of their points.
Source auxiliary: arXiv:2411.04551v3, §2.2, Step 1, the pieces of a
support on the circle are arcs. -/
theorem circlePoint_mem_convG {s₁ s s₂ : ℝ} (h₁ : s₁ ≤ s) (h₂ : s ≤ s₂) (hπ : s₂ - s₁ < π) :
    circlePoint s ∈ convG 2 {circlePoint s₁, circlePoint s₂} := by
  rcases h₁.eq_or_lt with rfl | h₁
  · exact subset_convG 2 _ (mem_insert _ _)
  have ha : 0 ≤ Real.sin (s₂ - s) :=
    Real.sin_nonneg_of_nonneg_of_le_pi (by linarith) (by linarith)
  have hb : 0 < Real.sin (s - s₁) := Real.sin_pos_of_pos_of_lt_pi (by linarith) (by linarith)
  have hD : 0 < Real.sin (s₂ - s₁) := Real.sin_pos_of_pos_of_lt_pi (by linarith) hπ
  have hab : 0 < Real.sin (s₂ - s) + Real.sin (s - s₁) := add_pos_of_nonneg_of_pos ha hb
  have hab' := hab.ne'
  have hD' := hD.ne'
  refine Or.inr ⟨(Real.sin (s₂ - s) / (Real.sin (s₂ - s) + Real.sin (s - s₁))) •
      (circlePoint s₁ : EucSpace 2) +
      (Real.sin (s - s₁) / (Real.sin (s₂ - s) + Real.sin (s - s₁))) •
      (circlePoint s₂ : EucSpace 2),
    segment_subset_convexHull (mem_image_of_mem _ (mem_insert (circlePoint s₁) {circlePoint s₂}))
      (mem_image_of_mem _ (mem_insert_of_mem (circlePoint s₁) rfl))
      ⟨_, _, div_nonneg ha hab.le, div_nonneg hb.le hab.le, by rw [← add_div, div_self hab'],
        rfl⟩,
    (Real.sin (s₂ - s) + Real.sin (s - s₁)) / Real.sin (s₂ - s₁), div_pos hab hD, ?_⟩
  ext k
  fin_cases k
  · simp only [Fin.zero_eta, PiLp.smul_apply, PiLp.add_apply, smul_eq_mul, circlePoint_apply_zero]
    field_simp
    simp only [Real.sin_sub]
    ring
  · simp only [Fin.mk_one, PiLp.smul_apply, PiLp.add_apply, smul_eq_mul, circlePoint_apply_one]
    field_simp
    simp only [Real.sin_sub]
    ring

/-- A functional of the plane along the circle:
`f (cos s, sin s) = cos s · f e₁ + sin s · f e₂`. -/
theorem apply_circlePoint (f : StrongDual ℝ (EucSpace 2)) (s : ℝ) :
    f (circlePoint s : EucSpace 2) =
      Real.cos s * f (axisX : EucSpace 2) + Real.sin s * f (axisY : EucSpace 2) := by
  rw [coe_circlePoint, map_add, map_smul, map_smul, smul_eq_mul, smul_eq_mul]

/-- **Windows.** A region of the circle that is open, closed under
geodesic hulls, and either the whole circle or inside an open half-plane,
carrying a law without atoms: the arc of one window of angles of length
at most `2π` lies in the region and carries the law and every given point
of the region. Source auxiliary: arXiv:2411.04551v3, §2.2, Step 1, the
pieces of the supports, on the circle. -/
theorem exists_circle_window (ν : Measure (SSphere 2)) [IsProbabilityMeasure ν]
    (hatom : ∀ y, ν {y} = 0) (U : Set (SSphere 2)) (hU : IsOpen U)
    (hconv : ∀ A ⊆ U, convG 2 A ⊆ U)
    (hcap : U = univ ∨ ∃ f : StrongDual ℝ (EucSpace 2), ∀ x ∈ U, 0 < f x) (hνU : ν U = 1)
    (T : Finset (SSphere 2)) (hT : ∀ y ∈ T, y ∈ U) :
    ∃ a b : ℝ, a < b ∧ b - a ≤ 2 * π ∧ circlePoint '' Ioo a b ⊆ U ∧
      ν (circlePoint '' Ioo a b) = 1 ∧ ∀ y ∈ T, ∃ θ ∈ Ioo a b, circlePoint θ = y := by
  rcases hcap with rfl | ⟨f, hf⟩
  · have := Perspective.infinite_sSphere (d := 2) le_rfl
    obtain ⟨w, hw⟩ := Infinite.exists_notMem_finset T
    obtain ⟨t, -, rfl⟩ := exists_circlePoint_eq w 0
    have hcompl : {circlePoint t}ᶜ ⊆ circlePoint '' Ioo (t - 2 * π) t := by
      intro y hy
      obtain ⟨s, hs, rfl⟩ := exists_circlePoint_eq y (t - π)
      refine ⟨s, ⟨by linarith [hs.1], lt_of_le_of_ne (by linarith [hs.2]) ?_⟩, rfl⟩
      rintro rfl
      exact hy rfl
    refine ⟨t - 2 * π, t, by linarith [Real.pi_pos], by linarith, subset_univ _, ?_,
      fun y hy => hcompl (mem_compl_singleton_iff.mpr fun h => hw (h ▸ hy))⟩
    refine le_antisymm prob_le_one ?_
    calc (1 : ℝ≥0∞) = ν {circlePoint t}ᶜ := by
          rw [measure_compl (measurableSet_singleton _) (measure_ne_top _ _), measure_univ,
            hatom, tsub_zero]
      _ ≤ _ := measure_mono hcompl
  obtain ⟨y₀, hy₀⟩ := nonempty_of_measure_ne_zero (hνU.trans_ne one_ne_zero)
  set z : ℂ := ⟨f (axisX : EucSpace 2), f (axisY : EucSpace 2)⟩
  have hz : z ≠ 0 := by
    intro h
    obtain ⟨s, -, rfl⟩ := exists_circlePoint_eq y₀ 0
    have h0 := hf _ hy₀
    have h1 : f (axisX : EucSpace 2) = 0 := congrArg Complex.re h
    have h2 : f (axisY : EucSpace 2) = 0 := congrArg Complex.im h
    rw [apply_circlePoint, h1, h2] at h0
    simp at h0
  have hn : ‖z‖ ≠ 0 := norm_ne_zero_iff.mpr hz
  have hcos : Real.cos (Complex.arg z) = f (axisX : EucSpace 2) / ‖z‖ := Complex.cos_arg hz
  have hsin : Real.sin (Complex.arg z) = f (axisY : EucSpace 2) / ‖z‖ := Complex.sin_arg z
  generalize Complex.arg z = φ at hcos hsin
  have hfs (s : ℝ) : f (circlePoint s : EucSpace 2) = ‖z‖ * Real.cos (s - φ) := by
    rw [apply_circlePoint, Real.cos_sub, hcos, hsin]
    field_simp
  set Θ := {s | s ∈ Ioo (φ - π / 2) (φ + π / 2) ∧ circlePoint s ∈ U}
  have hcover : ∀ y ∈ U, ∃ s ∈ Θ, circlePoint s = y := by
    intro y hy
    obtain ⟨s, hs, rfl⟩ := exists_circlePoint_eq y φ
    have hpos : 0 < Real.cos (s - φ) := by
      have h := hf _ hy
      rw [hfs] at h
      exact pos_of_mul_pos_right h (norm_nonneg _)
    have habs : |s - φ| < π / 2 := by
      by_contra hge
      have h := Real.cos_nonpos_of_pi_div_two_le_of_le (not_lt.mp hge)
        (by linarith [abs_le.mpr (⟨by linarith [hs.1], by linarith [hs.2]⟩ :
          -π ≤ s - φ ∧ s - φ ≤ π)])
      rw [Real.cos_abs] at h
      linarith
    exact ⟨s, ⟨⟨by linarith [(abs_lt.mp habs).1], by linarith [(abs_lt.mp habs).2]⟩, hy⟩, rfl⟩
  have hconvΘ : ∀ s₁ ∈ Θ, ∀ s₂ ∈ Θ, ∀ s, s₁ ≤ s → s ≤ s₂ → s ∈ Θ := by
    rintro s₁ ⟨hs₁, hU₁⟩ s₂ ⟨hs₂, hU₂⟩ s h₁ h₂
    exact ⟨⟨by linarith [hs₁.1], by linarith [hs₂.2]⟩,
      hconv _ (insert_subset_iff.mpr ⟨hU₁, singleton_subset_iff.mpr hU₂⟩)
        (circlePoint_mem_convG h₁ h₂ (by linarith [hs₁.1, hs₂.2]))⟩
  have hopen : IsOpen Θ := isOpen_Ioo.inter (hU.preimage continuous_circlePoint)
  have hside : ∀ s ∈ Θ, (∃ s' ∈ Θ, s' < s) ∧ ∃ s' ∈ Θ, s < s' := by
    intro s hs
    obtain ⟨δ, hδ, hball⟩ := Metric.isOpen_iff.mp hopen s hs
    rw [Real.ball_eq_Ioo] at hball
    exact ⟨⟨s - δ / 2, hball ⟨by linarith, by linarith⟩, by linarith⟩,
      ⟨s + δ / 2, hball ⟨by linarith, by linarith⟩, by linarith⟩⟩
  obtain ⟨s₀, hs₀, -⟩ := hcover y₀ hy₀
  have hne : Θ.Nonempty := ⟨s₀, hs₀⟩
  have hbdd₁ : BddBelow Θ := ⟨φ - π / 2, fun s hs => hs.1.1.le⟩
  have hbdd₂ : BddAbove Θ := ⟨φ + π / 2, fun s hs => hs.1.2.le⟩
  have hsub : Θ ⊆ Ioo (sInf Θ) (sSup Θ) := fun s hs => by
    obtain ⟨⟨s', hs', hlt⟩, ⟨s'', hs'', hgt⟩⟩ := hside s hs
    exact ⟨(csInf_le hbdd₁ hs').trans_lt hlt, hgt.trans_le (le_csSup hbdd₂ hs'')⟩
  have hsup : Ioo (sInf Θ) (sSup Θ) ⊆ Θ := fun s hs => by
    obtain ⟨s₁, hs₁, h₁⟩ := exists_lt_of_csInf_lt hne hs.1
    obtain ⟨s₂, hs₂, h₂⟩ := exists_lt_of_lt_csSup hne hs.2
    exact hconvΘ s₁ hs₁ s₂ hs₂ s h₁.le h₂.le
  have ha : φ - π / 2 ≤ sInf Θ := le_csInf hne fun s hs => hs.1.1.le
  have hb : sSup Θ ≤ φ + π / 2 := csSup_le hne fun s hs => hs.1.2.le
  have hUW : U ⊆ circlePoint '' Ioo (sInf Θ) (sSup Θ) := fun y hy => by
    obtain ⟨s, hs, rfl⟩ := hcover y hy
    exact ⟨s, hsub hs, rfl⟩
  refine ⟨sInf Θ, sSup Θ, (hsub hs₀).1.trans (hsub hs₀).2, by linarith [Real.pi_pos], ?_,
    le_antisymm prob_le_one (hνU.symm.le.trans (measure_mono hUW)), fun y hy => ?_⟩
  · rintro _ ⟨s, hs, rfl⟩
    exact (hsup hs).2
  · obtain ⟨s, hs, rfl⟩ := hcover y (hT y hy)
    exact ⟨s, hsub hs, rfl⟩

/-- The hypotheses of the arc lemmas are satisfiable: three equal angles,
and a law without atoms on the whole circle with one given point. -/
example : ∃ ν : Perspective.ProbSphere 2, (0 : ℝ) ≤ 0 ∧ (0 : ℝ) - 0 < π ∧
    (∀ y : SSphere 2, (ν : Measure (SSphere 2)) {y} = 0) ∧ IsOpen (univ : Set (SSphere 2)) ∧
    (∀ A ⊆ (univ : Set (SSphere 2)), convG 2 A ⊆ univ) ∧ (ν : Measure (SSphere 2)) univ = 1 ∧
    ∀ y ∈ ({axisX} : Finset (SSphere 2)), y ∈ (univ : Set (SSphere 2)) := by
  obtain ⟨ν, hν, -⟩ := exists_diffuse_probSphere 2 le_rfl isClosed_univ (by
    have := MeanField.isProbabilityMeasure_uniformLaw 2 (by norm_num)
    rw [measure_univ]
    exact one_pos)
  exact ⟨ν, le_rfl, by simp [Real.pi_pos], hν, isOpen_univ, fun _ _ => subset_univ _,
    measure_univ, fun _ _ => mem_univ _⟩

end Transformer.Interpolation
