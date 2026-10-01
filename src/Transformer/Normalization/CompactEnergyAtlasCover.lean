/-
# Finite neighborhoods covering all nearby noncentral energy fibers

Only nonconstant germs on the central energy level need coordinate
charts. Compactness then covers entire nearby noncentral source fibers.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2.
-/

import Transformer.Normalization.GradientEnergyImage
import Mathlib.Topology.MetricSpace.Pseudo.Defs

open Filter Set Topology

namespace Transformer.Normalization

/-- The central energy fiber after removing the open regions where the
energy is locally equal to its central value. These are exactly the points
at which a nonconstant germ must be prepared. Auxiliary for Appendix D.1,
`lem: loj`, of arXiv:2510.22026v2. -/
def nonflatEnergyLevel {N : ℕ} (E : EucSpace N → ℝ)
    (K : Set (EucSpace N)) (c : ℝ) : Set (EucSpace N) :=
  {w | w ∈ K ∧ E w = c} ∩ (interior {w | E w = c})ᶜ

/-- The part of a compact central fiber with nonconstant germs is
compact; relative continuity on the original source suffices. Auxiliary
for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem nonflatEnergyLevel_isCompact {N : ℕ} (E : EucSpace N → ℝ)
    (K : Set (EucSpace N)) (hK : IsCompact K) (hE : ContinuousOn E K) (c : ℝ) :
    IsCompact (nonflatEnergyLevel E K c) := by
  have hfiber : IsCompact {w | w ∈ K ∧ E w = c} :=
    hK.of_isClosed_subset (hK.isClosed.isClosed_eq hE continuousOn_const) (fun _ hw => hw.1)
  exact hfiber.inter_right isOpen_interior.isClosed_compl

/-- Membership in the central nonflat fiber supplies the actual
nonconstant-germ hypothesis needed by real analytic preparation.
Auxiliary for Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem nonflatEnergyLevel_nonconstant {N : ℕ} (E : EucSpace N → ℝ)
    (K : Set (EucSpace N)) (c : ℝ) {w : EucSpace N}
    (hw : w ∈ nonflatEnergyLevel E K c) : ¬ ∀ᶠ y in nhds w, E y = E w := by
  intro hconstant
  apply hw.2
  apply mem_interior_iff_mem_nhds.mpr
  exact hconstant.mono (fun y hy => hy.trans hw.1.2)

/-- An open cover of the nonflat part of one compact energy level
covers all sufficiently nearby noncentral energy fibers in the entire
source. Points in a flat central neighborhood cannot compete on a
noncentral level. Auxiliary for Appendix D.1, `lem: loj`, of
arXiv:2510.22026v2. -/
theorem compact_near_energy_level_covered {N : ℕ} (E : EucSpace N → ℝ)
    (K : Set (EucSpace N)) (hK : IsCompact K) (hE : ContinuousOn E K)
    (c : ℝ) (V : Set (EucSpace N)) (hV : IsOpen V)
    (hcover : nonflatEnergyLevel E K c ⊆ V) :
    ∃ epsilon > 0, ∀ w ∈ K, |E w - c| < epsilon → E w ≠ c → w ∈ V := by
  let W : Set (EucSpace N) := V ∪ interior {w | E w = c}
  have hW : IsOpen W := hV.union isOpen_interior
  have hcentral : ∀ w ∈ K, E w = c → w ∈ W := by
    intro w hw hwc
    by_cases hflat : w ∈ interior {w | E w = c}
    · exact Or.inr hflat
    · exact Or.inl (hcover ⟨⟨hw, hwc⟩, hflat⟩)
  have hT : IsCompact (K ∩ Wᶜ) := hK.inter_right hW.isClosed_compl
  have himage : IsCompact (E '' (K ∩ Wᶜ)) :=
    hT.image_of_continuousOn (hE.mono inter_subset_left)
  have hc : c ∉ E '' (K ∩ Wᶜ) := by
    rintro ⟨w, ⟨hwK, hwW⟩, hwc⟩
    exact hwW (hcentral w hwK hwc)
  obtain ⟨epsilon, he, hball⟩ := Metric.isOpen_iff.mp himage.isClosed.isOpen_compl c hc
  refine ⟨epsilon, he, ?_⟩
  intro w hw hsmall hne
  by_contra hwV
  have hwflat : w ∉ interior {w | E w = c} := fun h =>
    hne ((interior_subset (s := {w | E w = c})) h)
  have hwT : w ∈ K ∩ Wᶜ := ⟨hw, fun h => h.elim hwV hwflat⟩
  exact hball (by simpa only [Metric.mem_ball, Real.dist_eq] using hsmall) ⟨w, hwT, rfl⟩

/-- Any pointwise open coordinate neighborhoods of the nonflat central
fiber have a finite subfamily covering every sufficiently nearby
noncentral energy fiber. Thus a comparison on that finite union can
retain every point of the original compact source. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem finite_cover_near_energy_level {N : ℕ} (E : EucSpace N → ℝ)
    (K : Set (EucSpace N)) (hK : IsCompact K) (hE : ContinuousOn E K) (c : ℝ)
    (V : nonflatEnergyLevel E K c → Set (EucSpace N))
    (hV : ∀ w, IsOpen (V w)) (hwV : ∀ w, (w : EucSpace N) ∈ V w) :
    ∃ (t : Finset (nonflatEnergyLevel E K c)) (epsilon : ℝ), 0 < epsilon ∧
      ∀ w ∈ K, |E w - c| < epsilon → E w ≠ c → w ∈ ⋃ i ∈ t, V i := by
  obtain ⟨t, ht⟩ := (nonflatEnergyLevel_isCompact E K hK hE c).elim_finite_subcover V hV
    (fun w hw => mem_iUnion.mpr ⟨⟨w, hw⟩, hwV ⟨w, hw⟩⟩)
  obtain ⟨epsilon, he, hcovered⟩ := compact_near_energy_level_covered E K hK hE c
    (⋃ i ∈ t, V i) (isOpen_biUnion (fun i _ => hV i)) ht
  exact ⟨t, epsilon, he, hcovered⟩

/-- A nonconstant quadratic energy on the closed unit ball supplies
compactness, relative continuity, and actual open neighborhoods for
the finite-cover theorem. Auxiliary for Appendix D.1, `lem: loj`,
of arXiv:2510.22026v2. -/
example : ∃ (t : Finset (nonflatEnergyLevel (fun w : EucSpace 1 => ‖w‖ ^ 2)
      (Metric.closedBall 0 1) 0)) (epsilon : ℝ), 0 < epsilon ∧
    ∀ w ∈ Metric.closedBall (0 : EucSpace 1) 1,
      |‖w‖ ^ 2 - 0| < epsilon → ‖w‖ ^ 2 ≠ 0 →
        w ∈ ⋃ i ∈ t, Metric.ball (i : EucSpace 1) (1 / 2) := by
  apply finite_cover_near_energy_level (fun w : EucSpace 1 => ‖w‖ ^ 2)
    (Metric.closedBall 0 1) (isCompact_closedBall _ _) (continuous_norm.pow 2).continuousOn 0
    (fun i => Metric.ball (i : EucSpace 1) (1 / 2)) (fun _ => Metric.isOpen_ball)
  intro i
  exact Metric.mem_ball_self (by norm_num)

/-- The origin genuinely belongs to the nonflat quadratic central
fiber, so the nonconstant-germ hypothesis is inhabited. Auxiliary for
Appendix D.1, `lem: loj`, of arXiv:2510.22026v2. -/
theorem zero_mem_nonflat_quadratic_level : (0 : EucSpace 1) ∈ nonflatEnergyLevel (fun w => ‖w‖ ^ 2)
    (Metric.closedBall 0 1) 0 := by
  refine ⟨⟨by simp, by simp⟩, ?_⟩
  intro hflat
  have hzero : ∀ᶠ w in nhds (0 : EucSpace 1), ‖w‖ ^ 2 = 0 :=
    mem_interior_iff_mem_nhds.mp hflat
  obtain ⟨r, hr, hball⟩ := Metric.eventually_nhds_iff.mp hzero
  let v : EucSpace 1 := PiLp.single 2 (0 : Fin 1) (r / 2)
  have hnorm : ‖v‖ = r / 2 := by simp [v, abs_of_pos hr]
  have hv : ‖v‖ ^ 2 = 0 := hball (by
    simpa only [dist_zero_right, hnorm] using half_lt_self hr)
  rw [hnorm] at hv
  exact (sq_pos_of_pos (half_pos hr)).ne' hv

end Transformer.Normalization
