/-
# Finitely many small disjoint balls carrying almost all the mass

ArXiv:2411.04551v3, §2.2, proof of `prop: compression`, Step 2: "Pack
`𝓒_k^i` with `𝖭_k^i(δ) ≥ 1` disjoint open balls" whose union carries all
but `δ` of the mass. For a finite law on the sphere without atoms and an
open set, finitely many pairwise disjoint balls inside the open set,
each of radius at most `δ` and of mass at most `η`, carry all but `η` of
the mass of the open set.

The law is pushed to the ambient space, where the covering theorem of
Besicovitch gives a countable disjoint family of closed balls centered in
the open set covering almost all of it. The admissible radii are those whose
sphere is null, so that open and closed balls have the same mass, and
those small enough for the ball to stay in the open set and carry at most
`η`, which the absence of atoms allows. Finitely many of the balls carry
all but `η`.
-/

import Transformer.Interpolation.Basic
import Mathlib.MeasureTheory.Covering.BesicovitchVectorSpace
import Mathlib.Topology.Algebra.Module.Cardinality

open scoped ENNReal NNReal
open Real MeasureTheory Set Filter Topology

namespace Transformer.Interpolation

variable {d : ℕ}

/-- A ball of the sphere is the trace of the ambient ball with the same
center. -/
theorem ball_sphere_eq_preimage (z : SSphere d) (ρ : ℝ) :
    Metric.ball z ρ = (↑) ⁻¹' Metric.ball (z : EucSpace d) ρ := by
  ext y
  simp only [Metric.mem_ball, mem_preimage, Subtype.dist_eq]

/-- **Small disjoint balls.** A finite law on the sphere without atoms
and an open set: finitely many pairwise disjoint balls inside the open
set, of radii in `(0, δ]` and masses at most `η`, carry all but `η` of the
mass of the open set. Source: arXiv:2411.04551v3, §2.2, proof of
`prop: compression`, Step 2, with balls of small mass in place of a
packing of each piece of a partition. -/
theorem exists_small_disjoint_balls (ν : Measure (SSphere d)) [IsFiniteMeasure ν]
    (hatom : ∀ y : SSphere d, ν {y} = 0) (U : Set (SSphere d)) (hU : IsOpen U)
    (δ : ℝ) (hδ : 0 < δ) (η : ℝ≥0∞) (hη : 0 < η) :
    ∃ (P : ℕ) (z : Fin P → SSphere d) (ρ : Fin P → ℝ), (∀ p, 0 < ρ p ∧ ρ p ≤ δ) ∧
      (∀ p, Metric.ball (z p) (ρ p) ⊆ U) ∧
      Pairwise (Function.onFun Disjoint fun p => Metric.ball (z p) (ρ p)) ∧
      (∀ p, ν (Metric.ball (z p) (ρ p)) ≤ η) ∧
      ν U ≤ ∑ p, ν (Metric.ball (z p) (ρ p)) + η := by
  classical
  let : MeasurableSpace (EucSpace d) := borel _
  have : BorelSpace (EucSpace d) := ⟨rfl⟩
  let μ : Measure (EucSpace d) := ν.map (↑)
  have hemb : MeasurableEmbedding ((↑) : SSphere d → EucSpace d) :=
    Metric.isClosed_sphere.isClosedEmbedding_subtypeVal.measurableEmbedding
  have hμ (S : Set (EucSpace d)) : μ S = ν ((↑) ⁻¹' S) := hemb.map_apply ν S
  have : IsFiniteMeasure μ := Measure.isFiniteMeasure_map ν _
  let s : Set (EucSpace d) := (↑) '' U
  let f : EucSpace d → Set ℝ := fun x => {r | μ (Metric.sphere x r) = 0}
  have hf : ∀ x ∈ s, ∀ δ' > 0, (f x ∩ Ioo 0 δ').Nonempty := by
    intro x _ δ' hδ'
    have hcount : {r : ℝ | 0 < μ {y | dist y x = r}}.Countable :=
      Measure.countable_meas_level_set_pos (measurable_id.dist measurable_const)
    obtain ⟨r, hr, hrc⟩ := (hcount.dense_compl ℝ).inter_open_nonempty (Ioo 0 δ') isOpen_Ioo
      (nonempty_Ioo.mpr hδ')
    simp only [mem_compl_iff, mem_ofPred_eq, not_lt, nonpos_iff_eq_zero] at hrc
    exact ⟨r, hrc, hr⟩
  have hRex (x : EucSpace d) : ∃ R : ℝ, x ∈ s → 0 < R ∧ R ≤ δ ∧
      (∀ y : SSphere d, dist (y : EucSpace d) x ≤ R → y ∈ U) ∧
      μ (Metric.closedBall x R) < η := by
    by_cases hx : x ∈ s
    · obtain ⟨u, hu, rfl⟩ := hx
      obtain ⟨ε, hε, hεU⟩ := Metric.isOpen_iff.mp hU u hu
      have hlim := tendsto_measure_cthickening_of_isClosed (μ := μ)
        ⟨1, one_pos, measure_ne_top _ _⟩ (isClosed_singleton (x := (u : EucSpace d)))
      have h0 : μ {(u : EucSpace d)} = 0 := by
        rw [hμ]
        have hpre : ((↑) : SSphere d → EucSpace d) ⁻¹' {(u : EucSpace d)} = {u} := by
          ext y
          simp only [mem_preimage, mem_singleton_iff, Subtype.ext_iff]
        rw [hpre, hatom]
      rw [h0] at hlim
      obtain ⟨r₀, hr₀, hr₀η⟩ :=
        Metric.eventually_nhds_iff.mp (hlim.eventually (gt_mem_nhds hη))
      obtain ⟨R, hR0, hRε, hRδ, hRr⟩ : ∃ R : ℝ, 0 < R ∧ R ≤ ε / 2 ∧ R ≤ δ ∧ R < r₀ :=
        ⟨min (min (ε / 2) δ) (r₀ / 2), lt_min (lt_min (half_pos hε) hδ) (half_pos hr₀),
          (min_le_left _ _).trans (min_le_left _ _), (min_le_left _ _).trans (min_le_right _ _),
          (min_le_right _ _).trans_lt (half_lt_self hr₀)⟩
      refine ⟨R, fun _ => ⟨hR0, hRδ, fun y hy => hεU ?_, ?_⟩⟩
      · rw [Metric.mem_ball, Subtype.dist_eq]
        exact hy.trans_lt (hRε.trans_lt (half_lt_self hε))
      · have hR₀ : dist R 0 < r₀ := by
          rw [Real.dist_eq, sub_zero, abs_of_pos hR0]
          exact hRr
        have h := hr₀η hR₀
        rwa [Metric.cthickening_singleton _ hR0.le] at h
    · exact ⟨1, fun h => absurd h hx⟩
  choose R hR using hRex
  obtain ⟨t, r, ht, hts, hr, hnull, hdisj⟩ :=
    Besicovitch.exists_disjoint_closedBall_covering_ae μ f s hf R fun x hx => (hR x hx).1
  have hcov : μ s ≤ ∑' x : t, μ (Metric.closedBall x (r x)) := by
    rw [← measure_biUnion ht hdisj fun x _ => Metric.isClosed_closedBall.measurableSet]
    calc μ s ≤ μ (s ∩ ⋃ x ∈ t, Metric.closedBall x (r x)) +
          μ (s \ ⋃ x ∈ t, Metric.closedBall x (r x)) := measure_le_inter_add_sdiff _ _ _
      _ ≤ μ (⋃ x ∈ t, Metric.closedBall x (r x)) := by
          rw [hnull, add_zero]
          exact measure_mono inter_subset_right
  have hfin : ∃ F : Finset t, μ s ≤ ∑ x ∈ F, μ (Metric.closedBall x (r x)) + η := by
    by_cases hsη : μ s ≤ η
    · exact ⟨∅, by rw [Finset.sum_empty, zero_add]; exact hsη⟩
    · have hlt : μ s - η < μ s := ENNReal.sub_lt_self (measure_ne_top _ _)
        (hη.trans (not_le.mp hsη)).ne' hη.ne'
      rw [ENNReal.tsum_eq_iSup_sum] at hcov
      obtain ⟨F, hF⟩ := lt_iSup_iff.mp (hlt.trans_le hcov)
      exact ⟨F, tsub_le_iff_right.mp hF.le⟩
  obtain ⟨F, hF⟩ := hfin
  have hlift (x : t) : ∃ u : SSphere d, u ∈ U ∧ (u : EucSpace d) = x := hts x.2
  choose u huU hux using hlift
  have hνμ (x : t) (ρ : ℝ) : ν (Metric.ball (u x) ρ) = μ (Metric.ball (x : EucSpace d) ρ) := by
    rw [hμ, ← hux, ← ball_sphere_eq_preimage]
  let e := F.equivFin.symm
  refine ⟨F.card, fun p => u (e p), fun p => r (e p), fun p => ?_, fun p => ?_, ?_,
    fun p => ?_, ?_⟩
  · have h := hr _ (e p : t).2
    exact ⟨h.2.1, h.2.2.le.trans (hR _ (hts (e p : t).2)).2.1⟩
  · intro y hy
    have h := hr _ (e p : t).2
    refine (hR _ (hts (e p : t).2)).2.2.1 y ?_
    rw [Metric.mem_ball, Subtype.dist_eq, hux] at hy
    exact (hy.trans h.2.2).le
  · intro p q hpq
    have hne : ((e p : t) : EucSpace d) ≠ (e q : t) := fun h =>
      hpq (e.injective (Subtype.ext (Subtype.ext h)))
    have hcl : Disjoint (Metric.closedBall ((e p : t) : EucSpace d) (r (e p)))
        (Metric.closedBall ((e q : t) : EucSpace d) (r (e q))) :=
      hdisj (e p : t).2 (e q : t).2 hne
    refine disjoint_left.mpr fun y hyp hyq => ?_
    beta_reduce at hyp hyq
    rw [ball_sphere_eq_preimage, hux, mem_preimage] at hyp hyq
    exact disjoint_left.mp hcl (Metric.ball_subset_closedBall hyp)
      (Metric.ball_subset_closedBall hyq)
  · have h := hr _ (e p : t).2
    rw [hνμ]
    exact (measure_mono (Metric.ball_subset_closedBall.trans
      (Metric.closedBall_subset_closedBall h.2.2.le))).trans (hR _ (hts (e p : t).2)).2.2.2.le
  · have hU' : ν U = μ s := by
      rw [hμ]
      exact congrArg ν (Subtype.val_injective.preimage_image U).symm
    have hball (x : t) : μ (Metric.closedBall x (r x)) ≤ μ (Metric.ball x (r x)) := by
      have hz : μ (Metric.sphere (x : EucSpace d) (r x)) = 0 := (hr x x.2).1
      calc μ (Metric.closedBall x (r x))
          ≤ μ (Metric.ball x (r x)) + μ (Metric.sphere (x : EucSpace d) (r x)) := by
            rw [← Metric.ball_union_sphere]
            exact measure_union_le _ _
        _ = μ (Metric.ball x (r x)) := by rw [hz, add_zero]
    calc ν U ≤ ∑ x ∈ F, μ (Metric.closedBall x (r x)) + η := hU'.le.trans hF
      _ ≤ ∑ x ∈ F, μ (Metric.ball x (r x)) + η :=
          add_le_add (Finset.sum_le_sum fun x _ => hball x) le_rfl
      _ = ∑ p, ν (Metric.ball (u (e p)) (r (e p))) + η := by
          rw [← Finset.sum_coe_sort F, ← Equiv.sum_comp e]
          simp only [hνμ]

/-- The hypotheses of `exists_small_disjoint_balls` are satisfiable: the
zero law, the whole circle, and radius and mass bounds one. -/
example : (∀ y : SSphere 2, (0 : Measure (SSphere 2)) {y} = 0) ∧
    IsOpen (univ : Set (SSphere 2)) ∧ (0 : ℝ) < 1 ∧ (0 : ℝ≥0∞) < 1 :=
  ⟨fun _ => rfl, isOpen_univ, one_pos, one_pos⟩

end Transformer.Interpolation
