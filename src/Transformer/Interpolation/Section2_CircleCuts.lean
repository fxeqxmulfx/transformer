/-
# Cutting arcs of the circle at a prescribed mass

ArXiv:2411.04551v3, §2.2, proof of `prop: compression`, Steps 2–3, on
the circle (`d = 2`). Phases local in a region map a law without atoms by
a bijection, so the image has no atoms either. An arc of angular length
at most `2π` is a ball, and under a law without atoms the metric spheres
are null; so the mass of the arc from a fixed left end varies
continuously with its right end, and some right end gives any prescribed
mass up to that of a given arc.
-/

import Transformer.Interpolation.Section2_CircleArcs
import Transformer.Interpolation.Section2_LocalPhases
import Transformer.Interpolation.Section5_BallMass

open scoped ENNReal
open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- **Local phases keep laws without atoms.** A phase local in a region
maps a law without atoms to a law without atoms. Source auxiliary:
arXiv:2411.04551v3, §2.2, Step 2, the laws stay diffuse. -/
theorem IsLocalPhase.map_singleton_eq_zero {P : AutonomousPhase d} {D : Set (SSphere d)}
    {Φ : SSphere d → SSphere d} (hP : IsLocalPhase P D Φ) (ρ : Perspective.ProbSphere d)
    (hρ : ∀ y, (ρ : Measure (SSphere d)) {y} = 0) (y : SSphere d) :
    (P.2 ρ 1 : Measure (SSphere d)) {y} = 0 := by
  obtain ⟨x, rfl⟩ := hP.2.2.2.2.1.2 y
  rw [hP.2.2.2.2.2.2 ρ, Measure.map_apply hP.2.2.2.1.measurable (measurableSet_singleton _)]
  exact measure_mono_null (fun x' hx' => hP.2.2.2.2.1.1 hx') (hρ x)

/-- Lists of local phases keep laws without atoms. -/
theorem AllLocalIn.map_singleton_eq_zero {L : List (AutonomousPhase d)} {V : Set (SSphere d)}
    (hL : AllLocalIn L V) (ρ : Perspective.ProbSphere d)
    (hρ : ∀ y, (ρ : Measure (SSphere d)) {y} = 0) (y : SSphere d) :
    (phaseSequenceLaw L ρ : Measure (SSphere d)) {y} = 0 := by
  induction L generalizing ρ with
  | nil => exact hρ y
  | cons P L ih =>
    obtain ⟨Φ, hΦ⟩ := hL P List.mem_cons_self
    rw [phaseSequenceLaw_cons]
    exact ih (fun Q hQ => hL Q (List.mem_cons_of_mem P hQ)) _ (hΦ.map_singleton_eq_zero ρ hρ)

/-- **Arcs are balls.** The arc of an interval of angles of length at
most `2π` is the ball around the point at its middle angle with the
chord of its half length as radius. -/
theorem image_Ioo_eq_ball {l c : ℝ} (h : l ≤ c) (h' : c - l ≤ 2 * π) :
    circlePoint '' Ioo l c = Metric.ball (circlePoint ((l + c) / 2)) (chord ((c - l) / 2)) := by
  rw [ball_circlePoint _ _ (by linarith) (by linarith),
    show (l + c) / 2 - (c - l) / 2 = l by ring, show (l + c) / 2 + (c - l) / 2 = c by ring]

theorem measurableSet_image_Ioo {l c : ℝ} (h : l ≤ c) (h' : c - l ≤ 2 * π) :
    MeasurableSet (circlePoint '' Ioo l c) := by
  rw [image_Ioo_eq_ball h h']
  exact Metric.isOpen_ball.measurableSet

/-- **Cuts.** Under a finite law without atoms, for an interval of
angles `(l, r)` of length at most `2π`, every mass between zero and that
of its arc is the mass of the arc of `(l, c)` for some `c ∈ [l, r]`.
Source auxiliary: arXiv:2411.04551v3, §2.2, Step 2, the weights `α_k`
cut from the pieces. -/
theorem exists_cut_arc (ρ : Measure (SSphere 2)) [IsFiniteMeasure ρ] (hatom : ∀ y, ρ {y} = 0)
    {l r : ℝ} (hlr : l ≤ r) (h2π : r - l ≤ 2 * π) {τ : ℝ} (hτ0 : 0 ≤ τ)
    (hτ : τ ≤ ρ.real (circlePoint '' Ioo l r)) :
    ∃ c ∈ Icc l r, ρ.real (circlePoint '' Ioo l c) = τ := by
  have h1 : Continuous fun c : ℝ => circlePoint ((l + c) / 2) :=
    continuous_circlePoint.comp ((continuous_const.add continuous_id).div_const 2)
  have h2 : Continuous fun c : ℝ => chord ((c - l) / 2) :=
    continuous_chord.comp ((continuous_id.sub continuous_const).div_const 2)
  have hF : Continuous fun c : ℝ =>
      ρ.real (Metric.ball (circlePoint ((l + c) / 2)) (chord ((c - l) / 2))) := by
    have h := (continuous_ball_mass ρ (measure_sphere_circle_eq_zero ρ hatom)).comp
      (h1.prodMk h2)
    exact h
  have hFeq : ∀ c ∈ Icc l r,
      ρ.real (Metric.ball (circlePoint ((l + c) / 2)) (chord ((c - l) / 2))) =
        ρ.real (circlePoint '' Ioo l c) := fun c hc => by
    rw [image_Ioo_eq_ball hc.1 (by linarith [hc.2])]
  have hmem : τ ∈ Icc (ρ.real (Metric.ball (circlePoint ((l + l) / 2)) (chord ((l - l) / 2))))
      (ρ.real (Metric.ball (circlePoint ((l + r) / 2)) (chord ((r - l) / 2)))) := by
    rw [hFeq r ⟨hlr, le_rfl⟩, sub_self, zero_div, chord_zero, Metric.ball_zero, measureReal_empty]
    exact ⟨hτ0, hτ⟩
  obtain ⟨c, hc, hFc⟩ := intermediate_value_Icc hlr hF.continuousOn hmem
  exact ⟨c, hc, (hFeq c hc).symm.trans hFc⟩

/-- The hypotheses of the cut lemmas are satisfiable: a law without atoms
on the circle, the trivial local phase, the empty interval of angles and
the mass zero. -/
example : ∃ ρ : Perspective.ProbSphere 2, (∀ y : SSphere 2, (ρ : Measure (SSphere 2)) {y} = 0) ∧
    IsLocalPhase (ridgePhase (0 : Fin 2) 0 0 0) (∅ : Set (SSphere 2)) id ∧
    AllLocalIn [ridgePhase (0 : Fin 2) 0 0 0] (∅ : Set (SSphere 2)) ∧
    (0 : ℝ) ≤ 0 ∧ (0 : ℝ) - 0 ≤ 2 * π ∧
    (0 : ℝ) ≤ (ρ : Measure (SSphere 2)).real (circlePoint '' Ioo 0 0) := by
  obtain ⟨ρ, hρ, -⟩ := exists_diffuse_probSphere 2 le_rfl isClosed_univ (by
    have := MeanField.isProbabilityMeasure_uniformLaw 2 (by norm_num)
    rw [measure_univ]
    exact one_pos)
  have hP : IsLocalPhase (ridgePhase (0 : Fin 2) 0 0 0) (∅ : Set (SSphere 2)) id := by
    refine ⟨isAutonomousPhase_ridge 0 0 0 0, rfl, rfl, continuous_id, Function.bijective_id,
      fun _ _ => rfl, fun ρ => ?_⟩
    rw [Measure.map_id]
    change (ridgeLaw 0 0 0 ρ 1 : Measure (SSphere 2)) = _
    rw [ridgeLaw_eq_of_inactive_ae 0 0 0 ρ (ae_of_all _ fun _ => by simp) 1]
  exact ⟨ρ, hρ, hP, fun Q hQ => ⟨id, by rw [List.mem_singleton.mp hQ]; exact hP⟩, le_rfl,
    by linarith [Real.pi_pos], measureReal_nonneg⟩

end Transformer.Interpolation
