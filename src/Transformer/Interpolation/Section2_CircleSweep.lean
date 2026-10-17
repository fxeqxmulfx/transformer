/-
# Sweeping a law along a window of the circle

ArXiv:2411.04551v3, §2.2, proof of `prop: compression`, Steps 2–3, on
the circle (`d = 2`), where small balls cannot be routed past each other:
the mass is swept along a window of angles from the first point to the
last. At each point the arc around it is cut to the weight of the point,
and all the mass beyond the cut is collapsed into the arc around the next
point by a phase local in the rest of the window; a part once cut is
never moved again. Each collapse loses at most `ε` of the mass.
-/

import Transformer.Interpolation.Section2_CircleCuts
import Transformer.Interpolation.Section2_CollapsePhases

open scoped ENNReal
open Real MeasureTheory Set

namespace Transformer.Interpolation

/-- **The sweep.** Angles `θ 0 < ⋯ < θ m` whose arcs of angular radius
`s` are separated and end before `b`, in a window `(θ 0 - s, b)` of
length at most `2π`, and weights `α k ≥ 0` of total at most the mass of
the first arc plus a defect `D ≥ 0`, under a law without atoms: phases
local in the window leave in the closed arc of each angle a measurable
part of mass at least its weight less `D + m ε`. Source:
arXiv:2411.04551v3, §2.2, proof of `prop: compression`, Steps 2–3, on
the circle, with cuts and collapses along the window in place of the
routes of Step 3. -/
theorem exists_circle_sweep {s b ε : ℝ} (hs : 0 < s) (hε : 0 < ε) (hε1 : ε ≤ 1) (m : ℕ) :
    ∀ (θ α : Fin (m + 1) → ℝ) (ρ : Perspective.ProbSphere 2) (D : ℝ),
      (∀ k : Fin m, θ k.castSucc + s < θ k.succ - s) → (∀ k, θ k + s < b) →
      b - (θ 0 - s) ≤ 2 * π → (∀ k, 0 ≤ α k) → 0 ≤ D →
      (∀ y, (ρ : Measure (SSphere 2)) {y} = 0) →
      ∑ k, α k ≤ (ρ : Measure (SSphere 2)).real (circlePoint '' Ioo (θ 0 - s) (θ 0 + s)) + D →
      ∃ (L : List (AutonomousPhase 2)) (E : Fin (m + 1) → Set (SSphere 2)),
        AllLocalIn L (circlePoint '' Ioo (θ 0 - s) b) ∧ (∀ k, MeasurableSet (E k)) ∧
        (∀ k, E k ⊆ circlePoint '' Icc (θ k - s) (θ k + s)) ∧
        ∀ k, α k ≤ (phaseSequenceLaw L ρ : Measure (SSphere 2)).real (E k) + D + m * ε := by
  induction m with
  | zero =>
    intro θ α ρ D _ hb h2π _ _ _ hsum
    refine ⟨[], fun k => circlePoint '' Ioo (θ k - s) (θ k + s),
      fun _ h => absurd h List.not_mem_nil, fun k => ?_, fun k => image_mono Ioo_subset_Icc_self,
      fun k => ?_⟩ <;>
      obtain rfl := Subsingleton.elim (α := Fin 1) k 0
    · exact measurableSet_image_Ioo (by linarith) (by linarith [hb 0])
    · rw [Fin.sum_univ_one] at hsum
      simpa only [phaseSequenceLaw_nil, Nat.cast_zero, zero_mul, add_zero] using hsum
  | succ m ih =>
    intro θ α ρ D hgap hb h2π hα hD hatom hsum
    have hθ01 : θ 0 + s < θ (Fin.succ 0) - s := hgap 0
    have hsπ : s < π := by linarith [hb (Fin.succ 0)]
    set A₀ := circlePoint '' Ioo (θ 0 - s) (θ 0 + s)
    obtain ⟨τ, hτ⟩ : ∃ τ, τ = min (α 0) ((ρ : Measure (SSphere 2)).real A₀) := ⟨_, rfl⟩
    obtain ⟨c, hc, hcut⟩ := exists_cut_arc (ρ : Measure (SSphere 2)) hatom (τ := τ)
      (show θ 0 - s ≤ θ 0 + s by linarith) (by linarith)
      (by rw [hτ]; exact le_min (hα 0) measureReal_nonneg) (by rw [hτ]; exact min_le_right _ _)
    have hcb : c < b := by linarith [hc.2, hb (Fin.succ 0)]
    have hDom := image_Ioo_eq_ball hcb.le (by linarith [hc.1])
    have hA₁ := ball_circlePoint (θ (Fin.succ 0)) s hs.le hsπ.le
    obtain ⟨P, Φ, hP, hmass⟩ := exists_collapse_phase (0 : Fin 2) 1 (by decide)
      (circlePoint ((c + b) / 2)) (circlePoint (θ (Fin.succ 0))) (chord ((b - c) / 2))
      (chord_pos (by linarith) (by linarith [hc.1])) (chord_le_two _)
      (Metric.ball (circlePoint (θ (Fin.succ 0))) (chord s)) Metric.isOpen_ball
      (by rw [hA₁, ← hDom]; exact image_mono (Ioo_subset_Ioo (by linarith [hc.2])
        (hb (Fin.succ 0)).le))
      (Metric.mem_ball_self (chord_pos hs hsπ.le)) ε hε hε1 ρ
    rw [← hDom] at hP hmass
    rw [hA₁] at hmass
    set ρ₁ := P.2 ρ 1
    have hreal : (1 - ε) * (ρ : Measure (SSphere 2)).real (circlePoint '' Ioo c b) ≤
        (ρ₁ : Measure (SSphere 2)).real
          (circlePoint '' Ioo (θ (Fin.succ 0) - s) (θ (Fin.succ 0) + s)) := by
      have h := ENNReal.toReal_mono (measure_ne_top _ _) hmass
      rw [ENNReal.toReal_mul, ENNReal.toReal_ofReal (by linarith)] at h
      exact h
    have hcover : A₀ ⊆ circlePoint '' Ioo (θ 0 - s) c ∪
        ({circlePoint c} ∪ circlePoint '' Ioo c b) := by
      rintro _ ⟨t, ht, rfl⟩
      rcases lt_trichotomy t c with h | rfl | h
      · exact Or.inl ⟨t, ⟨ht.1, h⟩, rfl⟩
      · exact Or.inr (Or.inl rfl)
      · exact Or.inr (Or.inr ⟨t, ⟨h, by linarith [ht.2, hb (Fin.succ 0)]⟩, rfl⟩)
    have hA₀ : (ρ : Measure (SSphere 2)).real A₀ ≤
        τ + (ρ : Measure (SSphere 2)).real (circlePoint '' Ioo c b) := by
      have h0 : (ρ : Measure (SSphere 2)).real {circlePoint c} = 0 := by
        simp [Measure.real, hatom]
      calc _ ≤ _ := measureReal_mono hcover
        _ ≤ _ := (measureReal_union_le _ _).trans (add_le_add_right (measureReal_union_le _ _) _)
        _ = _ := by rw [hcut, h0, zero_add]
    have hsum' : ∑ k : Fin (m + 1), α k.succ ≤ (ρ₁ : Measure (SSphere 2)).real
        (circlePoint '' Ioo (θ (Fin.succ 0) - s) (θ (Fin.succ 0) + s)) + (D + ε) := by
      rw [Fin.sum_univ_succ] at hsum
      have hεD := mul_le_of_le_one_right hε.le
        (measureReal_le_one (μ := (ρ : Measure (SSphere 2))) (s := circlePoint '' Ioo c b))
      have h₁ := measureReal_nonneg (μ := (ρ₁ : Measure (SSphere 2)))
        (s := circlePoint '' Ioo (θ (Fin.succ 0) - s) (θ (Fin.succ 0) + s))
      rcases min_cases (α 0) ((ρ : Measure (SSphere 2)).real A₀) with ⟨h, -⟩ | ⟨h, hlt⟩ <;>
        rw [h] at hτ <;> linarith
    obtain ⟨L', E', hL', hE'm, hE'sub, hE'mass⟩ := ih (fun k => θ k.succ) (fun k => α k.succ) ρ₁
      (D + ε) (fun k => hgap k.succ) (fun k => hb k.succ) (by linarith)
      (fun k => hα k.succ) (by linarith) (hP.map_singleton_eq_zero ρ hatom) hsum'
    have hdisj₁ : Disjoint (circlePoint '' Ioo (θ 0 - s) c) (circlePoint '' Ioo c b) :=
      disjoint_image_circlePoint h2π (fun t ht => ⟨ht.1, ht.2.le.trans hcb.le⟩)
        (fun t ht => ⟨by linarith [hc.1, ht.1], ht.2.le⟩)
        (Set.disjoint_left.mpr fun t h₁ h₂ => lt_asymm h₁.2 h₂.1)
    have hdisj₂ : Disjoint (circlePoint '' Ioo (θ 0 - s) c)
        (circlePoint '' Ioo (θ (Fin.succ 0) - s) b) :=
      disjoint_image_circlePoint h2π (fun t ht => ⟨ht.1, ht.2.le.trans hcb.le⟩)
        (fun t ht => ⟨by linarith [ht.1], ht.2.le⟩)
        (Set.disjoint_left.mpr fun t h₁ h₂ => by linarith [h₁.2, h₂.1, hc.2])
    have hE₀ : τ ≤ (phaseSequenceLaw L' ρ₁ : Measure (SSphere 2)).real
        (circlePoint '' Ioo (θ 0 - s) c) := by
      rw [← hcut]
      exact (ENNReal.toReal_mono (measure_ne_top _ _) (hP.le_of_disjoint ρ _ hdisj₁)).trans
        (ENNReal.toReal_mono (measure_ne_top _ _) (hL'.le_of_disjoint ρ₁ _ hdisj₂))
    refine ⟨P :: L', Fin.cons (circlePoint '' Ioo (θ 0 - s) c) E', ?_,
      Fin.forall_fin_succ.mpr ⟨?_, fun j => ?_⟩, Fin.forall_fin_succ.mpr ⟨?_, fun j => ?_⟩,
      Fin.forall_fin_succ.mpr ⟨?_, fun j => ?_⟩⟩
    · intro Q hQ
      rcases List.mem_cons.mp hQ with rfl | hQ
      · exact ⟨Φ, hP.mono (image_mono (Ioo_subset_Ioo_left hc.1))⟩
      · obtain ⟨Ψ, hΨ⟩ := hL' Q hQ
        exact ⟨Ψ, hΨ.mono (image_mono (Ioo_subset_Ioo_left (by linarith)))⟩
    · exact measurableSet_image_Ioo hc.1 (by linarith [hc.2])
    · exact hE'm j
    · exact image_mono fun t ht => ⟨ht.1.le, ht.2.le.trans hc.2⟩
    · exact hE'sub j
    · have hα0 : α 0 ≤ ∑ k, α k := Finset.single_le_sum (fun k _ => hα k) (Finset.mem_univ 0)
      have hpos : 0 ≤ ((m : ℝ) + 1) * ε := by positivity
      simp only [Fin.cons_zero, phaseSequenceLaw_cons]
      push_cast
      rcases min_cases (α 0) ((ρ : Measure (SSphere 2)).real A₀) with ⟨h, -⟩ | ⟨h, -⟩ <;>
        rw [h] at hτ <;> linarith
    · have h := hE'mass j
      simp only [Fin.cons_succ, phaseSequenceLaw_cons]
      push_cast
      linarith

/-- The hypotheses of `exists_circle_sweep` are satisfiable: `s = 1`,
`ε = 1`, `b = 2`, one angle `0` of weight `0`, the defect `0`, and a law
without atoms on the circle. -/
example : ∃ ρ : Perspective.ProbSphere 2, (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 1 ∧
    (∀ k : Fin 0, (0 : ℝ) + 1 < (0 : ℝ) - 1 ∧ k = k) ∧ (0 : ℝ) + 1 < 2 ∧
    (2 : ℝ) - (0 - 1) ≤ 2 * π ∧ (∀ y : SSphere 2, (ρ : Measure (SSphere 2)) {y} = 0) ∧
    ∑ _k : Fin 1, (0 : ℝ) ≤
      (ρ : Measure (SSphere 2)).real (circlePoint '' Ioo (0 - 1) (0 + 1)) + 0 := by
  obtain ⟨ρ, hρ, -⟩ := exists_diffuse_probSphere 2 le_rfl isClosed_univ (by
    have := MeanField.isProbabilityMeasure_uniformLaw 2 (by norm_num)
    rw [measure_univ]
    exact one_pos)
  exact ⟨ρ, one_pos, le_rfl, fun k => k.elim0, by norm_num, by linarith [Real.pi_gt_three], hρ,
    by simp⟩

end Transformer.Interpolation
