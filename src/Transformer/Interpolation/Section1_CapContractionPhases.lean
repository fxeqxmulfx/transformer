module

public import Transformer.Interpolation.Section3_QuantitativeCenterClocks

/-!
# Cap contractions toward distinct targets, one target at a time

ArXiv:2411.04551v3, §3.2, Step 2 of selective contraction, and §5,
the clustering stage `θ₂` in the proof of `thm: targets.atoms`.
Distinct unit targets have closed caps of one common radius `r` that
the height `1 - r` over each target separates from the caps of all
the others. A balanced ridge neuron with output and gate along one
target and threshold `1 - r` fixes every law concentrated near another
target, and contracts a law concentrated near its own target.

Running one such neuron per listed target, for one common intrinsic
clock, brings every listed law into the requested radius, and a law
already within that radius stays there. The clock is the logarithmic
one of `exists_logarithmic_center_clock`, chosen from the radius alone.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d N : ℕ}

/-- A unit point within distance `r` of a unit point `p` has height at
least `1 - r²/2` over `p`. Source auxiliary: arXiv:2411.04551v3, §3.2,
the cap around a target. -/
theorem le_inner_of_dist_le (x p : SSphere d) (r : ℝ) (h : dist x p ≤ r) :
    1 - r ^ 2 / 2 ≤ inner ℝ (x : EucSpace d) (p : EucSpace d) := by
  have hsq := norm_sub_sq_real (x : EucSpace d) (p : EucSpace d)
  rw [mem_sphere_zero_iff_norm.mp x.2, mem_sphere_zero_iff_norm.mp p.2] at hsq
  have hn : ‖(x : EucSpace d) - (p : EucSpace d)‖ ≤ r := by
    rwa [Subtype.dist_eq, dist_eq_norm] at h
  have h0 := norm_nonneg ((x : EucSpace d) - (p : EucSpace d))
  nlinarith [mul_le_mul hn hn h0 (h0.trans hn)]

/-- Distinct unit targets have a common radius `r ≤ 1` whose closed
caps the height `1 - r` separates: a unit point within `r` of one
target has height at most `1 - r` over every other target. Source
auxiliary: arXiv:2411.04551v3, §3.2 and §5, disjoint target caps. -/
theorem exists_cap_separation_radius (Y : Idx N → SSphere d) (hY : Function.Injective Y) :
    ∃ r : ℝ, 0 < r ∧ r ≤ 1 ∧ ∀ j k, j ≠ k → ∀ x : SSphere d, dist x (Y j) ≤ r →
      inner ℝ (x : EucSpace d) (Y k : EucSpace d) ≤ 1 - r := by
  have hgap (j k : Idx N) (hjk : j ≠ k) :
      0 < (1 - inner ℝ (Y j : EucSpace d) (Y k : EucSpace d)) / 2 := by
    have hne : (Y j : EucSpace d) - (Y k : EucSpace d) ≠ 0 :=
      sub_ne_zero.mpr fun h => hjk (hY (Subtype.ext h))
    have hpos := norm_pos_iff.mpr hne
    have hsq := norm_sub_sq_real (Y j : EucSpace d) (Y k : EucSpace d)
    rw [mem_sphere_zero_iff_norm.mp (Y j).2, mem_sphere_zero_iff_norm.mp (Y k).2] at hsq
    nlinarith
  have hev : ∀ᶠ r in nhdsWithin (0 : ℝ) (Ioi 0), r ≤ 1 ∧
      ∀ j k, j ≠ k → r ≤ (1 - inner ℝ (Y j : EucSpace d) (Y k : EucSpace d)) / 2 := by
    refine Filter.Eventually.and ?_ ?_
    · exact ((eventually_lt_nhds one_pos).mono fun _ h => h.le).filter_mono nhdsWithin_le_nhds
    · refine Filter.eventually_all.2 fun j => Filter.eventually_all.2 fun k => ?_
      by_cases hjk : j = k
      · exact Filter.Eventually.of_forall fun _ h => absurd hjk h
      · exact ((eventually_lt_nhds (hgap j k hjk)).mono fun _ h _ => h.le).filter_mono
          nhdsWithin_le_nhds
  have hpos : ∀ᶠ r in nhdsWithin (0 : ℝ) (Ioi 0), r ∈ Ioi (0 : ℝ) := self_mem_nhdsWithin
  obtain ⟨r, ⟨hr1, hrg⟩, hr⟩ := (hev.and hpos).exists
  refine ⟨r, hr, hr1, fun j k hjk x hx => ?_⟩
  have hcs := real_inner_le_norm ((x : EucSpace d) - (Y j : EucSpace d)) (Y k : EucSpace d)
  rw [inner_sub_left, mem_sphere_zero_iff_norm.mp (Y k).2, mul_one] at hcs
  have hn : ‖(x : EucSpace d) - (Y j : EucSpace d)‖ ≤ r := by
    rwa [Subtype.dist_eq, dist_eq_norm] at hx
  linarith [hrg j k hjk]

/-- One balanced ridge neuron per listed target, with output and gate
along the target, threshold `c` and intrinsic clock `τ`. Source
auxiliary: arXiv:2411.04551v3, §3.2, Step 2, and §5, clustering. -/
noncomputable def capContractionPhases (i : Fin d) (Y : Idx N → SSphere d) (c τ : ℝ)
    (l : List (Idx N)) : List (AutonomousPhase d) :=
  l.map fun k => balancedRidgePhase i (Y k : EucSpace d) (-(Y k : EucSpace d)) (-c) τ

/-- There is one cap contraction per listed target. Source auxiliary:
arXiv:2411.04551v3, §5, switch accounting. -/
theorem capContractionPhases_length (i : Fin d) (Y : Idx N → SSphere d) (c τ : ℝ)
    (l : List (Idx N)) : (capContractionPhases i Y c τ l).length = l.length := by
  simp [capContractionPhases]

/-- Every cap contraction has its proved original law and zero scores.
Source auxiliary: arXiv:2411.04551v3, §1 and §3.2. -/
theorem capContractionPhases_valid (i : Fin d) (Y : Idx N → SSphere d) (c τ : ℝ)
    (l : List (Idx N)) :
    ∀ P ∈ capContractionPhases i Y c τ l, IsAutonomousPhase P ∧ P.1.B = 0 := by
  intro P hP
  obtain ⟨k, _, rfl⟩ := List.mem_map.mp hP
  exact ⟨isAutonomousPhase_balancedRidge _ _ _ _ _, balancedRidgePhase_B_zero _ _ _ _ _⟩

/-- A law within `r` of its target ends within `δ` of it after the
listed cap contractions, when its target is listed or the law already
lies within `δ`: its own neuron contracts it and the others fix it.
Source auxiliary: arXiv:2411.04551v3, §3.2, Step 2, the protected
inactive laws, and §5, clustering before matching. -/
theorem support_capContractionPhases (i : Fin d) (Y : Idx N → SSphere d) (r δ a c τ : ℝ)
    (hτ : 0 ≤ τ) (hδr : δ ≤ r)
    (hsep : ∀ j k, j ≠ k → ∀ x : SSphere d, dist x (Y j) ≤ r →
      inner ℝ (x : EucSpace d) (Y k : EucSpace d) ≤ c)
    (hcap : ∀ k (x : SSphere d), dist x (Y k) ≤ r →
      a ≤ inner ℝ (x : EucSpace d) (Y k : EucSpace d))
    (hcontract : ∀ k (ν : Perspective.ProbSphere d),
      (∀ x ∈ (ν : Measure (SSphere d)).support,
        a ≤ inner ℝ (x : EucSpace d) (Y k : EucSpace d)) →
      (ridgeLaw (Y k : EucSpace d) (-(Y k : EucSpace d)) (-c) ν τ :
        Measure (SSphere d)).support ⊆ Metric.closedBall (Y k) δ)
    (l : List (Idx N)) (j : Idx N) (ρ : Perspective.ProbSphere d)
    (hρ : (ρ : Measure (SSphere d)).support ⊆ Metric.closedBall (Y j) r)
    (hj : j ∈ l ∨ (ρ : Measure (SSphere d)).support ⊆ Metric.closedBall (Y j) δ) :
    (phaseSequenceLaw (capContractionPhases i Y c τ l) ρ : Measure (SSphere d)).support ⊆
      Metric.closedBall (Y j) δ := by
  induction l generalizing ρ with
  | nil =>
    rcases hj with hj | hj
    · simp at hj
    · exact hj
  | cons k l ih =>
    change (phaseSequenceLaw (capContractionPhases i Y c τ l)
      ((balancedRidgePhase i (Y k : EucSpace d) (-(Y k : EucSpace d)) (-c) τ).2 ρ 1) :
        Measure (SSphere d)).support ⊆ _
    rw [balancedRidgePhase_law i _ _ _ τ hτ, mul_one]
    by_cases hjk : j = k
    · subst hjk
      have hsmall := hcontract j ρ fun x hx => hcap j x (Metric.mem_closedBall.mp (hρ hx))
      exact ih _ (hsmall.trans (Metric.closedBall_subset_closedBall hδr)) (Or.inr hsmall)
    · have hfix : ridgeLaw (Y k : EucSpace d) (-(Y k : EucSpace d)) (-c) ρ τ = ρ := by
        apply ridgeLaw_eq_of_inactive_ae
        filter_upwards [Measure.support_mem_ae (μ := (ρ : Measure (SSphere d)))] with x hx
        have hh := hsep j k hjk x (Metric.mem_closedBall.mp (hρ hx))
        rw [inner_neg_right]
        linarith
      rw [hfix]
      refine ih ρ hρ ?_
      rcases hj with hj | hj
      · exact Or.inl ((List.mem_cons.mp hj).resolve_left hjk)
      · exact Or.inr hj

/-- A unit point is within distance zero of itself, so the height bound
has an instance. -/
example (p : SSphere 2) : dist p p ≤ 0 := by simp

/-- A single target is an injective family. -/
example : Function.Injective (fun _ : Idx 1 => basePoint 2 : Idx 1 → SSphere 3) :=
  Function.injective_of_subsingleton _

/-- A single target with radius one, height one half, threshold zero and
the logarithmic clock for radius one half meets every hypothesis of
`support_capContractionPhases` together. -/
example : ∃ (Y : Idx 1 → SSphere 2) (τ : ℝ), 0 ≤ τ ∧ (1 / 2 : ℝ) ≤ 1 ∧
    (∀ j k : Idx 1, j ≠ k → ∀ x : SSphere 2, dist x (Y j) ≤ 1 →
      inner ℝ (x : EucSpace 2) (Y k : EucSpace 2) ≤ 0) ∧
    (∀ k (x : SSphere 2), dist x (Y k) ≤ 1 →
      1 / 2 ≤ inner ℝ (x : EucSpace 2) (Y k : EucSpace 2)) ∧
    ∀ k (ν : Perspective.ProbSphere 2),
      (∀ x ∈ (ν : Measure (SSphere 2)).support,
        1 / 2 ≤ inner ℝ (x : EucSpace 2) (Y k : EucSpace 2)) →
      (ridgeLaw (Y k : EucSpace 2) (-(Y k : EucSpace 2)) (-0) ν τ :
        Measure (SSphere 2)).support ⊆ Metric.closedBall (Y k) (1 / 2) := by
  obtain ⟨C, _, hclock⟩ := exists_logarithmic_center_clock (d := 2) (1 / 2) 0 le_rfl (by norm_num)
  obtain ⟨τ, hτ, _, hcontract⟩ := hclock (1 / 2) (by norm_num) (by norm_num)
  refine ⟨fun _ => basePoint 1, τ, hτ.le, by norm_num,
    fun j k hjk => absurd (Subsingleton.elim j k) hjk, ?_, fun _ ν hν => hcontract _ ν hν⟩
  intro k x hx
  have h := le_inner_of_dist_le x (basePoint 1) 1 hx
  norm_num at h
  linarith

end Transformer.Interpolation
