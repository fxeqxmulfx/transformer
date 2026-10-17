/-
# Generic start and end points of the routes

ArXiv:2411.04551v3, §2.2, proof of `prop: compression`, Step 3: chains of
balls carry the mass of every packing ball to the ball around its anchor.
In dimension `d ≥ 3` the chains run along great circle arcs, and an arc
misses the start and end points of the other routes when no such point
lies in its plane. A point of a nonempty open set of the sphere avoids
finitely many planes through the origin, and a sequential choice gives
start and end points in prescribed open sets whose route planes are in
this general position.
-/

import Transformer.Interpolation.Section4_GenericQuadrantPoints

open Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- A nonzero vector that is a multiple of `v` puts `v` on the line of
`u`: avoiding the plane of `u` and `w` keeps `v` off the line of `u`.
Source auxiliary: arXiv:2411.04551v3, §2.2, Step 3. -/
theorem ne_smul_of_avoid_left (u v w : EucSpace d) (hu : u ≠ 0)
    (h : ∀ α β : ℝ, v ≠ α • u + β • w) : ∀ α : ℝ, u ≠ α • v := by
  intro α huv
  have hα : α ≠ 0 := by
    rintro rfl
    rw [zero_smul] at huv
    exact hu huv
  apply h α⁻¹ 0
  rw [zero_smul, add_zero, huv, smul_smul, inv_mul_cancel₀ hα, one_smul]

/-- The mirror of `ne_smul_of_avoid_left` for the second spanning
vector. Source auxiliary: arXiv:2411.04551v3, §2.2, Step 3. -/
theorem ne_smul_of_avoid_right (u v w : EucSpace d) (hw : w ≠ 0)
    (h : ∀ α β : ℝ, v ≠ α • u + β • w) : ∀ α : ℝ, w ≠ α • v :=
  ne_smul_of_avoid_left w v u hw fun α β => by rw [add_comm]; exact h β α

/-- **Generic point of an open set.** In dimension at least three, a
nonempty open set of the sphere has a point off finitely many planes
through the origin. Source auxiliary: arXiv:2411.04551v3, §2.2, Step 3. -/
theorem exists_mem_open_avoiding_planes (hd : 3 ≤ d) (O : Set (SSphere d)) (hO : IsOpen O)
    (hne : O.Nonempty) (F : Finset (EucSpace d × EucSpace d)) :
    ∃ w ∈ O, ∀ pq ∈ F, ∀ α β : ℝ, (w : EucSpace d) ≠ α • pq.1 + β • pq.2 := by
  obtain ⟨w₀, hw₀⟩ := hne
  obtain ⟨r, hr, hrO⟩ := Metric.isOpen_iff.mp hO w₀ hw₀
  have hD : Dense (⋂ pq : F,
      ((Submodule.span ℝ ({pq.1.1, pq.1.2} : Set (EucSpace d)) : Set (EucSpace d))ᶜ)) :=
    dense_iInter_of_isOpen (fun _ => (Submodule.span ℝ _).closed_of_finiteDimensional.isOpen_compl)
      (fun _ => dense_compl_proper_subspace _ (span_pair_ne_top_of_three_le hd _ _))
  have hρ : 0 < min r 1 / 2 := by positivity
  obtain ⟨v, hvball, hvD⟩ := hD.inter_open_nonempty (Metric.ball (w₀ : EucSpace d) (min r 1 / 2))
    Metric.isOpen_ball (Metric.nonempty_ball.mpr hρ)
  have hvw : ‖v - w₀‖ < min r 1 / 2 := by rw [← dist_eq_norm]; exact hvball
  have hvn : |‖v‖ - 1| ≤ ‖v - w₀‖ := by
    simpa only [mem_sphere_zero_iff_norm.mp w₀.2] using abs_norm_sub_norm_le v (w₀ : EucSpace d)
  have hv0 : 0 < ‖v‖ := by linarith [neg_abs_le (‖v‖ - 1), min_le_right r 1]
  let w : SSphere d := ⟨‖v‖⁻¹ • v, by
    rw [mem_sphere_zero_iff_norm, norm_smul, norm_inv, norm_norm, inv_mul_cancel₀ hv0.ne']⟩
  refine ⟨w, hrO ?_, fun pq hpq α β hw => ?_⟩
  · have h3 : (‖v‖⁻¹ - 1) * ‖v‖ = 1 - ‖v‖ := by rw [sub_mul, inv_mul_cancel₀ hv0.ne', one_mul]
    have hstep : ‖‖v‖⁻¹ • v - v‖ = |‖v‖ - 1| := by
      rw [show ‖v‖⁻¹ • v - v = (‖v‖⁻¹ - 1) • v by rw [sub_smul, one_smul], norm_smul,
        Real.norm_eq_abs, abs_sub_comm ‖v‖, ← h3, abs_mul, abs_of_nonneg (norm_nonneg v)]
    rw [Metric.mem_ball, Subtype.dist_eq, dist_eq_norm]
    calc ‖‖v‖⁻¹ • v - (w₀ : EucSpace d)‖ ≤ ‖‖v‖⁻¹ • v - v‖ + ‖v - w₀‖ :=
          norm_sub_le_norm_sub_add_norm_sub _ _ _
      _ < r := by linarith [min_le_left r 1]
  · have hvspan : v ∈ Submodule.span ℝ ({pq.1, pq.2} : Set (EucSpace d)) := by
      refine Submodule.mem_span_pair.mpr ⟨‖v‖ * α, ‖v‖ * β, ?_⟩
      have hv : v = ‖v‖ • (‖v‖⁻¹ • v) := by rw [smul_smul, mul_inv_cancel₀ hv0.ne', one_smul]
      change ‖v‖⁻¹ • v = α • pq.1 + β • pq.2 at hw
      rw [mul_smul, mul_smul, ← smul_add, ← hw, ← hv]
    exact mem_iInter.mp hvD ⟨pq, hpq⟩ hvspan

/-- **Sequential generic choice.** In dimension at least three, the
first `n` start and end points can be taken in prescribed nonempty open
sets so that no end point is on the line of its start point, no start
point is in the plane of another route, and no end point is in the plane
of another route. Source auxiliary: arXiv:2411.04551v3, §2.2, Step 3. -/
theorem exists_generic_pairs_nat (hd : 3 ≤ d) (O₁ O₂ : ℕ → Set (SSphere d))
    (hO₁ : ∀ p, IsOpen (O₁ p)) (hO₂ : ∀ p, IsOpen (O₂ p))
    (hne₁ : ∀ p, (O₁ p).Nonempty) (hne₂ : ∀ p, (O₂ p).Nonempty) (n : ℕ) :
    ∃ ω e : ℕ → SSphere d, (∀ p < n, ω p ∈ O₁ p ∧ e p ∈ O₂ p) ∧
      (∀ p < n, ∀ α : ℝ, (e p : EucSpace d) ≠ α • (ω p : EucSpace d)) ∧
      (∀ p < n, ∀ q < n, p ≠ q → ∀ α β : ℝ,
        (ω q : EucSpace d) ≠ α • (ω p : EucSpace d) + β • (e p : EucSpace d)) ∧
      (∀ p < n, ∀ q < n, p ≠ q → ∀ α β : ℝ,
        (e p : EucSpace d) ≠ α • (ω q : EucSpace d) + β • (e q : EucSpace d)) := by
  classical
  have hunit (x : SSphere d) : (x : EucSpace d) ≠ 0 := fun h => by
    simpa [h] using mem_sphere_zero_iff_norm.mp x.2
  induction n with
  | zero => exact ⟨fun _ => (hne₁ 0).some, fun _ => (hne₂ 0).some, by simp, by simp, by simp, by simp⟩
  | succ n ih =>
    obtain ⟨ω, e, hmem, hline, hstart, hend⟩ := ih
    let F₁ := (Finset.range n).image fun q => ((ω q : EucSpace d), (e q : EucSpace d))
    have hF₁ {q : ℕ} (hq : q < n) : ((ω q : EucSpace d), (e q : EucSpace d)) ∈ F₁ :=
      Finset.mem_image_of_mem _ (Finset.mem_range.mpr hq)
    obtain ⟨ωn, hωn, hωF⟩ := exists_mem_open_avoiding_planes hd (O₁ n) (hO₁ n) (hne₁ n) F₁
    let F₂ := insert ((ωn : EucSpace d), 0) (F₁ ∪
      (Finset.range n).image (fun q => ((ωn : EucSpace d), (ω q : EucSpace d))) ∪
      (Finset.range n).image (fun q => ((ωn : EucSpace d), (e q : EucSpace d))))
    obtain ⟨en, hen, heF⟩ := exists_mem_open_avoiding_planes hd (O₂ n) (hO₂ n) (hne₂ n) F₂
    have ha {q : ℕ} (hq : q < n) := hωF _ (hF₁ hq)
    have hb (α : ℝ) : (en : EucSpace d) ≠ α • (ωn : EucSpace d) := by
      have h := heF _ (Finset.mem_insert_self _ _) α 0
      rwa [smul_zero, add_zero] at h
    have hc {q : ℕ} (hq : q < n) := heF _ (Finset.mem_insert_of_mem
      (Finset.mem_union_left _ (Finset.mem_union_left _ (hF₁ hq))))
    have hd' {q : ℕ} (hq : q < n) := heF _ (Finset.mem_insert_of_mem (Finset.mem_union_left _
      (Finset.mem_union_right _ (Finset.mem_image_of_mem _ (Finset.mem_range.mpr hq)))))
    have he {q : ℕ} (hq : q < n) := heF _ (Finset.mem_insert_of_mem (Finset.mem_union_right _
      (Finset.mem_image_of_mem _ (Finset.mem_range.mpr hq))))
    have hω' {q : ℕ} (hq : q < n) : Function.update ω n ωn q = ω q :=
      Function.update_of_ne hq.ne _ _
    have he' {q : ℕ} (hq : q < n) : Function.update e n en q = e q :=
      Function.update_of_ne hq.ne _ _
    refine ⟨Function.update ω n ωn, Function.update e n en, fun p hp => ?_, fun p hp => ?_,
      fun p hp q hq hpq => ?_, fun p hp q hq hpq => ?_⟩
    · rcases Nat.lt_succ_iff_lt_or_eq.mp hp with hp | rfl
      · rw [hω' hp, he' hp]
        exact hmem p hp
      · rw [Function.update_self, Function.update_self]
        exact ⟨hωn, hen⟩
    · rcases Nat.lt_succ_iff_lt_or_eq.mp hp with hp | rfl
      · rw [hω' hp, he' hp]
        exact hline p hp
      · rw [Function.update_self, Function.update_self]
        exact hb
    · rcases Nat.lt_succ_iff_lt_or_eq.mp hp with hp | rfl <;>
        rcases Nat.lt_succ_iff_lt_or_eq.mp hq with hq | rfl
      · rw [hω' hp, he' hp, hω' hq]
        exact hstart p hp q hq hpq
      · rw [hω' hp, he' hp, Function.update_self]
        exact ha hp
      · rw [hω' hq, Function.update_self, Function.update_self]
        exact plane_exclusion_reversed _ _ _
          (ne_smul_of_avoid_left _ _ _ (hunit (ω q)) (ha hq)) (hd' hq)
      · exact absurd rfl hpq
    · rcases Nat.lt_succ_iff_lt_or_eq.mp hp with hp | rfl <;>
        rcases Nat.lt_succ_iff_lt_or_eq.mp hq with hq | rfl
      · rw [he' hp, hω' hq, he' hq]
        exact hend p hp q hq hpq
      · rw [he' hp, Function.update_self, Function.update_self]
        exact plane_exclusion_reversed _ _ _
          (ne_smul_of_avoid_right _ _ _ (hunit (e p)) (ha hp)) (he hp)
      · rw [hω' hq, he' hq, Function.update_self]
        exact hc hq
      · exact absurd rfl hpq

/-- **Generic routes.** In dimension at least three, start and end
points in prescribed nonempty open sets, indexed by `Fin P`, with every
end point off the line of its start point and the start and end points
of every route off the planes of the other routes. Source auxiliary:
arXiv:2411.04551v3, §2.2, Step 3. -/
theorem exists_generic_pairs (hd : 3 ≤ d) {P : ℕ} (O₁ O₂ : Fin P → Set (SSphere d))
    (hO₁ : ∀ p, IsOpen (O₁ p)) (hO₂ : ∀ p, IsOpen (O₂ p))
    (hne₁ : ∀ p, (O₁ p).Nonempty) (hne₂ : ∀ p, (O₂ p).Nonempty) :
    ∃ ω e : Fin P → SSphere d, (∀ p, ω p ∈ O₁ p ∧ e p ∈ O₂ p) ∧
      (∀ p, ∀ α : ℝ, (e p : EucSpace d) ≠ α • (ω p : EucSpace d)) ∧
      (∀ p q, p ≠ q → ∀ α β : ℝ,
        (ω q : EucSpace d) ≠ α • (ω p : EucSpace d) + β • (e p : EucSpace d)) ∧
      (∀ p q, p ≠ q → ∀ α β : ℝ,
        (e p : EucSpace d) ≠ α • (ω q : EucSpace d) + β • (e q : EucSpace d)) := by
  rcases Nat.eq_zero_or_pos P with rfl | hP
  · exact ⟨fun p => p.elim0, fun p => p.elim0, fun p => p.elim0, fun p => p.elim0,
      fun p => p.elim0, fun p => p.elim0⟩
  let ext (O : Fin P → Set (SSphere d)) (n : ℕ) : Set (SSphere d) :=
    if h : n < P then O ⟨n, h⟩ else O ⟨0, hP⟩
  have hext (O : Fin P → Set (SSphere d)) (p : Fin P) : ext O p = O p := dite_eq_left p.2
  have hopen (O : Fin P → Set (SSphere d)) (hO : ∀ p, IsOpen (O p)) (n : ℕ) :
      IsOpen (ext O n) := by simp only [ext]; split_ifs <;> exact hO _
  have hnon (O : Fin P → Set (SSphere d)) (hO : ∀ p, (O p).Nonempty) (n : ℕ) :
      (ext O n).Nonempty := by simp only [ext]; split_ifs <;> exact hO _
  obtain ⟨ω, e, hmem, hline, hstart, hend⟩ := exists_generic_pairs_nat hd (ext O₁) (ext O₂)
    (hopen O₁ hO₁) (hopen O₂ hO₂) (hnon O₁ hne₁) (hnon O₂ hne₂) P
  refine ⟨fun p => ω p, fun p => e p, fun p => ?_, fun p => hline p p.2,
    fun p q hpq => hstart p p.2 q q.2 (Fin.val_ne_of_ne hpq),
    fun p q hpq => hend p p.2 q q.2 (Fin.val_ne_of_ne hpq)⟩
  have h := hmem p p.2
  rwa [hext, hext] at h

/-- The hypotheses of `exists_mem_open_avoiding_planes`,
`exists_generic_pairs_nat` and `exists_generic_pairs` are satisfiable:
the whole sphere in dimension three. -/
example : 3 ≤ 3 ∧ IsOpen (univ : Set (SSphere 3)) ∧ (univ : Set (SSphere 3)).Nonempty :=
  ⟨le_rfl, isOpen_univ, ⟨⟨EuclideanSpace.single 0 1, by simp⟩, trivial⟩⟩

/-- The hypotheses of `ne_smul_of_avoid_left` and `ne_smul_of_avoid_right`
are satisfiable: a unit vector and a point off its line. -/
example : ∃ u v : SSphere 3, (u : EucSpace 3) ≠ 0 ∧
    ∀ α β : ℝ, (v : EucSpace 3) ≠ α • (u : EucSpace 3) + β • (u : EucSpace 3) := by
  let u : SSphere 3 := ⟨EuclideanSpace.single 0 1, by simp⟩
  obtain ⟨v, -, hv⟩ := exists_mem_open_avoiding_planes le_rfl univ isOpen_univ ⟨u, trivial⟩
    {((u : EucSpace 3), (u : EucSpace 3))}
  exact ⟨u, v, fun h => by simpa [h] using mem_sphere_zero_iff_norm.mp u.2,
    hv _ (Finset.mem_singleton_self _)⟩

end Transformer.Interpolation
