module

public import Transformer.Interpolation.Section3_PhaseLawMaps
public import Transformer.Interpolation.Section3_MeanAxisAttraction
public import Transformer.Interpolation.Section3_BarycenterContraction

/-!
# Intrinsic clocks for exact selective probability maps

ArXiv:2411.04551v3, §3.2, Steps 1–3 of
`lem: induction.barycenter` and `prop: separation`.
One positive attention clock isolates all unselected supports near the
two transverse poles. A common positive ridge clock contracts every
selected zero-transverse-mean law. Reverse attention restores each
unselected law exactly and gives the prescribed selected radius.

The second theorem constructs three actual autonomous phases for one
member of a finite positive-quadrant family with non-collinear means.
All weak laws and zero-score fields were proved for these concrete
phases; a prescribed physical horizon is supplied separately by the
finite-phase realization theorem.
-/

@[expose] public section

open Real MeasureTheory Set

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Actual intrinsic clocks give uniform selected contraction and exact
restoration of every other law. Source: arXiv:2411.04551v3, §3.2, Steps 1–3. -/
theorem exists_selective_phase_clocks {ι : Type*} [Fintype ι]
    (i : Fin d) (p w : SSphere d)
    (hwp : inner ℝ (w : EucSpace d) (p : EucSpace d) = 0)
    (η : ι → Perspective.ProbSphere d)
    (hη : ∀ k, meanAxisMoment (w : EucSpace d) (η k) ≠ 0)
    (hmiss : ∀ k, w ∉ (η k : Measure (SSphere d)).support ∧
      antipode d w ∉ (η k : Measure (SSphere d)).support)
    (A : Set (Perspective.ProbSphere d)) (a : ℝ) (ha : 0 < a)
    (hA : ∀ ν ∈ A, meanAxisMoment (w : EucSpace d) ν = 0 ∧
      ∀ x ∈ (ν : Measure (SSphere d)).support, a ≤ inner ℝ (x : EucSpace d) (p : EucSpace d))
    (r : ℝ) (hr : 0 < r) :
    ∃ c t₁ t₂ : ℝ, 0 < c ∧ 0 < t₁ ∧ 0 < t₂ ∧
      (∀ ν ∈ A, (phaseSequenceLaw (selectivePhaseList i p w c t₁ t₂) ν :
        Measure (SSphere d)).support ⊆ Metric.closedBall p r) ∧
      ∀ k, phaseSequenceLaw (selectivePhaseList i p w c t₁ t₂) (η k) = η k := by
  let c := a / 2
  have hc : 0 < c := by dsimp only [c]; positivity
  have hca : c < a := by dsimp only [c]; linarith
  have he : ∀ᶠ t : ℝ in Filter.atTop, ∀ k,
      (meanAxisLaw w 1 (η k) t : Measure (SSphere d)).support ⊆
        Metric.ball w (c / 2) ∪ Metric.ball (antipode d w) (c / 2) :=
    Filter.eventually_all.mpr (fun k => eventually_meanAxisLaw_two_poles w 1 (η k)
      zero_lt_one (hη k) (hmiss k).1 (hmiss k).2 (c / 2) (by positivity))
  obtain ⟨t₁, hpre, ht₁⟩ := (he.and (Filter.eventually_gt_atTop (0 : ℝ))).exists
  let δ := r / (2 * exp t₁)
  have hδ : 0 < δ := by dsimp only [δ]; positivity
  have hrad : δ * exp t₁ ≤ r := by
    have hh : δ * (2 * exp t₁) = r := div_mul_cancel₀ r (ne_of_gt (by positivity))
    have hp : 0 < δ * exp t₁ := mul_pos hδ (exp_pos _)
    nlinarith
  obtain ⟨t₂, ht₂, hcenter⟩ := exists_time_ridgeLaw_center_radius p a c δ hc.le hca hδ
  refine ⟨c, t₁, t₂, hc, ht₁, ht₂, ?_, ?_⟩
  · intro ν hν x hx
    have hs := selectivePhaseList_selected_radius i p w c t₁ t₂ δ ht₁.le hwp ν
      (hA ν hν).1 (hcenter ν (hA ν hν).2) hx
    exact hs.trans hrad
  · intro k
    have hfixed := ridgeLaw_inactive_near_orthogonal_poles p w (p : EucSpace d) c hc hwp
      (meanAxisLaw w 1 (η k) t₁) (hpre k) t₂
    exact selectivePhaseList_restore_fixed i p w c t₁ t₂ (η k) hfixed

/-- A target with its original mean direction contracts inside one
prescribed cap in three actual phases, while every other law is restored.
Source: arXiv:2411.04551v3, §3.2, `lem: induction.barycenter`. -/
theorem exists_target_contraction_phases {N : ℕ} (hd : 2 ≤ d) (j : Idx N)
    (μ : Idx N → Perspective.ProbSphere d)
    (hμ : ∀ k, (μ k : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hsep : ∀ k, k ≠ j → ∀ γ : ℝ, barycenter d (μ k) ≠ γ • barycenter d (μ j))
    (p : SSphere d) (hp : (p : EucSpace d) = barycenterDirection (μ j)) (r : ℝ) (hr : 0 < r) :
    ∃ L : List (AutonomousPhase d), L.length = 3 ∧
      (∀ P ∈ L, IsAutonomousPhase P ∧ P.1.B = 0) ∧
      (phaseSequenceLaw L (μ j) : Measure (SSphere d)).support ⊆ Metric.closedBall p r ∧
      ∀ k, k ≠ j → phaseSequenceLaw L (μ k) = μ k := by
  classical
  let i : Fin d := ⟨0, by omega⟩
  obtain ⟨w, hwm, hwothers⟩ := exists_barycenter_isolation_axis hd j μ (hμ j) hsep
  have hwp : inner ℝ (w : EucSpace d) (p : EucSpace d) = 0 := by
    rw [hp, barycenterDirection, real_inner_smul_right, hwm, mul_zero]
  have hppos (k : Fin d) : 0 < (p : EucSpace d) k := by
    rw [hp]
    exact barycenterDirection_coordinate_pos (μ j) (hμ j) k
  obtain ⟨a, ha, hfloor⟩ := exists_positiveQuadrant_pair_height_floor i p hppos (μ j) (μ j) (hμ j) (hμ j)
  let I := {k : Idx N // k ≠ j}
  let η : I → Perspective.ProbSphere d := fun k => μ k
  have hη (k : I) : meanAxisMoment (w : EucSpace d) (η k) ≠ 0 := hwothers k k.2
  have hmiss (k : I) : w ∉ (η k : Measure (SSphere d)).support ∧
      antipode d w ∉ (η k : Measure (SSphere d)).support :=
    orthogonal_poles_not_support i (barycenter d (μ j))
      (barycenter_coordinate_pos_of_positiveQuadrant (μ j) (hμ j)) w hwm (η k) (hμ k)
  let A : Set (Perspective.ProbSphere d) := {μ j}
  have hA : ∀ ν ∈ A, meanAxisMoment (w : EucSpace d) ν = 0 ∧
      ∀ x ∈ (ν : Measure (SSphere d)).support, a ≤ inner ℝ (x : EucSpace d) (p : EucSpace d) := by
    intro ν hν
    have he : ν = μ j := hν
    subst ν
    exact ⟨hwm, fun x hx => hfloor x (Or.inl hx)⟩
  obtain ⟨c, t₁, t₂, -, -, -, hselected, hrest⟩ :=
    exists_selective_phase_clocks i p w hwp η hη hmiss A a ha hA r hr
  exact ⟨selectivePhaseList i p w c t₁ t₂, selectivePhaseList_length i p w c t₁ t₂,
    selectivePhaseList_valid i p w c t₁ t₂, hselected (μ j) (by simp [A]),
    fun k hk => hrest ⟨k, hk⟩⟩

/-- Two positive circle atoms realize a selected law and a nonempty
family of other laws for the same transverse axis and positive gate
margin. The target pole is the original normalized barycenter. -/
example : ∃ (μ : Idx 2 → Perspective.ProbSphere 2) (p w : SSphere 2)
    (η : Fin 1 → Perspective.ProbSphere 2) (A : Set (Perspective.ProbSphere 2)),
    (2 : ℕ) ≤ 2 ∧
    (∀ k, (μ k : Measure (SSphere 2)).support ⊆ positiveQuadrant 2) ∧
    (∀ k : Idx 2, k ≠ 0 → ∀ γ : ℝ, barycenter 2 (μ k) ≠ γ • barycenter 2 (μ 0)) ∧
    (p : EucSpace 2) = barycenterDirection (μ 0) ∧
    inner ℝ (w : EucSpace 2) (p : EucSpace 2) = 0 ∧
    (∀ k, meanAxisMoment (w : EucSpace 2) (η k) ≠ 0) ∧
    (∀ k, w ∉ (η k : Measure (SSphere 2)).support ∧
      antipode 2 w ∉ (η k : Measure (SSphere 2)).support) ∧
    (0 : ℝ) < 1 / 2 ∧
    (∀ ν ∈ A, meanAxisMoment (w : EucSpace 2) ν = 0 ∧
      ∀ x ∈ (ν : Measure (SSphere 2)).support,
        (1 / 2 : ℝ) ≤ inner ℝ (x : EucSpace 2) (p : EucSpace 2)) ∧
    μ 0 ∈ A ∧ η 0 = μ 1 ∧ (0 : ℝ) < 1 / 10 := by
  let p : SSphere 2 := circlePt (3 / 5) (4 / 5) (by norm_num)
  let q : SSphere 2 := circlePt (4 / 5) (3 / 5) (by norm_num)
  let μ : Idx 2 → Perspective.ProbSphere 2 :=
    ![Perspective.diracProb 2 p, Perspective.diracProb 2 q]
  have hQ (k : Idx 2) : (μ k : Measure (SSphere 2)).support ⊆ positiveQuadrant 2 := by
    intro x hx
    fin_cases k
    · rw [eq_of_mem_support_dirac hx]
      exact circlePt_mem_positiveQuadrant _ (by norm_num) (by norm_num)
    · rw [eq_of_mem_support_dirac hx]
      exact circlePt_mem_positiveQuadrant _ (by norm_num) (by norm_num)
  have hsep (k : Idx 2) (hk : k ≠ 0) (γ : ℝ) :
      barycenter 2 (μ k) ≠ γ • barycenter 2 (μ 0) := by
    fin_cases k
    · exact (hk rfl).elim
    · intro he
      have h0 := congrArg (fun x : EucSpace 2 => x 0) he
      have h1 := congrArg (fun x : EucSpace 2 => x 1) he
      norm_num [μ, barycenter, Perspective.diracProb, p, q, circlePt] at h0 h1
      linarith
  have hp : (p : EucSpace 2) = barycenterDirection (μ 0) := by
    simp [barycenterDirection, μ, barycenter, Perspective.diracProb,
      mem_sphere_zero_iff_norm.mp p.2]
  obtain ⟨w, hwm, hother⟩ := exists_barycenter_isolation_axis (by norm_num : 2 ≤ (2 : ℕ))
    0 μ (hQ 0) hsep
  have hwp : inner ℝ (w : EucSpace 2) (p : EucSpace 2) = 0 := by
    rw [hp, barycenterDirection, real_inner_smul_right, hwm, mul_zero]
  let η : Fin 1 → Perspective.ProbSphere 2 := fun _ => μ 1
  let A : Set (Perspective.ProbSphere 2) := {μ 0}
  have hη (k : Fin 1) : meanAxisMoment (w : EucSpace 2) (η k) ≠ 0 :=
    hother 1 (by decide)
  have hmiss (k : Fin 1) : w ∉ (η k : Measure (SSphere 2)).support ∧
      antipode 2 w ∉ (η k : Measure (SSphere 2)).support :=
    orthogonal_poles_not_support 0 (barycenter 2 (μ 0))
      (barycenter_coordinate_pos_of_positiveQuadrant (μ 0) (hQ 0)) w hwm (η k) (hQ 1)
  have hA : ∀ ν ∈ A, meanAxisMoment (w : EucSpace 2) ν = 0 ∧
      ∀ x ∈ (ν : Measure (SSphere 2)).support,
        (1 / 2 : ℝ) ≤ inner ℝ (x : EucSpace 2) (p : EucSpace 2) := by
    intro ν hν
    have he : ν = μ 0 := hν
    subst ν
    refine ⟨hwm, fun x hx => ?_⟩
    have he : x = p := eq_of_mem_support_dirac hx
    rw [he, real_inner_self_eq_norm_sq, mem_sphere_zero_iff_norm.mp p.2]
    norm_num
  exact ⟨μ, p, w, η, A, le_rfl, hQ, hsep, hp, hwp, hη, hmiss,
    by norm_num, hA, rfl, rfl, by norm_num⟩

end Transformer.Interpolation
