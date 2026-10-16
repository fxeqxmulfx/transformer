/-
# Original barycenter contraction and geodesic accuracy

ArXiv:2411.04551v3, §3.2, `lem: induction.barycenter`.
The strictly positive target barycenter defines its source unit pole.
One unit axis orthogonal to this mean pairs nontrivially with every
other input mean. The selected law and its collinear companion have
zero transverse mean, while the remaining supports miss both poles.

Compactness gives the two selected original supports a common positive
height floor. The actual three-phase control contracts them to a small
chordal radius and restores the remaining laws. The original geodesic
hull estimate converts that radius to the source angular accuracy.
This proves the original conclusion for every original weak solution
in ambient dimension at least two, with three constant pieces.
-/

import Transformer.Interpolation.Section3_BarycenterAxes
import Transformer.Interpolation.Section3_SelectiveContraction
import Transformer.Interpolation.Section3_TargetCaps

open Real MeasureTheory Set
open scoped Topology

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Two original quadrant supports have one positive height floor about
any positive unit pole. Source: arXiv:2411.04551v3, §3.2, Step 2. -/
theorem exists_positiveQuadrant_pair_height_floor (i : Fin d) (p : SSphere d)
    (hp : ∀ k : Fin d, 0 < (p : EucSpace d) k)
    (μ ν : Perspective.ProbSphere d)
    (hμ : (μ : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hν : (ν : Measure (SSphere d)).support ⊆ positiveQuadrant d) :
    ∃ a : ℝ, 0 < a ∧ ∀ x ∈ (μ : Measure (SSphere d)).support ∪
      (ν : Measure (SSphere d)).support, a ≤ inner ℝ (x : EucSpace d) (p : EucSpace d) := by
  let K := (μ : Measure (SSphere d)).support ∪ (ν : Measure (SSphere d)).support
  have hK : IsCompact K :=
    Measure.isClosed_support.isCompact.union Measure.isClosed_support.isCompact
  have hKne : K.Nonempty :=
    (Measure.nonempty_support (IsProbabilityMeasure.ne_zero (μ : Measure (SSphere d)))).inl
  apply exists_positive_height_floor p K hK hKne
  intro x hx
  apply inner_pos_of_positive_coordinates i x ?_ (p : EucSpace d) hp
  rcases hx with hx | hx
  · exact hμ hx
  · exact hν hx

/-- Three actual pieces give the source angular-hull contraction and
exact restoration, with its original data hypotheses in dimension at
least two. Source: arXiv:2411.04551v3, §3.2, `lem: induction.barycenter`.
The source's basis cycle is replaced by one common transverse axis. -/
theorem exists_barycenter_contraction_controls {N : ℕ} (hd : 2 ≤ d)
    (T ε : ℝ) (j : Idx N) (μ₀ : Idx N → Perspective.ProbSphere d) (ν₀ : Perspective.ProbSphere d)
    (hT : 0 < T) (hε : 0 < ε)
    (hμQ : ∀ i, (μ₀ i : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hsep : ∀ i : Idx N, i ≠ j → ∀ γ : ℝ, barycenter d (μ₀ i) ≠ γ • barycenter d (μ₀ j))
    (hνQ : (ν₀ : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hcol : ∃ γ : ℝ, barycenter d ν₀ = γ • barycenter d (μ₀ j)) :
    ∃ θ : TimeParams d, PiecewiseConstant d θ T 3 ∧
      (∀ ρ : Perspective.ProbSphere d, ∃ μ : ℝ → Perspective.ProbSphere d, μ 0 = ρ ∧ cauchyPB d θ μ) ∧
      (∀ μ ν : ℝ → Perspective.ProbSphere d, μ 0 = μ₀ j → cauchyPB d θ μ →
        ν 0 = ν₀ → cauchyPB d θ ν →
        ∀ x ∈ convG d (ν T : Measure (SSphere d)).support ∪ convG d (μ T : Measure (SSphere d)).support,
          arccos (inner ℝ (x : EucSpace d) (‖barycenter d (μ₀ j)‖⁻¹ • barycenter d (μ₀ j))) < ε) ∧
      ∀ i : Idx N, i ≠ j → ∀ μ : ℝ → Perspective.ProbSphere d,
        μ 0 = μ₀ i → cauchyPB d θ μ → μ T = μ₀ i := by
  classical
  obtain ⟨p, hp, hppos⟩ := exists_positive_barycenter_pole (by omega : 1 ≤ d) (μ₀ j) (hμQ j)
  obtain ⟨w, hwm, hwothers⟩ := exists_barycenter_isolation_axis hd j μ₀ (hμQ j) hsep
  have hwp : inner ℝ (w : EucSpace d) (p : EucSpace d) = 0 := by
    rw [hp, real_inner_smul_right, hwm, mul_zero]
  have hmν : meanAxisMoment (w : EucSpace d) ν₀ = 0 := by
    obtain ⟨γ, hγ⟩ := hcol
    change inner ℝ (w : EucSpace d) (barycenter d ν₀) = 0
    rw [hγ, real_inner_smul_right, hwm, mul_zero]
  let i₀ : Fin d := ⟨0, by omega⟩
  obtain ⟨a, ha, hfloor⟩ := exists_positiveQuadrant_pair_height_floor
    i₀ p hppos (μ₀ j) ν₀ (hμQ j) hνQ
  let I := {i : Idx N // i ≠ j}
  let η : I → Perspective.ProbSphere d := fun i => μ₀ i
  have hη (i : I) : meanAxisMoment (w : EucSpace d) (η i) ≠ 0 := hwothers i i.2
  have hmiss (i : I) : w ∉ (η i : Measure (SSphere d)).support ∧
      antipode d w ∉ (η i : Measure (SSphere d)).support :=
    orthogonal_poles_not_support i₀ (barycenter d (μ₀ j))
      (barycenter_coordinate_pos_of_positiveQuadrant (μ₀ j) (hμQ j)) w hwm (η i) (hμQ i)
  let A : Set (Perspective.ProbSphere d) := {μ₀ j, ν₀}
  have hA : ∀ ρ ∈ A, meanAxisMoment (w : EucSpace d) ρ = 0 ∧
      ∀ x ∈ (ρ : Measure (SSphere d)).support, a ≤ inner ℝ (x : EucSpace d) (p : EucSpace d) := by
    intro ρ hρ
    rcases hρ with rfl | hρ
    · exact ⟨hwm, fun x hx => hfloor x (Or.inl hx)⟩
    · have he : ρ = ν₀ := hρ
      subst ρ
      exact ⟨hmν, fun x hx => hfloor x (Or.inr hx)⟩
  obtain ⟨r, hr, hhull⟩ := exists_chordal_radius_for_geodesic_hull p ε hε
  let i : Fin d := ⟨0, by omega⟩
  let k : Fin d := ⟨1, by omega⟩
  have hik : i ≠ k := by intro h; have hh := congrArg Fin.val h; simp [i, k] at hh
  obtain ⟨θ, hθ, hex, hselected, hrest⟩ := exists_selective_three_phase_control i k hik p w hwp
    η hη hmiss A a ha hA T r hT hr
  refine ⟨θ, hθ, hex, ?_, ?_⟩
  · intro μ ν hμ0 hμ hν0 hν x hx
    rw [← hp]
    rcases hx with hx | hx
    · exact hhull _ (fun y hy => hselected ν₀ (by simp [A]) ν hν0 hν hy) x hx
    · exact hhull _ (fun y hy => hselected (μ₀ j) (by simp [A]) μ hμ0 hμ hy) x hx
  · intro k hk μ h0 hμ
    exact hrest ⟨k, hk⟩ μ h0 hμ

/-- Two distinct positive circle atoms have non-collinear original
means. Their target and a genuine collinear companion satisfy all source
support hypotheses; normalizing the target mean gives a positive unit
pole for the common height floor. Both prescribed tolerances are positive. -/
example : ∃ (j : Idx 2) (μ : Idx 2 → Perspective.ProbSphere 2)
    (ν : Perspective.ProbSphere 2) (p : SSphere 2),
    (2 : ℕ) ≤ 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 / 10 ∧
    (∀ i, (μ i : Measure (SSphere 2)).support ⊆ positiveQuadrant 2) ∧
    (∀ i : Idx 2, i ≠ j → ∀ γ : ℝ, barycenter 2 (μ i) ≠ γ • barycenter 2 (μ j)) ∧
    (ν : Measure (SSphere 2)).support ⊆ positiveQuadrant 2 ∧
    (∃ γ : ℝ, barycenter 2 ν = γ • barycenter 2 (μ j)) ∧
    ∀ k : Fin 2, 0 < (p : EucSpace 2) k := by
  let a : SSphere 2 := ⟨EuclideanSpace.single 0 (3 / 5 : ℝ) + EuclideanSpace.single 1 (4 / 5 : ℝ), by
    rw [mem_sphere_zero_iff_norm, EuclideanSpace.norm_eq, Fin.sum_univ_two]; norm_num⟩
  let b : SSphere 2 := ⟨EuclideanSpace.single 0 (4 / 5 : ℝ) + EuclideanSpace.single 1 (3 / 5 : ℝ), by
    rw [mem_sphere_zero_iff_norm, EuclideanSpace.norm_eq, Fin.sum_univ_two]; norm_num⟩
  let μ (i : Idx 2) := if i = 0 then Perspective.diracProb 2 a else Perspective.diracProb 2 b
  have hs (x : SSphere 2) (hx : x ∈ positiveQuadrant 2) :
      (Perspective.diracProb 2 x : Measure (SSphere 2)).support ⊆ positiveQuadrant 2 := by
    intro y hy
    rw [eq_of_mem_support_dirac hy]
    exact hx
  have ha : a ∈ positiveQuadrant 2 := by intro i; fin_cases i <;> norm_num [a]
  have hb : b ∈ positiveQuadrant 2 := by intro i; fin_cases i <;> norm_num [b]
  have hQ (i : Idx 2) : (μ i : Measure (SSphere 2)).support ⊆ positiveQuadrant 2 := by
    by_cases hi : i = 0
    · simpa only [μ, ite_eq_left hi] using hs a ha
    · simpa only [μ, ite_eq_right hi] using hs b hb
  have hsep (i : Idx 2) (hi : i ≠ 0) (γ : ℝ) : barycenter 2 (μ i) ≠ γ • barycenter 2 (μ 0) := by
    fin_cases i
    · exact (hi rfl).elim
    · intro he
      have ha' : barycenter 2 (μ 0) = (a : EucSpace 2) := by simp [μ, barycenter, Perspective.diracProb]
      have hb' : barycenter 2 (μ 1) = (b : EucSpace 2) := by simp [μ, barycenter, Perspective.diracProb]
      change barycenter 2 (μ 1) = γ • barycenter 2 (μ 0) at he
      rw [ha', hb'] at he
      have h0 := congrArg (fun x : EucSpace 2 => x 0) he
      have h1 := congrArg (fun x : EucSpace 2 => x 1) he
      norm_num [a, b] at h0 h1
      linarith
  obtain ⟨p, -, hp⟩ := exists_positive_barycenter_pole (by norm_num : 1 ≤ (2 : ℕ)) (μ 0) (hQ 0)
  exact ⟨0, μ, μ 0, p, le_rfl, zero_lt_one, by norm_num, hQ, hsep, hQ 0, ⟨1, by simp⟩, hp⟩

end Transformer.Interpolation
