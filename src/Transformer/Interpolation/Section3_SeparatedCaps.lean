/-
# Disjoint quadrant caps and original barycenter directions

ArXiv:2411.04551v3, §3.2, `prop: separation`.
A finite family of distinct positive unit poles has disjoint small closed
chordal caps lying in the original positive quadrant. A single positive
radius is chosen for every pole and every distinct pair.

The original normalized barycenter of a probability supported in one
such cap stays in that same cap. Averaging preserves the positive axial
height, the mean has norm at most one, and normalization increases its
height. The original geodesic hull radius estimate is proved separately
in `Section3_TargetCaps`. These facts keep the finite contraction
induction's actual mean directions in distinct prescribed regions.
-/

import Transformer.Interpolation.Section3_BarycenterDirections
import Transformer.Interpolation.Section3_TargetCaps

open Real MeasureTheory Set Filter
open scoped Topology

namespace Transformer.Interpolation

variable {d : ℕ}

/-- Finitely many positive real tolerances admit a common positive
strict lower tolerance, also for an empty index type. Source auxiliary:
arXiv:2411.04551v3, §3.2, simultaneous cap selection. -/
theorem exists_common_positive_lt {ι : Type*} [Fintype ι] (ε : ι → ℝ)
    (hε : ∀ i, 0 < ε i) : ∃ r : ℝ, 0 < r ∧ ∀ i, r < ε i := by
  have he : ∀ᶠ r : ℝ in nhds 0, ∀ i, r < ε i :=
    Filter.eventually_all.mpr (fun i => gt_mem_nhds (hε i))
  obtain ⟨δ, hδ, hbound⟩ := Metric.eventually_nhds_iff.mp he
  refine ⟨δ / 2, half_pos hδ, ?_⟩
  apply hbound
  rw [Real.dist_eq, sub_zero, abs_of_pos (half_pos hδ)]
  exact half_lt_self hδ

/-- Distinct original quadrant poles have disjoint positive-radius
closed caps still in the quadrant. Source: arXiv:2411.04551v3, §3.2,
choosing the final separated neighborhoods in `prop: separation`. -/
theorem exists_separated_quadrant_caps {ι : Type*} [Fintype ι]
    (p : ι → SSphere d) (hp : ∀ i, p i ∈ positiveQuadrant d) (hinj : Function.Injective p) :
    ∃ r : ℝ, 0 < r ∧ r ≤ 1 ∧
      (∀ i, Metric.closedBall (p i) r ⊆ positiveQuadrant d) ∧
      ∀ i j, i ≠ j → Disjoint (Metric.closedBall (p i) r) (Metric.closedBall (p j) r) := by
  classical
  choose δ hδ hδQ using fun i => exists_ball_positiveQuadrant (p i) (hp i)
  let ε : Option (ι ⊕ (ι × ι)) → ℝ := fun k => match k with
    | none => 1
    | some (.inl i) => δ i
    | some (.inr (i, j)) => if i = j then 1 else dist (p i) (p j) / 2
  have hε (k : Option (ι ⊕ (ι × ι))) : 0 < ε k := by
    cases k with
    | none => exact zero_lt_one
    | some k =>
      cases k with
      | inl i => exact hδ i
      | inr ij =>
        rcases ij with ⟨i, j⟩
        dsimp only [ε]
        split_ifs with hij
        · exact zero_lt_one
        · exact div_pos (dist_pos.mpr (fun h => hij (hinj h))) (by norm_num)
  obtain ⟨r, hr, hsmall⟩ := exists_common_positive_lt ε hε
  refine ⟨r, hr, (hsmall none).le, ?_, ?_⟩
  · intro i x hx
    have hx' : dist x (p i) ≤ r := hx
    exact hδQ i (hx'.trans_lt (hsmall (some (.inl i))))
  · intro i j hij
    apply Set.disjoint_left.mpr
    intro x hx hy
    have hi : dist x (p i) ≤ r := hx
    have hj : dist x (p j) ≤ r := hy
    have hh := hsmall (some (.inr (i, j)))
    simp only [ε, ite_eq_right hij] at hh
    have hdist : 2 * r < dist (p i) (p j) := by linarith
    have htri := dist_triangle (p i) x (p j)
    rw [dist_comm (p i) x] at htri
    linarith

/-- The normalized original mean stays within a small closed chordal
cap supporting its law. Source: arXiv:2411.04551v3, §3.2, the barycenter
geometry in `prop: separation`. -/
theorem barycenterDirection_radius (p : SSphere d) (ν : Perspective.ProbSphere d) (r : ℝ)
    (hr : 0 ≤ r) (hr1 : r ≤ 1)
    (hν : (ν : Measure (SSphere d)).support ⊆ Metric.closedBall p r) :
    ‖barycenterDirection ν - (p : EucSpace d)‖ ≤ r := by
  let a := 1 - r ^ 2 / 2
  have ha : 0 < a := by dsimp only [a]; nlinarith
  have hheight (x : SSphere d) (hx : x ∈ (ν : Measure (SSphere d)).support) :
      a ≤ inner ℝ (x : EucSpace d) (p : EucSpace d) := by
    have hd : ‖(x : EucSpace d) - (p : EucSpace d)‖ ≤ r := hν hx
    have hn := norm_sub_sq_real (x : EucSpace d) (p : EucSpace d)
    rw [mem_sphere_zero_iff_norm.mp x.2, mem_sphere_zero_iff_norm.mp p.2] at hn
    dsimp only [a]
    nlinarith [norm_nonneg ((x : EucSpace d) - p)]
  have hs : ∀ᵐ x : SSphere d ∂(ν : Measure (SSphere d)), x ∈ (ν : Measure (SSphere d)).support :=
    Measure.support_mem_ae
  have hm := le_inner_barycenter d ν (p : EucSpace d) a (integrable_spherical_coe ν)
    (hs.mono fun x hx => hheight x hx)
  have hne : barycenter d ν ≠ 0 := by
    intro he
    rw [he, inner_zero_left] at hm
    linarith
  have hn : ‖barycenter d ν‖ ≤ 1 := norm_spherical_mean_le_one ν
  have hqpos : 0 < inner ℝ (barycenterDirection ν) (p : EucSpace d) := by
    rw [barycenterDirection, real_inner_smul_left]
    exact mul_pos (inv_pos.mpr (norm_pos_iff.mpr hne)) (ha.trans_le hm)
  have he : ‖barycenter d ν‖ * inner ℝ (barycenterDirection ν) (p : EucSpace d) =
      inner ℝ (barycenter d ν) (p : EucSpace d) := by
    rw [← real_inner_smul_left, norm_smul_barycenterDirection]
  have hcap : a ≤ inner ℝ (barycenterDirection ν) (p : EucSpace d) := by
    have hh := mul_le_mul_of_nonneg_right hn hqpos.le
    rw [he, one_mul] at hh
    exact hm.trans hh
  have hdist := norm_sub_sq_real (barycenterDirection ν) (p : EucSpace d)
  rw [norm_barycenterDirection ν hne, mem_sphere_zero_iff_norm.mp p.2] at hdist
  dsimp only [a] at hcap
  nlinarith [norm_nonneg (barycenterDirection ν - (p : EucSpace d))]

/-- An actual unit pole equal to the normalized mean belongs to the
same supporting cap. Source: arXiv:2411.04551v3, §3.2. -/
theorem barycenter_pole_mem_closedBall (p q : SSphere d) (ν : Perspective.ProbSphere d)
    (r : ℝ) (hr : 0 ≤ r) (hr1 : r ≤ 1)
    (hν : (ν : Measure (SSphere d)).support ⊆ Metric.closedBall p r)
    (hq : (q : EucSpace d) = barycenterDirection ν) : q ∈ Metric.closedBall p r := by
  change ‖(q : EucSpace d) - (p : EucSpace d)‖ ≤ r
  rw [hq]
  exact barycenterDirection_radius p ν r hr hr1 hν

/-- Two distinct positive circle points give genuine disjoint quadrant
caps. The original atom at the first pole realizes the support and
normalized-mean hypotheses at that same positive radius. -/
example : ∃ (ε : Fin 2 → ℝ) (p : Fin 2 → SSphere 2)
    (ν : Perspective.ProbSphere 2) (q : SSphere 2) (r : ℝ),
    (∀ i, 0 < ε i) ∧ (∀ i, p i ∈ positiveQuadrant 2) ∧ Function.Injective p ∧
    0 < r ∧ r ≤ 1 ∧ (ν : Measure (SSphere 2)).support ⊆ Metric.closedBall (p 0) r ∧
    (q : EucSpace 2) = barycenterDirection ν ∧
    (∀ i, Metric.closedBall (p i) r ⊆ positiveQuadrant 2) ∧
    ∀ i j, i ≠ j → Disjoint (Metric.closedBall (p i) r) (Metric.closedBall (p j) r) := by
  let a : SSphere 2 := circlePt (3 / 5) (4 / 5) (by norm_num)
  let b : SSphere 2 := circlePt (4 / 5) (3 / 5) (by norm_num)
  let p : Fin 2 → SSphere 2 := ![a, b]
  have hp (i : Fin 2) : p i ∈ positiveQuadrant 2 := by
    intro k
    fin_cases i <;> fin_cases k <;> norm_num [p, a, b, circlePt]
  have hinj : Function.Injective p := by
    intro i j he
    fin_cases i <;> fin_cases j
    · rfl
    · have hh := congrArg (fun x : SSphere 2 => (x : EucSpace 2) 0) he
      norm_num [p, a, b, circlePt] at hh
    · have hh := congrArg (fun x : SSphere 2 => (x : EucSpace 2) 0) he
      norm_num [p, a, b, circlePt] at hh
    · rfl
  obtain ⟨r, hr, hr1, hcap, hdisj⟩ := exists_separated_quadrant_caps p hp hinj
  let ν := Perspective.diracProb 2 (p 0)
  have hν : (ν : Measure (SSphere 2)).support ⊆ Metric.closedBall (p 0) r := by
    intro x hx
    rw [eq_of_mem_support_dirac hx]
    simpa only [Metric.mem_closedBall, dist_self] using hr.le
  have hq : (p 0 : EucSpace 2) = barycenterDirection ν := by
    simp [barycenterDirection, ν, barycenter, Perspective.diracProb,
      mem_sphere_zero_iff_norm.mp (p 0).2]
  exact ⟨(fun i => (i.val : ℝ) + 1), p, ν, p 0, r, (fun _ => by positivity),
    hp, hinj, hr, hr1, hν, hq, hcap, hdisj⟩

end Transformer.Interpolation
