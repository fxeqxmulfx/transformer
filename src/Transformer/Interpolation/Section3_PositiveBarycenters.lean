/-
# Positive barycenters and transverse hemisphere directions

ArXiv:2411.04551v3, §3.2, `lem: perturbation`, Part 2.

A probability supported in the strictly positive spherical quadrant
has a strictly positive barycenter in every coordinate. Its barycenter
therefore gives an open hemisphere containing both original supports.
In dimension at least two, an explicit two-coordinate vector supplies
a unit direction orthogonal to this hemisphere axis.

The dimension restriction follows from the original distinctness
premise: the zero-dimensional ambient sphere is empty, and the positive
quadrant in ambient dimension one is the singleton `+1`. Thus no
low-dimensional case is discarded when applying the ridge variation.
-/

import Transformer.Interpolation.Clustering
import Transformer.Interpolation.Section1_WeakContinuity
import Transformer.Interpolation.Section3_QuadrantGeometry

open Real MeasureTheory Filter Set
open scoped BigOperators

namespace Transformer.Interpolation

/-- Every original barycenter coordinate is strictly positive: its
integrand is positive almost everywhere and the probability has mass one.
Source: arXiv:2411.04551v3, §3.2, `lem: perturbation`. -/
theorem barycenter_coordinate_pos_of_positiveQuadrant {d : ℕ} (μ : Perspective.ProbSphere d)
    (hμ : (μ : Measure (SSphere d)).support ⊆ positiveQuadrant d) (i : Fin d) :
    0 < barycenter d μ i := by
  have hi : Integrable (fun x : SSphere d => (x : EucSpace d) i) (μ : Measure (SSphere d)) :=
    integrable_spherical_continuous μ _ (by fun_prop)
  have hsupp : ∀ᵐ (x : SSphere d) ∂(μ : Measure (SSphere d)),
      x ∈ (μ : Measure (SSphere d)).support := Measure.support_mem_ae
  have hpos : ∀ᵐ (x : SSphere d) ∂(μ : Measure (SSphere d)), 0 < (x : EucSpace d) i :=
    hsupp.mono fun x hx => hμ hx i
  have hmass : (μ : Measure (SSphere d)) (Function.support (fun x : SSphere d => (x : EucSpace d) i)) = 1 := by
    calc
      _ = (μ : Measure (SSphere d)) univ := measure_congr (hpos.mono fun x hx => by
        simp [Function.mem_support, hx.ne'])
      _ = 1 := measure_univ
  have hint : 0 < ∫ x : SSphere d, (x : EucSpace d) i ∂(μ : Measure (SSphere d)) :=
    (integral_pos_iff_support_of_nonneg_ae (hpos.mono fun x hx => hx.le) hi).mpr (by rw [hmass]; exact zero_lt_one)
  have heq := (EuclideanSpace.proj i : EucSpace d →L[ℝ] ℝ).integral_comp_comm (integrable_spherical_coe μ)
  change (∫ x : SSphere d, (x : EucSpace d) i ∂(μ : Measure (SSphere d))) = (barycenter d μ) i at heq
  exact heq ▸ hint

/-- Positive spherical points lie strictly inside the hemisphere of
any positive-coordinate axis. The supplied coordinate ensures nonemptiness.
Source: arXiv:2411.04551v3, §3.2, `lem: perturbation`. -/
theorem inner_pos_of_positive_coordinates {d : ℕ} (i : Fin d) (x : SSphere d)
    (hx : x ∈ positiveQuadrant d) (m : EucSpace d) (hm : ∀ j : Fin d, 0 < m j) :
    0 < inner ℝ (x : EucSpace d) m := by
  change 0 < ∑ j : Fin d, m j * (x : EucSpace d) j
  exact Finset.sum_pos' (fun j _ => (mul_pos (hm j) (hx j)).le)
    ⟨i, Finset.mem_univ i, mul_pos (hm i) (hx i)⟩

/-- Two coordinates give a nonzero vector orthogonal to a positive axis;
normalization supplies an actual unit transverse direction.
Source: arXiv:2411.04551v3, §3.2, the transverse perturbation. -/
theorem exists_unit_orthogonal_of_positive_coordinates {d : ℕ} (hd : 2 ≤ d)
    (m : EucSpace d) (hm : ∀ j : Fin d, 0 < m j) :
    ∃ p : EucSpace d, ‖p‖ = 1 ∧ inner ℝ p m = 0 := by
  let i : Fin d := ⟨0, by omega⟩
  let j : Fin d := ⟨1, by omega⟩
  have hij : i ≠ j := by intro h; have := congrArg Fin.val h; simp [i, j] at this
  let v : EucSpace d := m j • EuclideanSpace.single i 1 - m i • EuclideanSpace.single j 1
  have hv : v ≠ 0 := by
    intro he
    have he' := congrArg (fun x : EucSpace d => x i) he
    simp [v, hij] at he'
    exact (hm j).ne' he'
  refine ⟨‖v‖⁻¹ • v, norm_smul_inv_norm hv, ?_⟩
  rw [real_inner_smul_left]
  have hvm : inner ℝ v m = 0 := by
    simp [v, inner_sub_left, real_inner_smul_left, EuclideanSpace.inner_single_left, mul_comm]
  rw [hvm, mul_zero]

/-- The source's strict quadrant in ambient dimension one is `+1`.
Source: arXiv:2411.04551v3, §1.5 and §3.2, `lem: perturbation`. -/
theorem positiveQuadrant_one_point (x : SSphere 1) (hx : x ∈ positiveQuadrant 1) :
    x = basePoint 0 := by
  have hn := mem_sphere_zero_iff_norm.mp x.2
  rw [EuclideanSpace.norm_eq, Fin.sum_univ_one] at hn
  simp only [Real.norm_eq_abs, sq_abs, Real.sqrt_sq_eq_abs] at hn
  have hp : 0 < (x : EucSpace 1).ofLp 0 := hx 0
  rw [abs_of_pos hp] at hn
  apply Subtype.ext
  ext i
  have hi : i = (0 : Fin 1) := Subsingleton.elim i 0
  subst i
  simpa only [basePoint, PiLp.single_apply, ite_true] using hn

/-- Every probability in the one-point positive quadrant is the same
Dirac law. Source: arXiv:2411.04551v3, §3.2, `lem: perturbation`. -/
theorem eq_dirac_of_positiveQuadrant_one (μ : Perspective.ProbSphere 1)
    (hμ : (μ : Measure (SSphere 1)).support ⊆ positiveQuadrant 1) :
    μ = Perspective.diracProb 1 (basePoint 0) := by
  have hsupp : ∀ᵐ x ∂(μ : Measure (SSphere 1)), x ∈ (μ : Measure (SSphere 1)).support :=
    Measure.support_mem_ae
  have he : (id : SSphere 1 → SSphere 1) =ᵐ[(μ : Measure (SSphere 1))] (fun _ => basePoint 0) :=
    hsupp.mono fun x hx => positiveQuadrant_one_point x (hμ hx)
  apply Subtype.ext
  change (μ : Measure (SSphere 1)) = Measure.dirac (basePoint 0)
  simpa only [Measure.map_id, Measure.map_const, measure_univ, one_smul] using Measure.map_congr he

/-- The original distinct positive-quadrant laws force ambient dimension
at least two. Source: arXiv:2411.04551v3, §3.2, `lem: perturbation`. -/
theorem two_le_dim_of_distinct_positiveQuadrant {d : ℕ} (μ ν : Perspective.ProbSphere d)
    (hμ : (μ : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hν : (ν : Measure (SSphere d)).support ⊆ positiveQuadrant d) (hne : μ ≠ ν) :
    2 ≤ d := by
  by_contra hd
  have hd' : d ≤ 1 := by omega
  interval_cases d
  · obtain ⟨x, -⟩ := nonempty_of_measure_ne_zero (μ := (μ : Measure (SSphere 0)))
      (s := univ) (by rw [measure_univ]; exact one_ne_zero)
    have hx0 : (x : EucSpace 0) = 0 := by ext i; exact Fin.elim0 i
    have hn := mem_sphere_zero_iff_norm.mp x.2
    rw [hx0, norm_zero] at hn
    exact zero_ne_one hn
  · exact hne ((eq_dirac_of_positiveQuadrant_one μ hμ).trans
      (eq_dirac_of_positiveQuadrant_one ν hν).symm)

/-- The original support hypotheses supply all geometric inputs of the
weighted ridge separation: a transverse unit vector and a common open
hemisphere. Source: arXiv:2411.04551v3, §3.2, `lem: perturbation`. -/
theorem exists_transverse_hemisphere_of_positiveQuadrants {d : ℕ} (hd : 2 ≤ d)
    (μ ν : Perspective.ProbSphere d)
    (hμ : (μ : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hν : (ν : Measure (SSphere d)).support ⊆ positiveQuadrant d) :
    ∃ p : EucSpace d, ‖p‖ = 1 ∧ inner ℝ p (barycenter d ν) = 0 ∧
      (∀ᵐ x : SSphere d ∂(μ : Measure (SSphere d)), 0 < inner ℝ (x : EucSpace d) (barycenter d ν)) ∧
      (∀ᵐ x : SSphere d ∂(ν : Measure (SSphere d)), 0 < inner ℝ (x : EucSpace d) (barycenter d ν)) := by
  have hm := barycenter_coordinate_pos_of_positiveQuadrant ν hν
  obtain ⟨p, hp, hpm⟩ := exists_unit_orthogonal_of_positive_coordinates hd (barycenter d ν) hm
  let i : Fin d := ⟨0, by omega⟩
  have ha (ρ : Perspective.ProbSphere d)
      (hρ : (ρ : Measure (SSphere d)).support ⊆ positiveQuadrant d) :
      ∀ᵐ x : SSphere d ∂(ρ : Measure (SSphere d)), 0 < inner ℝ (x : EucSpace d) (barycenter d ν) := by
    have hsupp : ∀ᵐ x ∂(ρ : Measure (SSphere d)), x ∈ (ρ : Measure (SSphere d)).support :=
      Measure.support_mem_ae
    exact hsupp.mono fun x hx => inner_pos_of_positive_coordinates i x (hρ hx) _ hm
  exact ⟨p, hp, hpm, ha μ hμ, ha ν hν⟩

/-- The one-dimensional hypotheses are realized by the actual positive
unit point and its probability atom. -/
example : basePoint 0 ∈ positiveQuadrant 1 ∧
    ((Perspective.diracProb 1 (basePoint 0) : Perspective.ProbSphere 1) : Measure (SSphere 1)).support
    ⊆ positiveQuadrant 1 := by
  refine ⟨basePoint_mem_positiveQuadrant, fun x hx => ?_⟩
  have hx' : x ∈ (Measure.dirac (basePoint 0)).support := hx
  rw [eq_of_mem_support_dirac hx']
  exact basePoint_mem_positiveQuadrant

/-- Two distinct genuine positive-quadrant atoms and a positive axis
jointly realize the geometric and dimension hypotheses above. -/
example : ∃ (μ ν : Perspective.ProbSphere 2) (x : SSphere 2) (m : EucSpace 2),
    (2 : ℕ) ≤ 2 ∧ μ ≠ ν ∧ x ∈ positiveQuadrant 2 ∧ (∀ j : Fin 2, 0 < m j) ∧
    (μ : Measure (SSphere 2)).support ⊆ positiveQuadrant 2 ∧
    (ν : Measure (SSphere 2)).support ⊆ positiveQuadrant 2 := by
  let x := quadrantPoint (0 : Fin 2) 1 one_pos
  let y := quadrantPoint (0 : Fin 2) 2 (by norm_num)
  have hx : x ∈ positiveQuadrant 2 := quadrantPoint_mem _ _ _
  have hy : y ∈ positiveQuadrant 2 := quadrantPoint_mem _ _ _
  have hxy : x ≠ y := quadrantPoint_one_ne_two (0 : Fin 2) 1 (by decide)
  have hs (a : SSphere 2) (ha : a ∈ positiveQuadrant 2) :
      (Perspective.diracProb 2 a : Measure (SSphere 2)).support ⊆ positiveQuadrant 2 := by
    intro z hz
    have hz' : z ∈ (Measure.dirac a).support := hz
    rwa [eq_of_mem_support_dirac hz']
  refine ⟨Perspective.diracProb 2 x, Perspective.diracProb 2 y, x, (x : EucSpace 2), le_rfl,
    ?_, hx, hx, hs x hx, hs y hy⟩
  intro he
  have h := congrArg (fun ρ : Perspective.ProbSphere 2 => (ρ : Measure (SSphere 2))) he
  change Measure.dirac x = Measure.dirac y at h
  exact hxy (dirac_eq_dirac_iff.mp h)

/-- A point `(p, q)` of the unit circle.
Source: arXiv:2411.04551v3, §1.5, the original sphere. -/
noncomputable def circlePt (p q : ℝ) (h : p ^ 2 + q ^ 2 = 1) : SSphere 2 :=
  ⟨!₂[p, q], by
    rw [mem_sphere_zero_iff_norm, EuclideanSpace.norm_eq, Fin.sum_univ_two]
    simp [h]⟩

/-- Positive circle coordinates give a point in the original strict quadrant.
Source: arXiv:2411.04551v3, §1.5, `ℚ_1`. -/
theorem circlePt_mem_positiveQuadrant {p q : ℝ} (h : p ^ 2 + q ^ 2 = 1)
    (hp : 0 < p) (hq : 0 < q) : circlePt p q h ∈ positiveQuadrant 2 := by
  intro i
  fin_cases i <;> simpa [circlePt]

example : (3 / 5 : ℝ) ^ 2 + (4 / 5) ^ 2 = 1 ∧ (0 : ℝ) < 3 / 5 ∧ (0 : ℝ) < 4 / 5 := by
  norm_num

end Transformer.Interpolation
