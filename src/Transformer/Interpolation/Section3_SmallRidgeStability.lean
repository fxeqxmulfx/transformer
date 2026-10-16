/-
# Small actual ridge phases preserve finite geometric data

ArXiv:2411.04551v3, §3.2, the perturbation steps in `prop: separation`.
Joint continuity of the actual particle flow preserves an open support
region for a common short time on each compact original support.
Finite intersections give one time neighborhood for the entire family.

Actual barycenters are continuous by the original weak equation.
Non-collinear pairs with nonzero second vector are linearly independent;
that property is open in finite dimension. Thus one sufficiently small
ridge phase preserves every existing non-collinearity relation and all
positive-quadrant supports at once. This replaces the source's false
fixed-outside-hulls requirement by continuity of the original laws.
-/

import Transformer.Interpolation.Section3_RidgeWellPosed
import Transformer.Interpolation.Section3_ProjectionTransport
import Transformer.Interpolation.Section3_PositiveBarycenters
import Mathlib.Analysis.Normed.Module.FiniteDimension

open Real MeasureTheory Set Filter
open scoped Topology

namespace Transformer.Interpolation

variable {d : ℕ}

/-- A compact original set remains in an open region for a common short
time. Source: arXiv:2411.04551v3, §3.2, small perceptron perturbations. -/
theorem eventually_ridgeFlow_mem_open (p u : EucSpace d) (b : ℝ)
    (K O : Set (SSphere d)) (hK : IsCompact K) (hO : IsOpen O) (hKO : K ⊆ O) :
    ∀ᶠ t : ℝ in nhds 0, ∀ x ∈ K, ridgeFlow p u b x t ∈ O := by
  apply hK.eventually_forall_of_forall_eventually
  intro x hx
  have hc : ContinuousAt (fun q : ℝ × SSphere d => ridgeFlow p u b q.2 q.1) (0, x) :=
    (continuous_ridgeFlow p u b).continuousAt
  exact hc.tendsto.eventually (hO.mem_nhds (by simpa only [ridgeFlow_zero] using hKO hx))

/-- The entire actual transported support remains in its initial open
region for small time. Source: arXiv:2411.04551v3, §3.2. -/
theorem eventually_support_ridgeLaw_mem_open (p u : EucSpace d) (b : ℝ)
    (ν : Perspective.ProbSphere d) (O : Set (SSphere d)) (hO : IsOpen O)
    (hν : (ν : Measure (SSphere d)).support ⊆ O) :
    ∀ᶠ t : ℝ in nhds 0, (ridgeLaw p u b ν t : Measure (SSphere d)).support ⊆ O := by
  have he := eventually_ridgeFlow_mem_open p u b (ν : Measure (SSphere d)).support O
    Measure.isClosed_support.isCompact hO hν
  filter_upwards [he] with t ht
  rw [coe_ridgeLaw]
  have hs := support_map_subset_image (ν : Measure (SSphere d))
    (fun x => ridgeFlow p u b x t) (continuous_ridgeFlow_initial p u b t)
  intro y hy
  obtain ⟨x, hx, rfl⟩ := hs hy
  exact ht x hx

/-- One short-time neighborhood preserves every original quadrant
support in a finite family. Source: arXiv:2411.04551v3, §3.2. -/
theorem eventually_ridge_family_positiveQuadrant {ι : Type*} [Fintype ι]
    (p u : EucSpace d) (b : ℝ) (ν : ι → Perspective.ProbSphere d)
    (hν : ∀ k, (ν k : Measure (SSphere d)).support ⊆ positiveQuadrant d) :
    ∀ᶠ t : ℝ in nhds 0, ∀ k,
      (ridgeLaw p u b (ν k) t : Measure (SSphere d)).support ⊆ positiveQuadrant d := by
  exact Filter.eventually_all.mpr (fun k => eventually_support_ridgeLaw_mem_open p u b (ν k)
    (positiveQuadrant d) (isOpen_positiveQuadrant (d := d)) (hν k))

/-- The original ridge barycenter is continuous in its actual clock.
Source: arXiv:2411.04551v3, §1 and §3.2, `eq: cauchy.pb`. -/
theorem continuous_barycenter_ridgeLaw (i : Fin d) (p u : EucSpace d) (b : ℝ)
    (ν : Perspective.ProbSphere d) :
    Continuous (fun t => barycenter d (ridgeLaw p u b ν t)) :=
  continuous_cauchyPB_mean (fun _ => reluRidgeParams i p u b) _ (cauchyPB_ridgeLaw i p u b ν)

/-- Continuous pairs preserve non-collinearity near a point with nonzero
second vector. Auxiliary for arXiv:2411.04551v3, §3.2. -/
theorem eventually_noncolinear_of_continuous {α : Type*} [TopologicalSpace α]
    (x y : α → EucSpace d) (hx : Continuous x) (hy : Continuous y) (t₀ : α)
    (hy₀ : y t₀ ≠ 0) (hxy : ∀ γ : ℝ, x t₀ ≠ γ • y t₀) :
    ∀ᶠ t in nhds t₀, ∀ γ : ℝ, x t ≠ γ • y t := by
  have hc : Continuous (fun t => ![x t, y t]) := by
    apply continuous_pi
    intro k
    fin_cases k <;> simpa using (by assumption)
  have hli : LinearIndependent ℝ ![x t₀, y t₀] :=
    linearIndependent_fin2.mpr ⟨hy₀, fun γ => (hxy γ).symm⟩
  have he := (hc.tendsto t₀).eventually hli.eventually
  exact he.mono fun t ht γ => ((linearIndependent_fin2.mp ht).2 γ).symm

/-- Every existing non-collinearity relation survives one common short
actual ridge phase. Source: arXiv:2411.04551v3, §3.2. -/
theorem eventually_ridge_family_noncolinear {ι : Type*} [Fintype ι]
    (i : Fin d) (p u : EucSpace d) (b : ℝ) (ν : ι → Perspective.ProbSphere d)
    (hν : ∀ k, barycenter d (ν k) ≠ 0) :
    ∀ᶠ t : ℝ in nhds 0, ∀ k l,
      (∀ γ : ℝ, barycenter d (ν k) ≠ γ • barycenter d (ν l)) →
      ∀ γ : ℝ, barycenter d (ridgeLaw p u b (ν k) t) ≠
        γ • barycenter d (ridgeLaw p u b (ν l) t) := by
  apply Filter.eventually_all.mpr
  intro k
  apply Filter.eventually_all.mpr
  intro l
  by_cases hkl : ∀ γ : ℝ, barycenter d (ν k) ≠ γ • barycenter d (ν l)
  · have he := eventually_noncolinear_of_continuous
      (fun t => barycenter d (ridgeLaw p u b (ν k) t))
      (fun t => barycenter d (ridgeLaw p u b (ν l) t))
      (continuous_barycenter_ridgeLaw i p u b (ν k))
      (continuous_barycenter_ridgeLaw i p u b (ν l)) 0
      (by simpa only [ridgeLaw_zero] using hν l)
      (by simpa only [ridgeLaw_zero] using hkl)
    exact he.mono fun t ht _ => ht
  · exact Filter.Eventually.of_forall fun _ h => (hkl h).elim

/-- Both original quadrant supports and every existing non-collinearity
relation survive together. Source: arXiv:2411.04551v3, §3.2. -/
theorem eventually_ridge_family_stable {ι : Type*} [Fintype ι]
    (i : Fin d) (p u : EucSpace d) (b : ℝ) (ν : ι → Perspective.ProbSphere d)
    (hν : ∀ k, (ν k : Measure (SSphere d)).support ⊆ positiveQuadrant d) :
    ∀ᶠ t : ℝ in nhds 0,
      (∀ k, (ridgeLaw p u b (ν k) t : Measure (SSphere d)).support ⊆ positiveQuadrant d) ∧
      ∀ k l, (∀ γ : ℝ, barycenter d (ν k) ≠ γ • barycenter d (ν l)) →
        ∀ γ : ℝ, barycenter d (ridgeLaw p u b (ν k) t) ≠
          γ • barycenter d (ridgeLaw p u b (ν l) t) := by
  have hn (k : ι) : barycenter d (ν k) ≠ 0 := by
    intro he
    have hh := barycenter_coordinate_pos_of_positiveQuadrant (ν k) (hν k) i
    rw [he] at hh
    exact (lt_irrefl 0) hh
  exact (eventually_ridge_family_positiveQuadrant p u b ν hν).and
    (eventually_ridge_family_noncolinear i p u b ν hn)

/-- Two distinct positive circle atoms have non-collinear nonzero means
and compact genuine supports inside the open quadrant. A nonzero ridge
output and positive activation give actual small perturbations of these
data; the constant original mean curves also witness the continuity lemma. -/
example : ∃ (p : EucSpace 2) (b : ℝ)
    (ν : Fin 2 → Perspective.ProbSphere 2) (K O : Set (SSphere 2)),
    p ≠ 0 ∧ 0 < b ∧ IsCompact K ∧ K.Nonempty ∧ IsOpen O ∧ K ⊆ O ∧
    (∀ k, (ν k : Measure (SSphere 2)).support ⊆ positiveQuadrant 2) ∧
    (∀ k, barycenter 2 (ν k) ≠ 0) ∧
    (∀ γ : ℝ, barycenter 2 (ν 0) ≠ γ • barycenter 2 (ν 1)) ∧
    Continuous (fun _ : ℝ => barycenter 2 (ν 0)) ∧
    Continuous (fun _ : ℝ => barycenter 2 (ν 1)) := by
  let a : SSphere 2 := circlePt (3 / 5) (4 / 5) (by norm_num)
  let z : SSphere 2 := circlePt (4 / 5) (3 / 5) (by norm_num)
  let ν : Fin 2 → Perspective.ProbSphere 2 :=
    ![Perspective.diracProb 2 a, Perspective.diracProb 2 z]
  have hs (x : SSphere 2) (hx : x ∈ positiveQuadrant 2) :
      (Perspective.diracProb 2 x : Measure (SSphere 2)).support ⊆ positiveQuadrant 2 := by
    intro y hy
    rw [eq_of_mem_support_dirac hy]
    exact hx
  have hQ (k : Fin 2) : (ν k : Measure (SSphere 2)).support ⊆ positiveQuadrant 2 := by
    fin_cases k
    · exact hs a (circlePt_mem_positiveQuadrant _ (by norm_num) (by norm_num))
    · exact hs z (circlePt_mem_positiveQuadrant _ (by norm_num) (by norm_num))
  have hn (k : Fin 2) : barycenter 2 (ν k) ≠ 0 := by
    intro he
    have hh := barycenter_coordinate_pos_of_positiveQuadrant (ν k) (hQ k) 0
    rw [he] at hh
    exact (lt_irrefl 0) hh
  have hsep (γ : ℝ) : barycenter 2 (ν 0) ≠ γ • barycenter 2 (ν 1) := by
    intro he
    have h0 := congrArg (fun x : EucSpace 2 => x 0) he
    have h1 := congrArg (fun x : EucSpace 2 => x 1) he
    norm_num [ν, barycenter, Perspective.diracProb, a, z, circlePt] at h0 h1
    linarith
  let p : EucSpace 2 := basePoint 1
  have hp : p ≠ 0 := by
    intro he
    have hh := mem_sphere_zero_iff_norm.mp (basePoint 1).2
    change ‖p‖ = 1 at hh
    rw [he, norm_zero] at hh
    exact zero_ne_one hh
  exact ⟨p, 1, ν, (ν 0 : Measure (SSphere 2)).support, positiveQuadrant 2,
    hp, zero_lt_one, Measure.isClosed_support.isCompact,
    Measure.nonempty_support (IsProbabilityMeasure.ne_zero _), isOpen_positiveQuadrant,
    hQ 0, hQ, hn, hsep, continuous_const, continuous_const⟩

end Transformer.Interpolation
