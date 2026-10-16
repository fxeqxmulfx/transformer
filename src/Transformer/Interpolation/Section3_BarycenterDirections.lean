/-
# Original barycenter directions and finite separation relations

ArXiv:2411.04551v3, §3.2, `lem: induction.barycenter` and
`prop: separation`. The target pole is the normalized original mean.
The mean is recovered by multiplying this direction by its norm.
For positive-quadrant probabilities, equality of these directions is
exactly collinearity: all possible scalar factors are positive.

These identities concern the actual Bochner barycenter, including its
zero value where normalization returns zero. At nonzero original means,
the direction of the actual ridge law is continuous near time zero.
Finite intersections preserve every initially distinct direction during
one sufficiently short actual ridge phase.
-/

import Transformer.Interpolation.Section3_SmallRidgeStability

open Real MeasureTheory Filter Set
open scoped Topology

namespace Transformer.Interpolation

variable {d : ℕ}

/-- The normalized original barycenter appearing in the source target
ball. Source: arXiv:2411.04551v3, §3.2, `lem: induction.barycenter`. -/
noncomputable def barycenterDirection (ν : Perspective.ProbSphere d) : EucSpace d :=
  ‖barycenter d ν‖⁻¹ • barycenter d ν

/-- A nonzero original mean has a unit direction.
Source: arXiv:2411.04551v3, §3.2, the normalized target pole. -/
theorem norm_barycenterDirection (ν : Perspective.ProbSphere d) (hν : barycenter d ν ≠ 0) :
    ‖barycenterDirection ν‖ = 1 := norm_smul_inv_norm hν

/-- The original mean is recovered from its norm and direction, also
when it vanishes. Source: arXiv:2411.04551v3, §3.2. -/
theorem norm_smul_barycenterDirection (ν : Perspective.ProbSphere d) :
    ‖barycenter d ν‖ • barycenterDirection ν = barycenter d ν := by
  by_cases hν : barycenter d ν = 0
  · simp only [barycenterDirection, hν, norm_zero, smul_zero]
  · rw [barycenterDirection, smul_smul, mul_inv_cancel₀ (norm_ne_zero_iff.mpr hν), one_smul]

/-- Original positive-quadrant laws have positive direction coordinates.
Source: arXiv:2411.04551v3, §3.2, `lem: induction.barycenter`. -/
theorem barycenterDirection_coordinate_pos (ν : Perspective.ProbSphere d)
    (hν : (ν : Measure (SSphere d)).support ⊆ positiveQuadrant d) (i : Fin d) :
    0 < barycenterDirection ν i := by
  have hi := barycenter_coordinate_pos_of_positiveQuadrant ν hν i
  have hn : barycenter d ν ≠ 0 := by
    intro he
    rw [he] at hi
    exact (lt_irrefl 0) hi
  exact mul_pos (inv_pos.mpr (norm_pos_iff.mpr hn)) hi

/-- Positive rescaling of actual means preserves their source pole.
Source: arXiv:2411.04551v3, §3.2, collinear companion data. -/
theorem barycenterDirection_eq_of_pos_smul (μ ν : Perspective.ProbSphere d) (γ : ℝ)
    (hγ : 0 < γ) (hbar : barycenter d μ = γ • barycenter d ν) :
    barycenterDirection μ = barycenterDirection ν := by
  simp only [barycenterDirection]
  rw [hbar, norm_smul, Real.norm_eq_abs, abs_of_pos hγ, mul_inv_rev,
    smul_smul, mul_assoc, inv_mul_cancel₀ hγ.ne', mul_one]

/-- Equal actual directions imply collinear original means.
Source: arXiv:2411.04551v3, §3.2, barycenter classes in `prop: separation`. -/
theorem barycenter_colinear_of_direction_eq (μ ν : Perspective.ProbSphere d)
    (he : barycenterDirection μ = barycenterDirection ν) :
    ∃ γ : ℝ, barycenter d μ = γ • barycenter d ν := by
  refine ⟨‖barycenter d μ‖ * ‖barycenter d ν‖⁻¹, ?_⟩
  calc
    barycenter d μ = ‖barycenter d μ‖ • barycenterDirection μ :=
      (norm_smul_barycenterDirection μ).symm
    _ = ‖barycenter d μ‖ • barycenterDirection ν := congrArg (‖barycenter d μ‖ • ·) he
    _ = _ := by rw [barycenterDirection, smul_smul]

/-- Distinct source poles are exactly non-collinear original means for
quadrant data. Source: arXiv:2411.04551v3, §3.2, `prop: separation`. -/
theorem barycenterDirection_ne_iff_noncolinear (i : Fin d)
    (μ ν : Perspective.ProbSphere d)
    (hμ : (μ : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hν : (ν : Measure (SSphere d)).support ⊆ positiveQuadrant d) :
    barycenterDirection μ ≠ barycenterDirection ν ↔
      ∀ γ : ℝ, barycenter d μ ≠ γ • barycenter d ν := by
  constructor
  · intro hne γ he
    have hμi := barycenter_coordinate_pos_of_positiveQuadrant μ hμ i
    have hνi := barycenter_coordinate_pos_of_positiveQuadrant ν hν i
    have hh := congrArg (fun x : EucSpace d => x i) he
    change barycenter d μ i = γ * barycenter d ν i at hh
    have hγ : 0 < γ := by nlinarith
    exact hne (barycenterDirection_eq_of_pos_smul μ ν γ hγ he)
  · intro hne he
    obtain ⟨γ, hγ⟩ := barycenter_colinear_of_direction_eq μ ν he
    exact hne γ hγ

/-- A nonzero original mean gives continuity of its actual transported
direction near time zero. Source: arXiv:2411.04551v3, §3.2. -/
theorem continuousAt_barycenterDirection_ridgeLaw_zero (i : Fin d) (p u : EucSpace d)
    (b : ℝ) (ν : Perspective.ProbSphere d) (hν : barycenter d ν ≠ 0) :
    ContinuousAt (fun t => barycenterDirection (ridgeLaw p u b ν t)) 0 := by
  have hc := (continuous_barycenter_ridgeLaw i p u b ν).continuousAt (x := 0)
  exact (hc.norm.inv₀ (by simpa only [ridgeLaw_zero] using norm_ne_zero_iff.mpr hν)).smul hc

/-- Every distinct original direction survives one common short ridge
phase in a finite family. Source: arXiv:2411.04551v3, §3.2. -/
theorem eventually_ridge_family_direction_ne {ι : Type*} [Fintype ι]
    (i : Fin d) (p u : EucSpace d) (b : ℝ) (ν : ι → Perspective.ProbSphere d)
    (hν : ∀ k, barycenter d (ν k) ≠ 0) :
    ∀ᶠ t : ℝ in nhds 0, ∀ k l,
      barycenterDirection (ν k) ≠ barycenterDirection (ν l) →
      barycenterDirection (ridgeLaw p u b (ν k) t) ≠
        barycenterDirection (ridgeLaw p u b (ν l) t) := by
  apply Filter.eventually_all.mpr
  intro k
  apply Filter.eventually_all.mpr
  intro l
  by_cases hkl : barycenterDirection (ν k) ≠ barycenterDirection (ν l)
  · have hk := continuousAt_barycenterDirection_ridgeLaw_zero i p u b (ν k) (hν k)
    have hl := continuousAt_barycenterDirection_ridgeLaw_zero i p u b (ν l) (hν l)
    have he : ∀ᶠ t : ℝ in nhds 0,
        barycenterDirection (ridgeLaw p u b (ν k) t) -
          barycenterDirection (ridgeLaw p u b (ν l) t) ≠ 0 :=
      (hk.sub hl).eventually_ne (by simpa only [Pi.sub_apply, ridgeLaw_zero] using sub_ne_zero.mpr hkl)
    exact he.mono fun t ht _ h => ht (sub_eq_zero.mpr h)
  · exact Filter.Eventually.of_forall fun _ h => (hkl h).elim

/-- Two genuine positive circle atoms realize nonzero means and a
non-collinear pair. The same first law on both sides with scalar one
realizes the positive-rescaling and equal-direction premises. -/
example : ∃ (ν : Fin 2 → Perspective.ProbSphere 2) (γ : ℝ),
    0 < γ ∧ (∀ k, (ν k : Measure (SSphere 2)).support ⊆ positiveQuadrant 2) ∧
    (∀ k, barycenter 2 (ν k) ≠ 0) ∧
    (∀ q : ℝ, barycenter 2 (ν 0) ≠ q • barycenter 2 (ν 1)) ∧
    barycenter 2 (ν 0) = γ • barycenter 2 (ν 0) ∧
    barycenterDirection (ν 0) = barycenterDirection (ν 0) := by
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
  have hsep (q : ℝ) : barycenter 2 (ν 0) ≠ q • barycenter 2 (ν 1) := by
    intro he
    have h0 := congrArg (fun x : EucSpace 2 => x 0) he
    have h1 := congrArg (fun x : EucSpace 2 => x 1) he
    norm_num [ν, barycenter, Perspective.diracProb, a, z, circlePt] at h0 h1
    linarith
  exact ⟨ν, 1, zero_lt_one, hQ, hn, hsep, by simp, rfl⟩

end Transformer.Interpolation
