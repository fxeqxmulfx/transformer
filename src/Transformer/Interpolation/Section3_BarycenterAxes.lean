module

public import Transformer.Interpolation.Section3_CommonTransverseAxis
public import Transformer.Interpolation.Section3_PositiveBarycenters
public import Transformer.Interpolation.Section3_MeanAxisLaw

/-!
# Original positive means and one common isolation axis

ArXiv:2411.04551v3, §3.2, Step 1 of `lem: induction.barycenter`.
The positive-quadrant hypothesis gives a nonzero positive target mean.
Its normalization is an actual unit pole with positive coordinates.
A dummy transverse unit vector makes the finite avoidance family
nonempty even when there are no other input laws.

One common unit axis is orthogonal to the original target mean and has
nonzero pairing with every other original mean. The target and any
collinear companion are invariant under its actual attention law.
Both axis poles lie outside the positive quadrant and hence outside
every original supported input, as required for uniform attraction.
-/

@[expose] public section

open Real MeasureTheory Set
open scoped Topology

namespace Transformer.Interpolation

variable {d N : ℕ}

/-- The source normalized original barycenter is an actual positive unit
pole. Source: arXiv:2411.04551v3, §3.2, the target of `lem: induction.barycenter`. -/
theorem exists_positive_barycenter_pole (hd : 1 ≤ d) (ν : Perspective.ProbSphere d)
    (hν : (ν : Measure (SSphere d)).support ⊆ positiveQuadrant d) :
    ∃ p : SSphere d, (p : EucSpace d) = ‖barycenter d ν‖⁻¹ • barycenter d ν ∧
      ∀ i : Fin d, 0 < (p : EucSpace d) i := by
  let m := barycenter d ν
  have hmpos := barycenter_coordinate_pos_of_positiveQuadrant ν hν
  let i₀ : Fin d := ⟨0, by omega⟩
  have hm : m ≠ 0 := by
    intro he
    have hh := hmpos i₀
    change 0 < m i₀ at hh
    rw [he] at hh
    exact (lt_irrefl (0 : ℝ)) hh
  let p : SSphere d := ⟨‖m‖⁻¹ • m, mem_sphere_zero_iff_norm.mpr (norm_smul_inv_norm hm)⟩
  refine ⟨p, rfl, fun i => ?_⟩
  change 0 < ‖m‖⁻¹ * m i
  exact mul_pos (inv_pos.mpr (norm_pos_iff.mpr hm)) (hmpos i)

/-- One actual attention axis fixes the target and has nonzero original
mean coefficient on every other input. Source: arXiv:2411.04551v3, §3.2, Step 1. -/
theorem exists_barycenter_isolation_axis (hd : 2 ≤ d) (j : Idx N) (μ : Idx N → Perspective.ProbSphere d)
    (hμj : (μ j : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hsep : ∀ i : Idx N, i ≠ j → ∀ γ : ℝ, barycenter d (μ i) ≠ γ • barycenter d (μ j)) :
    ∃ w : SSphere d, inner ℝ (w : EucSpace d) (barycenter d (μ j)) = 0 ∧
      ∀ i : Idx N, i ≠ j → meanAxisMoment (w : EucSpace d) (μ i) ≠ 0 := by
  classical
  let m := barycenter d (μ j)
  have hmpos := barycenter_coordinate_pos_of_positiveQuadrant (μ j) hμj
  let i₀ : Fin d := ⟨0, by omega⟩
  have hm : m ≠ 0 := by
    intro he
    have hh := hmpos i₀
    change 0 < m i₀ at hh
    rw [he] at hh
    exact (lt_irrefl (0 : ℝ)) hh
  obtain ⟨u, hu, hum⟩ := exists_unit_orthogonal_of_positive_coordinates hd m hmpos
  have hune (γ : ℝ) : u ≠ γ • m := by
    intro he
    have hh : γ * inner ℝ m m = 0 := by rw [← real_inner_smul_left, ← he]; exact hum
    have hg : γ = 0 := (mul_eq_zero.mp hh).resolve_right (ne_of_gt (real_inner_self_pos.mpr hm))
    rw [he, hg, zero_smul, norm_zero] at hu
    exact zero_ne_one hu
  let I := Option {i : Idx N // i ≠ j}
  let M : I → EucSpace d := fun i => match i with
    | none => u
    | some k => barycenter d (μ k)
  have hM (i : I) (γ : ℝ) : M i ≠ γ • m := by
    cases i with
    | none => exact hune γ
    | some k => exact hsep k k.2 γ
  obtain ⟨w, hwm, hw⟩ := exists_common_unit_transverse_axis_of_nonzero m hm M hM
  exact ⟨w, hwm, fun i hi => hw (some ⟨i, hi⟩)⟩

/-- Original data collinear with a transverse target mean are stationary
under the actual rank-one law. Source: arXiv:2411.04551v3, §3.2, Step 1. -/
theorem meanAxisLaw_of_barycenter_colinear (w : SSphere d) (κ : ℝ) (m : EucSpace d)
    (hwm : inner ℝ (w : EucSpace d) m = 0) (ν : Perspective.ProbSphere d)
    (hν : ∃ γ : ℝ, barycenter d ν = γ • m) : meanAxisLaw w κ ν = fun _ => ν := by
  apply meanAxisLaw_of_moment_eq_zero
  obtain ⟨γ, hγ⟩ := hν
  change inner ℝ (w : EucSpace d) (barycenter d ν) = 0
  rw [hγ, real_inner_smul_right, hwm, mul_zero]

/-- Both transverse axis poles lie outside the original positive quadrant.
Source: arXiv:2411.04551v3, §3.2, Step 1. -/
theorem orthogonal_poles_not_positiveQuadrant (i : Fin d) (m : EucSpace d)
    (hm : ∀ k : Fin d, 0 < m k) (w : SSphere d) (hwm : inner ℝ (w : EucSpace d) m = 0) :
    w ∉ positiveQuadrant d ∧ antipode d w ∉ positiveQuadrant d := by
  constructor
  · intro hw
    have hp := inner_pos_of_positive_coordinates i w hw m hm
    rw [hwm] at hp
    exact (lt_irrefl (0 : ℝ)) hp
  · intro hw
    have hp := inner_pos_of_positive_coordinates i (antipode d w) hw m hm
    change 0 < inner ℝ (-(w : EucSpace d)) m at hp
    rw [inner_neg_left, hwm, neg_zero] at hp
    exact (lt_irrefl (0 : ℝ)) hp

/-- Original positive-quadrant supports miss both transverse poles.
Source: arXiv:2411.04551v3, §3.2, Step 1. -/
theorem orthogonal_poles_not_support (i : Fin d) (m : EucSpace d)
    (hm : ∀ k : Fin d, 0 < m k) (w : SSphere d) (hwm : inner ℝ (w : EucSpace d) m = 0)
    (ν : Perspective.ProbSphere d) (hν : (ν : Measure (SSphere d)).support ⊆ positiveQuadrant d) :
    w ∉ (ν : Measure (SSphere d)).support ∧ antipode d w ∉ (ν : Measure (SSphere d)).support := by
  have hq := orthogonal_poles_not_positiveQuadrant i m hm w hwm
  exact ⟨fun h => hq.1 (hν h), fun h => hq.2 (hν h)⟩

/-- Two distinct positive circle atoms have non-collinear original
means. The target itself is a genuine collinear companion, and the proved
isolation theorem supplies an actual orthogonal unit axis for these data. -/
example : ∃ (j : Idx 2) (μ : Idx 2 → Perspective.ProbSphere 2) (ν : Perspective.ProbSphere 2)
    (m : EucSpace 2) (w : SSphere 2),
    2 ≤ (2 : ℕ) ∧ (μ j : Measure (SSphere 2)).support ⊆ positiveQuadrant 2 ∧
    (ν : Measure (SSphere 2)).support ⊆ positiveQuadrant 2 ∧ m = barycenter 2 (μ j) ∧
    (∀ i : Fin 2, 0 < m i) ∧ inner ℝ (w : EucSpace 2) m = 0 ∧
    (∀ i : Idx 2, i ≠ j → ∀ γ : ℝ, barycenter 2 (μ i) ≠ γ • m) ∧
    (∃ γ : ℝ, barycenter 2 ν = γ • m) := by
  let a : SSphere 2 := ⟨EuclideanSpace.single 0 (3 / 5 : ℝ) + EuclideanSpace.single 1 (4 / 5 : ℝ), by
    rw [mem_sphere_zero_iff_norm, EuclideanSpace.norm_eq, Fin.sum_univ_two]; norm_num⟩
  let b : SSphere 2 := ⟨EuclideanSpace.single 0 (4 / 5 : ℝ) + EuclideanSpace.single 1 (3 / 5 : ℝ), by
    rw [mem_sphere_zero_iff_norm, EuclideanSpace.norm_eq, Fin.sum_univ_two]; norm_num⟩
  let μ (i : Idx 2) := if i = 0 then Perspective.diracProb 2 a else Perspective.diracProb 2 b
  have hQ : (μ 0 : Measure (SSphere 2)).support ⊆ positiveQuadrant 2 := by
    intro x hx
    have he : x = a := eq_of_mem_support_dirac hx
    rw [he]
    intro i
    fin_cases i <;> norm_num [a]
  have hsep (i : Idx 2) (hi : i ≠ 0) (γ : ℝ) : barycenter 2 (μ i) ≠ γ • barycenter 2 (μ 0) := by
    fin_cases i
    · exact (hi rfl).elim
    · intro he
      have ha : barycenter 2 (μ 0) = (a : EucSpace 2) := by simp [μ, barycenter, Perspective.diracProb]
      have hb : barycenter 2 (μ 1) = (b : EucSpace 2) := by simp [μ, barycenter, Perspective.diracProb]
      change barycenter 2 (μ 1) = γ • barycenter 2 (μ 0) at he
      rw [ha, hb] at he
      have h0 := congrArg (fun x : EucSpace 2 => x 0) he
      have h1 := congrArg (fun x : EucSpace 2 => x 1) he
      norm_num [a, b] at h0 h1
      linarith
  obtain ⟨w, hw, _⟩ := exists_barycenter_isolation_axis (by norm_num : 2 ≤ (2 : ℕ)) 0 μ hQ hsep
  exact ⟨0, μ, μ 0, barycenter 2 (μ 0), w, le_rfl, hQ, hQ, rfl,
    barycenter_coordinate_pos_of_positiveQuadrant (μ 0) hQ, hw, hsep, 1, by simp⟩

end Transformer.Interpolation
