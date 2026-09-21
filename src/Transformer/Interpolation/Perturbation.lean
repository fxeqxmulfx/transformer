/-
# Measure-to-measure interpolation — The pair of lemmas behind separation

Formalization of §3.2 of arXiv:2411.04551v3:

* `Lemma lem: induction.barycenter` — shrinking one measure and a companion
  into a small ball around the direction of its barycenter,
* `Lemma lem: perturbation` / `lem: colinearity` — making two measures with
  collinear barycenters "non-colinear".

As in `Transformer.Interpolation.Disentanglement`, "the solution" is read as: a
solution exists, and every solution satisfies the conclusion; `O(d · N)` is a
constant chosen before `d`, `N` and the data.
-/

import Transformer.Basic
import Transformer.Perspective.Section2_FlowMap
import Transformer.Interpolation.Basic
import Transformer.Interpolation.Clustering
import Transformer.Interpolation.HypPropagationFalse

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Interpolation

open Interpolation Perspective

/-- **Lemma (lem: induction.barycenter).** *Inductive barycenter shrinking.*

Let `μ_0^i ∈ 𝒫(ℚ_1^{d-1})` have pairwise non-collinear barycenters.  For any
`T, ε > 0` and any `ν_0 ∈ 𝒫(ℚ_1^{d-1})` whose barycenter is collinear with
`𝔼_{μ_0^j}[x]`, there is a piecewise-constant `θ : [0, T] → Θ` with at most
`O(d · N)` switches such that

  `conv_g supp ν(T) ∪ conv_g supp μ^j(T) ⊂ B(𝔼_{μ_0^j}[z] / ‖𝔼_{μ_0^j}[z]‖, ε)`,

while `μ^i(T) = μ_0^i` for `i ≠ j`.  The ball is geodesic (§1.5), i.e.
`arccos ⟨x, p⟩ < ε`.

The source writes the collinearity hypothesis and the hull with `μ_0^N`, the
ball and the last clause with `μ_0^j`, and has "Fix `j`" commented out; one
index `j` is used throughout here, which is the reading under which the four
occurrences refer to the same measure.

Not proved here.

Source: arXiv:2411.04551v3, §3, `lem: induction.barycenter`. -/
theorem induction_barycenter :
    ∃ C : ℕ, ∀ (d N : ℕ) (T ε : ℝ) (j : Idx N) (μ₀ : Idx N → ProbSphere d)
      (ν₀ : ProbSphere d), 0 < T → 0 < ε →
      (∀ i : Idx N, (μ₀ i : Measure (SSphere d)).support ⊆ positiveQuadrant d) →
      (∀ i i' : Idx N, i ≠ i' →
        ∀ γ : ℝ, barycenter d (μ₀ i) ≠ γ • barycenter d (μ₀ i')) →
      (ν₀ : Measure (SSphere d)).support ⊆ positiveQuadrant d →
      (∃ γ : ℝ, barycenter d ν₀ = γ • barycenter d (μ₀ j)) →
      ∃ (θ : TimeParams d) (K : ℕ),
        K ≤ C * (d * N) ∧ PiecewiseConstant d θ T K ∧
        (∀ i : Idx N, ∃ μ : ℝ → ProbSphere d, μ 0 = μ₀ i ∧ cauchyPB d θ μ) ∧
        (∃ ν : ℝ → ProbSphere d, ν 0 = ν₀ ∧ cauchyPB d θ ν) ∧
        (∀ μ ν : ℝ → ProbSphere d, μ 0 = μ₀ j → cauchyPB d θ μ →
          ν 0 = ν₀ → cauchyPB d θ ν →
          ∀ x ∈ convG d (ν T : Measure (SSphere d)).support ∪
              convG d (μ T : Measure (SSphere d)).support,
            Real.arccos (inner (𝕜 := ℝ) (x : EucSpace d)
              (‖barycenter d (μ₀ j)‖⁻¹ • barycenter d (μ₀ j))) < ε) ∧
        ∀ i : Idx N, i ≠ j → ∀ μ : ℝ → ProbSphere d,
          μ 0 = μ₀ i → cauchyPB d θ μ → μ T = μ₀ i := by
  sorry

/-- The hypotheses of `induction_barycenter` are satisfiable on a one-element
family: `N = 1` leaves the non-collinearity of *distinct* barycenters nothing to
check, `ν₀ = μ₀ 0 = δ_{basePoint 0}` is supported in `ℚ_1^0`, and the two
barycenters are collinear with `γ = 1`. -/
example :
    (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
      (∀ i : Idx 1,
        (((fun _ : Idx 1 => diracProb 1 (basePoint 0)) i : Measure (SSphere 1))).support
          ⊆ positiveQuadrant 1) ∧
      (∀ i i' : Idx 1, i ≠ i' →
        ∀ γ : ℝ, barycenter 1 ((fun _ : Idx 1 => diracProb 1 (basePoint 0)) i)
          ≠ γ • barycenter 1 ((fun _ : Idx 1 => diracProb 1 (basePoint 0)) i')) ∧
      ((diracProb 1 (basePoint 0) : ProbSphere 1) : Measure (SSphere 1)).support
        ⊆ positiveQuadrant 1 ∧
      (∃ γ : ℝ, barycenter 1 (diracProb 1 (basePoint 0))
        = γ • barycenter 1 ((fun _ : Idx 1 => diracProb 1 (basePoint 0)) 0)) := by
  have hQ : ((diracProb 1 (basePoint 0) : ProbSphere 1) : Measure (SSphere 1)).support
      ⊆ positiveQuadrant 1 := fun x hx => by
    have hx' : x ∈ (Measure.dirac (basePoint 0)).support := hx
    rw [eq_of_mem_support_dirac hx']
    exact basePoint_mem_positiveQuadrant
  exact ⟨one_pos, one_pos, fun _ => hQ, fun i i' hne => absurd (Subsingleton.elim i i') hne,
    hQ, 1, (one_smul ℝ _).symm⟩

variable (d : ℕ)

/-- **Lemma (lem: perturbation / lem: colinearity).**

Let `T > 0` and let `μ_0, ν_0 ∈ 𝒫(ℚ_1^{d-1})` be two different measures with
`𝔼_{μ_0}[x] = γ_1 𝔼_{ν_0}[x]` for some `γ_1 ∈ (0, 1]`.

1. If `γ_1 = 1`, then with `𝐕 ≡ 0` there are constant `𝐖, 𝐔` and `b` such that
   `𝔼_{μ(T)}[x] ≠ 𝔼_{ν(T)}[x]`; moreover the flow map `Φ^T : 𝕊^{d-1} → 𝕊^{d-1}`
   of the characteristics is Lipschitz, invertible, and satisfies
   `eq: identity.flow`: `Φ^T(x) = x` off `conv_g supp μ_0 ∪ conv_g supp ν_0`.
2. If `γ_1 ≠ 1`, then with `𝐁 ≡ 0` there are `(𝐕, 𝐖, 𝐔, b)`, piecewise
   constant with at most two switches, such that
   `𝔼_{μ(T)}[x] ≠ γ_2 𝔼_{ν(T)}[x]` for every `γ_2 ∈ ℝ`.

In case 1 the vector field is `eq: average.vf`, so `𝐁 ≡ 0` as well; with
`𝐕 ≡ 0` it does not depend on the measure, and a characteristic is a curve
`y` with `ẏ = 𝐯(t, y)` (`IsCharacteristic`) for any measure curve in the
attention slot.

Not proved here.

Source: arXiv:2411.04551v3, §3, `lem: perturbation`, `eq: identity.flow`. -/
theorem perturbation (T γ₁ : ℝ) (μ₀ ν₀ : ProbSphere d) (hT : 0 < T) (hne : μ₀ ≠ ν₀)
    (hμ₀ : (μ₀ : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hν₀ : (ν₀ : Measure (SSphere d)).support ⊆ positiveQuadrant d)
    (hγ₀ : 0 < γ₁) (hγ₁ : γ₁ ≤ 1) (hbary : barycenter d μ₀ = γ₁ • barycenter d ν₀) :
    (γ₁ = 1 →
      ∃ (W U : ParamMatrix d) (b : EucSpace d),
        let θ : TimeParams d := fun _ => { V := 0, B := 0, W := W, U := U, b := b }
        (∃ μ : ℝ → ProbSphere d, μ 0 = μ₀ ∧ cauchyPB d θ μ) ∧
        (∃ ν : ℝ → ProbSphere d, ν 0 = ν₀ ∧ cauchyPB d θ ν) ∧
        (∀ μ ν : ℝ → ProbSphere d, μ 0 = μ₀ → cauchyPB d θ μ →
          ν 0 = ν₀ → cauchyPB d θ ν → barycenter d (μ T) ≠ barycenter d (ν T)) ∧
        ∃ Φ : SSphere d → SSphere d, (∃ L, LipschitzWith L Φ) ∧ Function.Bijective Φ ∧
          (∀ (x : SSphere d) (ρ : ℝ → ProbSphere d) (y : ℝ → EucSpace d),
            y 0 = x → IsCharacteristic d θ ρ y →
              y T = Φ x) ∧
          ∀ x : SSphere d, x ∉ convG d (μ₀ : Measure (SSphere d)).support ∪
            convG d (ν₀ : Measure (SSphere d)).support → Φ x = x) ∧
    (γ₁ ≠ 1 →
      ∃ θ : TimeParams d,
        PiecewiseConstant d θ T 3 ∧ (∀ s : ℝ, (θ s).B = 0) ∧
        (∃ μ : ℝ → ProbSphere d, μ 0 = μ₀ ∧ cauchyPB d θ μ) ∧
        (∃ ν : ℝ → ProbSphere d, ν 0 = ν₀ ∧ cauchyPB d θ ν) ∧
        ∀ μ ν : ℝ → ProbSphere d, μ 0 = μ₀ → cauchyPB d θ μ →
          ν 0 = ν₀ → cauchyPB d θ ν →
          ∀ γ₂ : ℝ, barycenter d (μ T) ≠ γ₂ • barycenter d (ν T)) := by
  sorry

/-- A point `(p, q)` of the circle with `p² + q² = 1`. -/
noncomputable def circlePt (p q : ℝ) (h : p ^ 2 + q ^ 2 = 1) : SSphere 2 :=
  ⟨!₂[p, q], by
    rw [mem_sphere_zero_iff_norm, EuclideanSpace.norm_eq, Fin.sum_univ_two]
    simp [h]⟩

/-- A point of the circle with both coordinates positive lies in `ℚ_1^1`. -/
theorem circlePt_mem_positiveQuadrant {p q : ℝ} (h : p ^ 2 + q ^ 2 = 1)
    (hp : 0 < p) (hq : 0 < q) : circlePt p q h ∈ positiveQuadrant 2 := by
  intro i
  fin_cases i <;> simpa [circlePt]

/-- `½δ_x + ½δ_y` is supported in `{x, y}`: the complement of that finite set
is an open null set. -/
theorem mem_of_mem_support_halfDirac {d : ℕ} {x y z : SSphere d}
    (hz : z ∈ (halfDirac d x y : Measure (SSphere d)).support) : z = x ∨ z = y := by
  by_contra hne
  push Not at hne
  refine Measure.notMem_support_iff_exists.mpr ⟨{x, y}ᶜ, ?_, ?_⟩ hz
  · exact ((Set.toFinite _).isClosed).isOpen_compl.mem_nhds (by simp [hne.1, hne.2])
  · simp

/-- The barycenter of `½δ_x + ½δ_y` is the midpoint `½x + ½y`. -/
theorem barycenter_halfDirac {d : ℕ} (x y : SSphere d) :
    barycenter d (halfDirac d x y) = (2⁻¹ : ℝ) • (x : EucSpace d) + (2⁻¹ : ℝ) • (y : EucSpace d) := by
  rw [barycenter, coe_halfDirac, integral_add_measure, integral_smul_measure,
    integral_smul_measure, integral_dirac, integral_dirac]
  · simp
  · exact (integrable_dirac (by simp)).smul_measure (by simp)
  · exact (integrable_dirac (by simp)).smul_measure (by simp)

/-- The hypotheses of `perturbation` are satisfiable, with two different
measures: on `𝕊^1`, `μ_0 = ½δ_{(5,12)/13} + ½δ_{(12,5)/13}` and
`ν_0 = ½δ_{(3,4)/5} + ½δ_{(4,3)/5}`, supported in `ℚ_1^1`, whose barycenters
`(17/26)(1, 1)` and `(7/10)(1, 1)` are collinear with `γ_1 = 85/91 ∈ (0, 1]`.
In `d = 1` no witness exists, since `𝒫(ℚ_1^0) = {δ_{+1}}`. -/
example : ∃ (μ₀ ν₀ : ProbSphere 2) (γ₁ : ℝ),
    (0 : ℝ) < 1 ∧ μ₀ ≠ ν₀ ∧
    (μ₀ : Measure (SSphere 2)).support ⊆ positiveQuadrant 2 ∧
    (ν₀ : Measure (SSphere 2)).support ⊆ positiveQuadrant 2 ∧
    0 < γ₁ ∧ γ₁ ≤ 1 ∧ barycenter 2 μ₀ = γ₁ • barycenter 2 ν₀ := by
  have h1 : (5 / 13 : ℝ) ^ 2 + (12 / 13) ^ 2 = 1 := by norm_num
  have h2 : (12 / 13 : ℝ) ^ 2 + (5 / 13) ^ 2 = 1 := by norm_num
  have h3 : (3 / 5 : ℝ) ^ 2 + (4 / 5) ^ 2 = 1 := by norm_num
  have h4 : (4 / 5 : ℝ) ^ 2 + (3 / 5) ^ 2 = 1 := by norm_num
  have hne : ∀ {p q p' q' : ℝ} (h : p ^ 2 + q ^ 2 = 1) (h' : p' ^ 2 + q' ^ 2 = 1),
      p ≠ p' → circlePt p q h ≠ circlePt p' q' h' := fun h h' hp he =>
    hp (by simpa [circlePt] using congrArg (fun z : SSphere 2 => (z : EucSpace 2) 0) he)
  have hQ : ∀ {x y : SSphere 2}, x ∈ positiveQuadrant 2 → y ∈ positiveQuadrant 2 →
      (halfDirac 2 x y : Measure (SSphere 2)).support ⊆ positiveQuadrant 2 :=
    fun hx hy z hz => by
      rcases mem_of_mem_support_halfDirac hz with rfl | rfl
      exacts [hx, hy]
  refine ⟨halfDirac 2 (circlePt _ _ h1) (circlePt _ _ h2), halfDirac 2 (circlePt _ _ h3) (circlePt _ _ h4),
    85 / 91, one_pos, fun h => ?_,
    hQ (circlePt_mem_positiveQuadrant h1 (by norm_num) (by norm_num))
      (circlePt_mem_positiveQuadrant h2 (by norm_num) (by norm_num)),
    hQ (circlePt_mem_positiveQuadrant h3 (by norm_num) (by norm_num))
      (circlePt_mem_positiveQuadrant h4 (by norm_num) (by norm_num)),
    by norm_num, by norm_num, ?_⟩
  · have := congrArg (fun m : ProbSphere 2 => (m : Measure (SSphere 2)) {circlePt _ _ h3}) h
    simp [Pi.single_eq_of_ne (hne h1 h3 (by norm_num)),
      Pi.single_eq_of_ne (hne h2 h3 (by norm_num))] at this
    exact absurd (add_eq_zero.mp this.symm).1 (by simp)
  · rw [barycenter_halfDirac, barycenter_halfDirac]
    ext i
    fin_cases i <;> simp [circlePt] <;> norm_num

end Interpolation
end Transformer
