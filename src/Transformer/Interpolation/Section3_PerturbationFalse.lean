/-
# Part 1 of the perturbation lemma is false for atomic measures

Counterexample to arXiv:2411.04551v3, §3, `lem: perturbation` / `lem: colinearity`,
Part 1 and `eq: identity.flow`. Six rational points in the positive quadrant
of `S²` lie in `4x₀ = 3x₁`. Two different laws on them have the same
barycenter `(51/130, 34/65, 17/26)`. Any continuous map fixing points outside
their geodesic hulls is the identity, so its two pushforwards still have
equal barycenters. In particular, the asserted Lipschitz flow cannot exist.

The source's proof begins by choosing an open ball contained in the union
of the supports. These finite supports contain no such ball. Part 2, whose
barycenters differ, remains the open theorem `perturbation_noncolinear`.
-/

import Transformer.Interpolation.Section3_PerturbationGeometry

open scoped BigOperators
open Real MeasureTheory

namespace Transformer.Interpolation

open Perspective

/-- Lift a unit circle point into the plane `4x₀ = 3x₁` in `S²`.
Source: arXiv:2411.04551v3, §3, counterexample to `lem: perturbation`, Part 1. -/
noncomputable def liftedCirclePoint (p q : ℝ) (h : p ^ 2 + q ^ 2 = 1) : SSphere 3 :=
  ⟨!₂[3 * p / 5, 4 * p / 5, q], by
    apply mem_sphere_zero_iff_norm.mpr
    have hs : ‖(!₂[3 * p / 5, 4 * p / 5, q] : EucSpace 3)‖ ^ 2 = 1 := by
      rw [EuclideanSpace.real_norm_sq_eq, Fin.sum_univ_three]
      simp
      nlinarith [h]
    nlinarith [norm_nonneg (!₂[3 * p / 5, 4 * p / 5, q] : EucSpace 3)]⟩

/-- The lifts of three symmetric circle pairs: `(5,12)/13`, `(3,4)/5`,
and `(7,24)/25`, together with their coordinate swaps.
Source: arXiv:2411.04551v3, §3, counterexample to `lem: perturbation`. -/
noncomputable def perturbationAtom (i : Fin 6) : SSphere 3 :=
  liftedCirclePoint
    (![5 / 13, 12 / 13, 3 / 5, 4 / 5, 7 / 25, 24 / 25] i)
    (![12 / 13, 5 / 13, 4 / 5, 3 / 5, 24 / 25, 7 / 25] i)
    (by fin_cases i <;> norm_num)

/-- The even law on the first pair, with mean circle coordinates `17/26`.
Source: arXiv:2411.04551v3, §3, counterexample to `lem: perturbation`. -/
noncomputable def perturbationMu : ProbSphere 3 :=
  halfDirac 3 (perturbationAtom 0) (perturbationAtom 1)

/-- The mixture of the other two even laws with weights `11/26` and `15/26`.
Their mean circle coordinates are `7/10` and `31/50`, respectively, and
`(11/26)(7/10) + (15/26)(31/50) = 17/26`.
Source: arXiv:2411.04551v3, §3, counterexample to `lem: perturbation`. -/
noncomputable def perturbationNu : ProbSphere 3 :=
  ⟨(11 / 26 : ENNReal) • (halfDirac 3 (perturbationAtom 2) (perturbationAtom 3) :
      Measure (SSphere 3)) +
    (15 / 26 : ENNReal) • (halfDirac 3 (perturbationAtom 4) (perturbationAtom 5) :
      Measure (SSphere 3)), ⟨by
        simp only [Measure.add_apply, Measure.smul_apply, smul_eq_mul, measure_univ, mul_one]
        rw [← ENNReal.add_div]
        norm_num
        exact ENNReal.div_self (by norm_num) (by simp)⟩⟩

/-- Every atom satisfies the source's strict positive-quadrant hypothesis.
Source: arXiv:2411.04551v3, §3, `lem: perturbation`. -/
theorem perturbationAtom_positive (i : Fin 6) :
    perturbationAtom i ∈ positiveQuadrant 3 := by
  intro j
  fin_cases i <;> fin_cases j <;> norm_num [perturbationAtom, liftedCirclePoint]

/-- The first law is supported on these six points.
Source: arXiv:2411.04551v3, §3, `lem: perturbation`. -/
theorem perturbationMu_support :
    (perturbationMu : Measure (SSphere 3)).support ⊆ Set.range perturbationAtom := by
  intro x hx
  rcases mem_of_mem_support_halfDirac hx with rfl | rfl
  exacts [⟨0, rfl⟩, ⟨1, rfl⟩]

/-- The other law has an open null neighborhood outside these six points.
Source: arXiv:2411.04551v3, §3, `lem: perturbation`. -/
theorem perturbationNu_support :
    (perturbationNu : Measure (SSphere 3)).support ⊆ Set.range perturbationAtom := by
  intro x hx
  by_contra hne
  refine Measure.notMem_support_iff_exists.mpr ⟨(Set.range perturbationAtom)ᶜ, ?_, ?_⟩ hx
  · exact (Set.finite_range perturbationAtom).isClosed.isOpen_compl.mem_nhds hne
  · simp [perturbationNu, Set.mem_range]

/-- The laws differ on the singleton containing atom 2: its masses are
zero and `11/52`. Source: arXiv:2411.04551v3, §3, `lem: perturbation`. -/
theorem perturbationMu_ne_nu : perturbationMu ≠ perturbationNu := by
  have hne : ∀ i : Fin 6, i ≠ 2 → perturbationAtom i ≠ perturbationAtom 2 := by
    intro i hi he
    have he0 := congrArg (fun x : SSphere 3 => (x : EucSpace 3) 0) he
    fin_cases i <;> norm_num [perturbationAtom, liftedCirclePoint] at he0
    exact hi rfl
  intro h
  have he := congrArg (fun m : ProbSphere 3 => (m : Measure (SSphere 3))
    {perturbationAtom 2}) h
  simp [perturbationMu, perturbationNu,
    Pi.single_eq_of_ne (hne 0 (by decide)),
    Pi.single_eq_of_ne (hne 1 (by decide)),
    Pi.single_eq_of_ne (hne 3 (by decide)),
    Pi.single_eq_of_ne (hne 4 (by decide)),
    Pi.single_eq_of_ne (hne 5 (by decide))] at he

/-- Their actual Bochner barycenters agree, so the source's `γ₁ = 1` case
applies. Source: arXiv:2411.04551v3, §3, `lem: perturbation`. -/
theorem perturbationMu_bary_eq_nu :
    barycenter 3 perturbationMu = barycenter 3 perturbationNu := by
  have hi (i j : Fin 6) : Integrable (fun x : SSphere 3 => (x : EucSpace 3))
      (halfDirac 3 (perturbationAtom i) (perturbationAtom j) : Measure (SSphere 3)) := by
    rw [coe_halfDirac]
    exact ((integrable_dirac (by simp)).smul_measure (by simp)).add_measure
      ((integrable_dirac (by simp)).smul_measure (by simp))
  have hb : barycenter 3 perturbationNu = (11 / 26 : ℝ) •
      barycenter 3 (halfDirac 3 (perturbationAtom 2) (perturbationAtom 3)) +
      (15 / 26 : ℝ) •
      barycenter 3 (halfDirac 3 (perturbationAtom 4) (perturbationAtom 5)) := by
    change (∫ x : SSphere 3, (x : EucSpace 3) ∂
      ((11 / 26 : ENNReal) •
        (halfDirac 3 (perturbationAtom 2) (perturbationAtom 3) : Measure (SSphere 3)) +
       (15 / 26 : ENNReal) •
        (halfDirac 3 (perturbationAtom 4) (perturbationAtom 5) : Measure (SSphere 3)))) = _
    rw [integral_add_measure ((hi 2 3).smul_measure (c := (11 / 26 : ENNReal))
      (ENNReal.div_ne_top (by simp) (by norm_num)))
      ((hi 4 5).smul_measure (c := (15 / 26 : ENNReal))
      (ENNReal.div_ne_top (by simp) (by norm_num))),
      integral_smul_measure, integral_smul_measure]
    norm_num [ENNReal.toReal_div, barycenter]
  rw [hb]
  change barycenter 3 (halfDirac 3 (perturbationAtom 0) (perturbationAtom 1)) = _
  rw [barycenter_halfDirac, barycenter_halfDirac, barycenter_halfDirac]
  ext i
  fin_cases i <;> norm_num [perturbationAtom, liftedCirclePoint]

/-- **Part 1 of `lem: perturbation` is false.** These distinct probabilities
are supported in `ℚ₁²` and have equal barycenters. No continuous map fixing
the complement of their geodesic hulls can separate the barycenters of its
pushforwards. This rules out the source's Lipschitz invertible terminal
flow for every `T > 0`, including `T = 1`, with `γ₁ = 1`.

The pushforward integrals are exactly the terminal barycenters in the
characteristic representation used in the source's proof. No restriction
to atomic test functions or unproved solution input is made here.
Source: arXiv:2411.04551v3, §3, `lem: perturbation`, `eq: identity.flow`. -/
theorem perturbation_equal_barycenter_counterexample :
    ∃ μ₀ ν₀ : ProbSphere 3, μ₀ ≠ ν₀ ∧
      (μ₀ : Measure (SSphere 3)).support ⊆ positiveQuadrant 3 ∧
      (ν₀ : Measure (SSphere 3)).support ⊆ positiveQuadrant 3 ∧
      barycenter 3 μ₀ = barycenter 3 ν₀ ∧
      ¬ ∃ Φ : SSphere 3 → SSphere 3, Continuous Φ ∧
        (∀ x, x ∉ convG 3 (μ₀ : Measure (SSphere 3)).support ∪
          convG 3 (ν₀ : Measure (SSphere 3)).support → Φ x = x) ∧
        (∫ x, (x : EucSpace 3) ∂Measure.map Φ (μ₀ : Measure (SSphere 3))) ≠
          ∫ x, (x : EucSpace 3) ∂Measure.map Φ (ν₀ : Measure (SSphere 3)) := by
  have hQ : Set.range perturbationAtom ⊆ positiveQuadrant 3 := by
    rintro _ ⟨i, rfl⟩
    exact perturbationAtom_positive i
  have hp : ∀ x ∈ Set.range perturbationAtom, (x : EucSpace 3) ∈ perturbationPlane := by
    rintro _ ⟨i, rfl⟩
    fin_cases i <;> norm_num [perturbationPlane, perturbationAtom, liftedCirclePoint]
  refine ⟨perturbationMu, perturbationNu, perturbationMu_ne_nu,
    perturbationMu_support.trans hQ, perturbationNu_support.trans hQ,
    perturbationMu_bary_eq_nu, ?_⟩
  rintro ⟨Φ, hΦ, hfix, hne⟩
  have hid := eq_id_of_fix_outside_plane_hulls _ _
    (fun x hx => hp x (perturbationMu_support hx))
    (fun x hx => hp x (perturbationNu_support hx))
    (fun _ hx => hQ (perturbationMu_support hx) 0)
    (fun _ hx => hQ (perturbationNu_support hx) 0) Φ hΦ hfix
  subst Φ
  simp only [Measure.map_id] at hne
  exact hne perturbationMu_bary_eq_nu

/-- The original Part 1 hypotheses hold with `T = γ₁ = 1`; the measures are
different and have genuine positive-quadrant supports. -/
example : ∃ (T γ₁ : ℝ) (μ₀ ν₀ : ProbSphere 3), 0 < T ∧ 0 < γ₁ ∧ γ₁ ≤ 1 ∧ γ₁ = 1 ∧
    μ₀ ≠ ν₀ ∧ (μ₀ : Measure (SSphere 3)).support ⊆ positiveQuadrant 3 ∧
    (ν₀ : Measure (SSphere 3)).support ⊆ positiveQuadrant 3 ∧
    barycenter 3 μ₀ = γ₁ • barycenter 3 ν₀ := by
  rcases perturbation_equal_barycenter_counterexample with ⟨μ₀, ν₀, hne, hμ, hν, hb, _⟩
  exact ⟨1, 1, μ₀, ν₀, one_pos, one_pos, le_rfl, rfl, hne, hμ, hν, by simpa using hb⟩

end Transformer.Interpolation
