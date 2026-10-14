/-
# Kinetic theory for Transformers — the original U-shape theorem

Formalization of arXiv:2605.09213v1, §6, proof of `thm:U-shape`.
The original Fourier coefficients give an entire aggregate with strictly
positive curvature, a strictly negative initial slope and divergence at
infinity. Its actual unique positive minimum is transferred back through
the actual inverse of `Y` to the original soft-correction sum on `(0,1)`.

`Section6_ModeCurvature` proves the original coefficient sign criterion.
`Section6_WeightedProfiles` justifies the actual double series and sums
the original endpoint slopes. `Section6_InteriorMinimum` gives compact
attainment and uniqueness, and `Section6_SourcePosition` proves both
inverse identities on the original source-position domain.
-/

import Transformer.Kinetic.Section6_SoftLimits

open Set Filter Real
open scoped Topology

namespace Transformer
namespace Kinetic

/-- The actual unique minimum of the original Fourier aggregate transfers
to exactly one interior minimum of the original soft correction.
Source: arXiv:2605.09213v1, §6, the final bijection step in `thm:U-shape` (iii). -/
theorem softCorrection_unique_minimum {β lam M t : ℝ} (hβ : 0 < β) (hlam : 0 < lam)
    (hM : 0 < M) (ht : 0 < t)
    (hcurv : ∀ n, outputModeParameter β t n ≤ 3 - Real.sqrt 3)
    (hslope : ∀ n, outputModeParameter β t n ≤ 2 * (1 - Real.exp (-lam))) :
    ∃! s : ℝ, s ∈ Ioo (0 : ℝ) 1 ∧
      ∀ z ∈ Ioo (0 : ℝ) 1, softCorrection β lam M t s ≤ softCorrection β lam M t z := by
  obtain ⟨y, hy, huniq⟩ := softProfile_unique_minimum hβ hlam hM ht hcurv hslope
  let s := sourcePosition lam y
  have hs : s ∈ Ioo (0 : ℝ) 1 := ⟨sourcePosition_pos hlam y, sourcePosition_lt_one hlam hy.1⟩
  have hYs : Yfun lam 1 s = y := Yfun_one_sourcePosition hlam y
  refine ⟨s, ⟨hs, ?_⟩, ?_⟩
  · intro z hz
    rw [softCorrection_eq_softProfile hβ hlam hM ht hcurv hs.1 hs.2, hYs,
      softCorrection_eq_softProfile hβ hlam hM ht hcurv hz.1 hz.2]
    exact hy.2 (Yfun lam 1 z) (Yfun_one_pos hlam hz.1 hz.2)
  · intro z hz
    have hYz : 0 < Yfun lam 1 z := Yfun_one_pos hlam hz.1.1 hz.1.2
    have hmin : ∀ q : ℝ, 0 < q →
        softProfile β lam M t (Yfun lam 1 z) ≤ softProfile β lam M t q := by
      intro q hq
      have hq0 := sourcePosition_pos hlam q
      have hq1 := sourcePosition_lt_one hlam hq
      have hle := hz.2 (sourcePosition lam q) ⟨hq0, hq1⟩
      rwa [softCorrection_eq_softProfile hβ hlam hM ht hcurv hz.1.1 hz.1.2,
        softCorrection_eq_softProfile hβ hlam hM ht hcurv hq0 hq1,
        Yfun_one_sourcePosition hlam] at hle
    have heq := huniq (Yfun lam 1 z) ⟨hYz, hmin⟩
    calc z = sourcePosition lam (Yfun lam 1 z) := (sourcePosition_Yfun_one hlam hz.1.1).symm
      _ = s := congrArg (sourcePosition lam) heq

/-- **Theorem (thm:U-shape), Lost in the middle.** Let `λ > 0` and assume
`eq:affine-smallness`,

  `t sup_{n≥1} a_n ≤ min{3 - √3, 2(1 - e^{-λ})}`.

Then (i) *primacy*: `𝒮_t(σ₀) → +∞` as `σ₀ ↓ 0`; (ii) *recency*:
`𝒮_t'(1⁻) > 0`; (iii) *U-shape*: the original `𝒮_t` has a unique global
minimum in `(0,1)`.

**What the source says and what is changed here.** The manuscript's
standing assumptions `β > 0` and `t > 0` are explicit: at `β = 0` or
`t = 0` the actual correction vanishes, so (i) and (ii) fail. The source's
smallness footnote itself assumes positive time. The supremum is written
pointwise for all `n ≥ 1`; at nonnegative time this is the same bound,
using the proved boundedness from `aCoeff_tendsto_zero`.

The recency slope is the left limit of the actual derivative on `(0,1)`.
The printed Bessel formula is stated only for `σ₀ < σ`; its totalized
diagonal value is zero, while the actual positive series extension has
a positive left limit (`softCorrection_output_limit`, `softProfile_output_pos`).
An endpoint derivative at that totalized value cannot be positive
(`not_positive_endpoint_derivative`).

The source's non-strict slope bound allows zero slope in an individual
mode. The aggregate slope is nevertheless strictly negative: the actual
positive eigenvalues tend to zero, so some original mode satisfies the
strict bound. All infinite sums and derivatives are justified by the
proved entire coefficient aggregate. A positive linear coefficient proves
divergence directly, and the actual inverse `sourcePosition` transfers
the proved unique aggregate minimum to the original Fourier sum.

Source: arXiv:2605.09213v1, §1, `thm:U-shape`, `eq:affine-smallness`, and its proof in §6. -/
theorem u_shape (β lam M t : ℝ) (hβ : 0 < β) (hlam : 0 < lam) (hM : 0 < M)
    (ht : 0 < t)
    (hsmall : ∀ n : ℕ, 1 ≤ n →
      t * aCoeff β n ≤ min (3 - Real.sqrt 3) (2 * (1 - Real.exp (-lam)))) :
    Filter.Tendsto (fun σ₀ => softCorrection β lam M t σ₀)
        (nhdsWithin 0 (Set.Ioi 0)) Filter.atTop ∧
      (∃ D : ℝ, 0 < D ∧
        Filter.Tendsto (deriv (fun σ₀ => softCorrection β lam M t σ₀))
          (nhdsWithin 1 (Set.Iio 1)) (nhds D)) ∧
      (∃! s : ℝ, s ∈ Set.Ioo (0 : ℝ) 1 ∧
        ∀ σ₀ ∈ Set.Ioo (0 : ℝ) 1, softCorrection β lam M t s ≤ softCorrection β lam M t σ₀) := by
  have hcurv : ∀ n, outputModeParameter β t n ≤ 3 - Real.sqrt 3 :=
    fun n => (hsmall (n + 1) (by omega)).trans (min_le_left _ _)
  have hslope : ∀ n, outputModeParameter β t n ≤ 2 * (1 - Real.exp (-lam)) :=
    fun n => (hsmall (n + 1) (by omega)).trans (min_le_right _ _)
  exact ⟨softCorrection_primacy hβ hlam hM ht hcurv,
    softCorrection_recency hβ hlam hM ht hcurv hslope,
    softCorrection_unique_minimum hβ hlam hM ht hcurv hslope⟩

/-- The original U-shaped correction is strictly larger at every other
interior position than at its actual unique minimum.
Source: arXiv:2605.09213v1, §1, `thm:U-shape` (iii), and §6, strict convexity of `ℋ`. -/
theorem softCorrection_strict_minimum (β lam M t : ℝ) (hβ : 0 < β) (hlam : 0 < lam)
    (hM : 0 < M) (ht : 0 < t)
    (hsmall : ∀ n : ℕ, 1 ≤ n →
      t * aCoeff β n ≤ min (3 - Real.sqrt 3) (2 * (1 - Real.exp (-lam)))) :
    ∃ s ∈ Ioo (0 : ℝ) 1, ∀ z ∈ Ioo (0 : ℝ) 1, z ≠ s →
      softCorrection β lam M t s < softCorrection β lam M t z := by
  obtain ⟨s, hs, huniq⟩ := (u_shape β lam M t hβ hlam hM ht hsmall).2.2
  refine ⟨s, hs.1, fun z hz hne => lt_of_le_of_ne (hs.2 z hz) ?_⟩
  intro heq
  have hzmin : z ∈ Ioo (0 : ℝ) 1 ∧ ∀ q ∈ Ioo (0 : ℝ) 1,
      softCorrection β lam M t z ≤ softCorrection β lam M t q := by
    refine ⟨hz, fun q hq => ?_⟩
    rw [← heq]
    exact hs.2 q hq
  exact hne (huniq z hzmin)

/-- The actual minimum value of the original correction is positive.
Source: arXiv:2605.09213v1, §1, `eq:Sb-def`, `thm:U-shape` (iii), and §6, positive modes. -/
theorem softCorrection_positive_minimum (β lam M t : ℝ) (hβ : 0 < β) (hlam : 0 < lam)
    (hM : 0 < M) (ht : 0 < t)
    (hsmall : ∀ n : ℕ, 1 ≤ n →
      t * aCoeff β n ≤ min (3 - Real.sqrt 3) (2 * (1 - Real.exp (-lam)))) :
    ∃ s ∈ Ioo (0 : ℝ) 1, 0 < softCorrection β lam M t s ∧
      ∀ z ∈ Ioo (0 : ℝ) 1, softCorrection β lam M t s ≤ softCorrection β lam M t z := by
  obtain ⟨s, hs, -⟩ := (u_shape β lam M t hβ hlam hM ht hsmall).2.2
  have hcurv : ∀ n, outputModeParameter β t n ≤ 3 - Real.sqrt 3 :=
    fun n => (hsmall (n + 1) (by omega)).trans (min_le_left _ _)
  exact ⟨s, hs.1, softCorrection_pos hβ hlam hM ht hcurv hs.1.1 hs.1.2, hs.2⟩

/-- The hypotheses of the original `u_shape` and the minimum-transfer
theorem, including both minimum consequences, are satisfiable at the
original positive unit parameters, with
an actual positive time meeting both printed bounds.
Source: arXiv:2605.09213v1, §1, `eq:affine-smallness`. -/
example : (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ ∃ t : ℝ, 0 < t ∧ ∀ n : ℕ, 1 ≤ n →
    t * aCoeff 1 n ≤ min (3 - Real.sqrt 3) (2 * (1 - Real.exp (-(1 : ℝ)))) :=
  ⟨one_pos, one_pos, one_pos, exists_affine_smallness 1 1 one_pos⟩

end Kinetic
end Transformer
