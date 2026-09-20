/-
# §9.1 — Sharp configurations and the Cohn–Kumar dichotomy

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*, §9.1.

The survey reads the minimization of `𝖧_β` over `n`-point configurations of
the sphere through Cohn and Kumar's classification of universally optimal
configurations, and quotes the dichotomy: a global minimiser is a *sharp
configuration*, or it is the vertices of the 600-cell.

As the survey prints it the dichotomy is false, and `not_cohn_kumar_dichotomy`
refutes it.  What kills it is the clause `m > 1` of the printed definition of a
sharp configuration: an antipodal pair `{x, -x}` has one inner product between
distinct points, so it is excluded by that clause, and it is a global minimiser
of `𝖧_β` among two-point configurations for every `β ≥ 0`.  In dimension `2`
there are infinitely many antipodal pairs and only one `exceptional`
configuration to except, so no choice of the exception repairs it.
-/

import Transformer.Perspective.Section8_General
import Transformer.Perspective.Section3_SmallBeta
import Mathlib.Algebra.MvPolynomial.Degrees
import Mathlib.Data.Set.Card

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Perspective

variable (d : ℕ)

/-- A finite point set `𝒞 ⊂ 𝕊^{d-1}` is a *spherical `t`-design* for the
reference measure `σ` if

  `(1/#𝒞) Σ_{x ∈ 𝒞} p(x) = ∫ p dσ`

for every polynomial `p` of total degree `≤ t`.  The survey takes `σ` to be the
uniform measure on the sphere; this development does not fix a normalized
surface measure on `SSphere d`, so it is carried as a parameter.
Source: arXiv:2312.10794v5, §9.1. -/
def sphericalDesign
    (σ : Measure (SSphere d)) (t : ℕ) (𝒞 : Finset (SSphere d)) : Prop :=
  ∀ p : MvPolynomial (Fin d) ℝ, p.totalDegree ≤ t →
    ((𝒞.card : ℝ))⁻¹ * ∑ x ∈ 𝒞, MvPolynomial.eval (fun i => (x : EucSpace d) i) p
      = ∫ x, MvPolynomial.eval (fun i => (x : EucSpace d) i) p ∂σ

/-- A finite point set `𝒞 ⊂ 𝕊^{d-1}` is a *sharp configuration* if, for some
`m > 1`, there are `m` distinct inner products between pairwise distinct points
of `𝒞` and `𝒞` is a spherical `(2m-1)`-design.

**What the source says and what is changed here.**  The inner products are
counted between *pairwise distinct* points, which is what the survey's sentence
says; the earlier rendering took them over all of `𝒞 ×ˢ 𝒞`, adding the
diagonal value `⟨x, x⟩ = 1` and so inflating the count by one on every
configuration.  The clause `m > 1` is the survey's and is kept, so that
`not_cohn_kumar_dichotomy` refutes what is printed.
Source: arXiv:2312.10794v5, §9.1. -/
def sharpConfiguration (σ : Measure (SSphere d)) (𝒞 : Finset (SSphere d)) : Prop :=
  ∃ m : ℕ, 1 < m ∧
    { r : ℝ | ∃ x ∈ 𝒞, ∃ y ∈ 𝒞, x ≠ y ∧
        inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d) = r }.ncard = m ∧
    sphericalDesign d σ (2 * m - 1) 𝒞

/-- `⟨x, -x⟩ = -1` on the sphere, for `Perspective.antipode` of
`Transformer.Perspective.Section3_SmallBeta`.  Source: arXiv:2312.10794v5, §9.1. -/
theorem inner_antipode (x : SSphere d) :
    inner (𝕜 := ℝ) ((x : EucSpace d)) ((antipode d x : SSphere d) : EucSpace d) = -1 := by
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  show inner (𝕜 := ℝ) ((x : EucSpace d)) (-(x : EucSpace d)) = -1
  rw [inner_neg_right, real_inner_self_eq_norm_mul_norm, hx]
  norm_num

/-- A point of the sphere is not its own antipode, so `{x, -x}` really is a
two-point configuration.  Source: arXiv:2312.10794v5, §9.1. -/
theorem ne_antipode (x : SSphere d) : x ≠ antipode d x := by
  intro h
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  have h1 : inner (𝕜 := ℝ) ((x : EucSpace d)) ((x : EucSpace d)) = 1 := by
    rw [real_inner_self_eq_norm_mul_norm, hx]; ring
  have h2 := inner_antipode d x
  rw [← h] at h2
  rw [h1] at h2
  norm_num at h2

/-- The energy of a two-point configuration:

  `𝖧_β({u, v}) = 2 e^β + 2 e^{β ⟨u, v⟩}`.

Source: arXiv:2312.10794v5, §9.1. -/
theorem discreteEnergy_pair (β : ℝ) (u v : SSphere d) (huv : u ≠ v) :
    discreteEnergy d β {u, v}
      = 2 * Real.exp β
        + 2 * Real.exp (β * inner (𝕜 := ℝ) ((u : EucSpace d)) ((v : EucSpace d))) := by
  have hu : ‖(u : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp u.2
  have hv : ‖(v : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp v.2
  have huu : inner (𝕜 := ℝ) ((u : EucSpace d)) ((u : EucSpace d)) = 1 := by
    rw [real_inner_self_eq_norm_mul_norm, hu]; ring
  have hvv : inner (𝕜 := ℝ) ((v : EucSpace d)) ((v : EucSpace d)) = 1 := by
    rw [real_inner_self_eq_norm_mul_norm, hv]; ring
  rw [discreteEnergy, Finset.sum_pair huv, Finset.sum_pair huv, Finset.sum_pair huv,
    huu, hvv, real_inner_comm ((v : EucSpace d)) ((u : EucSpace d)), mul_one]
  ring

/-- The hypothesis of `discreteEnergy_pair` is satisfiable: a point of the
sphere and its antipode are distinct. -/
example : (basePoint 0 : SSphere 1) ≠ antipode 1 (basePoint 0) := ne_antipode 1 _

/-- **An antipodal pair minimises `𝖧_β` among two-point configurations**, for
every `β ≥ 0`: the energy of `{u, v}` is `2 e^β + 2 e^{β ⟨u, v⟩}` and
`⟨u, v⟩ ≥ -1`, with equality exactly for an antipodal pair.
Source: arXiv:2312.10794v5, §9.1. -/
theorem discreteEnergy_antipodal_min (β : ℝ) (hβ : 0 ≤ β) (x : SSphere d) :
    ∀ 𝒟 : Finset (SSphere d), 𝒟.card = 2 →
      discreteEnergy d β {x, antipode d x} ≤ discreteEnergy d β 𝒟 := by
  intro 𝒟 h𝒟
  obtain ⟨u, v, huv, rfl⟩ := Finset.card_eq_two.mp h𝒟
  rw [discreteEnergy_pair d β x (antipode d x) (ne_antipode d x),
    discreteEnergy_pair d β u v huv, inner_antipode]
  have hb : |inner (𝕜 := ℝ) ((u : EucSpace d)) ((v : EucSpace d))| ≤ 1 := by
    have h := abs_real_inner_le_norm (u : EucSpace d) (v : EucSpace d)
    rw [mem_sphere_zero_iff_norm.mp u.2, mem_sphere_zero_iff_norm.mp v.2, mul_one] at h
    exact h
  have hge : -(1 : ℝ) ≤ inner (𝕜 := ℝ) ((u : EucSpace d)) ((v : EucSpace d)) :=
    neg_le_of_abs_le hb
  have harg : β * (-1) ≤ β * inner (𝕜 := ℝ) ((u : EucSpace d)) ((v : EucSpace d)) := by
    nlinarith
  have := Real.exp_le_exp.mpr harg
  linarith

/-- The hypothesis `0 ≤ β` of `discreteEnergy_antipodal_min` is satisfiable. -/
example : (0 : ℝ) ≤ 0 := le_rfl

/-- **An antipodal pair is not a sharp configuration** in the printed sense:
its only inner product between distinct points is `-1`, so the count is `1`,
and the clause `m > 1` fails.  Source: arXiv:2312.10794v5, §9.1. -/
theorem not_sharpConfiguration_antipodal (σ : Measure (SSphere d)) (x : SSphere d) :
    ¬ sharpConfiguration d σ {x, antipode d x} := by
  rintro ⟨m, hm, hncard, -⟩
  have hsub : { r : ℝ | ∃ u ∈ ({x, antipode d x} : Finset (SSphere d)),
      ∃ v ∈ ({x, antipode d x} : Finset (SSphere d)), u ≠ v ∧
        inner (𝕜 := ℝ) (u : EucSpace d) (v : EucSpace d) = r } ⊆ {(-1 : ℝ)} := by
    rintro r ⟨u, hu, v, hv, huv, rfl⟩
    simp only [Finset.mem_insert, Finset.mem_singleton] at hu hv
    rcases hu with rfl | rfl <;> rcases hv with rfl | rfl
    · exact absurd rfl huv
    · simpa using inner_antipode d u
    · simpa [real_inner_comm] using inner_antipode d v
    · exact absurd rfl huv
  have hle := Set.ncard_le_ncard hsub (Set.finite_singleton _)
  rw [hncard, Set.ncard_singleton] at hle
  omega

/-- Two orthogonal unit vectors of `ℝ²`, hence two antipodal pairs of `𝕊^1`
that share no point.  Source: arXiv:2312.10794v5, §9.1. -/
theorem exists_orthogonal_pair :
    ∃ x y : SSphere 2, inner (𝕜 := ℝ) ((x : EucSpace 2)) ((y : EucSpace 2)) = 0 := by
  refine ⟨⟨EuclideanSpace.single (0 : Fin 2) (1 : ℝ), ?_⟩,
    ⟨EuclideanSpace.single (1 : Fin 2) (1 : ℝ), ?_⟩, ?_⟩
  · rw [mem_sphere_zero_iff_norm]; simp
  · rw [mem_sphere_zero_iff_norm]; simp
  · simp [EuclideanSpace.inner_single_left]

/-- **The Cohn–Kumar dichotomy is false as the survey states it.**

The survey quotes: *let `n ≥ 2`; any global minimum of `𝖧_β` among `𝒞 ⊂ 𝕊`
with `#𝒞 = n` is either a sharp configuration, or the vertices of a 600-cell*,
a sharp configuration being one with `m > 1` distinct inner products between
distinct points that is a spherical `(2m-1)`-design.

At `n = 2` on `𝕊^1` that is false, for every reference measure `σ` and every
`β ≥ 0`, and it stays false however the exceptional configuration is chosen:
an antipodal pair `{x, -x}` is a global minimiser of `𝖧_β` among two-point
configurations (`discreteEnergy_antipodal_min`), it has exactly one inner
product between distinct points, namely `-1`, so the clause `m > 1` excludes it
(`not_sharpConfiguration_antipodal`), and two orthogonal directions give two
antipodal pairs that cannot both be the one excepted configuration.

The clause `m > 1` is what fails: the antipodal pair at `n = 2` and the regular
simplex at `n = d + 1` have a single inner product between distinct points, and
they are exactly the configurations Cohn and Kumar's theorem is quoted for.
Nothing here is restated with the clause removed: the number `m` and the design
order `2m - 1` would then have to be read off a source this development does
not have, and a statement nobody can check against its source is not one of
this repository's.
Source: arXiv:2312.10794v5, §9.1, the theorem quoted from `cohn2007universally`. -/
theorem not_cohn_kumar_dichotomy (σ : Measure (SSphere 2)) (β : ℝ) (hβ : 0 ≤ β) :
    ¬ ∃ exceptional : Finset (SSphere 2),
        ∀ 𝒞 : Finset (SSphere 2), 𝒞.card = 2 →
          (∀ 𝒟 : Finset (SSphere 2), 𝒟.card = 2 →
            discreteEnergy 2 β 𝒞 ≤ discreteEnergy 2 β 𝒟) →
          sharpConfiguration 2 σ 𝒞 ∨ 𝒞 = exceptional := by
  rintro ⟨exceptional, h⟩
  obtain ⟨x, y, hortho⟩ := exists_orthogonal_pair
  have hxx : inner (𝕜 := ℝ) ((x : EucSpace 2)) ((x : EucSpace 2)) = 1 := by
    rw [real_inner_self_eq_norm_mul_norm, mem_sphere_zero_iff_norm.mp x.2]; ring
  rcases h _ (Finset.card_pair (ne_antipode 2 x))
    (discreteEnergy_antipodal_min 2 β hβ x) with hsx | hex
  · exact not_sharpConfiguration_antipodal 2 σ x hsx
  rcases h _ (Finset.card_pair (ne_antipode 2 y))
    (discreteEnergy_antipodal_min 2 β hβ y) with hsy | hey
  · exact not_sharpConfiguration_antipodal 2 σ y hsy
  have hmem : y ∈ ({x, antipode 2 x} : Finset (SSphere 2)) := by
    rw [hex, ← hey]; simp
  rcases Finset.mem_insert.mp hmem with hyx | hyx
  · rw [hyx, hxx] at hortho; norm_num at hortho
  · rw [Finset.mem_singleton] at hyx
    rw [hyx, inner_antipode] at hortho; norm_num at hortho

/-- The hypothesis `0 ≤ β` of `not_cohn_kumar_dichotomy` is satisfiable. -/
example : (0 : ℝ) ≤ 0 := le_rfl

end Perspective
end Transformer
