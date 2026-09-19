/-
# Appendix A — Proof of Theorem (p:beta0), part 1

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

This file formalizes the first half of Appendix A of the survey:

* `eq: E0`      — the `β = 0` energy `𝖤_0`,
* `e:gradfl`    — gradient ascent for `𝖤_0` on `(𝕊^{d-1})^n`, and the fact
                  that `e:Snonres0` *is* that gradient ascent,
* `eq: taylor`  — a non-trivial critical point of `𝖤_0` splits `[n]`.

The second half — the Hessian at a critical point, the "Russian trick",
`lem: yury.lemma` and the assembly of `p:beta0` — is in
`Perspective.AppendixA_Saddle`.
-/

import Transformer.Basic
import Transformer.Perspective.Section3_SmallBeta
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.InnerProductSpace.Calculus

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- **`β = 0` energy.**

  `𝖤_0(x_1,…,x_n) = (1/n) Σ_i Σ_j ⟨x_i, x_j⟩`. -/
noncomputable def E0 (X : SphereTuple d n) : ℝ :=
  ((n : ℝ)⁻¹) * ∑ i : Idx n, ∑ j : Idx n,
    inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d))

/-- The derivative of `𝖤_0` along a curve of tuples: with `ẋ_i(t) = b_i`,

  `d/dt 𝖤_0(X(t)) = (2/n) Σ_i ⟨b_i, Σ_j x_j(t)⟩`.

The factor `2` is the symmetry of the double sum, and it is what makes
`(2/n) Σ_j x_j` — and not `(1/n) Σ_j x_j` — the Euclidean gradient of `𝖤_0`
in the `i`-th variable. -/
theorem hasDerivAt_E0
    (Y : ℝ → SphereTuple d n) (b : Idx n → EucSpace d) (t : ℝ)
    (hY : ∀ i : Idx n, HasDerivAt (fun s => (Y s i : EucSpace d)) (b i) t) :
    HasDerivAt (fun s => E0 d n (Y s))
      ((2 * (n : ℝ)⁻¹) *
        ∑ i : Idx n, inner (𝕜 := ℝ) (b i) (∑ j : Idx n, (Y t j : EucSpace d))) t := by
  have hinner : ∀ i j : Idx n,
      HasDerivAt (fun s => inner (𝕜 := ℝ) ((Y s i : EucSpace d)) ((Y s j : EucSpace d)))
        (inner (𝕜 := ℝ) ((Y t i : EucSpace d)) (b j)
          + inner (𝕜 := ℝ) (b i) ((Y t j : EucSpace d))) t :=
    fun i j => HasDerivAt.inner ℝ (hY i) (hY j)
  have hsum := HasDerivAt.const_mul ((n : ℝ)⁻¹) (HasDerivAt.fun_sum
    (fun i (_ : i ∈ (Finset.univ : Finset (Idx n))) =>
      HasDerivAt.fun_sum
        (fun j (_ : j ∈ (Finset.univ : Finset (Idx n))) => hinner i j)))
  refine hsum.congr_deriv ?_
  have hswap :
      ∑ i : Idx n, ∑ j : Idx n, inner (𝕜 := ℝ) ((Y t i : EucSpace d)) (b j)
        = ∑ i : Idx n, ∑ j : Idx n, inner (𝕜 := ℝ) (b i) ((Y t j : EucSpace d)) := by
    rw [Finset.sum_comm]
    exact Finset.sum_congr rfl fun i _ =>
      Finset.sum_congr rfl fun j _ => real_inner_comm _ _
  have hsplit : ∀ i : Idx n,
      ∑ j : Idx n, (inner (𝕜 := ℝ) ((Y t i : EucSpace d)) (b j)
          + inner (𝕜 := ℝ) (b i) ((Y t j : EucSpace d)))
        = (∑ j : Idx n, inner (𝕜 := ℝ) ((Y t i : EucSpace d)) (b j))
            + ∑ j : Idx n, inner (𝕜 := ℝ) (b i) ((Y t j : EucSpace d)) :=
    fun i => Finset.sum_add_distrib
  rw [Finset.sum_congr rfl (fun i _ => hsplit i), Finset.sum_add_distrib, hswap]
  have hb : ∀ i : Idx n,
      inner (𝕜 := ℝ) (b i) (∑ j : Idx n, (Y t j : EucSpace d))
        = ∑ j : Idx n, inner (𝕜 := ℝ) (b i) ((Y t j : EucSpace d)) :=
    fun i => inner_sum _ _ _
  rw [Finset.sum_congr rfl (fun i _ => hb i)]
  ring

/-- The hypothesis of `hasDerivAt_E0` is satisfiable: a constant curve has
derivative `0`. -/
example (X : SphereTuple d n) (t : ℝ) :
    ∀ i : Idx n, HasDerivAt (fun _ : ℝ => (X i : EucSpace d)) 0 t :=
  fun _ => hasDerivAt_const _ _

/-- `⟨Proj_x y, y⟩ = ‖Proj_x y‖²` for a unit vector `x`: the projection is
orthogonal to `x`, so testing it against `y` and against itself agree. -/
theorem inner_proj_self {x y : EucSpace d} (hx : ‖x‖ = 1) :
    inner (𝕜 := ℝ) (proj d x y) y = ‖proj d x y‖ ^ 2 := by
  have hxx : inner (𝕜 := ℝ) x x = 1 := by
    rw [real_inner_self_eq_norm_mul_norm, hx]; ring
  simp only [proj, inner_sub_left, inner_sub_right, real_inner_smul_left,
    real_inner_smul_right, ← real_inner_self_eq_norm_sq, real_inner_comm x y, hxx]
  ring

/-- A unit vector is not its own antipode. -/
theorem ne_neg_self_of_norm_eq_one {x : EucSpace d} (hx : ‖x‖ = 1) : x ≠ -x := by
  intro h
  have h1 : inner (𝕜 := ℝ) x x = 1 := by
    rw [real_inner_self_eq_norm_mul_norm, hx]; ring
  have h2 : inner (𝕜 := ℝ) x x = -inner (𝕜 := ℝ) x x := by
    nth_rewrite 2 [h]
    rw [inner_neg_right]
  rw [h1] at h2
  norm_num at h2

/-- The hypothesis of `inner_proj_self` and of `ne_neg_self_of_norm_eq_one` is
satisfiable: the first standard basis vector of `ℝ¹` has norm `1`. -/
example : ‖(EuclideanSpace.single (0 : Fin 1) (1 : ℝ))‖ = 1 := by
  simp

/-- `Proj_x` is linear in the vector being projected; only the homogeneity is
needed below, to pull the factor `2/n` out of the gradient. -/
theorem proj_smul (a : ℝ) (x y : EucSpace d) :
    proj d x (a • y) = a • proj d x y := by
  simp only [proj, real_inner_smul_right, smul_sub, smul_smul]

/-- **Equation (e:gradfl).** *Gradient ascent for `𝖤_0` on `(𝕊^{d-1})^n`.*

  `ẋ_i(t) = Proj_{x_i(t)} ( ∇_{x_i} 𝖤_0(X(t)) ) = Proj_{x_i(t)} ( (2/n) Σ_j x_j(t) )`.

Written out rather than left abstract: on a product of spheres the Riemannian
gradient is the ambient gradient projected onto each tangent space, and the
ambient gradient of `𝖤_0` in the `i`-th variable is `(2/n) Σ_j x_j`
(`hasDerivAt_E0`). -/
def E0GradientAscent (X : ℝ → SphereTuple d n) : Prop :=
  ∀ t : ℝ, ∀ i : Idx n,
    HasDerivAt (fun s => (X s i : EucSpace d))
      (proj d ((X t i : EucSpace d))
        ((2 * (n : ℝ)⁻¹) • ∑ j : Idx n, (X t j : EucSpace d))) t

/-- *`e:Snonres0` is gradient ascent for `𝖤_0`, run at half speed.*

The `β = 0` dynamics is `ẋ_i = Proj_{x_i}((1/n) Σ_j x_j)`, and the gradient
ascent above is `ẋ_i = Proj_{x_i}((2/n) Σ_j x_j)`: the two vector fields are
proportional, so they have the same trajectories up to the time change
`t ↦ t/2`. -/
theorem beta0Dynamics_smul (X : ℝ → SphereTuple d n) (t : ℝ) (i : Idx n) :
    proj d ((X t i : EucSpace d))
        ((2 * (n : ℝ)⁻¹) • ∑ j : Idx n, (X t j : EucSpace d))
      = (2 : ℝ) • proj d ((X t i : EucSpace d))
          (((n : ℝ)⁻¹) • ∑ j : Idx n, (X t j : EucSpace d)) := by
  rw [mul_smul, proj_smul]

/-- *The energy increases along gradient ascent* (the `β = 0` case of
`eq: dissipation.softmax`):

  `d/dt 𝖤_0(X(t)) = (2/n)² Σ_i ‖Proj_{x_i(t)} Σ_j x_j(t)‖² ≥ 0`.

So `𝖤_0` is a Lyapunov function for `e:Snonres0`, which is what the
Łojasiewicz argument of Appendix A is applied to. -/
theorem hasDerivAt_E0_ascent
    (X : ℝ → SphereTuple d n) (hX : E0GradientAscent d n X) (t : ℝ) :
    HasDerivAt (fun s => E0 d n (X s))
      ((2 * (n : ℝ)⁻¹) ^ 2 * ∑ i : Idx n,
        ‖proj d ((X t i : EucSpace d)) (∑ j : Idx n, (X t j : EucSpace d))‖ ^ 2) t := by
  refine (hasDerivAt_E0 d n X _ t (hX t)).congr_deriv ?_
  have key : ∀ i : Idx n,
      inner (𝕜 := ℝ)
          (proj d ((X t i : EucSpace d))
            ((2 * (n : ℝ)⁻¹) • ∑ j : Idx n, (X t j : EucSpace d)))
          (∑ j : Idx n, (X t j : EucSpace d))
        = (2 * (n : ℝ)⁻¹) *
            ‖proj d ((X t i : EucSpace d)) (∑ j : Idx n, (X t j : EucSpace d))‖ ^ 2 := by
    intro i
    have hx : ‖(X t i : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp (X t i).2
    rw [proj_smul, real_inner_smul_left, inner_proj_self d hx]
  rw [Finset.sum_congr rfl fun i _ => key i, ← Finset.mul_sum]
  ring

/-- A *critical point* of `𝖤_0` on `(𝕊^{d-1})^n`: the Riemannian gradient
vanishes in every variable, i.e. `Σ_j x_j` is normal to the sphere at each
`x_i`.  This is the hypothesis "`X` is a critical point of `𝖤_0`" of
`eq: taylor` and of `lem: yury.lemma`. -/
def IsCriticalE0 (X : SphereTuple d n) : Prop :=
  ∀ i : Idx n, proj d ((X i : EucSpace d)) (∑ j : Idx n, (X j : EucSpace d)) = 0

/-- A critical point is *non-trivial* when the particles are not all at the
same point of the sphere — the trivial critical points being the consensus
configurations. -/
def NonTrivialTuple (X : SphereTuple d n) : Prop :=
  ∃ i j : Idx n, (X i : EucSpace d) ≠ (X j : EucSpace d)

/-- **Equation (eq: taylor).** If `(x_1,…,x_n) ∈ (𝕊^{d-1})^n` is a non-trivial
critical point of `𝖤_0`, then there exists `𝒮 ⊂ [n]` with

  `Σ_{i ∈ 𝒮} Σ_{j ∈ 𝒮^c} ⟨x_i, x_j⟩ < 0`.

A singleton always works.  Criticality says `S = Σ_j x_j` is `⟨x_i, S⟩ x_i`
for every `i`, so `|⟨x_i, S⟩| = ‖S‖`; were every `⟨x_i, S⟩` positive they
would all equal `‖S‖ > 0` and every `x_i` would be `S/‖S‖`, which is the
trivial critical point.  So some `⟨x_i, S⟩ ≤ 0`, and for `𝒮 = {i}` the double
sum is `⟨x_i, S⟩ - 1 ≤ -1`.

Source: arXiv:2312.10794v5, Appendix A, `eq: taylor`. -/
theorem taylor_eq
    (X : SphereTuple d n)
    (h_crit : IsCriticalE0 d n X)
    (h_nontriv : NonTrivialTuple d n X) :
    ∃ 𝒮 : Finset (Idx n),
      ∑ i ∈ 𝒮, ∑ j ∈ 𝒮ᶜ,
        inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d)) < 0 := by
  classical
  set S : EucSpace d := ∑ j : Idx n, (X j : EucSpace d) with hS
  have hnorm : ∀ i : Idx n, ‖(X i : EucSpace d)‖ = 1 := fun i =>
    mem_sphere_zero_iff_norm.mp (X i).2
  -- criticality says the barycentre is normal to the sphere at every particle
  have hpar : ∀ i : Idx n,
      S = (inner (𝕜 := ℝ) ((X i : EucSpace d)) S) • ((X i : EucSpace d)) := by
    intro i
    have h := h_crit i
    rw [proj, sub_eq_zero] at h
    exact h
  -- so `|⟨x_i, S⟩| = ‖S‖`, and were every one of them positive the particles
  -- would all sit at `S / ‖S‖`
  have hex : ∃ i : Idx n, inner (𝕜 := ℝ) ((X i : EucSpace d)) S ≤ 0 := by
    by_contra hcontra
    have hcon : ∀ i : Idx n, 0 < inner (𝕜 := ℝ) ((X i : EucSpace d)) S := fun i =>
      lt_of_not_ge fun h => hcontra ⟨i, h⟩
    have hlam : ∀ i : Idx n, inner (𝕜 := ℝ) ((X i : EucSpace d)) S = ‖S‖ := by
      intro i
      have h1 : ‖S‖ = |inner (𝕜 := ℝ) ((X i : EucSpace d)) S| * 1 := by
        conv_lhs => rw [hpar i]
        rw [norm_smul, Real.norm_eq_abs, hnorm i]
      rw [h1, abs_of_pos (hcon i), mul_one]
    obtain ⟨i, j, hij⟩ := h_nontriv
    refine hij ?_
    have hSpos : (0 : ℝ) < ‖S‖ := by
      have h := hcon i
      rwa [hlam i] at h
    have hi := hpar i
    have hj := hpar j
    rw [hlam i] at hi
    rw [hlam j] at hj
    exact smul_right_injective (EucSpace d) hSpos.ne'
      (show (‖S‖ : ℝ) • ((X i : EucSpace d)) = (‖S‖ : ℝ) • ((X j : EucSpace d)) by
        rw [← hi, ← hj])
  obtain ⟨i, hi⟩ := hex
  refine ⟨{i}, ?_⟩
  rw [Finset.sum_singleton]
  have hself : inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X i : EucSpace d)) = 1 := by
    rw [real_inner_self_eq_norm_mul_norm, hnorm i]; ring
  have htot : ∑ j : Idx n, inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d))
      = inner (𝕜 := ℝ) ((X i : EucSpace d)) S := by
    rw [hS, inner_sum]
  have hsplit := Finset.sum_compl_add_sum ({i} : Finset (Idx n))
    (fun j => inner (𝕜 := ℝ) ((X i : EucSpace d)) ((X j : EucSpace d)))
  rw [Finset.sum_singleton, hself, htot] at hsplit
  linarith

/-- The north pole `e₀` of `𝕊^0 ⊂ ℝ¹`. -/
noncomputable def northPole : SSphere 1 :=
  ⟨EuclideanSpace.single (0 : Fin 1) (1 : ℝ), by simp⟩

/-- The antipodal pair `(e₀, -e₀)` of `Perspective.antipodalPair` is a
non-trivial critical configuration of `𝖤_0`, its sum being `0`.  It is the
witness for the hypotheses of `taylor_eq` here and of `yury_lemma` and
`hessian_at_critical` in `Perspective.AppendixA_Saddle`. -/
theorem antipodalPair_critical_nonTrivial :
    IsCriticalE0 1 2 (antipodalPair 1 northPole) ∧
    NonTrivialTuple 1 2 (antipodalPair 1 northPole) := by
  refine ⟨fun i => ?_, 0, 1, ?_⟩
  · simp [antipodalPair, antipode, proj, Fin.sum_univ_two]
  · have hx : ‖((northPole : SSphere 1) : EucSpace 1)‖ = 1 :=
      mem_sphere_zero_iff_norm.mp northPole.2
    have h0 : ((antipodalPair 1 northPole 0 : SSphere 1) : EucSpace 1)
        = ((northPole : SSphere 1) : EucSpace 1) := rfl
    have h1 : ((antipodalPair 1 northPole 1 : SSphere 1) : EucSpace 1)
        = -((northPole : SSphere 1) : EucSpace 1) := rfl
    rw [h0, h1]
    exact ne_neg_self_of_norm_eq_one 1 hx

/-- The hypotheses of `taylor_eq` are satisfiable. -/
example : IsCriticalE0 1 2 (antipodalPair 1 northPole) ∧
    NonTrivialTuple 1 2 (antipodalPair 1 northPole) :=
  antipodalPair_critical_nonTrivial

end Perspective
end Transformer
