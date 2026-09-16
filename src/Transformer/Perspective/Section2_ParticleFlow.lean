/-
# §3.3 — the particle version of the Wasserstein gradient flow

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*, `rem: particle.gf`.

At an empirical measure `μ(t) = n⁻¹ Σ_i δ_{x_i(t)}` the interaction energy
`eq: interaction.energy` takes the form

  `𝖤_β(X) = (1/(2βn²)) Σ_i Σ_j e^{β⟨x_i,x_j⟩}`,

and `USA` is its gradient *ascent* flow for the round metric on
`(𝕊^{d-1})^n`, sped up by `n`:

  `Ẋ(t) = n ∇_X 𝖤_β(X(t))`.                                  (`e:dynonX`)

The Riemannian gradient on `(𝕊^{d-1})^n` is not carried by this development,
so — as in `Perspective.SAIsGradientFlow` — the identity is stated through
directional derivatives along curves, which is what `∇` abbreviates: the
differential of `𝖤_β` at `X(t)` in a tangent direction `b` is `n⁻¹ ⟨Ẋ(t), b⟩`.
Against `SAIsGradientFlow` the only difference is the metric: the round one
here, `e:scalarproduct` there.  That is exactly the difference between `USA`
and `SA`.
-/

import Transformer.Perspective.Section2_GradientFlow
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable (d n : ℕ)

/-- The empirical interaction energy of `rem: particle.gf`,

  `𝖤_β(X) = (1/(2βn²)) Σ_i Σ_j e^{β⟨x_i,x_j⟩}`,

i.e. `Perspective.particleEnergy` at `V = Id` carrying the `n⁻²` of the
empirical measure `n⁻¹ Σ_i δ_{x_i}`. -/
noncomputable def empiricalEnergy (β : ℝ) (X : SphereTuple d n) : ℝ :=
  ((n : ℝ) ^ 2)⁻¹ * particleEnergy d n β (ContinuousLinearMap.id ℝ (EucSpace d)) X

/-- `HasDerivAt.sum` for a sum written as a function of the point rather than
as a sum of functions.  Mathlib states the rule for `∑ i ∈ u, A i`; every use
below has the `fun y => ∑ i ∈ u, A i y` shape instead.

Source: Mathlib, `HasDerivAt.sum`. -/
theorem hasDerivAt_fun_sum {ι : Type*} (u : Finset ι)
    (A : ι → ℝ → ℝ) (A' : ι → ℝ) (x : ℝ)
    (h : ∀ i ∈ u, HasDerivAt (A i) (A' i) x) :
    HasDerivAt (fun y => ∑ i ∈ u, A i y) (∑ i ∈ u, A' i) x := by
  have heq : (fun y => ∑ i ∈ u, A i y) = ∑ i ∈ u, A i := by
    funext y; exact (Finset.sum_apply y u A).symm
  rw [heq]
  exact HasDerivAt.sum h

/-- The hypothesis of `hasDerivAt_fun_sum` is satisfiable: the empty family. -/
example (x : ℝ) : ∀ i ∈ (∅ : Finset ℕ),
    HasDerivAt (fun _ : ℝ => (0 : ℝ)) (0 : ℝ) x := by
  intro i hi
  exact absurd hi (Finset.notMem_empty i)

/-- **Equation (e:dynonX).**  *`USA` is a gradient ascent flow for the round
metric.*

For every curve `Y` through `X(t)` with velocity `b`,

  `d/ds 𝖤_β(Y(s))|_{s=0} = n⁻¹ Σ_i ⟨ẋ_i(t), b_i⟩`,

which is `Ẋ(t) = n ∇_X 𝖤_β(X(t))` read against the round metric
`⟨a, b⟩ = Σ_i ⟨a_i, b_i⟩`.  Equivalently, per particle,
`∂_i 𝖤_β(X(t)) = n⁻¹ ẋ_i(t)`.

Taking `b = Ẋ(t)` gives `d/dt 𝖤_β(X(t)) = n⁻¹ Σ_i ‖ẋ_i(t)‖² ≥ 0`, the
particle counterpart of `lem: dissipation`.

`β ≠ 0` is part of the statement and not an afterthought: at `β = 0` the
`(2β)⁻¹` in front of `𝖤_β` collapses it to the constant `0`, whose derivative
vanishes, while `USA` at `β = 0` still moves the particles.

Source: arXiv:2312.10794v5, §3.3, `rem: particle.gf`, `e:dynonX`. -/
theorem usa_isGradientFlow
    (β : ℝ) (hβ : β ≠ 0) (X : ℝ → SphereTuple d n) (hX : USA d n β X)
    (t : ℝ) (Y : ℝ → SphereTuple d n) (b : Idx n → EucSpace d)
    (hY : Y 0 = X t)
    (hb : ∀ i : Idx n, HasDerivAt (fun s => (Y s i : EucSpace d)) (b i) 0) :
    HasDerivAt (fun s => empiricalEnergy d n β (Y s))
      ((n : ℝ)⁻¹ *
        ∑ i : Idx n,
          inner (𝕜 := ℝ) (deriv (fun s => (X s i : EucSpace d)) t) (b i)) 0 := by
  classical
  have hY0 : ∀ i : Idx n, (Y 0 i : EucSpace d) = (X t i : EucSpace d) := by
    intro i; rw [hY]
  -- a curve on the sphere has a tangent velocity
  have htang : ∀ i : Idx n, inner (𝕜 := ℝ) ((X t i : EucSpace d)) (b i) = 0 := by
    intro i
    have hconst :
        (fun s => inner (𝕜 := ℝ) ((Y s i : EucSpace d)) ((Y s i : EucSpace d)))
          = fun _ : ℝ => (1 : ℝ) := by
      funext s
      rw [real_inner_self_eq_norm_mul_norm, mem_sphere_zero_iff_norm.mp (Y s i).2]
      ring
    have h1 : HasDerivAt
        (fun s => inner (𝕜 := ℝ) ((Y s i : EucSpace d)) ((Y s i : EucSpace d)))
        (inner (𝕜 := ℝ) ((Y 0 i : EucSpace d)) (b i)
          + inner (𝕜 := ℝ) (b i) ((Y 0 i : EucSpace d))) 0 :=
      (hb i).inner ℝ (hb i)
    rw [hconst] at h1
    have h2 := (hasDerivAt_const (0 : ℝ) (1 : ℝ)).unique h1
    rw [hY0 i] at h2
    have h3 : inner (𝕜 := ℝ) ((X t i : EucSpace d)) (b i)
        = inner (𝕜 := ℝ) (b i) ((X t i : EucSpace d)) := real_inner_comm _ _
    rw [h3]
    linarith
  -- the derivative of the double sum, term by term
  have hterm : ∀ i j : Idx n,
      HasDerivAt (fun s => Real.exp (β * inner (𝕜 := ℝ)
          ((Y s i : EucSpace d)) ((Y s j : EucSpace d))))
        (Real.exp (β * inner (𝕜 := ℝ)
              ((X t i : EucSpace d)) ((X t j : EucSpace d)))
          * (β * (inner (𝕜 := ℝ) ((X t i : EucSpace d)) (b j)
              + inner (𝕜 := ℝ) (b i) ((X t j : EucSpace d))))) 0 := by
    intro i j
    have hij : HasDerivAt
        (fun s => inner (𝕜 := ℝ) ((Y s i : EucSpace d)) ((Y s j : EucSpace d)))
        (inner (𝕜 := ℝ) ((Y 0 i : EucSpace d)) (b j)
          + inner (𝕜 := ℝ) (b i) ((Y 0 j : EucSpace d))) 0 :=
      (hb i).inner ℝ (hb j)
    have h := (hij.const_mul β).exp
    rw [hY0 i, hY0 j] at h
    exact h
  have hrows : ∀ i : Idx n, HasDerivAt
      (fun s => ∑ j : Idx n, Real.exp (β * inner (𝕜 := ℝ)
          ((Y s i : EucSpace d)) ((Y s j : EucSpace d))))
      (∑ j : Idx n, Real.exp (β * inner (𝕜 := ℝ)
            ((X t i : EucSpace d)) ((X t j : EucSpace d)))
          * (β * (inner (𝕜 := ℝ) ((X t i : EucSpace d)) (b j)
              + inner (𝕜 := ℝ) (b i) ((X t j : EucSpace d))))) 0 :=
    fun i => hasDerivAt_fun_sum _ _ _ _ fun j _ => hterm i j
  have hEeq : (fun s => empiricalEnergy d n β (Y s))
      = fun s => ((n : ℝ) ^ 2)⁻¹ * ((2 * β)⁻¹ *
          ∑ i : Idx n, ∑ j : Idx n, Real.exp (β * inner (𝕜 := ℝ)
            ((Y s i : EucSpace d)) ((Y s j : EucSpace d)))) := by
    funext s
    simp [empiricalEnergy, particleEnergy]
  rw [hEeq]
  refine HasDerivAt.congr_deriv
    (((hasDerivAt_fun_sum _ _ _ _ fun i _ => hrows i).const_mul
      ((2 * β)⁻¹)).const_mul (((n : ℝ) ^ 2)⁻¹)) ?_
  -- what is left is an identity between double sums
  have hderiv : ∀ i : Idx n, deriv (fun s => (X s i : EucSpace d)) t
      = proj d ((X t i : EucSpace d))
          (((n : ℝ)⁻¹) • ∑ j : Idx n,
            Real.exp (β * inner (𝕜 := ℝ)
              ((X t i : EucSpace d)) ((X t j : EucSpace d)))
              • ((X t j : EucSpace d))) := fun i => (hX t i).deriv
  obtain ⟨T, hT⟩ : ∃ T : ℝ, (∑ i : Idx n, ∑ j : Idx n,
      Real.exp (β * inner (𝕜 := ℝ)
        ((X t i : EucSpace d)) ((X t j : EucSpace d)))
        * inner (𝕜 := ℝ) ((X t j : EucSpace d)) (b i)) = T := ⟨_, rfl⟩
  have hrow : ∀ i : Idx n,
      inner (𝕜 := ℝ) (deriv (fun s => (X s i : EucSpace d)) t) (b i)
        = (n : ℝ)⁻¹ * ∑ j : Idx n,
            Real.exp (β * inner (𝕜 := ℝ)
              ((X t i : EucSpace d)) ((X t j : EucSpace d)))
              * inner (𝕜 := ℝ) ((X t j : EucSpace d)) (b i) := by
    intro i
    rw [hderiv i, proj, inner_sub_left, real_inner_smul_left, real_inner_smul_left,
      htang i, mul_zero, sub_zero, sum_inner]
    simp [real_inner_smul_left, Finset.mul_sum]
  have hRHS : (n : ℝ)⁻¹ * ∑ i : Idx n,
      inner (𝕜 := ℝ) (deriv (fun s => (X s i : EucSpace d)) t) (b i)
      = (n : ℝ)⁻¹ * ((n : ℝ)⁻¹ * T) := by
    rw [Finset.sum_congr rfl fun i _ => hrow i, ← Finset.mul_sum, hT]
  -- the `⟨x_i, b_j⟩` half of each term is the `⟨x_j, b_i⟩` half transposed
  have hcross : (∑ i : Idx n, ∑ j : Idx n,
        Real.exp (β * inner (𝕜 := ℝ)
          ((X t i : EucSpace d)) ((X t j : EucSpace d)))
          * inner (𝕜 := ℝ) ((X t i : EucSpace d)) (b j)) = T := by
    rw [← hT, Finset.sum_comm]
    exact Finset.sum_congr rfl fun i _ =>
      Finset.sum_congr rfl fun j _ => by rw [real_inner_comm ((X t j : EucSpace d))]
  have hstraight : (∑ i : Idx n, ∑ j : Idx n,
        Real.exp (β * inner (𝕜 := ℝ)
          ((X t i : EucSpace d)) ((X t j : EucSpace d)))
          * inner (𝕜 := ℝ) (b i) ((X t j : EucSpace d))) = T := by
    rw [← hT]
    exact Finset.sum_congr rfl fun i _ =>
      Finset.sum_congr rfl fun j _ => by rw [real_inner_comm (b i)]
  have hexpand : ∀ i j : Idx n,
      Real.exp (β * inner (𝕜 := ℝ)
          ((X t i : EucSpace d)) ((X t j : EucSpace d)))
        * (β * (inner (𝕜 := ℝ) ((X t i : EucSpace d)) (b j)
            + inner (𝕜 := ℝ) (b i) ((X t j : EucSpace d))))
      = β * (Real.exp (β * inner (𝕜 := ℝ)
            ((X t i : EucSpace d)) ((X t j : EucSpace d)))
          * inner (𝕜 := ℝ) ((X t i : EucSpace d)) (b j))
        + β * (Real.exp (β * inner (𝕜 := ℝ)
            ((X t i : EucSpace d)) ((X t j : EucSpace d)))
          * inner (𝕜 := ℝ) (b i) ((X t j : EucSpace d))) := by
    intro i j; ring
  have hkey : (2 * β)⁻¹ * (β * T + β * T) = T := by
    rw [show β * T + β * T = 2 * β * T by ring, ← mul_assoc,
      inv_mul_cancel₀ (mul_ne_zero two_ne_zero hβ), one_mul]
  rw [hRHS, Finset.sum_congr rfl fun i _ =>
    Finset.sum_congr rfl fun j _ => hexpand i j]
  simp only [Finset.sum_add_distrib, ← Finset.mul_sum]
  rw [hcross, hstraight, hkey, pow_two, mul_inv, mul_assoc]

/-- The single token of `𝕊^0 ⊂ ℝ^1`. -/
noncomputable def oneToken : SphereTuple 1 1 :=
  fun _ => ⟨EuclideanSpace.single (0 : Fin 1) (1 : ℝ), by simp⟩

/-- The hypotheses of `usa_isGradientFlow` are satisfiable: a single token is
a stationary solution of `USA`, since its velocity is `Proj_x x = 0`, and the
constant curve through it has velocity `b = 0`. -/
example :
    (1 : ℝ) ≠ 0 ∧ USA 1 1 1 (fun _ => oneToken) ∧
      ∀ i : Idx 1,
        HasDerivAt (fun s : ℝ => (((fun _ => oneToken) s i : EucSpace 1))) 0 0 := by
  refine ⟨one_ne_zero, ?_⟩
  have hx : ‖(oneToken 0 : EucSpace 1)‖ = 1 :=
    mem_sphere_zero_iff_norm.mp (oneToken 0).2
  have hinner : inner (𝕜 := ℝ) ((oneToken 0 : EucSpace 1)) ((oneToken 0 : EucSpace 1)) = 1 := by
    rw [real_inner_self_eq_norm_mul_norm, hx, mul_one]
  have hself : ∀ c : ℝ,
      proj 1 ((oneToken 0 : EucSpace 1)) (c • (oneToken 0 : EucSpace 1)) = 0 := by
    intro c
    rw [proj, real_inner_smul_right, hinner, mul_one, sub_self]
  refine ⟨fun t i => ?_, fun i => hasDerivAt_const (0 : ℝ) _⟩
  have hi : (oneToken i : EucSpace 1) = (oneToken 0 : EucSpace 1) := by
    rw [Subsingleton.elim i 0]
  simpa [hi, hinner, hself] using hasDerivAt_const t ((oneToken 0 : EucSpace 1))

end Perspective
end Transformer
