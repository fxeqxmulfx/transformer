/-
# §9 — General matrices

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

This file formalizes §9 of the survey:

* §9.1 — the *repulsive case* `V = -I_d`, with the connection to optimal
  sphere configurations;
* §9.2 — pure self-attention without projection: `eq: transf-1`,
  `eq:zifromxi`, `e:Rres`;
* §9.3 — *singular dynamics* in the `β → ∞` limit: `e:exp`, `e:maineq`,
  `e:Ci(t)`;
* §9.4 — diffusive regularization (consensus-based optimization analogy).
-/

import Transformer.Basic
import Transformer.Perspective.Section1_IPS
import Transformer.Perspective.Section2_FlowMap
import Mathlib.Algebra.MvPolynomial.Degrees
import Mathlib.Analysis.Normed.Algebra.Exponential

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Perspective

open Perspective

variable (d n : ℕ)

/-! ### §9.1 — The repulsive case `V = -I_d` -/

/-- The interaction energy `𝖤_β` rewritten via squared distances:

  `𝖤_β[μ] = (e^β / (2β)) ∫∫ exp(-β/2 ‖x - x'‖²) dμ(x) dμ(x')`.

Source: arXiv:2312.10794v5, §9.1. -/
noncomputable def squaredDistEnergy
    (β : ℝ) (μ : Perspective.ProbSphere d) : ℝ :=
  (Real.exp β / (2 * β)) *
    ∫ x, ∫ x',
      Real.exp (-(β / 2) * ‖(x : EucSpace d) - (x' : EucSpace d)‖ ^ 2)
        ∂(μ : Measure (SSphere d)) ∂(μ : Measure (SSphere d))

/-- **The kernel identity behind the rewriting.**  On the unit sphere
`‖x - y‖² = 2 - 2⟨x, y⟩`, so

  `e^{β ⟨x,y⟩} = e^β · e^{-(β/2) ‖x - y‖²}`:

the Gaussian-like kernel and the softmax kernel differ by the constant `e^β`.
Source: arXiv:2312.10794v5, §9.1. -/
theorem exp_inner_eq_exp_sqDist (β : ℝ) (x y : SSphere d) :
    Real.exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d))
      = Real.exp β * Real.exp (-(β / 2) * ‖(x : EucSpace d) - (y : EucSpace d)‖ ^ 2) := by
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  have hy : ‖(y : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp y.2
  have hd : ‖(x : EucSpace d) - (y : EucSpace d)‖ ^ 2
      = 2 - 2 * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d) := by
    rw [norm_sub_sq_real, hx, hy]; ring
  rw [← Real.exp_add, hd]
  congr 1
  ring

/-- **The rewriting itself.**  `squaredDistEnergy` and `interactionEnergy` are
the same functional, by `exp_inner_eq_exp_sqDist` under the double integral.
Source: arXiv:2312.10794v5, §9.1. -/
theorem squaredDistEnergy_eq_interactionEnergy (β : ℝ) (μ : Perspective.ProbSphere d) :
    squaredDistEnergy d β μ = interactionEnergy d β μ := by
  rw [squaredDistEnergy, interactionEnergy]
  simp_rw [exp_inner_eq_exp_sqDist d β, MeasureTheory.integral_const_mul]
  ring

/-- The energy of a finite configuration:

  `𝖧_β(𝒞) = Σ_{x ∈ 𝒞} Σ_{y ∈ 𝒞} e^{β ⟨x,y⟩}`,

the discrete counterpart of `2β 𝖤_β`. -/
noncomputable def discreteEnergy (β : ℝ) (𝒞 : Finset (SSphere d)) : ℝ :=
  ∑ x ∈ 𝒞, ∑ y ∈ 𝒞, Real.exp (β * inner (𝕜 := ℝ) (x : EucSpace d) (y : EucSpace d))

/-- `𝖧_β` is, up to the constant `e^β`, a sum of `f(‖x - y‖²)` over all pairs,
with `f(r) = e^{-(β/2) r}`.  This is what places the survey's energy inside the
Cohn–Kumar framework of potentials that decrease with distance. -/
theorem discreteEnergy_eq_sqDist (β : ℝ) (𝒞 : Finset (SSphere d)) :
    discreteEnergy d β 𝒞
      = Real.exp β * ∑ x ∈ 𝒞, ∑ y ∈ 𝒞,
          Real.exp (-(β / 2) * ‖(x : EucSpace d) - (y : EucSpace d)‖ ^ 2) := by
  rw [discreteEnergy, Finset.mul_sum]
  refine Finset.sum_congr rfl fun x _ => ?_
  rw [Finset.mul_sum]
  exact Finset.sum_congr rfl fun y _ => exp_inner_eq_exp_sqDist d β x y

/-- The potential `f(r) = e^{-(β/2) r}` of `discreteEnergy_eq_sqDist` is
strictly decreasing in the squared distance whenever `β > 0`: minimizing
`𝖧_β` pushes the points apart.  Source: arXiv:2312.10794v5, §9.1. -/
theorem exp_sqDist_strictAnti (β : ℝ) (hβ : 0 < β) :
    StrictAnti (fun r : ℝ => Real.exp (-(β / 2) * r)) := by
  intro a b hab
  exact Real.exp_lt_exp.mpr (by nlinarith)

/-- The hypothesis `0 < β` of `exp_sqDist_strictAnti` is satisfiable. -/
example : (0 : ℝ) < 1 := one_pos

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

/-- A finite point set `𝒞 ⊂ 𝕊^{d-1}` is a *sharp configuration* if there are
`m > 1` distinct pairwise inner products and `𝒞` is a spherical
`(2m-1)`-design. -/
def sharpConfiguration (σ : Measure (SSphere d)) (𝒞 : Finset (SSphere d)) : Prop :=
  ∃ m : ℕ, 1 < m ∧
    (Finset.image
      (fun p : SSphere d × SSphere d =>
        inner (𝕜 := ℝ) (p.1 : EucSpace d) (p.2 : EucSpace d))
      (𝒞 ×ˢ 𝒞)).card = m ∧
    sphericalDesign d σ (2 * m - 1) 𝒞

/-- **Cohn–Kumar theorem.**  Every global minimum of `𝖧_β` among finite
configurations `𝒞 ⊂ 𝕊^{d-1}` with `#𝒞 = n` is either a sharp configuration or
the vertices of the 600-cell (a 4-dimensional polytope with 120 vertices).

Stated, not proved: the theorem is a deep result of Cohn and Kumar, and the
600-cell is not constructed here — it enters as the parameter `exceptional`,
the configuration the dichotomy is allowed to except.
Source: arXiv:2312.10794v5, §9.1. -/
def CohnKumarDichotomy
    (σ : Measure (SSphere d)) (β : ℝ) (exceptional : Finset (SSphere d)) : Prop :=
  ∀ 𝒞 : Finset (SSphere d), 𝒞.card = n →
    (∀ 𝒟 : Finset (SSphere d), 𝒟.card = n →
        discreteEnergy d β 𝒞 ≤ discreteEnergy d β 𝒟) →
      sharpConfiguration d σ 𝒞 ∨ 𝒞 = exceptional

/-! ### §9.2 — Pure self-attention (no projection) -/

/-- **Equation (eq: transf-1).**  Pure self-attention on `ℝ^d`:

  `ẋ_i(t) = Z_{β,i}(t)⁻¹ Σ_j exp(β ⟨Q x_i, K x_j⟩) V x_j`. -/
def pureSA
    (β : ℝ) (Q K V : ParamMatrix d)
    (x : ℝ → Idx n → EucSpace d) : Prop :=
  ∀ t : ℝ, ∀ i : Idx n,
    HasDerivAt (fun s => x s i)
      ((∑ k : Idx n, Real.exp (β * inner (𝕜 := ℝ) (Q (x t i)) (K (x t k))))⁻¹
        • ∑ j : Idx n,
            Real.exp (β * inner (𝕜 := ℝ) (Q (x t i)) (K (x t j)))
            • V (x t j)) t

/-- **Equation (eq:zifromxi).**  The rescaling

  `z_i(t) = e^{-t V} x_i(t)`,

stated as the relation `x_i(t) = e^{t V} z_i(t)` between the two families, with
`e^{t V}` the exponential of the bounded operator `t V`.
Source: arXiv:2312.10794v5, §9.2. -/
def IsRescaling
    (V : ParamMatrix d) (x z : ℝ → Idx n → EucSpace d) : Prop :=
  ∀ (t : ℝ) (i : Idx n), x t i = NormedSpace.exp (t • V) (z t i)

/-- The relation is satisfiable: at `V = 0` the rescaling is the identity. -/
example (x : ℝ → Idx n → EucSpace d) : IsRescaling d n 0 x x := by
  intro t i
  have h : t • (0 : ParamMatrix d) = 0 := by ext v; simp
  show x t i = NormedSpace.exp (t • (0 : ParamMatrix d)) (x t i)
  rw [h, NormedSpace.exp_zero, one_apply_eq_self]

/-- **Equation (e:Rres).** Equation satisfied by the rescaled particles:

  `ż_i(t) = Z_{β,i}(t)⁻¹ Σ_j exp(β ⟨Q e^{tV} z_i, K e^{tV} z_j⟩)
                              · V (z_j(t) - z_i(t))`. -/
def rescaledEquation
    (β : ℝ) (Q K V : ParamMatrix d)
    (z : ℝ → Idx n → EucSpace d) : Prop :=
  ∀ t : ℝ, ∀ i : Idx n,
    HasDerivAt (fun s => z s i)
      ((∑ k : Idx n,
          Real.exp (β * inner (𝕜 := ℝ) (Q (z t i)) (K (z t k))))⁻¹
        • ∑ j : Idx n,
            Real.exp (β * inner (𝕜 := ℝ) (Q (z t i)) (K (z t j)))
            • V (z t j - z t i)) t

/-! ### §9.3 — Singular dynamics in the `β → ∞` limit -/

/-- **Equation (e:exp).**  Pre-limit ODE used to derive `e:maineq`:

  `ż_i(t) = Z_{β,i}(t)⁻¹ Σ_j exp(β ⟨Q z_i, K z_j⟩) V (z_j - z_i)`. -/
def preLimitODE
    (β : ℝ) (Q K V : ParamMatrix d)
    (z : ℝ → Idx n → EucSpace d) : Prop :=
  ∀ t : ℝ, ∀ i : Idx n,
    HasDerivAt (fun s => z s i)
      ((∑ k : Idx n,
          Real.exp (β * inner (𝕜 := ℝ) (Q (z t i)) (K (z t k))))⁻¹
        • ∑ j : Idx n,
            Real.exp (β * inner (𝕜 := ℝ) (Q (z t i)) (K (z t j)))
            • V (z t j - z t i)) t

/-- **Equation (e:Ci(t)).**  The *active set* of indices at `(t, i)`:

  `C_i(t) = { j ∈ [n] : ⟨Q z_i(t), K z_j(t)⟩ ≥ ⟨Q z_i(t), K z_k(t)⟩
                            for all k ∈ [n] }`. -/
noncomputable def activeSet
    (Q K : ParamMatrix d)
    (z : Idx n → EucSpace d) (i : Idx n) : Finset (Idx n) := by
  classical
  exact (Finset.univ : Finset (Idx n)).filter
    (fun j => ∀ k : Idx n,
      inner (𝕜 := ℝ) (Q (z i)) (K (z k))
        ≤ inner (𝕜 := ℝ) (Q (z i)) (K (z j)))

/-- **Equation (e:maineq).**  Limit ODE as `β → ∞`:

  `ż_i(t) = (1/|C_i(t)|) Σ_{j ∈ C_i(t)} V (z_j(t) - z_i(t))`. -/
def limitODE
    (Q K V : ParamMatrix d)
    (z : ℝ → Idx n → EucSpace d) : Prop :=
  ∀ t : ℝ, ∀ i : Idx n,
    let C := activeSet d n Q K (z t) i
    HasDerivAt (fun s => z s i)
      ((C.card : ℝ)⁻¹ • ∑ j ∈ C, V (z t j - z t i)) t

end Perspective
end Transformer
