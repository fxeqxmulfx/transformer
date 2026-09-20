/-
# §9 — General matrices

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

This file formalizes §9 of the survey:

* §9.1 — the *repulsive case* `V = -I_d`, with the connection to optimal
  sphere configurations (the Cohn–Kumar dichotomy itself is in
  `Transformer.Perspective.Section8_CohnKumar`);
* §9.2 — pure self-attention without projection: `eq: transf-1`,
  `eq:zifromxi`, `e:Rres`;
* §9.3 — *singular dynamics* in the `β → ∞` limit: `e:exp`, `e:maineq`,
  `e:Ci(t)`;
* §9.4 — diffusive regularization (consensus-based optimization analogy).
-/

import Transformer.Basic
import Transformer.Perspective.Section1_IPS
import Transformer.Perspective.Section2_FlowMap
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

/-- On the sphere every singleton has the same energy `e^β`, since
`⟨x, x⟩ = 1`: at `n = 1` the minimization problem of §9.1 is degenerate, every
configuration being a global minimiser.
Source: arXiv:2312.10794v5, §9.1. -/
theorem discreteEnergy_singleton (β : ℝ) (x : SSphere d) :
    discreteEnergy d β {x} = Real.exp β := by
  have hx : ‖(x : EucSpace d)‖ = 1 := mem_sphere_zero_iff_norm.mp x.2
  have hxx : inner (𝕜 := ℝ) ((x : EucSpace d)) ((x : EucSpace d)) = 1 := by
    rw [real_inner_self_eq_norm_mul_norm, hx]; ring
  rw [discreteEnergy, Finset.sum_singleton, Finset.sum_singleton, hxx, mul_one]

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
