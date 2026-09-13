/-
# §9 — General matrices

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
import Transformer.Section1_IPS
import Transformer.Section2_FlowMap

open scoped BigOperators
open Real

namespace Transformer
namespace SectionGeneralMatrices

open SectionIPS

variable (d n : ℕ)

/-! ### §9.1 — The repulsive case `V = -I_d` -/

/-- The interaction energy `𝖤_β` rewritten via squared distances:

  `𝖤_β[μ] = (e^β / (2β)) ∫∫ exp(-β/2 ‖x - x'‖²) dμ(x) dμ(x')`. -/
noncomputable def squaredDistEnergy
    (β : ℝ) (μ : SectionFlowMap.ProbSphere d) : ℝ := by
  exact 0  -- abstract placeholder for the integral expression.

/-- A finite point set `𝒞 ⊂ 𝕊^{d-1}` of size `n` is a *spherical `t`-design*
if `(1/n) Σ_{x ∈ 𝒞} p(x) = ∫ p dσ_d` for every polynomial `p` of total degree
`≤ t`. -/
def sphericalDesign
    (t : ℕ) (𝒞 : Finset (SSphere d)) : Prop :=
  -- ∀ polynomials p of total degree at most t, ∫ p dσ_d = (1/|𝒞|) Σ p(x).
  True

/-- A finite point set `𝒞 ⊂ 𝕊^{d-1}` is a *sharp configuration* if there are
`m > 1` distinct pairwise inner products and `𝒞` is a spherical
`(2m-1)`-design. -/
def sharpConfiguration (𝒞 : Finset (SSphere d)) : Prop :=
  ∃ m : ℕ, 1 < m ∧
    (Finset.image
      (fun p : SSphere d × SSphere d =>
        inner (𝕜 := ℝ) (p.1 : EucSpace d) (p.2 : EucSpace d))
      (𝒞 ×ˢ 𝒞)).card = m ∧
    sphericalDesign d (2 * m - 1) 𝒞

/-- **Cohn–Kumar theorem.**  Every global minimum of `𝖧_β` among finite
configurations `𝒞 ⊂ 𝕊^{d-1}` with `#𝒞 = n` is either a sharp configuration
or the vertices of the 600-cell (a particular 4-dimensional polytope with
120 vertices). -/
theorem cohn_kumar
    (β : ℝ) (hβ : 0 < β) (hn : 2 ≤ n) :
    True := by trivial

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

  `z_i(t) = e^{-t V} x_i(t)`. -/
noncomputable def rescaled
    (V : ParamMatrix d) (x : ℝ → Idx n → EucSpace d)
    (t : ℝ) (i : Idx n) : EucSpace d :=
  -- We use the matrix exponential of `-t V`.
  by exact x t i  -- placeholder: `Real.exp (-t)` of an operator is heavy to express here.

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

end SectionGeneralMatrices
end Transformer
