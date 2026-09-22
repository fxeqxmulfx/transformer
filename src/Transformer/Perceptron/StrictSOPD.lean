/-
# Perceptrons and attention's mean-field landscape — a sufficient condition for
  strictness

Formalization of `rem:strictSOPD-perceptron` of arXiv:2601.21366v2: on the
circle, enough angular convexity of the perceptron potential on `supp μ` makes
a stationary measure a *strict* SOPD critical point.

**What the source says and what is carried here.**

* `x(θ)` is `circlePoint θ`, the standard parametrization of `𝕊¹`; `x'(θ)` is
  `circleVel θ` and `x''(θ) = -x(θ)`, by `hasDerivAt_circleVel`.  The source's
  `∂_θ²(v_ϑ ∘ x)(θ)` is therefore literally
  `deriv^[2] (fun s => potential φ ω a (x s)) θ`, and no new definition is
  needed to state the condition.

* `eq:strictSOPD-sufficient` reads
  `∂_θ²(v_ϑ ∘ x)(θ) ≥ 2‖K_β''‖_∞/β + 2κ` on `supp μ`.  Here `K_β = e^{β cos ·}`
  and `‖K_β''‖_∞ = β e^β` — which is `isGreatest_abs_kernelK2`, proved — so the
  constant is `2e^β`, and that is the number the hypothesis below carries.

* The source's simplification of the condition under the `φ' = 2σ` convention
  of `eq: primitive.field` is `secondDeriv_potential_circlePoint` of
  `CircleDeriv.lean`, proved; `secondDeriv_ge_iff_simplified` turns it into
  the display the remark ends with,
  `Σ_j ω_j(σ'(a_j·x)(a_j·x')² + σ(a_j·x) a_j·x'') ≥ e^β + κ`.

* The remark's first sentence, "it is clear that the minimizer `μ⋆` is SOPD",
  is `isSOPD_of_isMin`.  The minimizer is quantified over inside the
  conclusion, as in `thm: any.d`; for `d ≥ 1` there is exactly one
  (`existsUnique_min_energy`, in `Minimizer`), so the statement is about `μ⋆`
  and nothing else.

* The remark's last paragraph — the analogue for `d ≥ 3`, with
  `λ_min(∇²v_ϑ)` in place of `∂_θ²(v_ϑ ∘ x)` and an unspecified constant
  `C_{β,d}` in place of `‖K_β''‖_∞` — is not carried: it names no constant one
  could state a theorem about.

Source: arXiv:2601.21366v2, `rem:strictSOPD-perceptron`, `eq:strictSOPD-sufficient`.
-/

import Transformer.Perceptron.KernelSup
import Transformer.Perceptron.CircleDeriv
import Transformer.Perceptron.Geodesic
import Transformer.Perceptron.MinStationary

open scoped BigOperators
open Real MeasureTheory

namespace Transformer
namespace Perceptron

/-! ### The minimizer -/

/-- **Remark (rem:strictSOPD-perceptron), first sentence.**  The global
minimizer of the coupled energy is a SOPD critical point: it satisfies
`eq: steady.state`, and along every `W₂`-geodesic issued from it the energy has
a global minimum at `t = 0`, so the second derivative there is non-negative.

The minimizer is quantified over inside the conclusion; see the module
docstring.  Note that `β > 0` is not decoration: at `β = 0` the prefactor
`(2β)⁻¹` annihilates the interaction energy while `energyGrad` keeps
`∫ e^{β x·y} Proj_x y dμ(y)`, and the statement is false.  The proof uses
only `β ≠ 0`.

The source's "it is clear" is carried in two halves.  Stationarity is
`isStationary_of_isMin` (`MinStationary`), which needs no continuity of `σ`,
unlike the source's `eq: steady.state`.  The second-order half is the source's
commented-out proof: `J(t) = E[ν t]` is minimal at `t = 0` and continuous there
(`continuous_energy`, `tendsto_geodesic`), so `J''(0) ≥ 0` whenever it exists
(`nonneg_of_hasDerivAt_deriv_of_isMin`).

Source: arXiv:2601.21366v2, `rem:strictSOPD-perceptron`, and the proof of
`prop: min.max`. -/
theorem isSOPD_of_isMin (d : ℕ) (β : ℝ) (hβ : 0 < β) (φ σ : ℝ → ℝ)
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s) (ω : Idx d → ℝ) (a : Idx d → EucSpace d) :
    ∀ μ : Perspective.ProbSphere d,
      (∀ ν : Perspective.ProbSphere d, energy β φ ω a μ ≤ energy β φ ω a ν) →
        IsSOPD β φ σ ω a μ := by
  intro μ hμ
  refine ⟨isStationary_of_isMin hβ.ne' hφ ω a hμ, fun ξ _ ν hν H hH => ?_⟩
  have h0 := geodesic_zero hν
  refine nonneg_of_hasDerivAt_deriv_of_isMin (J := fun t => energy β φ ω a (ν t))
    (fun t => ?_) ?_ hH
  · show energy β φ ω a (ν 0) ≤ energy β φ ω a (ν t)
    rw [h0]
    exact hμ (ν t)
  · show Filter.Tendsto (fun t => energy β φ ω a (ν t)) (nhds 0)
      (nhds (energy β φ ω a (ν 0)))
    rw [h0]
    exact ((continuous_energy β (continuous_of_hasDerivAt hφ) ω a).tendsto μ).comp
      (tendsto_geodesic hν)

/-- The hypotheses of `isSOPD_of_isMin` are satisfiable: `β = 1` and the
quadratic primitive `φ(s) = s²` of `2σ` for the linear `σ(s) = s`. -/
example : (0 : ℝ) < 1 ∧ ∀ s : ℝ, HasDerivAt (fun t : ℝ => t ^ 2) (2 * (fun s : ℝ => s) s) s :=
  ⟨one_pos, fun s => by simpa using hasDerivAt_pow 2 s⟩

/-! ### `eq:strictSOPD-sufficient` -/

/-- **`eq:strictSOPD-sufficient`.**  Let `d = 2` and `β > 0`, and let `μ` be
stationary.  If for some `κ > 0` the angular second derivative of the potential
is at least `2‖K_β''‖_∞/β + 2κ = 2e^β + 2κ` at every angle of `supp μ`, then
`μ` is a strict SOPD critical point.

The constant is `2e^β` because `‖K_β''‖_∞ = β e^β`, which is
`isGreatest_abs_kernelK2`.

Not proved here.

Source: arXiv:2601.21366v2, `eq:strictSOPD-sufficient`. -/
theorem isStrictSOPD_of_secondDeriv_ge (β : ℝ) (hβ : 0 < β) (φ σ σ' : ℝ → ℝ)
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s) (hσ : ∀ s : ℝ, HasDerivAt σ (σ' s) s)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (μ : Perspective.ProbSphere 2)
    (hμ : IsStationary β σ ω a μ) (κ : ℝ) (hκ : 0 < κ)
    (hge : ∀ θ : ℝ, circlePoint θ ∈ (μ : Measure (SSphere 2)).support →
      2 * Real.exp β + 2 * κ ≤
        deriv^[2] (fun s : ℝ => potential φ ω a (circlePoint s : EucSpace 2)) θ) :
    IsStrictSOPD β φ σ ω a μ := by
  sorry

/-- The hypotheses of `isStrictSOPD_of_secondDeriv_ge` are satisfiable: `β = 1`,
the quadratic primitive `φ(s) = s²` of `2σ` for the linear `σ(s) = s`, whose
derivative is `σ' ≡ 1`, the single neuron `ω_0 = -4` with `a_0 = -x` at
`x = basePoint 1`, and `μ = δ_x` — stationary because the drift at `x` is
radial.  Along the circle the potential is `v_ϑ(x(θ)) = -4cos²θ`, whose angular
second derivative is `8cos 2θ`, equal to `8` at the atom; and `2e + 2 < 8`
because `e < 3`, so `κ = 1` works. -/
example :
    (0 : ℝ) < 1 ∧ (∀ s : ℝ, HasDerivAt (fun t : ℝ => t ^ 2) (2 * s) s) ∧
      (∀ s : ℝ, HasDerivAt (fun t : ℝ => t) ((fun _ : ℝ => (1 : ℝ)) s) s) ∧
      IsStationary 1 (fun s : ℝ => s) (Pi.single 0 (-4 : ℝ))
        (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
        (Perspective.diracProb 2 (basePoint 1)) ∧
      (0 : ℝ) < 1 ∧
      ∀ θ : ℝ, circlePoint θ ∈
          (Perspective.diracProb 2 (basePoint 1) : Measure (SSphere 2)).support →
        2 * Real.exp 1 + 2 * 1 ≤
          deriv^[2] (fun s : ℝ => potential (fun t : ℝ => t ^ 2) (Pi.single 0 (-4 : ℝ))
            (Pi.single 0 (-((basePoint 1 : SSphere 2) : EucSpace 2)))
            (circlePoint s : EucSpace 2)) θ := by
  have hφ : ∀ s : ℝ, HasDerivAt (fun t : ℝ => t ^ 2) (2 * (fun s : ℝ => s) s) s :=
    fun s => by simpa using hasDerivAt_pow 2 s
  have hσ : ∀ s : ℝ, HasDerivAt (fun t : ℝ => t) ((fun _ : ℝ => (1 : ℝ)) s) s :=
    fun s => hasDerivAt_id' (𝕜 := ℝ) (x := s)
  have hu : inner (𝕜 := ℝ) ((basePoint 1 : SSphere 2) : EucSpace 2)
      ((basePoint 1 : SSphere 2) : EucSpace 2) = (1 : ℝ) := by
    rw [real_inner_self_eq_norm_sq, norm_basePoint_one]; norm_num
  have hw : inner (𝕜 := ℝ) secondAxis secondAxis = (1 : ℝ) := by
    rw [real_inner_self_eq_norm_sq, norm_secondAxis]; norm_num
  refine ⟨one_pos, hφ, hσ, ?_, one_pos, ?_⟩
  · refine isStationary_diracProb_of_radial 1 _ _ _ (basePoint 1) (-4) ?_
    rw [Fin.sum_univ_two]
    simp [inner_neg_left, ← neg_smul]
  · intro θ hθ
    have hx : circlePoint θ = basePoint 1 := Interpolation.eq_of_mem_support_dirac hθ
    have hcos : Real.cos θ = 1 := by
      have := inner_circlePoint ((basePoint 1 : SSphere 2) : EucSpace 2) θ
      rw [hx, hu, inner_basePoint_secondAxis] at this
      linarith
    have hwu : inner (𝕜 := ℝ) secondAxis ((basePoint 1 : SSphere 2) : EucSpace 2) = 0 := by
      rw [real_inner_comm]; exact inner_basePoint_secondAxis
    have hsin : Real.sin θ = 0 := by
      have := inner_circlePoint secondAxis θ
      rw [hx, hw, hwu] at this
      linarith
    have hvel : circleVel θ = secondAxis := by
      rw [circleVel, hsin, hcos, neg_zero, zero_smul, one_smul, zero_add]
    rw [secondDeriv_potential_circlePoint (fun t : ℝ => t ^ 2) (fun s : ℝ => s)
      (fun _ : ℝ => (1 : ℝ)) hφ hσ _ _ θ, hx, hvel, Fin.sum_univ_two]
    have h0 : inner (𝕜 := ℝ) (-((basePoint 1 : SSphere 2) : EucSpace 2)) secondAxis = 0 := by
      rw [inner_neg_left, inner_basePoint_secondAxis, neg_zero]
    have h1 : inner (𝕜 := ℝ) (-((basePoint 1 : SSphere 2) : EucSpace 2))
        ((basePoint 1 : SSphere 2) : EucSpace 2) = -1 := by
      rw [inner_neg_left, hu]
    simp only [Pi.single_eq_same, Pi.single_eq_of_ne (show (1 : Idx 2) ≠ 0 by decide),
      inner_zero_left, h0, h1]
    norm_num
    linarith [Real.exp_one_lt_three]

/-- The sufficient condition read in the source's own display: by
`secondDeriv_potential_circlePoint`, `eq:strictSOPD-sufficient` at an angle is
`Σ_j ω_j (σ'(a_j·x)(a_j·x')² + σ(a_j·x) a_j·x'') ≥ e^β + κ`. -/
theorem secondDeriv_ge_iff_simplified (β : ℝ) (φ σ σ' : ℝ → ℝ)
    (hφ : ∀ s : ℝ, HasDerivAt φ (2 * σ s) s) (hσ : ∀ s : ℝ, HasDerivAt σ (σ' s) s)
    (ω : Idx 2 → ℝ) (a : Idx 2 → EucSpace 2) (κ θ : ℝ) :
    2 * Real.exp β + 2 * κ ≤
        deriv^[2] (fun s : ℝ => potential φ ω a (circlePoint s : EucSpace 2)) θ ↔
      Real.exp β + κ ≤ ∑ j : Idx 2, ω j *
        (σ' (inner (𝕜 := ℝ) (a j) (circlePoint θ : EucSpace 2)) *
            inner (𝕜 := ℝ) (a j) (circleVel θ) ^ 2 +
          σ (inner (𝕜 := ℝ) (a j) (circlePoint θ : EucSpace 2)) *
            -inner (𝕜 := ℝ) (a j) (circlePoint θ : EucSpace 2)) := by
  rw [secondDeriv_potential_circlePoint φ σ σ' hφ hσ ω a θ]
  constructor <;> intro h <;> linarith

end Perceptron
end Transformer
