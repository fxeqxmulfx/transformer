/-
# Perceptrons and attention's mean-field landscape — the interaction kernel

Formalization of `lem: concavity` of arXiv:2601.21366v2: the angular kernel
`K_β(θ) = e^{β cos θ}` on the circle, its second derivative, the window
`(-θ_c(β), θ_c(β))` on which it is strictly concave, and the two Hessian
bounds the proof of `thm: bound` runs on.

**What the source says and what is carried here.**

* `θ_c(β) = arccos((√(1+4β²) - 1)/(2β))` is the source's definition, and
  `cos_thetaC` checks that the argument is in `[-1,1]`, so that `arccos` is
  not being read at a junk value.  The source says `θ_c(β) ∈ (-π, π]`; it is
  in fact in `(0, π/2)`, which is `thetaC_pos` and `thetaC_lt_pi_div_two`.

* "`K_β` is strictly concave on `(-θ_c, θ_c)`" is `StrictConcaveOn`, proved
  from `K_β'' < 0` there — which is `kernelK2_neg`, the statement that
  `β cos²θ + cos θ - β > 0` strictly inside the window.

* `θ_c(β) = β^{-1/2} + O(β^{-3/2})` is carried with the constant and the
  threshold quantified first, as everywhere in this formalization.

* `sup_{|θ| ≤ λθ_c} K_β''(θ)` and `max_{θ ∈ [θ_c, π]} K_β''(θ)` are carried as
  bounds valid at every such `θ`: a supremum bounded above is exactly that,
  and no junk value of `sSup` on an unbounded family can make the statement
  true by accident.

Source: arXiv:2601.21366v2, `lem: concavity`, `eq: K''`, `eq: theta_c`.
-/

import Mathlib.Analysis.SpecialFunctions.Trigonometric.Inverse
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv
import Mathlib.Analysis.Convex.Deriv

open Real Filter
open scoped Topology

namespace Transformer
namespace Perceptron

/-! ### The kernel and its second derivative -/

/-- **`K_β(θ) = e^{β cos θ}`**, the interaction kernel of `E_β` read in the
angle between two points of `𝕊^1`.

Source: arXiv:2601.21366v2, `lem: concavity`. -/
noncomputable def kernelK (β θ : ℝ) : ℝ := Real.exp (β * Real.cos θ)

/-- **`eq: K''`.**  `K_β''(θ) = e^{β cos θ}(β² sin²θ - β cos θ)`.

Source: arXiv:2601.21366v2, `eq: K''`. -/
noncomputable def kernelK2 (β θ : ℝ) : ℝ :=
  Real.exp (β * Real.cos θ) * (β ^ 2 * Real.sin θ ^ 2 - β * Real.cos θ)

theorem hasDerivAt_kernelK (β θ : ℝ) :
    HasDerivAt (kernelK β) (-(β * Real.sin θ) * Real.exp (β * Real.cos θ)) θ := by
  have h := ((Real.hasDerivAt_cos θ).const_mul β).exp
  rw [show kernelK β = fun t : ℝ => Real.exp (β * Real.cos t) from rfl]
  convert h using 1
  ring

theorem deriv_kernelK (β : ℝ) :
    deriv (kernelK β) = fun θ => -(β * Real.sin θ) * Real.exp (β * Real.cos θ) :=
  funext fun θ => (hasDerivAt_kernelK β θ).deriv

theorem hasDerivAt_deriv_kernelK (β θ : ℝ) :
    HasDerivAt (deriv (kernelK β)) (kernelK2 β θ) θ := by
  rw [deriv_kernelK]
  have hu : HasDerivAt (fun t : ℝ => -(β * Real.sin t)) (-(β * Real.cos θ)) θ :=
    ((Real.hasDerivAt_sin θ).const_mul β).neg
  have hv : HasDerivAt (fun t : ℝ => Real.exp (β * Real.cos t))
      (Real.exp (β * Real.cos θ) * (β * -Real.sin θ)) θ :=
    ((Real.hasDerivAt_cos θ).const_mul β).exp
  have := hu.mul hv
  convert this using 1
  rw [kernelK2]
  ring

/-- **`eq: K''` itself**: the second derivative of the kernel. -/
theorem deriv2_kernelK (β θ : ℝ) : deriv^[2] (kernelK β) θ = kernelK2 β θ :=
  (hasDerivAt_deriv_kernelK β θ).deriv

/-! ### The concavity window -/

/-- **`eq: theta_c`.**  `θ_c(β) = arccos((√(1+4β²) - 1)/(2β))`.

Source: arXiv:2601.21366v2, `eq: theta_c`. -/
noncomputable def thetaC (β : ℝ) : ℝ :=
  Real.arccos ((Real.sqrt (1 + 4 * β ^ 2) - 1) / (2 * β))

/-- The argument of `arccos` in `eq: theta_c` lies in `(0, 1)`. -/
theorem cosArg_mem_Ioo (β : ℝ) (hβ : 0 < β) :
    (Real.sqrt (1 + 4 * β ^ 2) - 1) / (2 * β) ∈ Set.Ioo (0 : ℝ) 1 := by
  have hs0 : 0 ≤ Real.sqrt (1 + 4 * β ^ 2) := Real.sqrt_nonneg _
  have hs : Real.sqrt (1 + 4 * β ^ 2) ^ 2 = 1 + 4 * β ^ 2 := Real.sq_sqrt (by positivity)
  have h1 : (1 : ℝ) < Real.sqrt (1 + 4 * β ^ 2) := by nlinarith
  have h2 : Real.sqrt (1 + 4 * β ^ 2) < 1 + 2 * β := by nlinarith
  constructor
  · exact div_pos (by linarith) (by linarith)
  · rw [div_lt_one (by linarith)]
    linarith

/-- `cos θ_c(β)` is the positive root `(√(1+4β²)-1)/(2β)` of
`β cos²θ + cos θ - β = 0`, as `eq:critical-root` says. -/
theorem cos_thetaC (β : ℝ) (hβ : 0 < β) :
    Real.cos (thetaC β) = (Real.sqrt (1 + 4 * β ^ 2) - 1) / (2 * β) := by
  obtain ⟨h0, h1⟩ := cosArg_mem_Ioo β hβ
  exact Real.cos_arccos (by linarith) h1.le

theorem thetaC_pos (β : ℝ) (hβ : 0 < β) : 0 < thetaC β :=
  Real.arccos_pos.mpr (cosArg_mem_Ioo β hβ).2

theorem thetaC_lt_pi_div_two (β : ℝ) (hβ : 0 < β) : thetaC β < π / 2 :=
  Real.arccos_lt_pi_div_two.mpr (cosArg_mem_Ioo β hβ).1

theorem thetaC_le_pi (β : ℝ) : thetaC β ≤ π := Real.arccos_le_pi _

/-- `cos θ_c(β)` is a root of `β c² + c - β`. -/
theorem quadratic_cos_thetaC (β : ℝ) (hβ : 0 < β) :
    β * Real.cos (thetaC β) ^ 2 + Real.cos (thetaC β) - β = 0 := by
  have hs : Real.sqrt (1 + 4 * β ^ 2) ^ 2 = 1 + 4 * β ^ 2 :=
    Real.sq_sqrt (by positivity)
  rw [cos_thetaC β hβ]
  set s := Real.sqrt (1 + 4 * β ^ 2) with hsdef
  have hβ' : β ≠ 0 := ne_of_gt hβ
  field_simp
  nlinarith [hs]

/-- **The kernel is strictly concave inside the window**: `K_β''(θ) < 0`
whenever `|θ| < θ_c(β)`.

Source: arXiv:2601.21366v2, `lem: concavity`. -/
theorem kernelK2_neg (β : ℝ) (hβ : 0 < β) {θ : ℝ} (hθ : |θ| < thetaC β) :
    kernelK2 β θ < 0 := by
  have hcos : Real.cos (thetaC β) < Real.cos θ := by
    rw [← Real.cos_abs θ]
    exact Real.cos_lt_cos_of_nonneg_of_le_pi (abs_nonneg θ) (thetaC_le_pi β) hθ
  have hc0 : 0 < Real.cos (thetaC β) := by
    rw [cos_thetaC β hβ]; exact (cosArg_mem_Ioo β hβ).1
  have hroot := quadratic_cos_thetaC β hβ
  have hsum : (0 : ℝ) < β * (Real.cos θ + Real.cos (thetaC β)) + 1 := by
    have := mul_pos hβ (show (0 : ℝ) < Real.cos θ + Real.cos (thetaC β) by linarith)
    linarith
  have hkey : 0 < β * Real.cos θ ^ 2 + Real.cos θ - β := by
    nlinarith [mul_pos (sub_pos.mpr hcos) hsum]
  rw [kernelK2, Real.sin_sq]
  have : β ^ 2 * (1 - Real.cos θ ^ 2) - β * Real.cos θ < 0 := by
    nlinarith [mul_pos hβ hkey]
  exact mul_neg_of_pos_of_neg (Real.exp_pos _) this

/-- **Lemma (lem: concavity), the concavity statement.**  `K_β` is strictly
concave on `(-θ_c(β), θ_c(β))`.

Source: arXiv:2601.21366v2, `lem: concavity`. -/
theorem strictConcaveOn_kernelK (β : ℝ) (hβ : 0 < β) :
    StrictConcaveOn ℝ (Set.Ioo (-thetaC β) (thetaC β)) (kernelK β) := by
  refine strictConcaveOn_of_deriv2_neg (convex_Ioo _ _)
    ((Real.continuous_exp.comp (continuous_const.mul Real.continuous_cos)).continuousOn)
    fun θ hθ => ?_
  rw [isOpen_Ioo.interior_eq] at hθ
  rw [deriv2_kernelK]
  exact kernelK2_neg β hβ (abs_lt.mpr ⟨hθ.1, hθ.2⟩)

/-- The hypotheses of `kernelK2_neg` and `strictConcaveOn_kernelK` are jointly
satisfiable: `β = 1` and `θ = 0`, the centre of the window. -/
example : (0 : ℝ) < 1 ∧ |(0 : ℝ)| < thetaC 1 :=
  ⟨one_pos, by simpa using thetaC_pos 1 one_pos⟩

/-! ### The asymptotics of the window -/

/-- **Lemma (lem: concavity), the small-`β` limit.**  `θ_c(β) → π/2` as
`β → 0⁺`.

Source: arXiv:2601.21366v2, `lem: concavity`. -/
theorem tendsto_thetaC_nhdsWithin_zero :
    Tendsto thetaC (𝓝[>] (0 : ℝ)) (𝓝 (π / 2)) := by
  have heq : ∀ β ∈ Set.Ioi (0 : ℝ),
      (Real.sqrt (1 + 4 * β ^ 2) - 1) / (2 * β)
        = 2 * β / (Real.sqrt (1 + 4 * β ^ 2) + 1) := by
    intro β hβ
    have hβ' : (0 : ℝ) < β := hβ
    have hs : Real.sqrt (1 + 4 * β ^ 2) ^ 2 = 1 + 4 * β ^ 2 :=
      Real.sq_sqrt (by positivity)
    have hpos : 0 < Real.sqrt (1 + 4 * β ^ 2) + 1 := by positivity
    field_simp
    nlinarith [hs]
  have hden : Tendsto (fun β : ℝ => Real.sqrt (1 + 4 * β ^ 2) + 1) (𝓝[>] (0 : ℝ))
      (𝓝 2) := by
    have hc : Continuous fun β : ℝ => Real.sqrt (1 + 4 * β ^ 2) + 1 :=
      (Real.continuous_sqrt.comp (by fun_prop)).add continuous_const
    have h := (hc.tendsto (0 : ℝ)).mono_left (nhdsWithin_le_nhds (s := Set.Ioi (0 : ℝ)))
    norm_num at h
    exact h
  have hnum : Tendsto (fun β : ℝ => 2 * β) (𝓝[>] (0 : ℝ)) (𝓝 0) := by
    have hc : Continuous fun β : ℝ => 2 * β := by fun_prop
    simpa using (hc.tendsto (0 : ℝ)).mono_left (nhdsWithin_le_nhds (s := Set.Ioi (0 : ℝ)))
  have hfrac : Tendsto (fun β : ℝ => (Real.sqrt (1 + 4 * β ^ 2) - 1) / (2 * β))
      (𝓝[>] (0 : ℝ)) (𝓝 0) := by
    refine Tendsto.congr' ?_ (by simpa using hnum.div hden (by norm_num))
    filter_upwards [self_mem_nhdsWithin] with β hβ using (heq β hβ).symm
  rw [show thetaC = fun β : ℝ => Real.arccos ((Real.sqrt (1 + 4 * β ^ 2) - 1) / (2 * β))
    from rfl]
  simpa [thetaC, Real.arccos_zero] using hfrac.arccos

/-- **Lemma (lem: concavity), the large-`β` expansion.**
`θ_c(β) = β^{-1/2} + O(β^{-3/2})` as `β → ∞`, written with `β^{-1/2} = 1/√β`
and `β^{-3/2} = 1/(β√β)` so that no real power is needed.

Not proved here.

Source: arXiv:2601.21366v2, `lem: concavity`. -/
theorem thetaC_asymptotics :
    ∃ C β₀ : ℝ, 0 < C ∧ 0 < β₀ ∧ ∀ β : ℝ, β₀ ≤ β →
      |thetaC β - 1 / Real.sqrt β| ≤ C / (β * Real.sqrt β) := by
  sorry

/-! ### The two Hessian bounds -/

/-- **`eq:kernel-hess-bound-lambda`.**  For every `λ ∈ (0,1)` there is `β₀(λ)`
with `K_β''(θ) ≤ -e^{-λ²/2}((1-λ²)/2) β e^β` for every `β ≥ β₀(λ)` and every
`|θ| ≤ λ θ_c(β)`.

Not proved here.

Source: arXiv:2601.21366v2, `eq:kernel-hess-bound-lambda`. -/
theorem kernelK2_le_of_abs_le_lambda_thetaC (lam : ℝ) (hlam : lam ∈ Set.Ioo (0 : ℝ) 1) :
    ∃ β₀ : ℝ, 0 < β₀ ∧ ∀ β : ℝ, β₀ ≤ β → ∀ θ : ℝ, |θ| ≤ lam * thetaC β →
      kernelK2 β θ ≤ -Real.exp (-lam ^ 2 / 2) * ((1 - lam ^ 2) / 2) * β * Real.exp β := by
  sorry

/-- The hypothesis of `kernelK2_le_of_abs_le_lambda_thetaC` is satisfiable:
`λ = 1/2`. -/
example : (1 : ℝ) / 2 ∈ Set.Ioo (0 : ℝ) 1 := ⟨by norm_num, by norm_num⟩

/-- **`eq:kernel-hess-bound-max`.**  There is `β₁` with
`K_β''(θ) ≤ 2β e^{β - 3/2}` for every `β ≥ β₁` and every `θ ∈ [θ_c(β), π]`.

Not proved here.

Source: arXiv:2601.21366v2, `eq:kernel-hess-bound-max`. -/
theorem kernelK2_le_of_thetaC_le :
    ∃ β₁ : ℝ, 0 < β₁ ∧ ∀ β : ℝ, β₁ ≤ β → ∀ θ ∈ Set.Icc (thetaC β) π,
      kernelK2 β θ ≤ 2 * β * Real.exp (β - 3 / 2) := by
  sorry

end Perceptron
end Transformer
