module

public import Transformer.Perspective.AppendixB_BetaInterval
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv

/-!
# The kernel `f_β` of self-attention on the circle

Polyanskiy, Rigollet, Yao — arXiv:2507.22857v1, *Synchronization of mean-field
models on the circle*, §5 (proof of `cor:conjecture`) and Appendices A–B.

On the circle, self-attention `eq:selfattention` is the mean-field model
`ẋ_i = -Σ_j f(x_i - x_j)` with `f(x) = sin x e^{β cos x}`. From §5 on the source
normalizes the kernel to

  `f_β(x) = sin x e^{β(cos x - 1)}`,

so that `f_β'(0) = 1`; the criterion of `thm:main` is unchanged by the positive
factor `e^{-β}`. This file has

* `f_β` and the three derivatives `f_β'`, `f_β''`, `f_β'''` that §5 writes out;
* their periodicity and parity;
* `f_β' = e^{-β} g_β`, with `g_β = g_β_2d β` the kernel of arXiv:2312.10794v5,
  Appendix B;
* `f_β' < 0` off `[-τ_β, τ_β]` on `𝕋`, where `τ_β = τ_β_star 2 β`
  `= arccos((√(1 + 4β²) - 1)/(2β))` is the `τ` that §5 sets.

The sign regions of `f_β'''` and the integral `∫ |f_β'''|₊` are in
`Perspective.SyncKernelQuartic` and `Perspective.SyncKernelRegions`.
-/

@[expose] public section

open Real

namespace Transformer
namespace Perspective

/-- **The kernel** `f_β(x) = sin x e^{β(cos x - 1)}`.

Source: arXiv:2507.22857v1, §5 and Appendices A–B. §2.3 writes
`f_β(x) = sin x e^{β cos x}`, which is `e^β` times this kernel. -/
noncomputable def f_β (β x : ℝ) : ℝ :=
  Real.sin x * Real.exp (β * (Real.cos x - 1))

/-- `f_β'(x) = (cos x - β sin² x) e^{β(cos x - 1)}`, as in §5. -/
noncomputable def f_β' (β x : ℝ) : ℝ :=
  (Real.cos x - β * Real.sin x ^ 2) * Real.exp (β * (Real.cos x - 1))

/-- `f_β''(x) = (-sin x - 3β sin x cos x + β² sin³ x) e^{β(cos x - 1)}`, as in §5. -/
noncomputable def f_β'' (β x : ℝ) : ℝ :=
  (-Real.sin x - 3 * β * Real.sin x * Real.cos x + β ^ 2 * Real.sin x ^ 3) *
    Real.exp (β * (Real.cos x - 1))

/-- `f_β'''(x) = -(cos x + 3β cos² x - 4β sin² x - 6β² cos x sin² x + β³ sin⁴ x)
e^{β(cos x - 1)}`, as in §5. -/
noncomputable def f_β''' (β x : ℝ) : ℝ :=
  -(Real.cos x + 3 * β * Real.cos x ^ 2 - 4 * β * Real.sin x ^ 2 -
      6 * β ^ 2 * Real.cos x * Real.sin x ^ 2 + β ^ 3 * Real.sin x ^ 4) *
    Real.exp (β * (Real.cos x - 1))

/-- The common factor `e^{β(cos x - 1)}` and its derivative `-β sin x e^{β(cos x - 1)}`. -/
theorem hasDerivAt_expCosSubOne (β x : ℝ) :
    HasDerivAt (fun y => Real.exp (β * (Real.cos y - 1)))
      (Real.exp (β * (Real.cos x - 1)) * (β * -Real.sin x)) x :=
  (((Real.hasDerivAt_cos x).sub_const 1).const_mul β).exp

theorem hasDerivAt_f_β (β x : ℝ) : HasDerivAt (f_β β) (f_β' β x) x := by
  refine ((Real.hasDerivAt_sin x).fun_mul (hasDerivAt_expCosSubOne β x)).congr_deriv ?_
  simp only [f_β']
  ring

theorem hasDerivAt_f_β' (β x : ℝ) : HasDerivAt (f_β' β) (f_β'' β x) x := by
  have hs := Real.hasDerivAt_sin x
  refine (((Real.hasDerivAt_cos x).fun_sub ((hs.fun_pow 2).const_mul β)).fun_mul
    (hasDerivAt_expCosSubOne β x)).congr_deriv ?_
  simp only [f_β'']
  push_cast
  ring

theorem hasDerivAt_f_β'' (β x : ℝ) : HasDerivAt (f_β'' β) (f_β''' β x) x := by
  have hs := Real.hasDerivAt_sin x
  have hc := Real.hasDerivAt_cos x
  have h := ((hs.fun_neg.fun_sub (((hs.const_mul 3).const_mul β).fun_mul hc)).fun_add
    ((hs.fun_pow 3).const_mul (β ^ 2))).fun_mul (hasDerivAt_expCosSubOne β x)
  convert h using 1
  · funext y
    simp only [f_β'']
    ring
  · simp only [f_β''']
    push_cast
    ring

theorem continuous_f_β''' (β : ℝ) : Continuous (f_β''' β) := by
  unfold f_β'''
  fun_prop

theorem f_β'_eq_exp_mul_g_β_2d (β x : ℝ) : f_β' β x = Real.exp (-β) * g_β_2d β x := by
  rw [f_β', g_β_2d, show β * (Real.cos x - 1) = β * Real.cos x + -β by ring, Real.exp_add]
  ring

@[simp] theorem f_β'_zero (β : ℝ) : f_β' β 0 = 1 := by
  simp [f_β']

@[simp] theorem f_β''_zero (β : ℝ) : f_β'' β 0 = 0 := by
  simp [f_β'']

@[simp] theorem f_β''_pi (β : ℝ) : f_β'' β π = 0 := by
  simp [f_β'']

theorem periodic_f_β (β : ℝ) : Function.Periodic (f_β β) (2 * π) := fun x => by
  simp [f_β]

theorem periodic_f_β' (β : ℝ) : Function.Periodic (f_β' β) (2 * π) := fun x => by
  simp [f_β']

theorem periodic_f_β''' (β : ℝ) : Function.Periodic (f_β''' β) (2 * π) := fun x => by
  simp [f_β''']

/-- `f_β''` is odd. -/
theorem f_β''_neg (β x : ℝ) : f_β'' β (-x) = -f_β'' β x := by
  simp only [f_β'', Real.sin_neg, Real.cos_neg]
  ring

/-- `f_β'''` is even. -/
theorem f_β'''_neg (β x : ℝ) : f_β''' β (-x) = f_β''' β x := by
  simp only [f_β''', Real.sin_neg, Real.cos_neg]
  ring

/-- `cos τ_β` solves `βc² + c - β = 0`: `cosTauStar_quadratic` at `d = 2`. -/
theorem cosTauStar_two_quadratic {β : ℝ} (hβ : 0 < β) :
    β * cosTauStar 2 β ^ 2 + cosTauStar 2 β - β = 0 := by
  have h := cosTauStar_quadratic 2 β hβ
  norm_num at h
  linarith

/-- `τ_β ∈ (0, π/2)`: `cos τ_β ∈ (0, 1)`, since `c = 1` does not solve
`βc² + c - β = 0`. -/
theorem τ_β_star_two_mem {β : ℝ} (hβ : 0 < β) : τ_β_star 2 β ∈ Set.Ioo 0 (π / 2) := by
  refine ⟨Real.arccos_pos.mpr (lt_of_le_of_ne (cosTauStar_le_one 2 β hβ (by norm_num))
    fun h => ?_), (τ_β_star_spec 2 β hβ (by norm_num)).1.2⟩
  have hq := cosTauStar_two_quadratic hβ
  rw [h] at hq
  linarith

/-- **`f_β' < 0` off `[-τ_β, τ_β]`.**  For `β > 0` and `τ_β = τ_β_star 2 β`
`= arccos((√(1 + 4β²) - 1)/(2β))`, `f_β'(u) < 0` for every `u ∈ (τ_β, 2π - τ_β)`,
that is for every `u ∉ [-τ_β, τ_β]` on `𝕋`.

With `c = cos u` and `c_β = cos τ_β`, `f_β'(u) e^{-β(cos u - 1)} = βc² + c - β
= (c - c_β)(β(c + c_β) + 1)`; the first factor is negative since `u` is farther
than `τ_β` from `0`, the second positive since `β(c_β - 1) + 1 = 1/(1 + c_β)`.

Source: arXiv:2507.22857v1, §5, proof of `cor:conjecture`, "it is clear that we
can set `τ = arccos((√(1+4β²) - 1)/(2β))`"; `thm:main` asks for `f'(x) < 0` for all
`x ∉ [-τ, τ]`. -/
theorem f_β'_neg {β : ℝ} (hβ : 0 < β) :
    ∀ u ∈ Set.Ioo (τ_β_star 2 β) (2 * π - τ_β_star 2 β), f_β' β u < 0 := by
  intro u hu
  have hq := cosTauStar_two_quadratic hβ
  set c := cosTauStar 2 β with hc
  have hc0 : 0 < c := cosTauStar_pos 2 β hβ (by norm_num)
  have hτ : Real.cos (τ_β_star 2 β) = c :=
    Real.cos_arccos (by linarith) (cosTauStar_le_one 2 β hβ (by norm_num))
  have hτ0 : 0 ≤ τ_β_star 2 β := Real.arccos_nonneg _
  -- `cos u < cos τ_β`: on `(τ_β, π]` directly, on `(π, 2π - τ_β)` through `2π - u`
  have hcos : Real.cos u < c := by
    rw [← hτ]
    rcases le_or_gt u π with h | h
    · exact Real.cos_lt_cos_of_nonneg_of_le_pi hτ0 h hu.1
    · rw [← Real.cos_two_pi_sub u]
      exact Real.cos_lt_cos_of_nonneg_of_le_pi hτ0 (by linarith) (by linarith [hu.2])
  have h1 : (β * (c - 1) + 1) * (1 + c) = 1 := by linear_combination hq
  have h2 : 0 < β * (c - 1) + 1 := lt_of_not_ge fun h => by nlinarith
  have hfac : 0 < β * (Real.cos u + c) + 1 := by
    nlinarith [mul_nonneg hβ.le (show 0 ≤ Real.cos u + 1 by linarith [Real.neg_one_le_cos u])]
  have hsplit : Real.cos u - β * Real.sin u ^ 2 =
      (Real.cos u - c) * (β * (Real.cos u + c) + 1) := by
    rw [Real.sin_sq]
    linear_combination hq
  rw [f_β', hsplit]
  exact mul_neg_of_neg_of_pos (mul_neg_of_neg_of_pos (by linarith) hfac) (Real.exp_pos _)

/-- The hypothesis of `f_β'_neg` is satisfiable, and its interval is not empty:
it contains `π`, as `τ_β < π/2`. -/
example : f_β' 1 π < 0 :=
  f_β'_neg one_pos π ⟨by linarith [(τ_β_star_two_mem one_pos).2, Real.pi_pos],
    by linarith [(τ_β_star_two_mem one_pos).2, Real.pi_pos]⟩

end Perspective
end Transformer
