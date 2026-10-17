/-
# Critical points of `𝖤_β` on the circle: stationarity, cut-stability, maxima

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5, §7 and
Appendix B; Polyanskiy, Rigollet, Yao — arXiv:2507.22857v1, §2–§5.

The predicates of Open Problem `o:strictsaddle` (`Perspective.strict_saddle`),
and its reduction to the system `ẋ_i = -Σ_j f(x_i - x_j)` (S1) of
arXiv:2507.22857v1 with the kernel of its §5, `f(x) = sin x · e^{β (cos x - 1)}`,
`f'(x) = (cos x - β sin² x) e^{β (cos x - 1)}` (`hasDerivAt_sin_mul_exp`).
A critical point of `𝖤_β` is stationary (C1)
(`sum_sin_mul_exp_eq_zero_of_isCritical`).  A point that is not a strict
saddle has no positive second derivative
(`torusHessianNonPos_of_not_isStrictSaddle`), so it is cut-stable (C2) for
every set `S` (`cutStable_of_torusHessianNonPos`), and a negative cut sum
makes a strict saddle (`isStrictSaddleTorusEnergy_of_cut_neg`).  For `β > 0`,
synchronized angles are global maxima (`isGlobalMaxTorusEnergy_of_synchronized`).
-/

import Transformer.Perspective.AppendixB_Taylor

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable (n : ℕ)

/-- *A critical point of `𝖤_β` on `𝕋^n`*: every partial derivative vanishes.

The partial derivative is the one `hasDerivAt_torusEnergy` computes, so this
says `Σ_j e^{β cos(θ_j - θ_i)} sin(θ_j - θ_i) = 0` for every `i`, written as
a `HasDerivAt` so that no `deriv` of a non-differentiable function can slip
in.

Source: arXiv:2312.10794v5, §7, `o:strictsaddle`. -/
def IsCriticalTorusEnergy (β : ℝ) (θ : Angles n) : Prop :=
  ∀ i : Idx n,
    HasDerivAt (fun x : ℝ => torusEnergy n β (Function.update θ i x)) 0 (θ i)

/-- *A global maximum of `𝖤_β` on `𝕋^n`.*  The dynamics `eq:onangles` is a
gradient **ascent** (`angularUSA_is_gradient_flow`), so the maxima are the
attractors and the open problem is about everything else. -/
def IsGlobalMaxTorusEnergy (β : ℝ) (θ : Angles n) : Prop :=
  ∀ φ : Angles n, torusEnergy n β φ ≤ torusEnergy n β θ

/-- *A strict saddle of `𝖤_β`*: some straight line through `θ` sees a
strictly positive second derivative, so the ascent leaves along it.

This is the negation of `TorusHessianNonPos` sharpened from `¬ (c ≤ 0)` to an
exhibited direction, which is what "strict" asks for. -/
def IsStrictSaddleTorusEnergy (β : ℝ) (θ : Angles n) : Prop :=
  ∃ (v : Angles n) (c : ℝ), SecondDerivTorusEnergy n β θ v c ∧ 0 < c

/-- A strict saddle has no non-positive Hessian, which is the form Appendix B
uses its hypothesis in: `taylor2_inequality` and `taylor3_inequality` say
nothing at a strict saddle. -/
theorem not_torusHessianNonPos_of_strictSaddle (β : ℝ) (θ : Angles n)
    (h : IsStrictSaddleTorusEnergy n β θ) : ¬ TorusHessianNonPos n β θ := by
  obtain ⟨v, c, hc, hcpos⟩ := h
  exact fun hhess => absurd (hhess v c hc) (not_le.mpr hcpos)

/-- **A point that is not a strict saddle has no positive second
derivative** of `𝖤_β` along any straight line. -/
theorem torusHessianNonPos_of_not_isStrictSaddle (β : ℝ) (θ : Angles n)
    (h : ¬ IsStrictSaddleTorusEnergy n β θ) : TorusHessianNonPos n β θ := fun v c hc =>
  le_of_not_gt fun hpos => h ⟨v, c, hc, hpos⟩

/-- The kernel `f(x) = sin x · e^{β (cos x - 1)}` of §5 of arXiv:2507.22857v1
has the derivative `f'(x) = (cos x - β sin² x) e^{β (cos x - 1)}`, as there. -/
theorem hasDerivAt_sin_mul_exp (β x : ℝ) :
    HasDerivAt (fun y => sin y * exp (β * (cos y - 1)))
      ((cos x - β * sin x ^ 2) * exp (β * (cos x - 1))) x := by
  have h := (hasDerivAt_sin x).mul (((hasDerivAt_cos x).sub_const 1).const_mul β).exp
  convert h using 1
  ring

/-- **Stationarity (C1).**  At a critical point of `𝖤_β`, `β ≠ 0`,

  `Σ_j f(θ_i - θ_j) = 0` for every `i`, with `f(x) = sin x · e^{β (cos x - 1)}`:

the partial derivative `n⁻² Σ_j e^{β cos(θ_j - θ_i)} sin(θ_j - θ_i)` of
`hasDerivAt_torusEnergy` is `-n⁻² e^β Σ_j f(θ_i - θ_j)`.

Source: arXiv:2507.22857v1, §3, (C1), for the kernel `f` that §5 takes for
(T1): `e^{-β}` times the kernel `sin x · e^{β cos x}` of (T1), whose dynamics
is `n` times `eq:onangles` of arXiv:2312.10794v5, the gradient ascent on `𝖤_β`
(`angularUSA_is_gradient_flow`).  Positive factors do not change (C1). -/
theorem sum_sin_mul_exp_eq_zero_of_isCritical (β : ℝ) (hβ : β ≠ 0) (θ : Angles n)
    (h : IsCriticalTorusEnergy n β θ) (i : Idx n) :
    ∑ j : Idx n, sin (θ i - θ j) * exp (β * (cos (θ i - θ j) - 1)) = 0 := by
  have hn : ((n : ℝ) ^ 2)⁻¹ ≠ 0 := inv_ne_zero (pow_ne_zero 2 (Nat.cast_ne_zero.2 i.pos.ne'))
  have h0 := (h i).unique (hasDerivAt_torusEnergy n β hβ θ i)
  rw [one_div] at h0
  have hsum : ∑ j : Idx n, exp (β * cos (θ j - θ i)) * sin (θ j - θ i) = 0 :=
    (mul_eq_zero.1 h0.symm).resolve_left hn
  have hterm : ∀ j : Idx n, sin (θ i - θ j) * exp (β * (cos (θ i - θ j) - 1)) =
      -exp (-β) * (exp (β * cos (θ j - θ i)) * sin (θ j - θ i)) := fun j => by
    rw [show θ j - θ i = -(θ i - θ j) by ring, cos_neg, sin_neg, mul_sub, mul_one, exp_sub,
      exp_neg]
    ring
  rw [Finset.sum_congr rfl fun j _ => hterm j, ← Finset.mul_sum, hsum, mul_zero]

/-- **Cut-stability (C2) for every `S`.**  At a point with no positive second
derivative, `β ≠ 0`,

  `Σ_{i ∈ S} Σ_{j ∉ S} f'(θ_i - θ_j) ≥ 0` for every `S ⊆ [n]`,

with `f'(x) = (cos x - β sin² x) e^{β (cos x - 1)}`: this is `e^{-β}` times the
sum of `g_β` that `taylor3_inequality` bounds.

Source: arXiv:2507.22857v1, §3, (C2), and §4, `remark:negative_semidef`, for
the kernel of §5.  Changed: (C2) asks only for sets `S` of coincident
particles; here it holds for every `S`, which is what the proof of
`lemma:twoconsecutive_general` uses, and what `remark:negative_semidef`
gives with `c_i = 1_{i ∈ S}`.  A non-positive Hessian along every straight
line (`TorusHessianNonPos`) stands for the negative semi-definite Hessian. -/
theorem cutStable_of_torusHessianNonPos (β : ℝ) (hβ : β ≠ 0) (θ : Angles n)
    (h : TorusHessianNonPos n β θ) (S : Finset (Idx n)) :
    0 ≤ ∑ i ∈ S, ∑ j ∈ Sᶜ,
      (cos (θ i - θ j) - β * sin (θ i - θ j) ^ 2) * exp (β * (cos (θ i - θ j) - 1)) := by
  rcases Nat.eq_zero_or_pos n with rfl | hn
  · rw [Finset.eq_empty_of_isEmpty S, Finset.sum_empty]
  have hg := taylor3_inequality n β hβ (Nat.cast_ne_zero.2 hn.ne') θ S h
  have hterm : ∀ i j : Idx n,
      (cos (θ i - θ j) - β * sin (θ i - θ j) ^ 2) * exp (β * (cos (θ i - θ j) - 1)) =
        exp (-β) * g_β_2d β (θ i - θ j) := fun i j => by
    rw [g_β_2d, mul_sub β, mul_one, exp_sub, exp_neg]
    ring
  simp_rw [hterm, ← Finset.mul_sum]
  exact mul_nonneg (exp_pos _).le hg

/-- **A negative cut sum makes a strict saddle**: if, for `β ≠ 0`, some
`S ⊆ [n]` has `Σ_{i ∈ S} Σ_{j ∉ S} f'(θ_i - θ_j) < 0`, then `θ` is a strict
saddle of `𝖤_β`: along `1_S` the second derivative is `-e^β n⁻²` times the
cut sum (`secondDeriv_torusEnergy_block`). -/
theorem isStrictSaddleTorusEnergy_of_cut_neg (β : ℝ) (hβ : β ≠ 0) (θ : Angles n)
    (S : Finset (Idx n)) (hS : ∑ i ∈ S, ∑ j ∈ Sᶜ,
      (cos (θ i - θ j) - β * sin (θ i - θ j) ^ 2) * exp (β * (cos (θ i - θ j) - 1)) < 0) :
    IsStrictSaddleTorusEnergy n β θ := by
  by_contra hns
  exact absurd (cutStable_of_torusHessianNonPos n β hβ θ
    (torusHessianNonPos_of_not_isStrictSaddle n β θ hns) S) (not_le.mpr hS)

/-- **Synchronized angles are global maxima** of `𝖤_β` for `β > 0`: if every
`θ_i - θ_j` lies in `2πℤ`, every term `e^{β cos(φ_i - φ_j)}` of `𝖤_β(φ)` is at
most `e^β`, its value at `θ`.

Source: arXiv:2507.22857v1, §2.3, after `eq:energy_sad`, a potential of which
`𝖤_β` is `1/(2n²)` times: "the global optimizer is clearly the synchronized
configuration", `x_1 = ⋯ = x_n` on `𝕋 = ℝ/2πℤ`, the limit in
`def:synchronization`. -/
theorem isGlobalMaxTorusEnergy_of_synchronized (β : ℝ) (hβ : 0 < β) (θ : Angles n)
    (hθ : ∀ i j, ∃ k : ℤ, θ i - θ j = k * (2 * π)) : IsGlobalMaxTorusEnergy n β θ := by
  intro φ
  have hcos : ∀ i j, cos (θ i - θ j) = 1 := fun i j =>
    (hθ i j).elim fun k hk => by rw [hk, cos_int_mul_two_pi]
  simp only [torusEnergy]
  refine mul_le_mul_of_nonneg_left (Finset.sum_le_sum fun i _ => Finset.sum_le_sum fun j _ => ?_)
    (by positivity)
  rw [hcos i j, mul_one]
  exact exp_le_exp.2 (mul_le_of_le_one_right hβ.le (cos_le_one _))

/-- The hypotheses of `sum_sin_mul_exp_eq_zero_of_isCritical` and
`isGlobalMaxTorusEnergy_of_synchronized` hold at `θ = 0` for every `n` and
`β = 1`: it is critical, and every difference `θ_i - θ_j` is `0 · 2π`. -/
example (n : ℕ) : (1 : ℝ) ≠ 0 ∧ (0 : ℝ) < 1 ∧ IsCriticalTorusEnergy n 1 (fun _ => 0) ∧
    ∀ i j : Idx n, ∃ k : ℤ, (fun _ => (0 : ℝ)) i - (fun _ => (0 : ℝ)) j = k * (2 * π) := by
  refine ⟨one_ne_zero, one_pos, fun i => ?_, fun _ _ => ⟨0, by simp⟩⟩
  simpa using hasDerivAt_torusEnergy n 1 one_ne_zero (fun _ => 0) i

/-- The hypotheses of `torusHessianNonPos_of_not_isStrictSaddle` and
`cutStable_of_torusHessianNonPos` hold for one particle, `β = 1`, `θ = 0`:
`𝖤_1` is constant on `𝕋¹`, so every second derivative vanishes. -/
example : (1 : ℝ) ≠ 0 ∧ TorusHessianNonPos 1 1 (fun _ => 0) ∧
    ¬ IsStrictSaddleTorusEnergy 1 1 (fun _ => 0) := by
  have h : TorusHessianNonPos 1 1 (fun _ => 0) := by
    rintro v c ⟨f', hf', hc⟩
    have hconst : (fun u : ℝ => torusEnergy 1 1 (fun i => (0 : ℝ) + u * v i))
        = fun _ : ℝ => torusEnergy 1 1 (fun _ : Idx 1 => (0 : ℝ)) := by
      funext u
      simp [torusEnergy]
    have hf0 : f' = fun _ : ℝ => (0 : ℝ) := by
      funext s
      exact (hf' s).unique (by rw [hconst]; exact hasDerivAt_const s _)
    rw [hf0] at hc
    exact (hc.unique (hasDerivAt_const (0 : ℝ) (0 : ℝ))).le
  exact ⟨one_ne_zero, h, fun hs => not_torusHessianNonPos_of_strictSaddle 1 1 _ hs h⟩

/-- The hypotheses of `isStrictSaddleTorusEnergy_of_cut_neg` hold at `β = 1`,
`θ = (0, π)`, `S = {0}`, where the cut sum is `f'(-π) = -e^{-2} < 0`; the strict
saddle it gives is the hypothesis of `not_torusHessianNonPos_of_strictSaddle`. -/
example : IsStrictSaddleTorusEnergy 2 1 ![0, π] := by
  refine isStrictSaddleTorusEnergy_of_cut_neg 2 1 one_ne_zero _ {0} ?_
  rw [show ({0} : Finset (Idx 2))ᶜ = {1} by decide, Finset.sum_singleton, Finset.sum_singleton]
  norm_num [exp_pos]

end Perspective
end Transformer
