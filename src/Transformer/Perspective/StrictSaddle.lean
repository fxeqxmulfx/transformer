module

public import Transformer.Perspective.CircleCritical
public import Transformer.Perspective.CircleSync
public import Transformer.Perspective.SyncKernelBetaOne
public import Transformer.Perspective.SyncKernelSmallBeta

/-!
# Open Problem `o:strictsaddle` — the critical points of `𝖤_β` on `𝕋^n`

Geshkovski, Letrouit, Polyanskiy, Rigollet — arXiv:2312.10794v5,
*A mathematical perspective on Transformers*.

§7 asks whether, apart from the global maxima, every critical point of the
toroidal energy `𝖤_β` is a strict saddle — a point at which some direction
sees a strictly positive second derivative, so that gradient ascent escapes.

The survey leaves it open. Polyanskiy, Rigollet and Yao answer it for every
`β > 0` (arXiv:2507.22857v1, §5, proof of `cor:conjecture`, which covers
`β ≥ -0.16`), and `strict_saddle` follows their proof:

* a critical point is stationary (C1) for the kernel
  `f_β(x) = sin x e^{β(cos x - 1)}` (`sum_sin_mul_exp_eq_zero_of_isCritical`),
  and one that is not a strict saddle is cut-stable (C2) for every `S`
  (`cutStable_of_torusHessianNonPos`);
* `f_β` meets the criterion of `thm:main` with `τ = τ_β`, since
  `τ_β ∫_{-π}^{π} |f_β'''|₊ < 4 = 4 f_β'(0)`
  (`τ_mul_integral_posPart_f_β'''_lt_four` for `β ≥ 1`,
  `τ_mul_integral_posPart_f_β'''_lt_four_of_lt_one` for `0 < β < 1`);
* so `thm:main` (`synchronized_of_integral_le`) synchronizes it, and
  synchronized angles are global maxima
  (`isGlobalMaxTorusEnergy_of_synchronized`).

The vocabulary is Appendix B's: `Perspective.SecondDerivTorusEnergy` is the
second derivative along a straight line of `𝕋^n`, and
`Perspective.TorusHessianNonPos` its non-positivity, which is what a critical
point that is *not* a strict saddle satisfies. The predicates of the
statement are in `Perspective.CircleCritical`.
-/

@[expose] public section

open scoped BigOperators
open Real

namespace Transformer
namespace Perspective

variable (n : ℕ)

/-- **Open Problem (o:strictsaddle), answered.** *Non-maximal critical points
are strict saddles.*

For `β > 0`, every critical point of `𝖤_β` on `𝕋^n` that is not a global
maximum is a strict saddle: some straight line through it sees a strictly
positive second derivative of `𝖤_β`.

The proof is by contradiction. A critical point that is not a strict saddle
is stationary (C1) and cut-stable (C2) for the kernel `f_β`. `f_β` meets the
criterion of `thm:main` with `τ = τ_β`, so the point is synchronized, and
hence a global maximum.

The two-particle antipodal configuration is the picture: at `n = 2`,
`θ = (0, π)` is critical and a global minimum, and moving one particle gives
the second derivative `e^{-β}/4 > 0`.

Source: arXiv:2312.10794v5, §7, `o:strictsaddle`. The survey asks it for
`𝖤_β` with `β > 0` (§2, and §7 after `eq:onangles`), with no bound on `n`,
and leaves it open. The proof is that of arXiv:2507.22857v1, §5,
`cor:conjecture`, for `β > 0`. It uses `thm:main` (§2), the (C2) of §4 for
every `S`, and the bound `τ_β ∫ |f_β'''|₊ < 4` of §5 and Appendices A–B,
proved here separately for `β ≥ 1` and for `0 < β < 1`. -/
theorem strict_saddle (β : ℝ) (hβ : 0 < β) :
    ∀ θ : Angles n, IsCriticalTorusEnergy n β θ →
      ¬ IsGlobalMaxTorusEnergy n β θ → IsStrictSaddleTorusEnergy n β θ := by
  intro θ hcrit hmax
  by_contra hns
  have hτ := τ_β_star_two_mem hβ
  have hlt : τ_β_star 2 β * ∫ u in -π..π, max (f_β''' β u) 0 < 4 := by
    rcases lt_or_ge β 1 with h | h
    · exact τ_mul_integral_posPart_f_β'''_lt_four_of_lt_one hβ h
    · exact τ_mul_integral_posPart_f_β'''_lt_four h
  have hint : τ_β_star 2 β * ∫ u in -π..π, max (f_β''' β u) 0 ≤
      4 * (1 + τ_β_star 2 β / (2 * π)) * f_β' β 0 := by
    rw [f_β'_zero, mul_one]
    have : 0 ≤ τ_β_star 2 β / (2 * π) := div_nonneg hτ.1.le (by positivity)
    linarith
  exact hmax (isGlobalMaxTorusEnergy_of_synchronized n β hβ θ
    (synchronized_of_integral_le (periodic_f_β β) (hasDerivAt_f_β β) (hasDerivAt_f_β' β)
      (hasDerivAt_f_β'' β) (continuous_f_β''' β) hτ.1 (f_β'_neg hβ) (by simp) hint
      (sum_sin_mul_exp_eq_zero_of_isCritical n β hβ.ne' θ hcrit)
      (cutStable_of_torusHessianNonPos n β hβ.ne' θ
        (torusHessianNonPos_of_not_isStrictSaddle n β θ hns))))

/-- The hypotheses of `strict_saddle` hold at `β = 1`, `n = 2`, `θ = (0, π)`:
every `sin(θ_j - θ_i)` vanishes, so `θ` is critical, and
`𝖤_1(θ) = (e + e⁻¹)/4 < e/2 = 𝖤_1(0)`. -/
example : (0 : ℝ) < 1 ∧ IsCriticalTorusEnergy 2 1 ![0, π] ∧
    ¬ IsGlobalMaxTorusEnergy 2 1 ![0, π] := by
  refine ⟨one_pos, fun i => (hasDerivAt_torusEnergy 2 1 one_ne_zero _ i).congr_deriv ?_,
    fun h => ?_⟩
  · fin_cases i <;> simp [Fin.sum_univ_two]
  · have h0 := h fun _ => 0
    simp only [torusEnergy, Fin.sum_univ_two] at h0
    norm_num at h0
    linarith [Real.exp_lt_exp.2 (show (-1 : ℝ) < 1 by norm_num)]

end Perspective
end Transformer
