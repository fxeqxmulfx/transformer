module

public import Transformer.Landscape.SaddleAvoidance.Orbit
public import Transformer.Landscape.SaddleAvoidance.Rates

/-!
# Saddle avoidance — rates along an orbit with a Hessian bound

The rates of `Landscape.SaddleAvoidance.Rates` under the condition a run can check
along its own orbit `x_k = (gradientStep f η)^[k] x`:

> (S) `‖∇²f(y)‖ ≤ L` at every `y` of every segment `[x_k, x_{k+1}]`, with `η L < 1`.

Let `f` be `C²`, `0 < η`, and let an orbit satisfying (S) converge to `w`, where the
Łojasiewicz inequality holds with exponent `a`. By `iterate_succ_le_sub_of_segment`
every step lowers `f` by at least `η (1 - η L) ‖∇f(x_k)‖²`, so the two estimates
hold (`exists_rate_estimates_of_isLojasiewiczAt_of_segment`), `f x_k - f w` and
`‖x_k - w‖` decay geometrically for `a ≤ 1/2`
(`exists_le_mul_pow_of_isLojasiewiczAt_of_segment`), and as `(k + 1) ^ (-1/(2a - 1))`
and `(k + 1) ^ (-(1 - a)/(2a - 1))` for `1/2 < a < 1`
(`exists_le_mul_rpow_of_isLojasiewiczAt_of_segment`). No global Lipschitz constant
of `∇f` is asked, unlike in `prop:rates` of arXiv:1602.04915.
-/

@[expose] public section

open Filter
open scoped Topology

namespace Transformer.Landscape.SaddleAvoidance

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [FiniteDimensional ℝ E]
  {f : E → ℝ} {L η a : ℝ}

/-- **The two estimates behind the rates under (S).**  Let `f` be `C²`, `0 < η`,
`η L < 1`, and let an orbit of gradient descent with `‖∇²f‖ ≤ L` on every segment
`[x_k, x_{k+1}]` converge to `w`, where the Łojasiewicz inequality holds with
`0 < a < 1`. Then `f` stays above `f w` along it, and for some `c > 0` and `K ≥ 0`,
from some step on, `r_k = f x_k - f w` satisfies `r_{k+1} ≤ r_k - c r_k ^ (2a)` and
`‖x_k - w‖ ≤ K r_k ^ (1 - a)`. -/
theorem exists_rate_estimates_of_isLojasiewiczAt_of_segment (hf : ContDiff ℝ 2 f)
    (hη : 0 < η) (hηL : η * L < 1) (ha0 : 0 < a) (ha1 : a < 1) {w x : E}
    (hw : IsLojasiewiczAt f w a)
    (hS : ∀ k, ∀ y ∈ segment ℝ ((gradientStep f η)^[k] x) ((gradientStep f η)^[k + 1] x),
      ‖fderiv ℝ (gradient f) y‖ ≤ L)
    (hx : Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w)) :
    (∀ k, f w ≤ f ((gradientStep f η)^[k] x)) ∧ ∃ c > 0, ∃ K ≥ 0, ∀ᶠ k in atTop,
      f ((gradientStep f η)^[k + 1] x) - f w ≤ f ((gradientStep f η)^[k] x) - f w -
          c * (f ((gradientStep f η)^[k] x) - f w) ^ (2 * a) ∧
        ‖(gradientStep f η)^[k] x - w‖ ≤
          K * (f ((gradientStep f η)^[k] x) - f w) ^ (1 - a) :=
  exists_rate_estimates_of_isLojasiewiczAt_of_le_sub hη (by linarith) ha0 ha1 hw
    hf.continuous.continuousAt (iterate_succ_le_sub_of_segment hf hS) hx

/-- **Geometric rate under (S)** (`0 < a ≤ 1/2`). Under the hypotheses of
`exists_rate_estimates_of_isLojasiewiczAt_of_segment`, with an exponent `a ≤ 1/2`
instead, for some `C` and `0 < b < 1`, `f x_k - f w ≤ C b ^ k` and
`‖x_k - w‖ ≤ C b ^ k` for every `k`. -/
theorem exists_le_mul_pow_of_isLojasiewiczAt_of_segment (hf : ContDiff ℝ 2 f)
    (hη : 0 < η) (hηL : η * L < 1) (ha : a ≤ 1 / 2) {w x : E} (hw : IsLojasiewiczAt f w a)
    (hS : ∀ k, ∀ y ∈ segment ℝ ((gradientStep f η)^[k] x) ((gradientStep f η)^[k + 1] x),
      ‖fderiv ℝ (gradient f) y‖ ≤ L)
    (hx : Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w)) :
    ∃ C b : ℝ, 0 < b ∧ b < 1 ∧ ∀ k : ℕ, f ((gradientStep f η)^[k] x) - f w ≤ C * b ^ k ∧
      ‖(gradientStep f η)^[k] x - w‖ ≤ C * b ^ k :=
  exists_le_mul_pow_of_isLojasiewiczAt_of_le_sub hη (by linarith) ha hw
    hf.continuous.continuousAt (iterate_succ_le_sub_of_segment hf hS) hx

/-- **Polynomial rate under (S)** (`1/2 < a < 1`). Under the hypotheses of
`exists_le_mul_pow_of_isLojasiewiczAt_of_segment`, with an exponent `1/2 < a < 1`
instead, for some `C` and every `k`, `f x_k - f w ≤ C (k + 1) ^ (-1/(2a - 1))` and
`‖x_k - w‖ ≤ C (k + 1) ^ (-(1 - a)/(2a - 1))`. -/
theorem exists_le_mul_rpow_of_isLojasiewiczAt_of_segment (hf : ContDiff ℝ 2 f)
    (hη : 0 < η) (hηL : η * L < 1) (ha : 1 / 2 < a) (ha1 : a < 1) {w x : E}
    (hw : IsLojasiewiczAt f w a)
    (hS : ∀ k, ∀ y ∈ segment ℝ ((gradientStep f η)^[k] x) ((gradientStep f η)^[k + 1] x),
      ‖fderiv ℝ (gradient f) y‖ ≤ L)
    (hx : Tendsto (fun k => (gradientStep f η)^[k] x) atTop (𝓝 w)) :
    ∃ C : ℝ, ∀ k : ℕ,
      f ((gradientStep f η)^[k] x) - f w ≤ C * ((k : ℝ) + 1) ^ (-(1 / (2 * a - 1))) ∧
        ‖(gradientStep f η)^[k] x - w‖ ≤ C * ((k : ℝ) + 1) ^ (-((1 - a) / (2 * a - 1))) :=
  exists_le_mul_rpow_of_isLojasiewiczAt_of_le_sub hη (by linarith) ha ha1 hw
    hf.continuous.continuousAt (iterate_succ_le_sub_of_segment hf hS) hx

/-- `‖x‖² / 2` on the line has the Hessian `1`, so (S) holds on every orbit with
`L = 1` and `η = 1/2`; the orbit `2⁻ᵏ` of `1` converges to `0`, where the
Łojasiewicz inequality holds with `a = 1/2`, and so with `a = 3/4`. -/
example := exists_rate_estimates_of_isLojasiewiczAt_of_segment
  (f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2) (L := 1) (η := 1 / 2)
  (contDiff_const.mul (contDiff_norm_sq ℝ)) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) isLojasiewiczAt_half_mul_norm_sq (x := 1)
  (fun _ y _ => by
    rw [gradient_half_mul_norm_sq, fderiv_id]
    exact ContinuousLinearMap.norm_id_le)
  (tendsto_iterate_gradientStep_half_mul_norm_sq (E := ℝ) (η := 1 / 2) (by norm_num) 1)

example := exists_le_mul_pow_of_isLojasiewiczAt_of_segment
  (f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2) (L := 1) (η := 1 / 2)
  (contDiff_const.mul (contDiff_norm_sq ℝ)) (by norm_num) (by norm_num) le_rfl
  isLojasiewiczAt_half_mul_norm_sq (x := 1)
  (fun _ y _ => by
    rw [gradient_half_mul_norm_sq, fderiv_id]
    exact ContinuousLinearMap.norm_id_le)
  (tendsto_iterate_gradientStep_half_mul_norm_sq (E := ℝ) (η := 1 / 2) (by norm_num) 1)

example := exists_le_mul_rpow_of_isLojasiewiczAt_of_segment
  (f := fun x : ℝ => (1 / 2 : ℝ) * ‖x‖ ^ 2) (L := 1) (η := 1 / 2) (a := 3 / 4)
  (contDiff_const.mul (contDiff_norm_sq ℝ)) (by norm_num) (by norm_num) (by norm_num)
  (by norm_num) (isLojasiewiczAt_half_mul_norm_sq.mono (by norm_num)) (x := 1)
  (fun _ y _ => by
    rw [gradient_half_mul_norm_sq, fderiv_id]
    exact ContinuousLinearMap.norm_id_le)
  (tendsto_iterate_gradientStep_half_mul_norm_sq (E := ℝ) (η := 1 / 2) (by norm_num) 1)

end Transformer.Landscape.SaddleAvoidance
