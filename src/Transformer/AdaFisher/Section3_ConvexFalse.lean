/-
# AdaFisher: the printed convex convergence claim is false

arXiv:2405.16397v3, §3.4, Proposition 3.3, Appendix A.2.
The source allows a fixed step `α ≤ 1/L`, independent of the smallest
Fisher eigenvalue. The quadratic `J(x)=x²/2` with damping 0.001 and α=1
has multiplier -999 and diverges. The example also refutes its 1/k bound.
-/

import Transformer.AdaFisher.Section3_Algorithm
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Convex.Strong
import Mathlib.Analysis.SpecificLimits.Basic

open Filter Topology

noncomputable section

namespace Transformer.AdaFisher

/-- Smooth strongly convex objective for Proposition 3.3's counterexample. -/
def quadratic (x : ℝ) : ℝ := x ^ 2 / 2

/-- Scalar eigendirection of the unmomented update in Proposition 3.3. -/
def quadraticRun (η f initial : ℝ) : ℕ → ℝ
  | 0 => initial
  | t + 1 => quadraticRun η f initial t - η * quadraticRun η f initial t / f

/-- This objective has the paper's actual gradient `∇J(x)=x`, Proposition 3.3. -/
theorem quadratic_hasDerivAt (x : ℝ) : HasDerivAt quadratic x x := by
  change HasDerivAt (fun y : ℝ => y ^ 2 / 2) x x
  convert ((hasDerivAt_id x).pow 2).div_const 2 using 1 <;> simp

/-- The exact quadratic expansion gives both strong convexity modulus one
and smoothness constant one, §3.4, Proposition 3.3. -/
theorem quadratic_exact_model (x y : ℝ) :
    quadratic y = quadratic x + x * (y - x) + (y - x) ^ 2 / 2 := by
  simp only [quadratic]
  ring

/-- The quadratic gradient is 1-Lipschitz, Proposition 3.3. -/
theorem quadratic_gradient_lipschitz (x y : ℝ) : |x - y| ≤ 1 * |x - y| := by
  rw [one_mul]

/-- Zero is the global minimizer in the counterexample to Proposition 3.3. -/
theorem quadratic_minimum (x : ℝ) : quadratic 0 ≤ quadratic x := by
  simp only [quadratic, zero_pow (by decide : 2 ≠ 0), zero_div]
  positivity

/-- The counterexample meets the source's strong convexity assumption
literally, §3.4, Proposition 3.3, with modulus one. -/
theorem quadratic_strongConvexOn : StrongConvexOn Set.univ 1 quadratic := by
  rw [strongConvexOn_iff_convex]
  convert convexOn_const (0 : ℝ) convex_univ using 1
  funext x
  simp [quadratic, Real.norm_eq_abs, sq_abs]
  ring

/-- The scalar iterate in a fixed Fisher eigendirection is geometric,
§3.4, Proposition 3.3. -/
theorem quadraticRun_eq (η f initial : ℝ) (t : ℕ) :
    quadraticRun η f initial t = (1 - η / f) ^ t * initial := by
  induction t with
  | zero => simp [quadraticRun]
  | succ t ih =>
    rw [quadraticRun, ih, pow_succ]
    ring

/-- A nonconstant min-max factor has the small eigenvalue 0.001 used in
the counterexample, Proposition 3.2 and Proposition 3.3. Thus the example
does not rely on the undefined constant-vector normalization case. -/
theorem convex_counterexample_fisher :
    fisherDiagonal (1 / 1000) (fun i : Fin 2 => (i : ℝ))
      (fun i : Fin 2 => (i : ℝ)) (0, 0) = 1 / 1000 := by
  norm_num [fisherDiagonal]

/-- Counterexample to Proposition 3.3's printed O(1/k) inequality,
Appendix A.2: at k=1, α=L=1 the left side is 998001/2 and the alleged
bound is 1/2. The objective is smooth, strongly convex, and minimized at 0. -/
theorem convex_rate_counterexample :
    (0 : ℝ) < 1 / 1000 ∧ (1 : ℝ) ≤ 1 / 1 ∧
    quadratic (quadraticRun 1 (1 / 1000) 1 1) - quadratic 0 >
      (1 - 0) ^ 2 / (2 * 1 * 1) := by
  norm_num [quadratic, quadraticRun]

/-- Proposition 3.3's allowed step can actually diverge, not merely violate
its rate constant: the absolute iterate tends to infinity for the valid
smooth strongly convex quadratic and the efficient Fisher eigendirection. -/
theorem convex_iterates_diverge :
    Tendsto (fun t => |quadraticRun 1 (1 / 1000) 1 t|) atTop atTop := by
  have heq : (fun t => |quadraticRun 1 (1 / 1000) 1 t|) =
      (fun t : ℕ => (999 : ℝ) ^ t) := by
    funext t
    rw [quadraticRun_eq, abs_mul, abs_pow]
    norm_num
  rw [heq]
  exact tendsto_pow_atTop_atTop_of_one_lt (by norm_num)

end Transformer.AdaFisher
