/-
# Inner products and asymptotics at the origin

The bookkeeping the second-order expansion of `𝖤_β` runs on: Cauchy–Schwarz
read as a statement about `O` and `o`, and the two monomial comparisons
`t³ = o(t²)`, `t² = o(t)`.  Nothing here mentions the survey.

Used by `Perspective.AppendixB_Intrinsic`.
-/

import Transformer.Basic
import Mathlib.Analysis.Asymptotics.Lemmas

open scoped BigOperators
open Asymptotics Filter

namespace Transformer
namespace Perspective

variable (d : ℕ)

section Asymptotic

variable {l : Filter ℝ}

/-- Cauchy–Schwarz as an asymptotic statement: a `O(g)` paired with a `O(k)`
is a `O(g k)`. -/
theorem isBigO_inner {f h : ℝ → EucSpace d} {g k : ℝ → ℝ}
    (hf : f =O[l] g) (hh : h =O[l] k) :
    (fun t => inner (𝕜 := ℝ) (f t) (h t)) =O[l] fun t => g t * k t := by
  have h1 : (fun t => inner (𝕜 := ℝ) (f t) (h t)) =O[l] fun t => ‖f t‖ * ‖h t‖ :=
    Asymptotics.isBigO_of_le l fun t => by
      simp only [Real.norm_eq_abs, abs_mul, abs_norm]
      exact abs_real_inner_le_norm (f t) (h t)
  refine h1.trans ((hf.norm_norm.mul hh.norm_norm).trans ?_)
  exact Asymptotics.isBigO_of_le l fun t => by simp [Real.norm_eq_abs]

/-- A `o(g)` paired with a `O(k)` is a `o(g k)`. -/
theorem isLittleO_inner {f h : ℝ → EucSpace d} {g k : ℝ → ℝ}
    (hf : f =o[l] g) (hh : h =O[l] k) :
    (fun t => inner (𝕜 := ℝ) (f t) (h t)) =o[l] fun t => g t * k t := by
  have h1 : (fun t => inner (𝕜 := ℝ) (f t) (h t)) =O[l] fun t => ‖f t‖ * ‖h t‖ :=
    Asymptotics.isBigO_of_le l fun t => by
      simp only [Real.norm_eq_abs, abs_mul, abs_norm]
      exact abs_real_inner_le_norm (f t) (h t)
  refine h1.trans_isLittleO ((hf.norm_norm.mul_isBigO hh.norm_norm).trans_isBigO ?_)
  exact Asymptotics.isBigO_of_le l fun t => by simp [Real.norm_eq_abs]

/-- Pairing with a fixed vector on the right preserves `O`. -/
theorem isBigO_inner_const_left {f : ℝ → EucSpace d} {g : ℝ → ℝ} (a : EucSpace d)
    (hf : f =O[l] g) : (fun t => inner (𝕜 := ℝ) (f t) a) =O[l] g := by
  have hcomm : (fun t => inner (𝕜 := ℝ) (f t) a) = fun t => innerSL ℝ a (f t) := by
    funext t; exact real_inner_comm a (f t)
  rw [hcomm]
  exact ((innerSL ℝ a).isBigO_comp f l).trans hf

/-- Pairing with a fixed vector on the left preserves `O`. -/
theorem isBigO_inner_const_right {f : ℝ → EucSpace d} {g : ℝ → ℝ} (a : EucSpace d)
    (hf : f =O[l] g) : (fun t => inner (𝕜 := ℝ) a (f t)) =O[l] g :=
  ((innerSL ℝ a).isBigO_comp f l).trans hf

/-- Pairing with a fixed vector on the right preserves `o`. -/
theorem isLittleO_inner_const_left {f : ℝ → EucSpace d} {g : ℝ → ℝ} (a : EucSpace d)
    (hf : f =o[l] g) : (fun t => inner (𝕜 := ℝ) (f t) a) =o[l] g := by
  have hcomm : (fun t => inner (𝕜 := ℝ) (f t) a) = fun t => innerSL ℝ a (f t) := by
    funext t; exact real_inner_comm a (f t)
  rw [hcomm]
  exact ((innerSL ℝ a).isBigO_comp f l).trans_isLittleO hf

/-- Pairing with a fixed vector on the left preserves `o`. -/
theorem isLittleO_inner_const_right {f : ℝ → EucSpace d} {g : ℝ → ℝ} (a : EucSpace d)
    (hf : f =o[l] g) : (fun t => inner (𝕜 := ℝ) a (f t)) =o[l] g :=
  ((innerSL ℝ a).isBigO_comp f l).trans_isLittleO hf

end Asymptotic

/-- `t³` is `o(t²)` at the origin. -/
theorem isLittleO_cube_sq : (fun t : ℝ => t ^ 3) =o[nhds 0] fun t : ℝ => t ^ 2 := by
  rw [Asymptotics.isLittleO_iff]
  intro ε hε
  refine Metric.eventually_nhds_iff.mpr ⟨ε, hε, fun t ht => ?_⟩
  have habs : |t| < ε := by simpa [Real.dist_eq] using ht
  rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_pow, abs_pow]
  have h2 : (0 : ℝ) ≤ |t| ^ 2 := by positivity
  calc |t| ^ 3 = |t| * |t| ^ 2 := by ring
    _ ≤ ε * |t| ^ 2 := by nlinarith

/-- `t²` is `o(t)` at the origin. -/
theorem isLittleO_sq_id : (fun t : ℝ => t ^ 2) =o[nhds 0] fun t : ℝ => t := by
  rw [Asymptotics.isLittleO_iff]
  intro ε hε
  refine Metric.eventually_nhds_iff.mpr ⟨ε, hε, fun t ht => ?_⟩
  have habs : |t| < ε := by simpa [Real.dist_eq] using ht
  rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_pow]
  nlinarith [abs_nonneg t]

/-- `t · c` is `O(t)` at the origin. -/
theorem isBigO_mul_const (c : ℝ) : (fun t : ℝ => t * c) =O[nhds 0] fun t : ℝ => t := by
  refine Asymptotics.isBigO_iff.mpr ⟨|c|, ?_⟩
  filter_upwards with t
  rw [Real.norm_eq_abs, Real.norm_eq_abs, abs_mul, mul_comm]

/-- `t • v` is `O(t)` at the origin. -/
theorem isBigO_smul_const (v : EucSpace d) :
    (fun t : ℝ => t • v) =O[nhds 0] fun t : ℝ => t := by
  refine Asymptotics.isBigO_iff.mpr ⟨‖v‖, ?_⟩
  filter_upwards with t
  rw [norm_smul, mul_comm]

end Perspective
end Transformer
