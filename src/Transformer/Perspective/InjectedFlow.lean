/-
# Weighted flows with the input mixed back in — no collapse

**Not a statement of any paper.**  A companion to `lem: hemisphere.clustering`
of arXiv:2312.10794v5 (§6.1, `cone_collapse`), for the flow

  `ẋ_i = Proj_{x_i}( Σ_j a_ij(t) x_j + z_i )`   (`IsInjectedFlow`),

a weighted flow (`IsWeightedFlow`) with a constant vector `z_i` added to the
drive of particle `i`.  If two of the `z_i` are linearly independent, the
particles never collapse, whatever the weights, as long as their absolute row
sums are at most some `A`: they cannot stay within `δ` of each other for a time
`T`, where `δ` and `T` depend on `A` and on `z` only
(`IsInjectedFlow.spread`, in `Perspective.InjectedSpread`; this file holds the
estimates it is made of).

The proof is one Lyapunov-type function.  For `u = z_k`, `v = z_l`
independent, `ψ = ⟨x_k, u⟩ + ⟨x_l, v⟩` is bounded by `‖u‖ + ‖v‖`.  While the
particles are `δ`-close, the attention term of `ψ̇` is `O(δ)`, because
`Proj_x` kills `x` itself, and what remains is `‖Proj_x u‖² + ‖Proj_x v‖²` at
one point `x`, which no unit `x` makes small: it is at least
`(‖u‖²‖v‖² - ⟨u, v⟩²) / (2(‖u‖² + ‖v‖²))` (`gram_le_tangent`).  So `ψ`
would grow without bound.
-/

import Transformer.Perspective.ConeChart
import Mathlib.Analysis.Calculus.Deriv.MeanValue

open scoped BigOperators

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- **A weighted flow with injection:** `ẋ_i = Proj_{x_i}(Σ_j a_ij(t) x_j + z_i)`,
the constant vector `z_i` added to the drive of particle `i` inside the
projection, where `eq: albert` adds the feed-forward layer.

Source: none — posed here; `IsWeightedFlow` (arXiv:2312.10794v5, §6.1) with the
term `z_i` added inside the projection, at the place where `eq: albert`
(arXiv:2312.10794v5, §2.3) adds the feed-forward term `w σ(a x_i + b)`.  Unlike
that term, the same map for every particle, `z_i` depends on the particle. -/
def IsInjectedFlow (X : ℝ → SphereTuple d n) (a : ℝ → Idx n → Idx n → ℝ)
    (z : Idx n → EucSpace d) : Prop :=
  ∀ (t : ℝ) (i : Idx n), HasDerivAt (fun s => (X s i : EucSpace d))
    (proj d (X t i : EucSpace d) (∑ j : Idx n, a t i j • (X t j : EucSpace d) + z i)) t

/-- **Strict Cauchy–Schwarz** for two linearly independent vectors:
`⟨u, v⟩² < ‖u‖² ‖v‖²`. -/
theorem sq_inner_lt_of_linearIndependent {u v : EucSpace d}
    (h : LinearIndependent ℝ ![u, v]) :
    (inner (𝕜 := ℝ) u v) ^ 2 < ‖u‖ ^ 2 * ‖v‖ ^ 2 := by
  have hu : u ≠ 0 := by simpa using h.ne_zero 0
  have hv : v ≠ 0 := by simpa using h.ne_zero 1
  have hne : |inner (𝕜 := ℝ) u v| ≠ ‖u‖ * ‖v‖ := by
    intro heq
    obtain ⟨r, -, hr⟩ := (norm_inner_eq_norm_iff hu hv).1 (by rwa [Real.norm_eq_abs])
    have := LinearIndependent.pair_iff.1 h r (-1) (by rw [hr]; simp)
    norm_num at this
  have hlt := lt_of_le_of_ne (abs_real_inner_le_norm u v) hne
  nlinarith [abs_nonneg (inner (𝕜 := ℝ) u v), sq_abs (inner (𝕜 := ℝ) u v)]

/-- The hypothesis of `sq_inner_lt_of_linearIndependent` is satisfiable: the
two standard basis vectors of `ℝ²`. -/
example : LinearIndependent ℝ
    ![(EuclideanSpace.single 0 1 : EucSpace 2), EuclideanSpace.single 1 1] := by
  refine linearIndependent_of_ne_zero_of_inner_eq_zero (fun i => ?_) fun i j hij => ?_
  · fin_cases i <;> simp
  · fin_cases i <;> fin_cases j <;> simp_all [EuclideanSpace.inner_single_left]

/-- **Cauchy–Schwarz against a unit vector:** `⟨x, w⟩² ≤ ‖w‖²` for `‖x‖ = 1`,
i.e. `‖Proj_x w‖² = ‖w‖² - ⟨x, w⟩²` is not negative. -/
theorem sq_inner_le_norm_sq {x : EucSpace d} (hx : ‖x‖ = 1) (w : EucSpace d) :
    (inner (𝕜 := ℝ) x w) ^ 2 ≤ ‖w‖ ^ 2 := by
  have h := real_inner_mul_inner_self_le x w
  rw [real_inner_self_eq_norm_sq, real_inner_self_eq_norm_sq, hx, one_pow, one_mul] at h
  linarith

/-- The hypothesis of `sq_inner_le_norm_sq` is satisfiable: the first standard
basis vector of `ℝ¹` has norm `1`. -/
example : ‖(EuclideanSpace.single (0 : Fin 1) (1 : ℝ))‖ = 1 := by
  simp

/-- **No unit vector is almost parallel to two independent ones.**  For
`‖x‖ = 1`, the tangential parts of `u` and `v` at `x` satisfy

  `‖u‖²‖v‖² - ⟨u, v⟩² ≤ 2 (‖u‖² + ‖v‖²) (‖Proj_x u‖² + ‖Proj_x v‖²)`,

with `‖Proj_x u‖² = ‖u‖² - ⟨x, u⟩²`.  The certificate is
`‖⟨x, v⟩ u + ⟨x, u⟩ v - 2 ⟨x, u⟩⟨x, v⟩ x‖² ≥ 0`, the vector
`⟨x, v⟩ Proj_x u + ⟨x, u⟩ Proj_x v`. -/
theorem gram_le_tangent {x : EucSpace d} (hx : ‖x‖ = 1) (u v : EucSpace d) :
    ‖u‖ ^ 2 * ‖v‖ ^ 2 - (inner (𝕜 := ℝ) u v) ^ 2 ≤
      2 * (‖u‖ ^ 2 + ‖v‖ ^ 2) *
        ((‖u‖ ^ 2 - (inner (𝕜 := ℝ) x u) ^ 2) + (‖v‖ ^ 2 - (inner (𝕜 := ℝ) x v) ^ 2)) := by
  have hp := sub_nonneg.2 (sq_inner_le_norm_sq hx u)
  have hq := sub_nonneg.2 (sq_inner_le_norm_sq hx v)
  have hw := real_inner_self_nonneg (x := (inner (𝕜 := ℝ) x v) • u
    + (inner (𝕜 := ℝ) x u) • v - (2 * inner (𝕜 := ℝ) x u * inner (𝕜 := ℝ) x v) • x)
  simp only [inner_sub_left, inner_sub_right, inner_add_left, inner_add_right,
    real_inner_smul_left, real_inner_smul_right, real_inner_self_eq_norm_sq, norm_smul,
    Real.norm_eq_abs, mul_pow, sq_abs, hx, mul_one,
    real_inner_comm x u, real_inner_comm x v, real_inner_comm u v] at hw
  nlinarith [mul_nonneg (sq_nonneg (inner (𝕜 := ℝ) x u)) hp,
    mul_nonneg (sq_nonneg (inner (𝕜 := ℝ) x v)) hq, mul_nonneg hp hq,
    sq_nonneg (inner (𝕜 := ℝ) u v - inner (𝕜 := ℝ) x u * inner (𝕜 := ℝ) x v)]

/-- The hypothesis of `gram_le_tangent` is satisfiable: the first standard
basis vector of `ℝ¹` has norm `1`. -/
example : ‖(EuclideanSpace.single (0 : Fin 1) (1 : ℝ))‖ = 1 := by
  simp

/-- The drive `u` splits off the tangential energy it feeds in:
`⟨Proj_x (S + u), u⟩ = ⟨Proj_x S, u⟩ + (‖u‖² - ⟨x, u⟩²)`. -/
theorem inner_proj_add_self (x S u : EucSpace d) :
    inner (𝕜 := ℝ) (proj d x (S + u)) u =
      inner (𝕜 := ℝ) (proj d x S) u + (‖u‖ ^ 2 - (inner (𝕜 := ℝ) x u) ^ 2) := by
  simp only [proj, inner_sub_left, inner_add_left, inner_add_right, real_inner_smul_left,
    real_inner_self_eq_norm_sq]
  ring

/-- **The attention term is small near consensus.**  If the points `y_j` are
within `δ` of the unit vector `x` and the weights have absolute sum at most
`A`, then `|⟨Proj_x (Σ_j a_j y_j), u⟩| ≤ A δ ‖u‖`: `Proj_x` kills the part
`(Σ_j a_j) x`, and what is left is `Σ_j a_j (y_j - x)`. -/
theorem abs_inner_proj_sum_le {x : EucSpace d} (hx : ‖x‖ = 1) (a : Idx n → ℝ)
    (y : Idx n → EucSpace d) {A δ : ℝ} (hA : ∑ j, |a j| ≤ A) (hδ : 0 ≤ δ)
    (hy : ∀ j, ‖y j - x‖ ≤ δ) (u : EucSpace d) :
    |inner (𝕜 := ℝ) (proj d x (∑ j, a j • y j)) u| ≤ A * δ * ‖u‖ := by
  have hsplit : proj d x (∑ j, a j • y j) = proj d x (∑ j, a j • (y j - x)) := by
    have h1 : ∑ j, a j • y j = ∑ j, a j • (y j - x) + (∑ j, a j) • x := by
      simp only [smul_sub, Finset.sum_sub_distrib, Finset.sum_smul, sub_add_cancel]
    rw [h1]
    simp only [proj, inner_add_right, real_inner_smul_right, real_inner_self_eq_norm_sq, hx,
      one_pow, mul_one, add_smul]
    abel
  have hsum : ‖∑ j, a j • (y j - x)‖ ≤ A * δ :=
    calc ‖∑ j, a j • (y j - x)‖ ≤ ∑ j, ‖a j • (y j - x)‖ := norm_sum_le _ _
      _ = ∑ j, |a j| * ‖y j - x‖ := by simp only [norm_smul, Real.norm_eq_abs]
      _ ≤ ∑ j, |a j| * δ :=
          Finset.sum_le_sum fun j _ => mul_le_mul_of_nonneg_left (hy j) (abs_nonneg _)
      _ = (∑ j, |a j|) * δ := (Finset.sum_mul _ _ _).symm
      _ ≤ A * δ := mul_le_mul_of_nonneg_right hA hδ
  rw [hsplit]
  calc |inner (𝕜 := ℝ) (proj d x (∑ j, a j • (y j - x))) u|
      ≤ ‖proj d x (∑ j, a j • (y j - x))‖ * ‖u‖ := abs_real_inner_le_norm _ _
    _ ≤ ‖∑ j, a j • (y j - x)‖ * ‖u‖ :=
        mul_le_mul_of_nonneg_right (norm_proj_le hx _) (norm_nonneg _)
    _ ≤ A * δ * ‖u‖ := mul_le_mul_of_nonneg_right hsum (norm_nonneg _)

/-- The hypotheses of `abs_inner_proj_sum_le` are satisfiable: no weights at
all, `A = δ = 0`, around the first standard basis vector of `ℝ¹`. -/
example : ‖(EuclideanSpace.single (0 : Fin 1) (1 : ℝ))‖ = 1 ∧
    ∑ j : Idx 0, |(fun _ => (0 : ℝ)) j| ≤ 0 ∧ (0 : ℝ) ≤ 0 := by
  simp

/-- **Nearby unit vectors see `v` at nearly the same height:** if
`‖y - x‖ ≤ δ` for unit `x` and `y`, then `⟨y, v⟩² ≤ ⟨x, v⟩² + 2 ‖v‖² δ`, the
difference of the squares being `⟨y - x, v⟩ ⟨y + x, v⟩`. -/
theorem sq_inner_le_sq_inner_add {x y : EucSpace d} (hx : ‖x‖ = 1) (hy : ‖y‖ = 1) {δ : ℝ}
    (hxy : ‖y - x‖ ≤ δ) (v : EucSpace d) :
    (inner (𝕜 := ℝ) y v) ^ 2 ≤ (inner (𝕜 := ℝ) x v) ^ 2 + 2 * ‖v‖ ^ 2 * δ := by
  have hd : |inner (𝕜 := ℝ) y v - inner (𝕜 := ℝ) x v| ≤ δ * ‖v‖ := by
    rw [← inner_sub_left]
    exact (abs_real_inner_le_norm _ _).trans (mul_le_mul_of_nonneg_right hxy (norm_nonneg _))
  have hs : |inner (𝕜 := ℝ) y v + inner (𝕜 := ℝ) x v| ≤ 2 * ‖v‖ := by
    have h1 := abs_real_inner_le_norm y v
    have h2 := abs_real_inner_le_norm x v
    rw [hy, one_mul] at h1
    rw [hx, one_mul] at h2
    linarith [abs_add_le (inner (𝕜 := ℝ) y v) (inner (𝕜 := ℝ) x v)]
  have h := mul_le_mul hs hd (abs_nonneg _) (by positivity)
  rw [← abs_mul, ← sq_sub_sq] at h
  linarith [le_abs_self ((inner (𝕜 := ℝ) y v) ^ 2 - (inner (𝕜 := ℝ) x v) ^ 2)]

/-- The hypotheses of `sq_inner_le_sq_inner_add` are satisfiable: `x = y` the
first standard basis vector of `ℝ¹`, and `δ = 0`. -/
example : ‖(EuclideanSpace.single (0 : Fin 1) (1 : ℝ))‖ = 1 ∧
    ‖(EuclideanSpace.single (0 : Fin 1) (1 : ℝ)) - EuclideanSpace.single 0 1‖ ≤ 0 := by
  simp

end Perspective
end Transformer
