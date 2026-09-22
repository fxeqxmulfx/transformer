/-
# Injected flows do not collapse

**Not a statement of any paper.**  The theorem that `Perspective.InjectedFlow`
prepares: along `ẋ_i = Proj_{x_i}(Σ_j a_ij(t) x_j + z_i)` with two linearly
independent `z_k`, `z_l` and absolute row sums at most `A`, the particles
cannot stay pairwise within `δ` of each other for a time `T`, where `δ` and
`T` depend on `A`, `z_k` and `z_l` only (`IsInjectedFlow.spread`).  This is
the opposite of cone collapse (`lem: hemisphere.clustering` of
arXiv:2312.10794v5, §6.1), which the same flow undergoes at `z = 0`.
-/

import Transformer.Perspective.InjectedFlow

open scoped BigOperators

namespace Transformer
namespace Perspective

variable {d n : ℕ}

/-- **The Lyapunov function climbs near consensus.**  If the points `X_i` are
pairwise within `δ` and the weight rows `a`, `b` have absolute sums at most
`A`, then

  `⟨Proj_{x_k}(Σ_j a_j x_j + u), u⟩ + ⟨Proj_{x_l}(Σ_j b_j x_j + v), v⟩ ≥ c - K δ`

with `c = (‖u‖²‖v‖² - ⟨u, v⟩²) / (2(‖u‖² + ‖v‖²))` and
`K = A (‖u‖ + ‖v‖) + 2 ‖v‖²`.  Along an injected flow the left side is the
derivative of `⟨x_k, u⟩ + ⟨x_l, v⟩` for `u = z_k`, `v = z_l`. -/
theorem le_inner_proj_add_inner_proj (X : SphereTuple d n) (a b : Idx n → ℝ) (k l : Idx n)
    {A δ : ℝ} (ha : ∑ j, |a j| ≤ A) (hb : ∑ j, |b j| ≤ A) (hδ : 0 ≤ δ)
    (hX : ∀ i j, ‖(X i : EucSpace d) - X j‖ ≤ δ) (u v : EucSpace d) :
    (‖u‖ ^ 2 * ‖v‖ ^ 2 - (inner (𝕜 := ℝ) u v) ^ 2) / (2 * (‖u‖ ^ 2 + ‖v‖ ^ 2))
        - (A * (‖u‖ + ‖v‖) + 2 * ‖v‖ ^ 2) * δ ≤
      inner (𝕜 := ℝ) (proj d (X k) (∑ j, a j • (X j : EucSpace d) + u)) u
        + inner (𝕜 := ℝ) (proj d (X l) (∑ j, b j • (X j : EucSpace d) + v)) v := by
  have hxk := norm_coe_tuple X k
  have hxl := norm_coe_tuple X l
  have h1 : -(A * δ * ‖u‖) ≤ inner (𝕜 := ℝ) (proj d (X k) (∑ j, a j • (X j : EucSpace d))) u :=
    (abs_le.1 (abs_inner_proj_sum_le hxk a _ ha hδ (fun j => hX j k) u)).1
  have h2 : -(A * δ * ‖v‖) ≤ inner (𝕜 := ℝ) (proj d (X l) (∑ j, b j • (X j : EucSpace d))) v :=
    (abs_le.1 (abs_inner_proj_sum_le hxl b _ hb hδ (fun j => hX j l) v)).1
  have h3 : (‖u‖ ^ 2 * ‖v‖ ^ 2 - (inner (𝕜 := ℝ) u v) ^ 2) / (2 * (‖u‖ ^ 2 + ‖v‖ ^ 2)) ≤
      (‖u‖ ^ 2 - (inner (𝕜 := ℝ) (X k : EucSpace d) u) ^ 2)
        + (‖v‖ ^ 2 - (inner (𝕜 := ℝ) (X k : EucSpace d) v) ^ 2) :=
    div_le_of_le_mul₀ (by positivity)
      (by linarith [sq_inner_le_norm_sq hxk u, sq_inner_le_norm_sq hxk v])
      (by linarith [gram_le_tangent hxk u v])
  have h4 := sq_inner_le_sq_inner_add hxk hxl (hX l k) v
  rw [inner_proj_add_self, inner_proj_add_self]
  linarith

/-- The hypotheses of `le_inner_proj_add_inner_proj` are satisfiable: one
particle, no weights, `A = δ = 0`. -/
example : ∑ j : Idx 1, |(fun _ => (0 : ℝ)) j| ≤ 0 ∧ (0 : ℝ) ≤ 0 ∧
    ∀ i j : Idx 1, ‖(((fun _ => basePoint 0 : SphereTuple 1 1) i : EucSpace 1))
      - (fun _ => basePoint 0 : SphereTuple 1 1) j‖ ≤ 0 := by
  simp

/-- **Theorem (no collapse under injection).**  Let `z_k`, `z_l` be linearly
independent and `A ≥ 0`.  There are `δ, T > 0`, depending on `A`, `z_k` and
`z_l` only, such that along every injected flow
`ẋ_i = Proj_{x_i}(Σ_j a_ij(t) x_j + z_i)` whose weights have absolute row sums
at most `A`, every time window of length `T` contains a time at which two
particles are more than `δ` apart.

In particular the particles do not converge to one point, from any initial
configuration, and `δ`, `T` do not depend on the weights: for softmax attention
(row sums `1`) they are uniform in `β` and in `Q(t)`, `K(t)`.

The proof: `ψ = ⟨x_k, z_k⟩ + ⟨x_l, z_l⟩` satisfies `|ψ| ≤ ‖z_k‖ + ‖z_l‖`, and
while the particles are `δ`-close, `ψ̇ ≥ c - K δ = c / 2`
(`le_inner_proj_add_inner_proj`, `δ = c / (2K)`), so over a time
`T = 8 (‖z_k‖ + ‖z_l‖) / c` it would rise by `4 (‖z_k‖ + ‖z_l‖)`.

Source: none — posed here, as the counterpart of `lem: hemisphere.clustering`
(arXiv:2312.10794v5, §6.1) for the flow of `IsInjectedFlow`. -/
theorem IsInjectedFlow.spread {z : Idx n → EucSpace d} {k l : Idx n}
    (hz : LinearIndependent ℝ ![z k, z l]) {A : ℝ} (hA : 0 ≤ A) :
    ∃ δ T : ℝ, 0 < δ ∧ 0 < T ∧
      ∀ (X : ℝ → SphereTuple d n) (a : ℝ → Idx n → Idx n → ℝ),
        (∀ t i, ∑ j, |a t i j| ≤ A) → IsInjectedFlow X a z →
        ∀ t₀ : ℝ, ∃ t ∈ Set.Icc t₀ (t₀ + T), ∃ i j : Idx n,
          δ < ‖(X t i : EucSpace d) - X t j‖ := by
  have hu : 0 < ‖z k‖ := norm_pos_iff.2 (by simpa using hz.ne_zero 0)
  have hv : 0 < ‖z l‖ := norm_pos_iff.2 (by simpa using hz.ne_zero 1)
  obtain ⟨c, hc_def⟩ : ∃ c, c = (‖z k‖ ^ 2 * ‖z l‖ ^ 2 - (inner (𝕜 := ℝ) (z k) (z l)) ^ 2)
      / (2 * (‖z k‖ ^ 2 + ‖z l‖ ^ 2)) := ⟨_, rfl⟩
  obtain ⟨K, hK_def⟩ : ∃ K, K = A * (‖z k‖ + ‖z l‖) + 2 * ‖z l‖ ^ 2 := ⟨_, rfl⟩
  obtain ⟨T, hT_def⟩ : ∃ T, T = 8 * (‖z k‖ + ‖z l‖) / c := ⟨_, rfl⟩
  have hc : 0 < c := by
    rw [hc_def]
    exact div_pos (sub_pos.2 (sq_inner_lt_of_linearIndependent hz)) (by positivity)
  have hK : 0 < K := by rw [hK_def]; positivity
  have hT : 0 < T := by rw [hT_def]; positivity
  refine ⟨c / (2 * K), T, by positivity, hT, fun X a ha hX t₀ => ?_⟩
  by_contra hcon
  push Not at hcon
  obtain ⟨ψ, hψ_def⟩ : ∃ ψ : ℝ → ℝ, ψ = fun s => inner (𝕜 := ℝ) (X s k : EucSpace d) (z k)
      + inner (𝕜 := ℝ) (X s l : EucSpace d) (z l) := ⟨_, rfl⟩
  have hψ : ∀ s, HasDerivAt ψ
      (inner (𝕜 := ℝ) (proj d (X s k) (∑ j, a s k j • (X s j : EucSpace d) + z k)) (z k)
        + inner (𝕜 := ℝ) (proj d (X s l) (∑ j, a s l j • (X s j : EucSpace d) + z l)) (z l))
      s := fun s => by
    rw [hψ_def]
    convert ((hX s k).inner ℝ (hasDerivAt_const s (z k))).fun_add
      ((hX s l).inner ℝ (hasDerivAt_const s (z l))) using 1
    simp only [inner_zero_right, zero_add]
  have hle : t₀ ≤ t₀ + T := le_add_of_nonneg_right hT.le
  have hmvt := (convex_Icc t₀ (t₀ + T)).mul_sub_le_image_sub_of_le_deriv (f := ψ)
    (fun s _ => (hψ s).continuousAt.continuousWithinAt)
    (fun s _ => (hψ s).differentiableAt.differentiableWithinAt) (C := c / 2)
    (fun s hs => by
      have h := le_inner_proj_add_inner_proj (X s) (a s k) (a s l) k l (ha s k) (ha s l)
        (by positivity) (hcon s (interior_subset hs)) (z k) (z l)
      rw [← hc_def, ← hK_def] at h
      have hKδ : K * (c / (2 * K)) = c / 2 := by
        rw [← mul_div_assoc, mul_comm K c, mul_div_mul_right c 2 hK.ne']
      rw [(hψ s).deriv]
      linarith)
    t₀ (Set.left_mem_Icc.2 hle) (t₀ + T) (Set.right_mem_Icc.2 hle) hle
  have hbound : ∀ s, |ψ s| ≤ ‖z k‖ + ‖z l‖ := fun s => by
    have h1 := abs_real_inner_le_norm (X s k : EucSpace d) (z k)
    have h2 := abs_real_inner_le_norm (X s l : EucSpace d) (z l)
    rw [norm_coe_tuple, one_mul] at h1 h2
    rw [hψ_def]
    exact (abs_add_le _ _).trans (add_le_add h1 h2)
  have hcT : c / 2 * (t₀ + T - t₀) = 4 * (‖z k‖ + ‖z l‖) := by
    rw [add_sub_cancel_left, hT_def]
    field_simp
    ring
  obtain ⟨h0, -⟩ := abs_le.1 (hbound t₀)
  obtain ⟨-, h1⟩ := abs_le.1 (hbound (t₀ + T))
  linarith

/-- The hypotheses of `IsInjectedFlow.spread` are satisfiable: the two
standard basis vectors of `ℝ²` as `z_0`, `z_1`, and `A = 1`. -/
example : LinearIndependent ℝ
    ![(![EuclideanSpace.single 0 1, EuclideanSpace.single 1 1] : Idx 2 → EucSpace 2) 0,
      (![EuclideanSpace.single 0 1, EuclideanSpace.single 1 1] : Idx 2 → EucSpace 2) 1] ∧
    (0 : ℝ) ≤ 1 := by
  refine ⟨?_, zero_le_one⟩
  refine linearIndependent_of_ne_zero_of_inner_eq_zero (fun i => ?_) fun i j hij => ?_
  · fin_cases i <;> simp
  · fin_cases i <;> fin_cases j <;> simp_all [EuclideanSpace.inner_single_left]

end Perspective
end Transformer
