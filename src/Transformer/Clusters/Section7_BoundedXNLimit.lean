/-
# The emergence of clusters in self-attention dynamics — `l:boundedxn`, the barrier

The heart of point (3) of the proof of `l:boundedxn` (§7 of
arXiv:2305.05465v6), on the scalar curves of `e:Idnonresca`: when the largest
token `x_N` is bounded and the next one `x_J` runs off like `-c e^t`,
`e:convdeladiff` holds, `x_N (x_N - x_J) → +∞`
(`tendsto_mul_sub_of_bounded`).  Once `x_N (x_N - x_J) ≤ κ` at a large time,
`e:epsxjxn` makes it decrease at speed at least `2` for good, which drives the
nonnegative quantity below `0`.

Plugged into the weights, every `P_Nk`, `k ≠ N`, tends to `0`
(`tendsto_softmaxWeight_of_bounded`).

Source: arXiv:2305.05465v6, proof of `l:boundedxn`, point (3).
-/

import Transformer.Clusters.Section7_BoundedXNCore

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {m : ℕ}

/-- **`e:convdeladiff`.**  If the largest token `x_N` stays in `[-R, R]` and
the largest other one `x_J` stays below `-c e^t` for large `t`, then
`x_N (x_N - x_J) → +∞`.

Source: arXiv:2305.05465v6, proof of `l:boundedxn`, `e:convdeladiff`. -/
theorem tendsto_mul_sub_of_bounded (x : ℝ → Idx (m + 1) → ℝ)
    (hder : ∀ t k, HasDerivAt (fun s => x s k)
      (∑ j, Perspective.softmaxWeight (fun l => x t k * x t l) j * x t j) t)
    (N J L : Idx (m + 1)) (hJN : J ≠ N) (hmax : ∀ t, 0 ≤ t → ∀ k, x t k ≤ x t N)
    (hJ : ∀ t, 0 ≤ t → ∀ k, k ≠ N → x t k ≤ x t J) (hmin : ∀ t, 0 ≤ t → ∀ k, x t L ≤ x t k)
    {R : ℝ} (hR : ∀ t, 0 ≤ t → |x t N| ≤ R) {c : ℝ} (hc : 0 < c)
    (hJgrow : ∀ᶠ t in atTop, x t J ≤ -(c * Real.exp t)) :
    Tendsto (fun t => x t N * (x t N - x t J)) atTop atTop := by
  rw [tendsto_atTop]
  intro κ
  set φ : ℝ → ℝ := fun t => x t N * (x t N - x t J)
  set C := |x 0 L|
  have hR0 : 0 ≤ R := (abs_nonneg _).trans (hR 0 le_rfl)
  have hlow : ∀ t, 0 ≤ t → -(C * Real.exp t) ≤ x t L := fun t ht => by
    have := mul_exp_le_of_min x hder L hmin t ht
    nlinarith [neg_abs_le (x 0 L), Real.exp_pos t]
  have hN0 := nonneg_of_bounded_max x hder N hmax fun t ht => (abs_le.1 (hR t ht)).1
  set ε := 1 / (Real.exp κ + ((m : ℝ) + 1))
  have hε : 0 < ε := by positivity
  set U := (2 * R ^ 2 + 2 * R * C + 2) / (ε * c ^ 2) + 1
  obtain ⟨T, hT⟩ := eventually_atTop.1 ((hJgrow.and
    (tendsto_exp_atTop.eventually (eventually_ge_atTop U))).and (eventually_ge_atTop 0))
  have hφ : ∀ t, HasDerivAt φ
      ((∑ j, Perspective.softmaxWeight (fun l => x t N * x t l) j * x t j) *
          (2 * x t N - x t J) -
        x t N * ∑ j, Perspective.softmaxWeight (fun l => x t J * x t l) j * x t j) t := by
    intro t
    convert (hder t N).mul ((hder t N).sub (hder t J)) using 1
    simp only [Pi.sub_apply]; ring
  have hslow : ∀ t, T ≤ t → φ t ≤ κ →
      (∑ j, Perspective.softmaxWeight (fun l => x t N * x t l) j * x t j) *
          (2 * x t N - x t J) -
        x t N * ∑ j, Perspective.softmaxWeight (fun l => x t J * x t l) j * x t j ≤ -2 := by
    intro t ht hκ
    obtain ⟨⟨hJt, hU⟩, ht0⟩ := hT t ht
    have hce : 0 < c * Real.exp t := by positivity
    have h1 := deriv_mul_sub_le (x t) N J L hJN (hJ t ht0) (hmin t ht0) (hN0 t ht0)
      (by linarith) hκ
    have hy := (abs_le.1 (hR t ht0)).2
    have hy0 := hN0 t ht0
    have hL := hlow t ht0
    set u := Real.exp t
    have hU1 : 1 ≤ u := le_trans (le_add_of_nonneg_left (by positivity)) hU
    have hz : c ^ 2 * u ^ 2 ≤ x t J ^ 2 := by nlinarith
    have hUe : ε * c ^ 2 * U = 2 * R ^ 2 + 2 * R * C + 2 + ε * c ^ 2 := by
      simp only [U]; field_simp
    have hq : 2 * R ^ 2 + 2 * R * C * u + 2 ≤ ε * c ^ 2 * u ^ 2 := by
      have h2 : ε * c ^ 2 * U * u ≤ ε * c ^ 2 * u * u :=
        mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hU (by positivity)) (by positivity)
      have e1 : ε * c ^ 2 * U * u = (2 * R ^ 2 + 2 * R * C + 2 + ε * c ^ 2) * u := by rw [hUe]
      nlinarith [mul_le_mul_of_nonneg_left hU1 (sq_nonneg R),
        mul_nonneg (mul_nonneg hε.le (sq_nonneg c)) (zero_le_one.trans hU1)]
    have hyL : -(2 * x t N * x t L) ≤ 2 * R * C * u := by
      have hCu : 0 ≤ C * u := by positivity
      have : x t N * -x t L ≤ R * (C * u) :=
        (mul_le_mul_of_nonneg_left (by linarith) hy0).trans (mul_le_mul_of_nonneg_right hy hCu)
      nlinarith
    have hyR : x t N ^ 2 ≤ R ^ 2 := by nlinarith
    nlinarith [mul_le_mul_of_nonneg_left hz hε.le]
  have hpos : ∀ t, 0 ≤ t → 0 ≤ φ t := fun t ht =>
    mul_nonneg (hN0 t ht) (sub_nonneg.2 (hmax t ht J))
  refine eventually_atTop.2 ⟨T, fun t₀ ht₀ => ?_⟩
  by_contra hlt
  push Not at hlt
  set B : ℝ → ℝ := fun t => κ - (t - t₀)
  have hB : ∀ y, HasDerivAt B (-1) y := fun y => by
    simpa using ((hasDerivAt_id y).sub_const t₀).const_sub κ
  set t₁ := t₀ + |κ| + 1
  have hle : φ t₁ ≤ B t₁ := by
    refine image_le_of_deriv_right_lt_deriv_boundary' (a := t₀) (b := t₁) (B' := fun _ => -1)
      (continuous_iff_continuousAt.2 fun y => (hφ y).continuousAt).continuousOn
      (fun y _ => (hφ y).hasDerivWithinAt) (by simp only [B, sub_self, sub_zero]; exact hlt.le)
      (by fun_prop)
      (fun y _ => (hB y).hasDerivWithinAt) ?_ ⟨by linarith [abs_nonneg κ], le_rfl⟩
    rintro y ⟨hy, -⟩ hφB
    have := hslow y (ht₀.trans hy) (by rw [hφB]; simp only [B]; linarith)
    linarith
  have h0 := hpos t₁ (by linarith [abs_nonneg κ, (hT t₀ ht₀).2])
  simp only [B, t₁] at hle
  linarith [le_abs_self κ]

/-- **Point (3) of `l:boundedxn`.**  Under the hypotheses of
`tendsto_mul_sub_of_bounded`, `P_Nk(t) → 0` for every `k ≠ N`.

Source: arXiv:2305.05465v6, proof of `l:boundedxn`, point (3). -/
theorem tendsto_softmaxWeight_of_bounded (x : ℝ → Idx (m + 1) → ℝ)
    (hder : ∀ t k, HasDerivAt (fun s => x s k)
      (∑ j, Perspective.softmaxWeight (fun l => x t k * x t l) j * x t j) t)
    (N J L : Idx (m + 1)) (hJN : J ≠ N) (hmax : ∀ t, 0 ≤ t → ∀ k, x t k ≤ x t N)
    (hJ : ∀ t, 0 ≤ t → ∀ k, k ≠ N → x t k ≤ x t J) (hmin : ∀ t, 0 ≤ t → ∀ k, x t L ≤ x t k)
    {R : ℝ} (hR : ∀ t, 0 ≤ t → |x t N| ≤ R) {c : ℝ} (hc : 0 < c)
    (hJgrow : ∀ᶠ t in atTop, x t J ≤ -(c * Real.exp t)) (k : Idx (m + 1)) (hk : k ≠ N) :
    Tendsto (fun t => Perspective.softmaxWeight (fun l => x t N * x t l) k) atTop (𝓝 0) := by
  have hφ := tendsto_mul_sub_of_bounded x hder N J L hJN hmax hJ hmin hR hc hJgrow
  have hN0 := nonneg_of_bounded_max x hder N hmax fun t ht => (abs_le.1 (hR t ht)).1
  refine tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds
    (Real.tendsto_exp_neg_atTop_nhds_zero.comp hφ)
    (Eventually.of_forall fun t => Perspective.softmaxWeight_nonneg _ _) ?_
  filter_upwards [eventually_ge_atTop 0] with t ht
  refine (softmaxWeight_le_exp (x t) N k).trans (Real.exp_le_exp.2 ?_)
  have := mul_le_mul_of_nonneg_left (hJ t ht k hk) (hN0 t ht)
  nlinarith

end Clusters
end Transformer
