/-
# The emergence of clusters in self-attention dynamics — `l:boundedother`, the escape

The heart of the proof of `l:boundedother` (§7 of arXiv:2305.05465v6), on the
scalar curves of `e:Idnonresca`.  With `y = x_{i₀} e^t` and
`a_k(t) = x_k(t) e^{-t}`, the equation of the bounded token is exactly
`ẏ = e^{2t} g_{a(t)}(y) + y` (`hasDerivAt_rescaled`), `g` the softmax mean.

If `a(t) → γ` and `g_γ(θ) > 0`, then `y` is eventually below `θ`
(`eventually_rescaled_lt`): once `y ≥ θ`, monotonicity of `g` gives
`ẏ ≥ (δ/4) e^{2t}`, so `y ≥ θ + (δ/8)(e^{2t} - e^{2t₁})` and `x_{i₀} = y e^{-t}`
is unbounded.  With the mirror image, `y → θ₀` (`tendsto_rescaled`).

**What the source says and what is carried here.**  The source's argument
runs through `x_{i₀} → 0` and `|y| = o(e^t)`; the barrier above contradicts
the boundedness of `x_{i₀}` directly, and the limit `x_{i₀} → 0` is not used.

Source: arXiv:2305.05465v6, proof of `l:boundedother`.
-/

import Transformer.Clusters.Section7_BoundedOtherMean

open scoped BigOperators
open Real Filter Topology

namespace Transformer
namespace Clusters

variable {m : ℕ}

/-- **The rescaled equation.**  `y = x_i e^t` solves `ẏ = e^{2t} g_{a}(y) + y`
with `a_k = x_k e^{-t}`.

Source: arXiv:2305.05465v6, proof of `l:boundedother`, the equation of `ẏ_{i₀}`. -/
theorem hasDerivAt_rescaled (x : ℝ → Idx (m + 1) → ℝ)
    (hder : ∀ t k, HasDerivAt (fun s => x s k)
      (∑ j, Perspective.softmaxWeight (fun l => x t k * x t l) j * x t j) t)
    (i : Idx (m + 1)) (t : ℝ) :
    HasDerivAt (fun s => x s i * Real.exp s)
      (Real.exp t ^ 2 * softmaxMean (fun k => x t k / Real.exp t) (x t i * Real.exp t) +
        x t i * Real.exp t) t := by
  have he := Real.exp_pos t
  have hs : (fun l => x t i * Real.exp t * (x t l / Real.exp t)) = fun l => x t i * x t l :=
    funext fun l => by field_simp
  convert (hder t i).mul (Real.hasDerivAt_exp t) using 1
  rw [softmaxMean, hs, Finset.mul_sum, Finset.sum_mul]
  congr 1
  exact Finset.sum_congr rfl fun k _ => by field_simp

/-- **The escape.**  If `x_i` is bounded, `x_k e^{-t} → γ_k` for every `k`,
and `g_γ(θ) > 0`, then `x_i e^t < θ` for all large `t`.

Source: arXiv:2305.05465v6, proof of `l:boundedother`, "if
`y_{i₀}(t) > θ₀ + ε` for some large time". -/
theorem eventually_rescaled_lt (x : ℝ → Idx (m + 1) → ℝ)
    (hder : ∀ t k, HasDerivAt (fun s => x s k)
      (∑ j, Perspective.softmaxWeight (fun l => x t k * x t l) j * x t j) t)
    (hm : 0 < m) (i : Idx (m + 1)) {R : ℝ} (hR : ∀ t, 0 ≤ t → |x t i| ≤ R)
    (γ : Idx (m + 1) → ℝ) (hγ : ∀ k, Tendsto (fun t => x t k / Real.exp t) atTop (𝓝 (γ k)))
    {θ : ℝ} (hθ : 0 < softmaxMean γ θ) :
    ∀ᶠ t in atTop, x t i * Real.exp t < θ := by
  set δ := softmaxMean γ θ
  have hconv : Tendsto (fun t => softmaxMean (fun k => x t k / Real.exp t) θ) atTop (𝓝 δ) :=
    tendsto_softmaxMean (a := fun t k => x t k / Real.exp t) hγ θ
  obtain ⟨T, hT⟩ := eventually_atTop.1 (((hconv.eventually (lt_mem_nhds (half_lt_self hθ))).and
    ((tendsto_exp_atTop.const_mul_atTop (by positivity : 0 < δ / 4)).eventually
      (eventually_gt_atTop (-θ)))).and (eventually_ge_atTop 0))
  refine eventually_atTop.2 ⟨T, fun t₁ ht₁ => ?_⟩
  by_contra hge
  push Not at hge
  set C := δ / 8 * Real.exp t₁ ^ 2
  set B : ℝ → ℝ := fun s => θ + (δ / 8 * Real.exp s ^ 2 - C)
  have hB : ∀ s, HasDerivAt B (δ / 4 * Real.exp s ^ 2) s := fun s => by
    have := (((Real.hasDerivAt_exp s).pow 2).const_mul (δ / 8)).sub_const C |>.const_add θ
    convert this using 1
    simp only [Nat.cast_ofNat]; ring
  have hy := hasDerivAt_rescaled x hder i
  have hlow : ∀ t, t₁ ≤ t → B t ≤ x t i * Real.exp t := fun t ht => by
    refine image_le_of_deriv_right_lt_deriv_boundary' (a := t₁) (b := t)
      (f' := fun s => δ / 4 * Real.exp s ^ 2)
      (continuous_iff_continuousAt.2 fun s => (hB s).continuousAt).continuousOn
      (fun s _ => (hB s).hasDerivWithinAt) (by simp only [B, C, sub_self, add_zero]; exact hge)
      (continuous_iff_continuousAt.2 fun s => (hy s).continuousAt).continuousOn
      (fun s _ => (hy s).hasDerivWithinAt) ?_ ⟨ht, le_rfl⟩
    rintro s ⟨hs, -⟩ hBs
    obtain ⟨⟨hG, hθs⟩, hs0⟩ := hT s (ht₁.trans hs)
    have hes : Real.exp t₁ ^ 2 ≤ Real.exp s ^ 2 :=
      pow_le_pow_left₀ (Real.exp_pos t₁).le (Real.exp_le_exp.2 hs) 2
    have hθy : θ ≤ x s i * Real.exp s := by
      rw [← hBs]
      have := mul_le_mul_of_nonneg_left hes (by positivity : (0 : ℝ) ≤ δ / 8)
      simp only [B, C]; linarith
    have hGy := softmaxMean_mono (fun k => x s k / Real.exp s) hm hθy
    have he1 : 1 ≤ Real.exp s := Real.one_le_exp hs0
    have hsq : 0 < Real.exp s ^ 2 := by positivity
    have hee : Real.exp s ≤ Real.exp s ^ 2 := by
      rw [sq]; exact le_mul_of_one_le_left (by positivity) he1
    have h1 := mul_le_mul_of_nonneg_left hee (by positivity : (0 : ℝ) ≤ δ / 4)
    have h2 := mul_le_mul_of_nonneg_left hGy hsq.le
    have h3 := mul_lt_mul_of_pos_left hG hsq
    show δ / 4 * Real.exp s ^ 2 <
      Real.exp s ^ 2 * softmaxMean (fun k => x s k / Real.exp s) (x s i * Real.exp s) +
        x s i * Real.exp s
    linarith
  set V := 8 * (|R| + |θ| + C + 1) / δ
  obtain ⟨t, ht, hV⟩ := ((eventually_ge_atTop (max t₁ 0)).and
    (tendsto_exp_atTop.eventually (eventually_ge_atTop V))).exists
  have h1 := hlow t (le_of_max_le_left ht)
  have ht0 : 0 ≤ t := le_of_max_le_right ht
  have hRt := (abs_le.1 (hR t ht0)).2
  have he1 : 1 ≤ Real.exp t := Real.one_le_exp ht0
  have hC : 0 ≤ C := by positivity
  have hVe : δ / 8 * V = |R| + |θ| + C + 1 := by simp only [V]; field_simp
  have h2 : (|R| + |θ| + C + 1) * Real.exp t ≤ δ / 8 * Real.exp t ^ 2 := by
    have := mul_le_mul_of_nonneg_left hV (by positivity : (0 : ℝ) ≤ δ / 8 * Real.exp t)
    have h3 : δ / 8 * V * Real.exp t = (|R| + |θ| + C + 1) * Real.exp t := by rw [hVe]
    rw [sq]; linarith
  have h3 := mul_le_mul_of_nonneg_right (hRt.trans (le_abs_self R)) (Real.exp_pos t).le
  have h4 := mul_le_mul_of_nonneg_left he1 (by positivity : (0 : ℝ) ≤ |θ| + C + 1)
  have h5 : B t = θ + (δ / 8 * Real.exp t ^ 2 - C) := rfl
  linarith [neg_abs_le θ]

/-- **`y_{i₀} → θ₀`.**  If `x_i` is bounded, `x_k e^{-t} → γ_k` for every
`k`, and some rate is positive and some negative, then `x_i e^t` converges.

Source: arXiv:2305.05465v6, proof of `l:boundedother`, "`y_{i₀}(t) → θ₀`". -/
theorem tendsto_rescaled (x : ℝ → Idx (m + 1) → ℝ)
    (hder : ∀ t k, HasDerivAt (fun s => x s k)
      (∑ j, Perspective.softmaxWeight (fun l => x t k * x t l) j * x t j) t)
    (hm : 0 < m) (i : Idx (m + 1)) {R : ℝ} (hR : ∀ t, 0 ≤ t → |x t i| ≤ R)
    (γ : Idx (m + 1) → ℝ) (hγ : ∀ k, Tendsto (fun t => x t k / Real.exp t) atTop (𝓝 (γ k)))
    {p q : Idx (m + 1)} (hp : 0 < γ p) (hq : γ q < 0) :
    ∃ θ₀, Tendsto (fun t => x t i * Real.exp t) atTop (𝓝 θ₀) := by
  obtain ⟨θ₀, hθ₀⟩ := exists_softmaxMean_root γ hp hq
  refine ⟨θ₀, tendsto_order.2 ⟨fun a ha => ?_, fun b hb =>
    eventually_rescaled_lt x hder hm i hR γ hγ ((hθ₀ b).2 hb)⟩⟩
  have hder' : ∀ t k, HasDerivAt (fun s => -x s k)
      (∑ j, Perspective.softmaxWeight (fun l => -x t k * -x t l) j * -x t j) t := by
    intro t k
    convert (hder t k).fun_neg using 1
    simp only [mul_neg, neg_mul, neg_neg, Finset.sum_neg_distrib]
  have hpos : 0 < softmaxMean (fun k => -γ k) (-a) := by
    rw [softmaxMean_neg, neg_neg]; linarith [(hθ₀ a).1 ha]
  have := eventually_rescaled_lt (fun t k => -x t k) hder' hm i
    (fun t ht => by simpa only [abs_neg] using hR t ht) (fun k => -γ k)
    (fun k => by simpa only [neg_div] using (hγ k).neg) hpos
  filter_upwards [this] with t ht
  linarith

end Clusters
end Transformer
