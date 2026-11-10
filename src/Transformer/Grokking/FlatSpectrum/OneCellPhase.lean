module

public import Mathlib.Analysis.Convex.StrictConvexSpace
public import Transformer.Grokking.FlatSpectrum.OneCellRay

/-!
# The value that minimizes the norm has the opposite phase

Source: `review/flat_spectrum.md`, *One unseen cell* (added), item OC4.

With `κ ≠ 0`, `s = |κ|`, `g > 0` and `s² + g ≤ c`, a value `x` that minimizes the norm `N` of
`OneCellRay` is `−θ κ` with `g / (2c − g) ≤ θ ≤ g / (c − g)`
(`exists_neg_mul_of_forall_cellNorm_le`):

- off the closed ray opposite to `κ`, `|x − κ| < |x| + s`, and the point of the ray with the
  same modulus has a smaller `N`;
- on the ray `N² = 2c − s² + r² + 2 √Δ(r)`, and a modulus `r < g s / (2c − g)` is not a
  minimum, by `exists_ray_lt` (`le_mul_of_forall_cellNorm_le`);
- beyond `ρ = g s / (c − g)` both `r²` and `Δ(r) = Δ(ρ) + (c − g) (r − ρ)²` increase
  (`mul_sub_le_of_forall_cellNorm_le`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

/-- **Before `g s / (2c − g)` the norm decreases along the ray** (`review/flat_spectrum.md`,
OC4): a minimizing value `−θ κ` with `θ ≥ 0` has `g ≤ θ (2c − g)`. -/
lemma le_mul_of_forall_cellNorm_le {c g θ : ℝ} {κ : ℂ} (hg : 0 < g) (hκ : κ ≠ 0)
    (hcg : ‖κ‖ ^ 2 + g ≤ c) (hθ : 0 ≤ θ)
    (hmin : ∀ y, cellNorm c g κ (-(θ : ℂ) * κ) ≤ cellNorm c g κ y) : g ≤ θ * (2 * c - g) := by
  have hs : 0 < ‖κ‖ := norm_pos_iff.2 hκ
  by_contra hlt
  rw [not_le] at hlt
  have hr0 : 0 ≤ θ * ‖κ‖ := mul_nonneg hθ hs.le
  obtain ⟨r₁, hr₁, hF⟩ := exists_ray_lt hg hs hcg hr0
    (by linarith [mul_pos (sub_pos.2 hlt) hs] : θ * ‖κ‖ * (2 * c - g) < g * ‖κ‖)
  have hx : -(θ : ℂ) * κ = -((θ * ‖κ‖ / ‖κ‖ : ℝ) : ℂ) * κ := by
    rw [mul_div_assoc, div_self hs.ne', mul_one]
  have h1 := hmin (-((r₁ / ‖κ‖ : ℝ) : ℂ) * κ)
  rw [hx, cellNorm_ray hκ hcg hr0, cellNorm_ray hκ hcg hr₁.le] at h1
  exact absurd h1 (not_le.2 (Real.sqrt_lt_sqrt (ray_outer_nonneg hg.le hcg r₁) (by linarith)))

/-- **Beyond `g s / (c − g)` the norm increases along the ray** (`review/flat_spectrum.md`,
OC4): a minimizing value `−θ κ` has `θ (c − g) ≤ g`. -/
lemma mul_sub_le_of_forall_cellNorm_le {c g θ : ℝ} {κ : ℂ} (hg : 0 ≤ g) (hκ : κ ≠ 0)
    (hcg : ‖κ‖ ^ 2 + g ≤ c)
    (hmin : ∀ y, cellNorm c g κ (-(θ : ℂ) * κ) ≤ cellNorm c g κ y) : θ * (c - g) ≤ g := by
  have hs : 0 < ‖κ‖ := norm_pos_iff.2 hκ
  have hcg' : 0 < c - g := by nlinarith
  by_contra hlt
  rw [not_le] at hlt
  obtain ⟨ρ, hρ⟩ : ∃ ρ : ℝ, ρ = g * ‖κ‖ / (c - g) := ⟨_, rfl⟩
  have hρ0 : 0 ≤ ρ := by rw [hρ]; exact div_nonneg (mul_nonneg hg hs.le) hcg'.le
  have hrel : ρ * (c - g) = g * ‖κ‖ := by
    rw [hρ]
    field_simp
  obtain ⟨r, hr⟩ : ∃ r : ℝ, r = θ * ‖κ‖ := ⟨_, rfl⟩
  have hrρ : ρ < r := by
    refine lt_of_mul_lt_mul_right ?_ hcg'.le
    rw [hrel, hr]
    nlinarith [mul_lt_mul_of_pos_right hlt hs]
  have hΔ : c * (c + (r ^ 2 - ‖κ‖ ^ 2)) - g * (r + ‖κ‖) ^ 2 =
      c * (c + (ρ ^ 2 - ‖κ‖ ^ 2)) - g * (ρ + ‖κ‖) ^ 2 + (c - g) * (r - ρ) ^ 2 := by
    linear_combination (2 * (r - ρ)) * hrel
  have hx : -(θ : ℂ) * κ = -((r / ‖κ‖ : ℝ) : ℂ) * κ := by
    rw [hr, mul_div_assoc, div_self hs.ne', mul_one]
  have h1 := hmin (-((ρ / ‖κ‖ : ℝ) : ℂ) * κ)
  rw [hx, cellNorm_ray hκ hcg (hρ0.trans hrρ.le), cellNorm_ray hκ hcg hρ0] at h1
  have h2 : √(c * (c + (ρ ^ 2 - ‖κ‖ ^ 2)) - g * (ρ + ‖κ‖) ^ 2) ≤
      √(c * (c + (r ^ 2 - ‖κ‖ ^ 2)) - g * (r + ‖κ‖) ^ 2) :=
    Real.sqrt_le_sqrt (by rw [hΔ]; nlinarith [mul_nonneg hcg'.le (sq_nonneg (r - ρ))])
  have h3 : ρ ^ 2 < r ^ 2 := pow_lt_pow_left₀ hrρ hρ0 two_ne_zero
  exact absurd h1 (not_le.2 (Real.sqrt_lt_sqrt (ray_outer_nonneg hg hcg ρ) (by linarith)))

/-- **A value that minimizes the norm is on the open ray opposite to `κ`**
(`review/flat_spectrum.md`, OC4): `x = −θ κ` with `0 < θ`, `g ≤ θ (2c − g)` and
`θ (c − g) ≤ g`. -/
theorem exists_neg_mul_of_forall_cellNorm_le {c g : ℝ} {κ x : ℂ} (hg : 0 < g) (hκ : κ ≠ 0)
    (hcg : ‖κ‖ ^ 2 + g ≤ c) (hmin : ∀ y, cellNorm c g κ x ≤ cellNorm c g κ y) :
    ∃ θ : ℝ, 0 < θ ∧ g ≤ θ * (2 * c - g) ∧ θ * (c - g) ≤ g ∧ x = -(θ : ℂ) * κ := by
  have hs : 0 < ‖κ‖ := norm_pos_iff.2 hκ
  obtain ⟨θ, hθ, rfl⟩ : ∃ θ : ℝ, 0 ≤ θ ∧ x = -(θ : ℂ) * κ := by
    by_cases hsame : SameRay ℝ x (-κ)
    · obtain h0 | h0 | ⟨r₁, r₂, hr₁, hr₂, h12⟩ := hsame
      · exact ⟨0, le_rfl, by simp [h0]⟩
      · exact absurd (neg_eq_zero.1 h0) hκ
      · have h3 : (r₁ : ℂ) * x = -((r₂ : ℂ) * κ) := by
          simpa [Complex.real_smul] using h12
        have hr₁' : (r₁ : ℂ) ≠ 0 := Complex.ofReal_ne_zero.2 hr₁.ne'
        refine ⟨r₂ / r₁, (div_pos hr₂ hr₁).le, ?_⟩
        rw [Complex.ofReal_div]
        field_simp
        linear_combination h3
    · exfalso
      have hlt : ‖x - κ‖ < ‖x‖ + ‖κ‖ := by
        have h := not_sameRay_iff_norm_add_lt.1 hsame
        rwa [← sub_eq_add_neg, norm_neg] at h
      have h1 := hmin (-((‖x‖ / ‖κ‖ : ℝ) : ℂ) * κ)
      rw [cellNorm_ray hκ hcg (norm_nonneg x), cellNorm] at h1
      have hsq : ‖x - κ‖ ^ 2 < (‖x‖ + ‖κ‖) ^ 2 :=
        pow_lt_pow_left₀ hlt (norm_nonneg _) two_ne_zero
      have hΔ : c * (c + (‖x‖ ^ 2 - ‖κ‖ ^ 2)) - g * (‖x‖ + ‖κ‖) ^ 2 <
          |c * (c + (‖x‖ ^ 2 - ‖κ‖ ^ 2)) - g * ‖x - κ‖ ^ 2| :=
        lt_of_lt_of_le (by nlinarith) (le_abs_self _)
      have hroot := Real.sqrt_lt_sqrt (ray_nonneg hs hcg ‖x‖) hΔ
      exact absurd h1 (not_le.2 (Real.sqrt_lt_sqrt
        (ray_outer_nonneg hg.le hcg ‖x‖) (by linarith)))
  have hlow := le_mul_of_forall_cellNorm_le hg hκ hcg hθ hmin
  refine ⟨θ, lt_of_le_of_ne hθ fun h0 => ?_, hlow,
    mul_sub_le_of_forall_cellNorm_le hg.le hκ hcg hmin, rfl⟩
  rw [← h0, zero_mul] at hlow
  exact absurd hg (not_lt.2 hlow)

/-- The numbers of the Hadamard matrix of size two, `c = 2`, `g = 1`, `κ = 1`: a minimizing
value exists, and it is `−θ` with `1 / 3 ≤ θ ≤ 1`. -/
example : ∃ x : ℂ, ∃ θ : ℝ, 0 < θ ∧ 1 ≤ θ * (2 * 2 - 1) ∧ θ * (2 - 1) ≤ 1 ∧
    x = -(θ : ℂ) * 1 := by
  obtain ⟨x, hx⟩ := exists_forall_cellNorm_le (c := 2) (g := 1) (κ := 1) (by norm_num)
  exact ⟨x, exists_neg_mul_of_forall_cellNorm_le one_pos one_ne_zero (by norm_num) hx⟩

example : ∃ θ : ℝ, 1 ≤ θ * (2 * 2 - 1) ∧ θ * (2 - 1) ≤ 1 := by
  obtain ⟨x, hx⟩ := exists_forall_cellNorm_le (c := 2) (g := 1) (κ := 1) (by norm_num)
  obtain ⟨θ, h0, -, -, rfl⟩ :=
    exists_neg_mul_of_forall_cellNorm_le one_pos one_ne_zero (by norm_num) hx
  exact ⟨θ, le_mul_of_forall_cellNorm_le one_pos one_ne_zero (by norm_num) h0.le hx,
    mul_sub_le_of_forall_cellNorm_le zero_le_one one_ne_zero (by norm_num) hx⟩

end Transformer.Grokking.FlatSpectrum
