module

public import Mathlib.Analysis.Normed.Group.Bounded
public import Mathlib.Topology.Order.Compact
public import Transformer.Grokking.FlatSpectrum.OneCellNorm

/-!
# The norm as a function of the value of the changed cell

Source: `review/flat_spectrum.md`, *One unseen cell* (added), item OC4.

Write `x` for the value of the changed cell, `κ` for its value in `K`, `s = |κ| > 0`, and let
`s² + g ≤ c`, `g > 0`. The norm of OC3 depends on `x` through

    N(x) = √(2c + |x|² − s² + 2 √|c (c + |x|² − s²) − g |x − κ|²|)           (`cellNorm`).

On the ray opposite to `κ`, at `|x| = r`, the inner square root is of
`Δ(r) = c (c + r² − s²) − g (r + s)²`, and this is not negative (`ray_nonneg`, `cellNorm_ray`):
`(c − g) Δ(r) = c² (c − g − s²) + ((c − g) r − g s)²`.

Near zero the norm decreases along this ray (`exists_ray_lt`). With `w² = Δ(0) ≤ c²`,
`Δ(r) = w² + ((c − g) r − 2 g s) r`. For `0 ≤ r < r₁ = g s / (w + c − g)` both `√Δ(r)` and
`√Δ(r₁)` are at most `w`, and `Δ(r) − Δ(r₁) > w (r₁² − r²)`, so
`r₁² + 2 √Δ(r₁) < r² + 2 √Δ(r)`. The lemma asks for `r < g s / (2c − g)`, which is at most `r₁`.

`N` is continuous and `N(x) ≥ |x|`, so some value minimizes it (`exists_forall_cellNorm_le`).
-/

@[expose] public section

namespace Transformer.Grokking.FlatSpectrum

/-- **On the ray opposite to `κ` the inner square root is of a nonnegative number**
(`review/flat_spectrum.md`, OC4). -/
lemma ray_nonneg {c g s : ℝ} (hs : 0 < s) (hcg : s ^ 2 + g ≤ c) (r : ℝ) :
    0 ≤ c * (c + (r ^ 2 - s ^ 2)) - g * (r + s) ^ 2 := by
  have h1 : 0 < c - g := by nlinarith
  have h2 : (c - g) * (c * (c + (r ^ 2 - s ^ 2)) - g * (r + s) ^ 2) =
      c ^ 2 * (c - g - s ^ 2) + ((c - g) * r - g * s) ^ 2 := by ring
  have h3 : 0 ≤ (c - g) * (c * (c + (r ^ 2 - s ^ 2)) - g * (r + s) ^ 2) := by
    rw [h2]
    exact add_nonneg (mul_nonneg (sq_nonneg c) (by linarith)) (sq_nonneg _)
  exact nonneg_of_mul_nonneg_right h3 h1

example (r : ℝ) := ray_nonneg (c := 2) (g := 1) (s := 1) one_pos (by norm_num) r

/-- The number under the outer square root is not negative on the ray. -/
lemma ray_outer_nonneg {c g s : ℝ} (hg : 0 ≤ g) (hcg : s ^ 2 + g ≤ c) (r : ℝ) :
    0 ≤ 2 * c + (r ^ 2 - s ^ 2) + 2 * √(c * (c + (r ^ 2 - s ^ 2)) - g * (r + s) ^ 2) := by
  linarith [Real.sqrt_nonneg (c * (c + (r ^ 2 - s ^ 2)) - g * (r + s) ^ 2), sq_nonneg r,
    sq_nonneg s]

example (r : ℝ) := ray_outer_nonneg (c := 2) (g := 1) (s := 1) zero_le_one (by norm_num) r

/-- **Near zero the norm decreases along the opposite ray** (`review/flat_spectrum.md`, OC4):
for `0 ≤ r < g s / (2c − g)` some `r₁ > 0` has `r₁² + 2 √Δ(r₁) < r² + 2 √Δ(r)`, where
`Δ(ρ) = c (c + ρ² − s²) − g (ρ + s)²`. -/
lemma exists_ray_lt {c g s r : ℝ} (hg : 0 < g) (hs : 0 < s) (hcg : s ^ 2 + g ≤ c) (hr0 : 0 ≤ r)
    (hr : r * (2 * c - g) < g * s) :
    ∃ r₁ : ℝ, 0 < r₁ ∧ r₁ ^ 2 + 2 * √(c * (c + (r₁ ^ 2 - s ^ 2)) - g * (r₁ + s) ^ 2) <
      r ^ 2 + 2 * √(c * (c + (r ^ 2 - s ^ 2)) - g * (r + s) ^ 2) := by
  have hA : 0 < c - g := by linarith [pow_pos hs 2]
  have hc0 : 0 < c := by linarith
  have hcs : g ≤ c - s ^ 2 := by linarith
  have hΔ0 : 0 < c * (c - s ^ 2) - g * s ^ 2 := by
    linarith [mul_le_mul_of_nonneg_left hcs hc0.le, mul_le_mul_of_nonneg_left hcs hg.le,
      mul_pos hg hg]
  obtain ⟨w, hw, hw2⟩ : ∃ w : ℝ, 0 < w ∧ w ^ 2 = c * (c - s ^ 2) - g * s ^ 2 :=
    ⟨_, Real.sqrt_pos.2 hΔ0, Real.sq_sqrt hΔ0.le⟩
  have hwc : w ≤ c := (pow_le_pow_iff_left₀ hw.le hc0.le two_ne_zero).1 (by
    rw [hw2]
    linarith [mul_nonneg hc0.le (sq_nonneg s), mul_nonneg hg.le (sq_nonneg s)])
  obtain ⟨r₁, hr₁⟩ : ∃ r₁ : ℝ, r₁ = g * s / (w + (c - g)) := ⟨_, rfl⟩
  have hr₁pos : 0 < r₁ := by rw [hr₁]; positivity
  have hrel₁ : r₁ * (w + (c - g)) = g * s := by
    rw [hr₁]
    field_simp
  have hrel : r * (w + (c - g)) < g * s := by linarith [mul_nonneg hr0 (sub_nonneg.2 hwc)]
  have hrr : r < r₁ := by
    refine lt_of_mul_lt_mul_right ?_ (add_pos hw hA).le
    rw [hrel₁]
    exact hrel
  have hΔ (ρ : ℝ) : c * (c + (ρ ^ 2 - s ^ 2)) - g * (ρ + s) ^ 2 =
      w ^ 2 + ((c - g) * ρ - 2 * g * s) * ρ := by
    rw [hw2]
    ring
  obtain ⟨a, ha⟩ : ∃ a : ℝ, a = √(c * (c + (r ^ 2 - s ^ 2)) - g * (r + s) ^ 2) := ⟨_, rfl⟩
  obtain ⟨b, hb⟩ : ∃ b : ℝ, b = √(c * (c + (r₁ ^ 2 - s ^ 2)) - g * (r₁ + s) ^ 2) := ⟨_, rfl⟩
  have ha2 : a ^ 2 = w ^ 2 + ((c - g) * r - 2 * g * s) * r := by
    rw [ha, Real.sq_sqrt (ray_nonneg hs hcg r), hΔ]
  have hb2 : b ^ 2 = w ^ 2 + ((c - g) * r₁ - 2 * g * s) * r₁ := by
    rw [hb, Real.sq_sqrt (ray_nonneg hs hcg r₁), hΔ]
  have ha0 : 0 ≤ a := by rw [ha]; exact Real.sqrt_nonneg _
  have hb0 : 0 ≤ b := by rw [hb]; exact Real.sqrt_nonneg _
  have h8 : (c - g) * r - 2 * g * s ≤ 0 := by linarith [mul_nonneg hr0 hw.le, mul_pos hg hs]
  have h9 : (c - g) * r₁ - 2 * g * s ≤ 0 := by linarith [mul_pos hr₁pos hw, mul_pos hg hs]
  have haw : a ≤ w := (pow_le_pow_iff_left₀ ha0 hw.le two_ne_zero).1 (by
    rw [ha2]
    linarith [mul_nonneg hr0 (neg_nonneg.2 h8)])
  have hbw : b ≤ w := (pow_le_pow_iff_left₀ hb0 hw.le two_ne_zero).1 (by
    rw [hb2]
    linarith [mul_nonneg hr₁pos.le (neg_nonneg.2 h9)])
  have h5 : (r₁ + r) * w < 2 * g * s - (c - g) * (r + r₁) := by linarith
  have hkey : (r₁ - r) * ((r₁ + r) * w) < a ^ 2 - b ^ 2 := by
    rw [ha2, hb2]
    linarith [mul_lt_mul_of_pos_left h5 (sub_pos.2 hrr)]
  have hab : b < a := by
    refine lt_of_pow_lt_pow_left₀ 2 ha0 ?_
    linarith [mul_nonneg (sub_nonneg.2 hrr.le) (mul_nonneg (add_nonneg hr₁pos.le hr0) hw.le)]
  have h6 : (r₁ - r) * (r₁ + r) < 2 * (a - b) := by
    refine lt_of_mul_lt_mul_right ?_ hw.le
    linarith [mul_le_mul_of_nonneg_left (add_le_add haw hbw) (sub_nonneg.2 hab.le)]
  refine ⟨r₁, hr₁pos, ?_⟩
  rw [← ha, ← hb]
  linarith

example := exists_ray_lt (c := 2) (g := 1) (s := 1) (r := 0) one_pos one_pos (by norm_num)
  le_rfl (by norm_num)

/-- **The norm as a function of the value `x` of the changed cell**
(`review/flat_spectrum.md`, OC4). -/
noncomputable def cellNorm (c g : ℝ) (κ x : ℂ) : ℝ :=
  √(2 * c + (‖x‖ ^ 2 - ‖κ‖ ^ 2) + 2 * √|c * (c + (‖x‖ ^ 2 - ‖κ‖ ^ 2)) - g * ‖x - κ‖ ^ 2|)

/-- The norm on the ray opposite to `κ`, at the modulus `r`. -/
lemma cellNorm_ray {c g : ℝ} {κ : ℂ} (hκ : κ ≠ 0) (hcg : ‖κ‖ ^ 2 + g ≤ c)
    {r : ℝ} (hr : 0 ≤ r) : cellNorm c g κ (-((r / ‖κ‖ : ℝ) : ℂ) * κ) =
      √(2 * c + (r ^ 2 - ‖κ‖ ^ 2) + 2 * √(c * (c + (r ^ 2 - ‖κ‖ ^ 2)) - g * (r + ‖κ‖) ^ 2)) := by
  have hs : 0 < ‖κ‖ := norm_pos_iff.2 hκ
  have h1 : ‖-((r / ‖κ‖ : ℝ) : ℂ) * κ‖ = r := by
    rw [norm_mul, norm_neg, Complex.norm_real, Real.norm_of_nonneg (div_nonneg hr hs.le),
      div_mul_cancel₀ _ hs.ne']
  have h2 : ‖-((r / ‖κ‖ : ℝ) : ℂ) * κ - κ‖ = r + ‖κ‖ := by
    have h3 : -((r / ‖κ‖ : ℝ) : ℂ) * κ - κ = -(((r / ‖κ‖ + 1 : ℝ) : ℂ) * κ) := by
      push_cast
      ring
    rw [h3, norm_neg, norm_mul, Complex.norm_real, Real.norm_of_nonneg (by positivity), add_mul,
      div_mul_cancel₀ _ hs.ne', one_mul]
  rw [cellNorm, h1, h2, abs_of_nonneg (ray_nonneg hs hcg r)]

example := cellNorm_ray (c := 2) (g := 1) (κ := 1) one_ne_zero (by norm_num)
  (r := 1) zero_le_one

/-- The norm is at least the modulus of the value. -/
lemma norm_le_cellNorm {c g : ℝ} {κ : ℂ} (hc : ‖κ‖ ^ 2 ≤ 2 * c) (x : ℂ) :
    ‖x‖ ≤ cellNorm c g κ x :=
  Real.le_sqrt_of_sq_le (by
    linarith [Real.sqrt_nonneg |c * (c + (‖x‖ ^ 2 - ‖κ‖ ^ 2)) - g * ‖x - κ‖ ^ 2|])

example (x : ℂ) := norm_le_cellNorm (c := 2) (g := 1) (κ := 1) (by norm_num) x

/-- **Some value minimizes the norm** (`review/flat_spectrum.md`, OC4). -/
lemma exists_forall_cellNorm_le {c g : ℝ} {κ : ℂ} (hc : ‖κ‖ ^ 2 ≤ 2 * c) :
    ∃ x, ∀ y, cellNorm c g κ x ≤ cellNorm c g κ y := by
  have hcont : Continuous (cellNorm c g κ) := by
    unfold cellNorm
    fun_prop
  exact hcont.exists_forall_le
    (Filter.tendsto_atTop_mono (norm_le_cellNorm hc) tendsto_norm_cocompact_atTop)

example := exists_forall_cellNorm_le (c := 2) (g := 1) (κ := 1) (by norm_num)

end Transformer.Grokking.FlatSpectrum
