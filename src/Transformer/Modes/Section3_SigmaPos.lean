/-
# The number of modes of a Gaussian KDE — `Σ_t` is positive definite

The whitening `Σ_t^{-1/2}` of `eq:Yi` in arXiv:2412.09080v3 exists only when
`Σ_t` is positive definite, which the source takes for granted.  It holds for
every `β > 0`:

* `G(t)` and `G'(t)` are bounded in `X` for `β > 0` (`abs_bigG_le`,
  `abs_bigG'_le`), so every moment below is finite.
* `Var G(t) > 0`: otherwise the continuous `G(t, ·)` is constant, since
  `N(0,1)` charges every open set; but it vanishes at `X = t` and not at
  `X = t - 1`.
* `det Σ_t > 0`: by the equality case of Cauchy–Schwarz,
  `det Σ_t = Var G · 𝔼[(G' - 𝔼G' - λ(G - 𝔼G))²]` with `λ = Cov/Var G`, and the
  second factor vanishes only if `G' = α + c G` identically in `u = t - X`,
  impossible since `G'` is even in `u` and `G` odd: `α = G'(u=0) = 1` while
  `G'(u) + G'(-u) = 0` at `βu² = 1`.

Source: arXiv:2412.09080v3, `eq:Yi` and the sentence after it.
-/

import Transformer.Modes.Section3_Edgeworth

open Real MeasureTheory ProbabilityTheory

namespace Transformer
namespace Modes

/-- `G(t)` is bounded by `1 + 2/β`. -/
theorem abs_bigG_le {β : ℝ} (hβ : 0 < β) (t x : ℝ) : |bigG β t x| ≤ 1 + 2 / β := by
  set u := t - x
  have he := Real.add_one_le_exp (β / 2 * u ^ 2)
  have hpos : 0 < Real.exp (β / 2 * u ^ 2) := Real.exp_pos _
  have hu : |u| ≤ (1 + 2 / β) * (β / 2 * u ^ 2 + 1) := by
    have h1 : (1 + 2 / β) * (β / 2 * u ^ 2 + 1) = 1 + u ^ 2 + β / 2 * u ^ 2 + 2 / β := by
      field_simp; ring
    have h2 : |u| ≤ 1 + u ^ 2 := by nlinarith [abs_nonneg u, sq_abs u]
    have h3 : 0 ≤ β / 2 * u ^ 2 + 2 / β := by positivity
    linarith
  rw [bigG, abs_mul, Real.abs_exp, show -(β / 2) * u ^ 2 = -(β / 2 * u ^ 2) by ring,
    Real.exp_neg, inv_mul_le_iff₀ hpos]
  exact hu.trans ((mul_le_mul_of_nonneg_left he (by positivity)).trans_eq (mul_comm _ _))

/-- `G'(t)` is bounded by `2`. -/
theorem abs_bigG'_le {β : ℝ} (hβ : 0 < β) (t x : ℝ) : |bigG' β t x| ≤ 2 := by
  set u := t - x
  have he := Real.add_one_le_exp (β / 2 * u ^ 2)
  have hpos : 0 < Real.exp (β / 2 * u ^ 2) := Real.exp_pos _
  have hu : |1 - β * u ^ 2| ≤ 2 * (β / 2 * u ^ 2 + 1) := by
    rw [abs_le]; constructor <;> nlinarith [sq_nonneg u]
  rw [bigG', abs_mul, Real.abs_exp, show -(β / 2) * u ^ 2 = -(β / 2 * u ^ 2) by ring,
    Real.exp_neg, inv_mul_le_iff₀ hpos]
  exact hu.trans ((mul_le_mul_of_nonneg_left he (by positivity)).trans_eq (mul_comm _ _))

/-- `N(0,1)` charges every nonempty open set. -/
theorem gaussianReal_pos_of_isOpen {U : Set ℝ} (hU : IsOpen U) (hne : U.Nonempty) :
    0 < gaussianReal 0 1 U := by
  refine pos_iff_ne_zero.2 fun h => (hU.measure_pos volume hne).ne' ?_
  exact gaussianReal_absolutelyContinuous' 0 one_ne_zero h

/-- A continuous nonnegative integrable function, positive at one point, has a
positive integral under `N(0,1)`. -/
theorem integral_gaussianReal_pos {f : ℝ → ℝ} (hf : Continuous f) (h0 : 0 ≤ f)
    (hi : Integrable f (gaussianReal 0 1)) {x₀ : ℝ} (hx₀ : 0 < f x₀) :
    0 < ∫ x, f x ∂gaussianReal 0 1 := by
  refine (integral_pos_iff_support_of_nonneg h0 hi).2 (gaussianReal_pos_of_isOpen ?_ ⟨x₀, ?_⟩)
  · exact hf.isOpen_support
  · exact hx₀.ne'

/-- Linearity of `𝔼` on the quadratic polynomials in two square-integrable
functions. -/
theorem integral_quad {μ : Measure ℝ} [IsProbabilityMeasure μ] {f g : ℝ → ℝ}
    (hf : Integrable f μ) (hg : Integrable g μ) (hff : Integrable (fun x => f x ^ 2) μ)
    (hfg : Integrable (fun x => f x * g x) μ) (hgg : Integrable (fun x => g x ^ 2) μ)
    (c₀ c₁ c₂ c₃ c₄ c₅ : ℝ) :
    ∫ x, (c₀ + c₁ * f x + c₂ * g x + c₃ * f x ^ 2 + c₄ * (f x * g x) + c₅ * g x ^ 2) ∂μ =
      c₀ + c₁ * ∫ x, f x ∂μ + c₂ * ∫ x, g x ∂μ + c₃ * ∫ x, f x ^ 2 ∂μ
        + c₄ * ∫ x, f x * g x ∂μ + c₅ * ∫ x, g x ^ 2 ∂μ := by
  have i1 := hf.const_mul c₁
  have i2 := hg.const_mul c₂
  have i3 := hff.const_mul c₃
  have i4 := hfg.const_mul c₄
  have i5 := hgg.const_mul c₅
  have i0 : Integrable (fun _ : ℝ => c₀) μ := integrable_const _
  have j1 : Integrable (fun x => c₀ + c₁ * f x) μ := i0.add i1
  have j2 : Integrable (fun x => c₀ + c₁ * f x + c₂ * g x) μ := j1.add i2
  have j3 : Integrable (fun x => c₀ + c₁ * f x + c₂ * g x + c₃ * f x ^ 2) μ := j2.add i3
  have j4 : Integrable
      (fun x => c₀ + c₁ * f x + c₂ * g x + c₃ * f x ^ 2 + c₄ * (f x * g x)) μ := j3.add i4
  rw [integral_add j4 i5, integral_add j3 i4, integral_add j2 i3, integral_add j1 i2,
    integral_add i0 i1, integral_const, integral_const_mul, integral_const_mul,
    integral_const_mul, integral_const_mul, integral_const_mul]
  simp

/-- A continuous bounded function is integrable under `N(0,1)`. -/
theorem integrable_gaussianReal_of_bound {f : ℝ → ℝ} (hf : Continuous f) (C : ℝ)
    (hC : ∀ x, |f x| ≤ C) : Integrable f (gaussianReal 0 1) :=
  Integrable.of_bound hf.aestronglyMeasurable C (ae_of_all _ fun x => by simpa using hC x)

theorem continuous_bigG (β t : ℝ) : Continuous (bigG β t) := by unfold bigG; fun_prop

theorem continuous_bigG' (β t : ℝ) : Continuous (bigG' β t) := by unfold bigG'; fun_prop

/-- The five moments of `(G(t), G'(t))` are finite, and linear in quadratic
expressions: `𝔼 P(G, G') = P(moments)`. -/
theorem integral_quad_bigG {β : ℝ} (hβ : 0 < β) (t c₀ c₁ c₂ c₃ c₄ c₅ : ℝ) :
    ∫ x, (c₀ + c₁ * bigG β t x + c₂ * bigG' β t x + c₃ * bigG β t x ^ 2
        + c₄ * (bigG β t x * bigG' β t x) + c₅ * bigG' β t x ^ 2) ∂gaussianReal 0 1 =
      c₀ + c₁ * meanG β t + c₂ * meanG' β t + c₃ * sqMeanG β t
        + c₄ * mulMeanGG' β t + c₅ * sqMeanG' β t := by
  have hG := abs_bigG_le hβ t
  have hH := abs_bigG'_le hβ t
  have cG := continuous_bigG β t
  have cH := continuous_bigG' β t
  refine integral_quad (integrable_gaussianReal_of_bound cG _ hG)
    (integrable_gaussianReal_of_bound cH _ hH)
    (integrable_gaussianReal_of_bound (cG.pow 2) ((1 + 2 / β) ^ 2) fun x => ?_)
    (integrable_gaussianReal_of_bound (cG.mul cH) ((1 + 2 / β) * 2) fun x => ?_)
    (integrable_gaussianReal_of_bound (cH.pow 2) (2 ^ 2) fun x => ?_) _ _ _ _ _ _
  · rw [abs_pow]; exact pow_le_pow_left₀ (abs_nonneg _) (hG x) 2
  · rw [abs_mul]; exact mul_le_mul (hG x) (hH x) (abs_nonneg _) (by positivity)
  · rw [abs_pow]; exact pow_le_pow_left₀ (abs_nonneg _) (hH x) 2

/-- `integral_quad_bigG` for any function that is pointwise such a quadratic
expression. -/
theorem integral_eq_of_quad {β : ℝ} (hβ : 0 < β) (t : ℝ) {f : ℝ → ℝ} (c₀ c₁ c₂ c₃ c₄ c₅ : ℝ)
    (hf : ∀ x, f x = c₀ + c₁ * bigG β t x + c₂ * bigG' β t x + c₃ * bigG β t x ^ 2
        + c₄ * (bigG β t x * bigG' β t x) + c₅ * bigG' β t x ^ 2) :
    ∫ x, f x ∂gaussianReal 0 1 =
      c₀ + c₁ * meanG β t + c₂ * meanG' β t + c₃ * sqMeanG β t
        + c₄ * mulMeanGG' β t + c₅ * sqMeanG' β t := by
  rw [integral_congr_ae (ae_of_all _ hf)]
  exact integral_quad_bigG hβ t c₀ c₁ c₂ c₃ c₄ c₅

/-- **`Var G(t) > 0`** for `β > 0`. -/
theorem sigmaFst_pos {β : ℝ} (hβ : 0 < β) (t : ℝ) : 0 < sigmaFst β t := by
  set m := meanG β t
  have e : sigmaFst β t = ∫ x, (bigG β t x - m) ^ 2 ∂gaussianReal 0 1 := by
    rw [integral_eq_of_quad hβ t (m ^ 2) (-2 * m) 0 1 0 0 fun x => by ring, sigmaFst]
    ring
  have hG0 : bigG β t t = 0 := by simp [bigG]
  have hG1 : bigG β t (t - 1) = Real.exp (-(β / 2)) := by simp [bigG]
  rw [e]
  refine integral_gaussianReal_pos ((continuous_bigG β t).sub continuous_const |>.pow 2)
    (fun x => sq_nonneg _) (integrable_gaussianReal_of_bound
      ((continuous_bigG β t).sub continuous_const |>.pow 2) ((1 + 2 / β + |m|) ^ 2) fun x => ?_)
    (x₀ := if m = 0 then t - 1 else t) ?_
  · rw [abs_pow]
    refine pow_le_pow_left₀ (abs_nonneg _) ((abs_sub _ _).trans ?_) 2
    linarith [abs_bigG_le hβ t x]
  · split_ifs with hm
    · rw [hm, hG1, sub_zero]; exact pow_pos (Real.exp_pos _) 2
    · rw [hG0, zero_sub, neg_sq]; positivity

/-- **`det Σ_t > 0`** for `β > 0`: the equality case of Cauchy–Schwarz for
`G(t) - 𝔼G(t)` and `G'(t) - 𝔼G'(t)` does not occur. -/
theorem sigmaDet_pos {β : ℝ} (hβ : 0 < β) (t : ℝ) : 0 < sigmaDet β t := by
  have ha := sigmaFst_pos hβ t
  set m := meanG β t
  set m' := meanG' β t
  set l := sigmaCov β t / sigmaFst β t
  set k := l * m - m'
  set W : ℝ → ℝ := fun x => k - l * bigG β t x + bigG' β t x
  have e : sigmaDet β t = sigmaFst β t * ∫ x, W x ^ 2 ∂gaussianReal 0 1 := by
    rw [integral_eq_of_quad hβ t (k ^ 2) (-2 * k * l) (2 * k) (l ^ 2) (-2 * l) 1
      fun x => by simp only [W]; ring]
    simp only [k, l]
    have ha' := ha.ne'
    unfold sigmaDet sigmaCov sigmaSnd at *
    unfold sigmaFst at ha' ⊢
    field_simp
    ring
  have cW : Continuous W :=
    (continuous_const.sub (continuous_const.mul (continuous_bigG β t))).add (continuous_bigG' β t)
  -- `W` vanishes nowhere identically: `W(t) = k + 1`, `W(t ∓ u) = k ∓ l e u` at `βu² = 1`.
  set u := 1 / Real.sqrt β
  have hu : β * u ^ 2 = 1 := by
    rw [div_pow, Real.sq_sqrt hβ.le]; field_simp
  have h0 : W t = k + 1 := by simp [W, bigG, bigG']
  have h1 : W (t - u) = k - l * (Real.exp (-(β / 2) * u ^ 2) * u) := by
    simp only [W, bigG, bigG', sub_sub_cancel, hu, sub_self, mul_zero, add_zero]
  have h2 : W (t + u) = k - l * (Real.exp (-(β / 2) * u ^ 2) * -u) := by
    simp only [W, bigG, bigG', sub_add_cancel_left, neg_sq, hu, sub_self, mul_zero, add_zero]
  have hx : ∃ x₀, W x₀ ≠ 0 := by
    by_contra h
    push Not at h
    have := h t; have := h (t - u); have := h (t + u)
    linarith
  obtain ⟨x₀, hx₀⟩ := hx
  obtain ⟨C, hC⟩ : ∃ C, ∀ x, |W x| ≤ C :=
    ⟨|k| + |l| * (1 + 2 / β) + 2, fun x => by
      simp only [W]
      calc |k - l * bigG β t x + bigG' β t x|
          ≤ |k| + |l| * |bigG β t x| + |bigG' β t x| := by
            rw [← abs_mul]; exact (abs_add_le _ _).trans (by gcongr; exact abs_sub _ _)
        _ ≤ _ := by gcongr; exacts [abs_bigG_le hβ t x, abs_bigG'_le hβ t x]⟩
  rw [e]
  refine mul_pos ha (integral_gaussianReal_pos (cW.pow 2) (fun x => sq_nonneg _)
    (integrable_gaussianReal_of_bound (cW.pow 2) (C ^ 2) fun x => ?_)
    (x₀ := x₀) (lt_of_le_of_ne (sq_nonneg _) (pow_ne_zero 2 hx₀).symm))
  rw [abs_pow]; exact pow_le_pow_left₀ (abs_nonneg _) (hC x) 2

/-- The hypothesis of the results above is satisfiable: `β = 1`. -/
example : 0 < sigmaFst 1 0 ∧ 0 < sigmaDet 1 0 := ⟨sigmaFst_pos one_pos 0, sigmaDet_pos one_pos 0⟩

end Modes
end Transformer
