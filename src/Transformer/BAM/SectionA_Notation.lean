module

public import Mathlib.MeasureTheory.Integral.Gamma
public import Mathlib.Probability.Distributions.Gaussian.Real

/-!
# Appendix A: the positional priors

Bianchessi, Aguirre, Barros, Kupssinskü, arXiv:2505.22842v4, Appendix A, "Positional Priors".
The paper gives each prior up to a constant; here they are the normalized densities.

* `uniform a b x`: `Uniform(a, b, x) ∝ 𝕀[a ≤ x ≤ b]` on the integers; it sums to `1`
  (`sum_uniform`).
* `laplace μ α x`: `Laplace(μ, α, x) ∝ exp(−|x − μ| / α)`.
* `ggd μ α β x`: `GGD(μ, α, β, x) ∝ exp(−|(x − μ) / α|^β)`, a probability density for `β > 0`,
  `α > 0` (`integral_ggd`).

The claims of Appendix A on the GGD:

* "when `β = 1` it recovers the Laplace distribution" (`ggd_one`);
* "when `β = 2` it becomes Normal", with variance `α² / 2` (`ggd_two`);
* "larger values of `β` yield sharper, more localized priors": past the scale, `|x − μ| > α`, the
  weight `exp(−|(x − μ) / α|^β)` falls as `β` grows (`ggdWeight_lt_of_scale_lt`), and within it,
  it rises (`ggdWeight_lt_of_lt_scale`);
* "lower values of `β` yield large tailed distributions": for `β₁ < β₂` the ratio of the weights
  tends to `∞` (`tendsto_ggdWeight_div`).
-/

@[expose] public section

namespace Transformer.BAM

open Finset Filter MeasureTheory Topology

/-- The discrete uniform distribution on the integers `a, …, b`:
`Uniform(a, b, x) = 𝕀[a ≤ x ≤ b] / (b − a + 1)` (Appendix A). -/
noncomputable def uniform (a b x : ℤ) : ℝ :=
  if a ≤ x ∧ x ≤ b then 1 / ((b - a + 1 : ℤ) : ℝ) else 0

/-- The uniform distribution on `a, …, b` sums to `1` (Appendix A). -/
theorem sum_uniform {a b : ℤ} (h : a ≤ b) : ∑ x ∈ Icc a b, uniform a b x = 1 := by
  have hpos : (0 : ℝ) < ((b - a + 1 : ℤ) : ℝ) := by exact_mod_cast (by omega : (0 : ℤ) < b - a + 1)
  have hterm : ∀ x ∈ Icc a b, uniform a b x = 1 / ((b - a + 1 : ℤ) : ℝ) := fun x hx => by
    simp [uniform, (mem_Icc.1 hx).1, (mem_Icc.1 hx).2]
  have hcard : ((#(Icc a b) : ℕ) : ℝ) = ((b - a + 1 : ℤ) : ℝ) := by
    rw [Int.card_Icc, ← Int.cast_natCast, Int.toNat_of_nonneg (by omega)]
    push_cast
    ring
  rw [sum_congr rfl hterm, sum_const, nsmul_eq_mul, hcard]
  field_simp

/-- The hypothesis of `sum_uniform` holds for `a = 1`, `b = 3`. -/
example : ∑ x ∈ Icc (1 : ℤ) 3, uniform 1 3 x = 1 := sum_uniform (by norm_num)

/-- The Laplace density with location `μ` and scale `α`: `exp(−|x − μ| / α) / (2α)`
(Appendix A). -/
noncomputable def laplace (μ α x : ℝ) : ℝ := Real.exp (-|x - μ| / α) / (2 * α)

/-- The generalized Gaussian density with location `μ`, scale `α` and shape `β`:
`β / (2α Γ(1/β)) · exp(−|(x − μ) / α|^β)` (Appendix A). -/
noncomputable def ggd (μ α β x : ℝ) : ℝ :=
  β / (2 * α * Real.Gamma (1 / β)) * Real.exp (-|(x - μ) / α| ^ β)

/-- `∫ exp(−|(x − μ) / α|^β) dx = 2α Γ(1/β + 1)`. -/
theorem integral_exp_neg_abs_rpow {α β : ℝ} (hα : 0 < α) (hβ : 0 < β) (μ : ℝ) :
    ∫ x, Real.exp (-|(x - μ) / α| ^ β) = 2 * α * Real.Gamma (1 / β + 1) := by
  rw [integral_sub_right_eq_self (fun x => Real.exp (-|x / α| ^ β)) μ,
    Measure.integral_comp_div (fun x => Real.exp (-|x| ^ β)) α,
    integral_comp_abs (f := fun x => Real.exp (-x ^ β)), integral_exp_neg_rpow hβ,
    abs_of_pos hα, smul_eq_mul]
  ring

/-- **The GGD is a probability density** for `β > 0` and `α > 0` (Appendix A). -/
theorem integral_ggd {α β : ℝ} (hα : 0 < α) (hβ : 0 < β) (μ : ℝ) : ∫ x, ggd μ α β x = 1 := by
  have hΓ : Real.Gamma (1 / β) ≠ 0 := (Real.Gamma_pos_of_pos (one_div_pos.2 hβ)).ne'
  simp only [ggd]
  rw [integral_const_mul, integral_exp_neg_abs_rpow hα hβ,
    Real.Gamma_add_one (one_div_pos.2 hβ).ne']
  field_simp

/-- The hypotheses of `integral_ggd` hold for `α = 1`, `β = 1/2`. -/
example (μ : ℝ) : ∫ x, ggd μ 1 (1 / 2) x = 1 := integral_ggd one_pos (by norm_num) μ

/-- **`β = 1` recovers the Laplace distribution** (Appendix A). -/
theorem ggd_one {α : ℝ} (hα : 0 < α) (μ x : ℝ) : ggd μ α 1 x = laplace μ α x := by
  simp [ggd, laplace, abs_of_pos hα, div_eq_mul_inv, mul_comm]

/-- The hypothesis of `ggd_one` holds for `α = 2`. -/
example (μ x : ℝ) : ggd μ 2 1 x = laplace μ 2 x := ggd_one two_pos μ x

/-- The Laplace density integrates to `1` (Appendix A). -/
theorem integral_laplace {α : ℝ} (hα : 0 < α) (μ : ℝ) : ∫ x, laplace μ α x = 1 := by
  simpa [ggd_one hα] using integral_ggd hα one_pos μ

/-- The hypothesis of `integral_laplace` holds for `α = 1`. -/
example (μ : ℝ) : ∫ x, laplace μ 1 x = 1 := integral_laplace one_pos μ

/-- **`β = 2` gives the Normal distribution**, of variance `α² / 2` (Appendix A). -/
theorem ggd_two {α : ℝ} (hα : 0 < α) (μ x : ℝ) :
    ggd μ α 2 x = ProbabilityTheory.gaussianPDFReal μ (α ^ 2 / 2).toNNReal x := by
  have hπ : 0 < Real.pi := Real.pi_pos
  have hs : √(2 * Real.pi * (α ^ 2 / 2)) = √Real.pi * α := by
    rw [show 2 * Real.pi * (α ^ 2 / 2) = Real.pi * α ^ 2 by ring, Real.sqrt_mul hπ.le,
      Real.sqrt_sq hα.le]
  rw [ProbabilityTheory.gaussianPDFReal, Real.coe_toNNReal _ (by positivity), hs, ggd,
    Real.Gamma_one_half_eq, Real.rpow_two, sq_abs, div_pow]
  have : 0 < √Real.pi := Real.sqrt_pos.2 hπ
  congr 1
  · field_simp
  · congr 1
    field_simp

/-- The hypothesis of `ggd_two` holds for `α = 1`. -/
example (μ x : ℝ) : ggd μ 1 2 x = ProbabilityTheory.gaussianPDFReal μ (1 ^ 2 / 2 : ℝ).toNNReal x :=
  ggd_two one_pos μ x

/-- **Larger `β` is more localized.** Past the scale, `|x − μ| > α`, the GGD weight
`exp(−|(x − μ) / α|^β)` is smaller for the larger shape (Appendix A). -/
theorem ggdWeight_lt_of_scale_lt {μ α x β₁ β₂ : ℝ} (hα : 0 < α) (hx : α < |x - μ|)
    (h : β₁ < β₂) : Real.exp (-|(x - μ) / α| ^ β₂) < Real.exp (-|(x - μ) / α| ^ β₁) := by
  have h1 : 1 < |(x - μ) / α| := by rwa [abs_div, abs_of_pos hα, one_lt_div hα]
  exact Real.exp_lt_exp.2 (neg_lt_neg (Real.rpow_lt_rpow_of_exponent_lt h1 h))

/-- The hypotheses of `ggdWeight_lt_of_scale_lt` hold at `x = 3` for `α = 1`, `μ = 0`. -/
example : Real.exp (-|((3 : ℝ) - 0) / 1| ^ (2 : ℝ)) < Real.exp (-|((3 : ℝ) - 0) / 1| ^ (1 : ℝ)) :=
  ggdWeight_lt_of_scale_lt one_pos (by norm_num) one_lt_two

/-- **Larger `β` is sharper within the scale.** For `0 < |x − μ| < α` the GGD weight is larger
for the larger shape (Appendix A). -/
theorem ggdWeight_lt_of_lt_scale {μ α x β₁ β₂ : ℝ} (hα : 0 < α) (hx₀ : x ≠ μ) (hx : |x - μ| < α)
    (h : β₁ < β₂) : Real.exp (-|(x - μ) / α| ^ β₁) < Real.exp (-|(x - μ) / α| ^ β₂) := by
  have h0 : 0 < |(x - μ) / α| := abs_pos.2 (div_ne_zero (sub_ne_zero.2 hx₀) hα.ne')
  have h1 : |(x - μ) / α| < 1 := by rwa [abs_div, abs_of_pos hα, div_lt_one hα]
  exact Real.exp_lt_exp.2 (neg_lt_neg (Real.rpow_lt_rpow_of_exponent_gt h0 h1 h))

/-- The hypotheses of `ggdWeight_lt_of_lt_scale` hold at `x = 1/2` for `α = 1`, `μ = 0`. -/
example : Real.exp (-|((1 / 2 : ℝ) - 0) / 1| ^ (1 : ℝ)) <
    Real.exp (-|((1 / 2 : ℝ) - 0) / 1| ^ (2 : ℝ)) :=
  ggdWeight_lt_of_lt_scale one_pos (by norm_num) (by norm_num) one_lt_two

/-- **Lower `β` has the heavier tail.** For `β₁ < β₂` with `β₂ > 0`, the ratio of the GGD
weights of shapes `β₁` and `β₂` tends to `∞` (Appendix A). -/
theorem tendsto_ggdWeight_div {μ α β₁ β₂ : ℝ} (hα : 0 < α) (h : β₁ < β₂) (hβ₂ : 0 < β₂) :
    Tendsto (fun x => Real.exp (-|(x - μ) / α| ^ β₁) / Real.exp (-|(x - μ) / α| ^ β₂))
      atTop atTop := by
  have ht : Tendsto (fun x : ℝ => |(x - μ) / α|) atTop atTop :=
    tendsto_abs_atTop_atTop.comp (((tendsto_atTop_add_const_right _ (-μ) tendsto_id).congr
      fun x => (sub_eq_add_neg x μ).symm).atTop_div_const hα)
  have hlim : Tendsto (fun t : ℝ => t ^ β₂ * (1 - t ^ (-(β₂ - β₁)))) atTop atTop :=
    (tendsto_rpow_atTop hβ₂).atTop_mul_pos one_pos
      (by simpa using (tendsto_rpow_neg_atTop (sub_pos.2 h)).const_sub 1)
  refine (Real.tendsto_exp_atTop.comp (hlim.comp ht)).congr' ?_
  filter_upwards [ht.eventually_gt_atTop 0] with x hx
  rw [Function.comp_apply, Function.comp_apply, ← Real.exp_sub, mul_sub, mul_one,
    ← Real.rpow_add hx]
  ring_nf

/-- The hypotheses of `tendsto_ggdWeight_div` hold for the shapes `1/2 < 1` (GGD against
Laplace). -/
example (μ : ℝ) : Tendsto (fun x => Real.exp (-|(x - μ) / 1| ^ (1 / 2 : ℝ)) /
    Real.exp (-|(x - μ) / 1| ^ (1 : ℝ))) atTop atTop :=
  tendsto_ggdWeight_div one_pos (by norm_num) one_pos

end Transformer.BAM
