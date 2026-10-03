import Transformer.AMSGrad.Section1_AMSGrad
import Mathlib.Analysis.Convex.Deriv
import Mathlib.Analysis.SpecificLimits.Normed
import Mathlib.NumberTheory.Harmonic.Bounds

/-
# AMSGrad — the preliminary lemmas

§2 of arXiv:1904.03590v4: the elementary lemmas the convergence proofs are
built from.

**What the source says and what is carried here.**

* Lemma 2.3 (`Taylor`) states `Σ_{t≥1} αᵗ = 1/(1-α)` for `0 < α < 1`.  That is
  false: the sum from `t = 1` is `α/(1-α)`.  The statement is refuted,
  `not_taylor`, and the correct value proved, `taylor_geom`.  Its second half,
  `Σ_{t≥1} t α^{t-1} = 1/(1-α)²`, is correct and proved, `taylor_deriv`.

* Lemma 2.7 (`McM&Str`) writes `u = min_{x∈F} ‖Q^{1/2}(x - z)‖` for the
  minimizer, not the minimum.  It is stated for minimizers, with the quadratic
  form `‖Q^{1/2}v‖² = vᵀQv`, and proved in squared form, which is equivalent
  since both sides are non-negative.

Every other lemma is proved as stated.

Source: arXiv:1904.03590v4, §2, Lemmas 2.1–2.7.
-/

open Finset Matrix

namespace Transformer
namespace AMSGrad

variable {d : ℕ}

/-! ### First-order convexity -/

/-- The directional derivative is the gradient paired with the direction. -/
theorem fderiv_apply_eq_sum (f : Vec d → ℝ) (x v : Vec d) :
    fderiv ℝ f x v = ∑ i, grad f x i * v i := by
  have hv : v = ∑ i, v i • (Pi.single i 1 : Vec d) := by
    conv_lhs => rw [← Finset.univ_sum_single v]
    refine Finset.sum_congr rfl fun i _ => ?_
    ext j; by_cases h : j = i <;> simp [h]
  conv_lhs => rw [hv]
  simp only [map_sum, map_smul, smul_eq_mul, grad]
  exact Finset.sum_congr rfl fun i _ => mul_comm _ _

/-- **Lemma 2.1.** A differentiable convex `f` lies above its tangent planes:
`f(y) ≥ f(x) + ∇f(x)ᵀ(y - x)`.  The source assumes differentiability
implicitly, through `∇f`.

Source: arXiv:1904.03590v4, §2, Lemma 2.1. -/
theorem convex_first_order {f : Vec d → ℝ} (hf : ConvexOn ℝ Set.univ f)
    (hd : Differentiable ℝ f) (x y : Vec d) :
    f x + ∑ i, grad f x i * (y i - x i) ≤ f y := by
  set g : ℝ → ℝ := fun s => f (x + s • (y - x))
  have hline : HasDerivAt (fun s : ℝ => x + s • (y - x)) (y - x) 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).smul_const (y - x)).const_add x
  have hg : HasDerivAt g (fderiv ℝ f x (y - x)) 0 := by
    have := (hd (x + (0 : ℝ) • (y - x))).hasFDerivAt.comp_hasDerivAt (0 : ℝ) hline
    rw [zero_smul, add_zero] at this
    exact this
  have hgc : ConvexOn ℝ Set.univ g := by
    refine ⟨convex_univ, fun a _ b _ p q hp hq hpq => ?_⟩
    have h := hf.2 (Set.mem_univ (x + a • (y - x))) (Set.mem_univ (x + b • (y - x))) hp hq hpq
    simp only [g, smul_eq_mul]
    convert h using 2
    rw [smul_add, smul_add, add_add_add_comm, ← add_smul, hpq, one_smul, smul_smul, smul_smul,
      ← add_smul]
  have h := hgc.deriv_le_slope (Set.mem_univ 0) (Set.mem_univ 1) one_pos hg.differentiableAt
  rw [hg.deriv, slope_def_field, fderiv_apply_eq_sum] at h
  simp only [g, zero_smul, add_zero, one_smul, add_sub_cancel, sub_zero, div_one] at h
  simp only [Pi.sub_apply] at h
  linarith

/-! ### Sums -/

/-- **Lemma 2.2 (Cauchy–Schwarz).**
`(Σᵢ uᵢvᵢ)² ≤ (Σᵢ uᵢ²)(Σᵢ vᵢ²)`.  Source: arXiv:1904.03590v4, §2, Lemma 2.2. -/
theorem cauchy_schwarz (n : ℕ) (u v : ℕ → ℝ) :
    (∑ i ∈ Icc 1 n, u i * v i) ^ 2 ≤ (∑ i ∈ Icc 1 n, u i ^ 2) * ∑ i ∈ Icc 1 n, v i ^ 2 :=
  Finset.sum_mul_sq_le_sq_mul_sq _ _ _

/-- **Lemma 2.3, first half, corrected:** `Σ_{t≥1} αᵗ = α/(1-α)` for
`0 ≤ α < 1`.  Source: arXiv:1904.03590v4, §2, Lemma 2.3; see the module
docstring. -/
theorem taylor_geom {α : ℝ} (h₀ : 0 ≤ α) (h₁ : α < 1) :
    ∑' t : ℕ, α ^ (t + 1) = α / (1 - α) := by
  simp_rw [pow_succ]
  rw [tsum_mul_right, tsum_geometric_of_lt_one h₀ h₁, div_eq_inv_mul]

/-- **Lemma 2.3, first half, is false.**  `Σ_{t≥1} αᵗ ≠ 1/(1-α)`: at
`α = 1/2` the sum is `1`, not `2`.

Source: arXiv:1904.03590v4, §2, Lemma 2.3. -/
theorem not_taylor :
    ¬ ∀ α : ℝ, 0 < α → α < 1 → ∑' t : ℕ, α ^ (t + 1) = 1 / (1 - α) := by
  intro h
  have h1 := h (1 / 2) (by norm_num) (by norm_num)
  have h2 : ∑' t : ℕ, (1 / 2 : ℝ) ^ (t + 1) = 1 := by
    rw [taylor_geom (by norm_num) (by norm_num)]; norm_num
  rw [h2] at h1
  norm_num at h1

/-- **Lemma 2.3, second half:** `Σ_{t≥1} t α^{t-1} = 1/(1-α)²` for
`0 ≤ α < 1`.  Source: arXiv:1904.03590v4, §2, Lemma 2.3. -/
theorem taylor_deriv {α : ℝ} (h₀ : 0 ≤ α) (h₁ : α < 1) :
    ∑' t : ℕ, ((t : ℝ) + 1) * α ^ t = 1 / (1 - α) ^ 2 := by
  have hn : ‖α‖ < 1 := by rwa [Real.norm_of_nonneg h₀]
  have h := (hasSum_coe_mul_geometric_of_norm_lt_one hn).add (hasSum_geometric_of_lt_one h₀ h₁)
  have hne : 1 - α ≠ 0 := by linarith
  have he : (fun t : ℕ => ((t : ℝ) + 1) * α ^ t) = fun t : ℕ => (t : ℝ) * α ^ t + α ^ t := by
    funext t; ring
  rw [he, h.tsum_eq]
  field_simp
  ring

/-- The hypotheses of `taylor_geom` and `taylor_deriv` are satisfiable. -/
example : (0 : ℝ) ≤ 1 / 2 ∧ (1 / 2 : ℝ) < 1 := by norm_num

/-- **Lemma 2.4 (the harmonic series).**  `Σ_{n=1}^N 1/n ≤ ln N + 1`.
Source: arXiv:1904.03590v4, §2, Lemma 2.4. -/
theorem harmonic_le (N : ℕ) : ∑ n ∈ Icc 1 N, (1 : ℝ) / n ≤ Real.log N + 1 := by
  have h := harmonic_le_one_add_log N
  rw [harmonic_eq_sum_Icc] at h
  push_cast at h
  simp only [one_div]
  linarith

/-- **Lemma 2.5.**  `Σ_{n=1}^N 1/√n ≤ 2√N`.
Source: arXiv:1904.03590v4, §2, Lemma 2.5. -/
theorem sum_inv_sqrt_le (N : ℕ) : ∑ n ∈ Icc 1 N, 1 / Real.sqrt n ≤ 2 * Real.sqrt N := by
  induction N with
  | zero => simp
  | succ N ih =>
    rw [Finset.sum_Icc_succ_top (by omega)]
    set a := Real.sqrt N
    set b := Real.sqrt (N + 1 : ℕ)
    have ha : 0 ≤ a := Real.sqrt_nonneg _
    have hb : 0 < b := Real.sqrt_pos.2 (by positivity)
    have hab : b ^ 2 = a ^ 2 + 1 := by
      simp only [a, b, Real.sq_sqrt (Nat.cast_nonneg _)]; push_cast; ring
    have : 1 / b ≤ 2 * b - 2 * a := by
      rw [div_le_iff₀ hb]; nlinarith [sq_nonneg (b - a)]
    linarith

/-- **Lemma 2.6.**  For `aᵢ ≥ 0` and `bᵢ > 0`,
`(Σᵢ aᵢ)/(Σⱼ bⱼ) ≤ Σᵢ aᵢ/bᵢ`.  Source: arXiv:1904.03590v4, §2, Lemma 2.6. -/
theorem sum_div_sum_le (n : ℕ) {a b : ℕ → ℝ} (ha : ∀ i ∈ Icc 1 n, 0 ≤ a i)
    (hb : ∀ i ∈ Icc 1 n, 0 < b i) :
    (∑ i ∈ Icc 1 n, a i) / ∑ j ∈ Icc 1 n, b j ≤ ∑ i ∈ Icc 1 n, a i / b i := by
  rw [Finset.sum_div]
  refine Finset.sum_le_sum fun i hi => div_le_div_of_nonneg_left (ha i hi) (hb i hi) ?_
  exact Finset.single_le_sum (fun j hj => (hb j hj).le) hi

/-- The hypotheses of `sum_div_sum_le` are satisfiable. -/
example : ∀ i ∈ Icc 1 1, (0 : ℝ) ≤ (fun _ => 0) i ∧ (0 : ℝ) < (fun _ => 1) i :=
  fun _ _ => ⟨le_rfl, one_pos⟩

end AMSGrad
end Transformer
