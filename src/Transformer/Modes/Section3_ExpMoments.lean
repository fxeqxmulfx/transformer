import Transformer.Modes.Section3_Cumulants
import Mathlib.Analysis.Calculus.ParametricIntegral
import Mathlib.MeasureTheory.Order.Group.Lattice

/-
# The number of modes of a Gaussian KDE — the mixed moments of `eq:psi`

§3.1 of arXiv:2412.09080v3 (`sec:error-3`) reads the cumulant `κ^α` as a mixed
derivative at `0` of `log 𝔼 e^{⟨u, Y⟩}`, so before that reading means anything
the mixed moments

  `expMoment μ i j u v = ∫ x₁^i x₂^j e^{u x₁ + v x₂} dμ`

have to exist for small `u, v` and to differentiate under the integral, one
index at a time.  Both come from one inequality: a monomial weight is absorbed
by four exponentials of slightly larger parameters (`abs_pow_mul_exp_le`),
which is the integrable majorant in the first case and the domination the
parametric-integral theorem asks for in the second.

The absorption and the differentiation are stated for an arbitrary pair `X, Y`
of real random variables on an arbitrary measure space, since nothing in them
is about `ℝ²`; `expMoment` and its API are the case `X = x₁`, `Y = x₂`.

Source: arXiv:2412.09080v3, §3.1, after `eq:psi`.
-/

open Real MeasureTheory Filter
open scoped Topology

namespace Transformer
namespace Modes

section General

variable {Ω : Type*} [MeasurableSpace Ω] {μ : Measure Ω} {X Y : Ω → ℝ} {E : ℝ}

/-- **A monomial weight is absorbed by exponentials of larger parameters.**
For `δ > 0`, `|z|^i |w|^j e^{uz+vw}` is at most a constant times the four
exponentials `e^{(u±δ)z + (v±δ)w}`.  This is the majorant behind the existence
of the mixed moments of `eq:psi` and behind their differentiation.

Source: arXiv:2412.09080v3, §3.1, after `eq:psi`. -/
theorem abs_pow_mul_exp_le {δ : ℝ} (hδ : 0 < δ) (i j : ℕ) (u v z w : ℝ) :
    |z| ^ i * |w| ^ j * exp (u * z + v * w) ≤
      ((i.factorial : ℝ) / δ ^ i) * ((j.factorial : ℝ) / δ ^ j) *
        (exp ((u + δ) * z + (v + δ) * w) + exp ((u + δ) * z + (v - δ) * w) +
          exp ((u - δ) * z + (v + δ) * w) + exp ((u - δ) * z + (v - δ) * w)) := by
  have key : ∀ (n : ℕ) (y : ℝ),
      |y| ^ n ≤ ((n.factorial : ℝ) / δ ^ n) * (exp (δ * y) + exp (-(δ * y))) := by
    intro n y
    have hfac : (0 : ℝ) < (n.factorial : ℝ) := by exact_mod_cast n.factorial_pos
    have hpow : (0 : ℝ) < δ ^ n := pow_pos hδ n
    have hxy : (0 : ℝ) ≤ δ * |y| := by positivity
    have h1 : (δ * |y|) ^ n / (n.factorial : ℝ) ≤ exp (δ * |y|) :=
      Real.pow_div_factorial_le_exp _ hxy n
    have h2 : exp (δ * |y|) ≤ exp (δ * y) + exp (-(δ * y)) := by
      rw [show δ * |y| = |δ * y| by rw [abs_mul, abs_of_pos hδ]]
      exact Real.exp_abs_le _
    rw [div_le_iff₀ hfac] at h1
    have h3 : (δ * |y|) ^ n ≤ (exp (δ * y) + exp (-(δ * y))) * (n.factorial : ℝ) :=
      h1.trans (mul_le_mul_of_nonneg_right h2 hfac.le)
    calc |y| ^ n = (δ * |y|) ^ n / δ ^ n := by
          rw [mul_pow, mul_comm, mul_div_assoc, div_self hpow.ne', mul_one]
      _ ≤ (exp (δ * y) + exp (-(δ * y))) * (n.factorial : ℝ) / δ ^ n := by gcongr
      _ = ((n.factorial : ℝ) / δ ^ n) * (exp (δ * y) + exp (-(δ * y))) := by ring
  have hz := key i z
  have hw := key j w
  have h1 : |z| ^ i * |w| ^ j ≤
      (((i.factorial : ℝ) / δ ^ i) * (exp (δ * z) + exp (-(δ * z)))) *
        (((j.factorial : ℝ) / δ ^ j) * (exp (δ * w) + exp (-(δ * w)))) :=
    mul_le_mul hz hw (by positivity) (le_trans (by positivity) hz)
  refine le_trans (mul_le_mul_of_nonneg_right h1 (exp_pos _).le) (le_of_eq ?_)
  simp only [add_mul, sub_mul, Real.exp_add, Real.exp_sub, Real.exp_neg]
  ring

/-- **The mixed moments exist.**  If `e^{uX + vY}` is integrable on the square
`|u|, |v| < E`, then so is every `|X|^i |Y|^j e^{uX + vY}` there.

Source: arXiv:2412.09080v3, §3.1, after `eq:psi`. -/
theorem integrable_abs_pow_mul_exp (hX : Measurable X) (hY : Measurable Y)
    (hE : ∀ u v : ℝ, |u| < E → |v| < E → Integrable (fun ω => exp (u * X ω + v * Y ω)) μ)
    {u v : ℝ} (hu : |u| < E) (hv : |v| < E) (i j : ℕ) :
    Integrable (fun ω => |X ω| ^ i * |Y ω| ^ j * exp (u * X ω + v * Y ω)) μ := by
  have hu0 : 0 < E - |u| := sub_pos.2 hu
  have hv0 : 0 < E - |v| := sub_pos.2 hv
  set δ := min (E - |u|) (E - |v|) / 2 with hδdef
  have hδ : 0 < δ := by rw [hδdef]; exact div_pos (lt_min hu0 hv0) two_pos
  have habs : ∀ t : ℝ, |t| < E → δ ≤ (E - |t|) / 2 → |t + δ| < E ∧ |t - δ| < E := by
    intro t ht hδt
    have h1 : |δ| = δ := abs_of_pos hδ
    exact ⟨(abs_add_le t δ).trans_lt (by rw [h1]; linarith),
      (abs_sub t δ).trans_lt (by rw [h1]; linarith)⟩
  obtain ⟨hu1, hu2⟩ := habs u hu (by rw [hδdef]; gcongr; exact min_le_left _ _)
  obtain ⟨hv1, hv2⟩ := habs v hv (by rw [hδdef]; gcongr; exact min_le_right _ _)
  have hbound : Integrable (fun ω => ((i.factorial : ℝ) / δ ^ i) * ((j.factorial : ℝ) / δ ^ j) *
      (exp ((u + δ) * X ω + (v + δ) * Y ω) + exp ((u + δ) * X ω + (v - δ) * Y ω) +
        exp ((u - δ) * X ω + (v + δ) * Y ω) + exp ((u - δ) * X ω + (v - δ) * Y ω))) μ :=
    ((((hE _ _ hu1 hv1).add (hE _ _ hu1 hv2)).add (hE _ _ hu2 hv1)).add
      (hE _ _ hu2 hv2)).const_mul _
  refine hbound.mono' ?_ ?_
  · exact (((hX.abs.pow_const i).mul (hY.abs.pow_const j)).mul
      ((hX.const_mul u).add (hY.const_mul v)).exp).aestronglyMeasurable
  · refine Eventually.of_forall fun ω => ?_
    rw [Real.norm_eq_abs, abs_of_nonneg (by positivity)]
    exact abs_pow_mul_exp_le hδ i j u v (X ω) (Y ω)

/-- The hypotheses of `integrable_abs_pow_mul_exp` are satisfiable: the Dirac
mass at the origin, where every function is integrable. -/
example : Measurable (Prod.fst : ℝ × ℝ → ℝ) ∧ Measurable (Prod.snd : ℝ × ℝ → ℝ) ∧
    (∀ u v : ℝ, |u| < 1 → |v| < 1 →
      Integrable (fun x : ℝ × ℝ => exp (u * x.1 + v * x.2)) (Measure.dirac 0)) ∧
    |(0 : ℝ)| < 1 ∧ |(0 : ℝ)| < 1 :=
  ⟨measurable_fst, measurable_snd, fun _ _ _ _ => integrable_dirac enorm_lt_top, by norm_num, by norm_num⟩

/-- The signed form of `integrable_abs_pow_mul_exp`. -/
theorem integrable_pow_mul_exp (hX : Measurable X) (hY : Measurable Y)
    (hE : ∀ u v : ℝ, |u| < E → |v| < E → Integrable (fun ω => exp (u * X ω + v * Y ω)) μ)
    {u v : ℝ} (hu : |u| < E) (hv : |v| < E) (i j : ℕ) :
    Integrable (fun ω => X ω ^ i * Y ω ^ j * exp (u * X ω + v * Y ω)) μ := by
  refine (integrable_abs_pow_mul_exp hX hY hE hu hv i j).mono' ?_
    (Eventually.of_forall fun ω => le_of_eq ?_)
  · exact (((hX.pow_const i).mul (hY.pow_const j)).mul
      ((hX.const_mul u).add (hY.const_mul v)).exp).aestronglyMeasurable
  · rw [Real.norm_eq_abs, abs_mul, abs_mul, abs_pow, abs_pow, Real.abs_exp]

/-- The hypotheses of `integrable_pow_mul_exp` are satisfiable: the Dirac mass
at the origin. -/
example : Measurable (Prod.fst : ℝ × ℝ → ℝ) ∧ Measurable (Prod.snd : ℝ × ℝ → ℝ) ∧
    (∀ u v : ℝ, |u| < 1 → |v| < 1 →
      Integrable (fun x : ℝ × ℝ => exp (u * x.1 + v * x.2)) (Measure.dirac 0)) ∧
    |(0 : ℝ)| < 1 ∧ |(0 : ℝ)| < 1 :=
  ⟨measurable_fst, measurable_snd, fun _ _ _ _ => integrable_dirac enorm_lt_top, by norm_num, by norm_num⟩

/-- **The mixed moments differentiate under the integral**: one more power of
`X` in the integrand is one derivative in the parameter of `X`.

Source: arXiv:2412.09080v3, §3.1, after `eq:psi`. -/
theorem hasDerivAt_integral_pow_mul_exp (hX : Measurable X) (hY : Measurable Y)
    (hE : ∀ u v : ℝ, |u| < E → |v| < E → Integrable (fun ω => exp (u * X ω + v * Y ω)) μ)
    {u v : ℝ} (hu : |u| < E) (hv : |v| < E) (i j : ℕ) :
    HasDerivAt (fun s => ∫ ω, X ω ^ i * Y ω ^ j * exp (s * X ω + v * Y ω) ∂μ)
      (∫ ω, X ω ^ (i + 1) * Y ω ^ j * exp (u * X ω + v * Y ω) ∂μ) u := by
  have hu0 : 0 < E - |u| := sub_pos.2 hu
  set δ := (E - |u|) / 2 with hδdef
  have hδ : 0 < δ := by rw [hδdef]; linarith
  have h1 : |δ| = δ := abs_of_pos hδ
  have hu1 : |u + δ| < E := (abs_add_le u δ).trans_lt (by rw [h1, hδdef]; linarith)
  have hu2 : |u - δ| < E := (abs_sub u δ).trans_lt (by rw [h1, hδdef]; linarith)
  refine (hasDerivAt_integral_of_dominated_loc_of_deriv_le (μ := μ) (x₀ := u)
    (F := fun s ω => X ω ^ i * Y ω ^ j * exp (s * X ω + v * Y ω))
    (F' := fun s ω => X ω ^ (i + 1) * Y ω ^ j * exp (s * X ω + v * Y ω))
    (bound := fun ω => |X ω| ^ (i + 1) * |Y ω| ^ j * exp ((u + δ) * X ω + v * Y ω) +
      |X ω| ^ (i + 1) * |Y ω| ^ j * exp ((u - δ) * X ω + v * Y ω))
    (Metric.ball_mem_nhds u hδ) (Eventually.of_forall fun s => ?_)
    (integrable_pow_mul_exp hX hY hE hu hv i j) ?_ ?_
    ((integrable_abs_pow_mul_exp hX hY hE hu1 hv (i + 1) j).add
      (integrable_abs_pow_mul_exp hX hY hE hu2 hv (i + 1) j)) ?_).2
  · exact (((hX.pow_const i).mul (hY.pow_const j)).mul
      ((hX.const_mul s).add (hY.const_mul v)).exp).aestronglyMeasurable
  · exact (((hX.pow_const (i + 1)).mul (hY.pow_const j)).mul
      ((hX.const_mul u).add (hY.const_mul v)).exp).aestronglyMeasurable
  · refine Eventually.of_forall fun ω s hs => ?_
    have hs' : |s - u| < δ := by simpa [Real.dist_eq] using hs
    have hsplit : exp (s * X ω) ≤ exp ((u + δ) * X ω) + exp ((u - δ) * X ω) := by
      obtain ⟨hs1, hs2⟩ := abs_lt.1 hs'
      rcases le_total 0 (X ω) with h | h
      · exact le_add_of_le_of_nonneg (exp_le_exp.2
          (mul_le_mul_of_nonneg_right (by linarith) h)) (exp_pos _).le
      · exact le_add_of_nonneg_of_le (exp_pos _).le (exp_le_exp.2
          (mul_le_mul_of_nonpos_right (by linarith) h))
    rw [Real.norm_eq_abs, abs_mul, abs_mul, abs_pow, abs_pow, Real.abs_exp]
    calc |X ω| ^ (i + 1) * |Y ω| ^ j * exp (s * X ω + v * Y ω)
        = |X ω| ^ (i + 1) * |Y ω| ^ j * exp (s * X ω) * exp (v * Y ω) := by
          rw [Real.exp_add]; ring
      _ ≤ |X ω| ^ (i + 1) * |Y ω| ^ j * (exp ((u + δ) * X ω) + exp ((u - δ) * X ω)) *
            exp (v * Y ω) := by
          have h0 : (0 : ℝ) ≤ |X ω| ^ (i + 1) * |Y ω| ^ j := by positivity
          gcongr
      _ = _ := by simp only [Real.exp_add]; ring
  · refine Eventually.of_forall fun ω s _ => ?_
    have hlin : HasDerivAt (fun t : ℝ => t * X ω + v * Y ω) (X ω) s := by
      simpa using ((hasDerivAt_id s).mul_const (X ω)).add_const (v * Y ω)
    have := hlin.exp.const_mul (X ω ^ i * Y ω ^ j)
    convert this using 1
    ring

/-- The hypotheses of `hasDerivAt_integral_pow_mul_exp` are satisfiable: the
Dirac mass at the origin. -/
example : Measurable (Prod.fst : ℝ × ℝ → ℝ) ∧ Measurable (Prod.snd : ℝ × ℝ → ℝ) ∧
    (∀ u v : ℝ, |u| < 1 → |v| < 1 →
      Integrable (fun x : ℝ × ℝ => exp (u * x.1 + v * x.2)) (Measure.dirac 0)) ∧
    |(0 : ℝ)| < 1 ∧ |(0 : ℝ)| < 1 :=
  ⟨measurable_fst, measurable_snd, fun _ _ _ _ => integrable_dirac enorm_lt_top, by norm_num, by norm_num⟩

end General

end Modes
end Transformer
