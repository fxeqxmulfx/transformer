import Transformer.Modes.Section3_SumCharFun
import Mathlib.Analysis.Calculus.Taylor
import Mathlib.Analysis.Complex.RealDeriv

/-!
# Taylor remainders for the actual phase exponential

The small-frequency argument in §5.4 of arXiv:2412.09080v3 uses a Taylor
expansion of the characteristic function. Its `eq:br-9.10` is a bound for
the normalized sum and its derivatives; the statements below supply its
unscaled exponential remainder, without claiming that full estimate.

We expand `exp(i u s)` for the real parameter `s` on `[0, 1]`. Every
iterated derivative has norm `|u|^k`, because the exponential has unit
modulus. The mean-value Taylor theorem therefore bounds the remainder
by `|u|^(n+1) / n!`, for every real `u`. The constant is deliberately
nonsharp; the quadratic and cubic cases suffice for finite third and
fourth moments. Integrating preserves these bounds and requires only
the next absolute moment, as in the hypotheses of §3 `thm:br`.
-/

open Real MeasureTheory Filter
open scoped BigOperators
namespace Transformer.Modes

/-- Iterated real derivatives of the complex phase exponential.
This retains a real parameter while the target is complex, as needed in
arXiv:2412.09080v3, §5.4, the Taylor step leading to `eq:br-9.10`. -/
theorem iteratedDeriv_complexExp_real (c : ℂ) (n : ℕ) :
    iteratedDeriv n (fun s : ℝ => Complex.exp (c * (s : ℂ))) =
      fun s : ℝ => c ^ n * Complex.exp (c * (s : ℂ)) := by
  induction n with
  | zero => simp [iteratedDeriv_zero]
  | succ n ih =>
    rw [iteratedDeriv_succ, ih]
    funext s
    have hc : HasDerivAt (fun r : ℝ => Complex.exp (c * (r : ℂ)))
        (Complex.exp (c * (s : ℂ)) * c) s := by
      simpa using (((hasDerivAt_id s).ofReal_comp.const_mul c).cexp)
    rw [(hc.const_mul (c ^ n)).deriv]
    rw [pow_succ]
    ring

/-- Taylor evaluation along the segment from zero to one is the usual
exponential polynomial. Source: arXiv:2412.09080v3, §5.4, the phase
expansion underlying `eq:br-9.10`. -/
theorem taylorWithinEval_complexExp_real (c : ℂ) (n : ℕ) :
    taylorWithinEval (fun s : ℝ => Complex.exp (c * (s : ℂ)))
      n (Set.Icc 0 1) 0 1 =
        ∑ k ∈ Finset.range (n + 1), c ^ k / (k.factorial : ℂ) := by
  have hu : UniqueDiffOn ℝ (Set.Icc (0 : ℝ) 1) := uniqueDiffOn_Icc (by norm_num)
  have hc : ContDiff ℝ ⊤ (fun s : ℝ => Complex.exp (c * (s : ℂ))) := by
    exact (contDiff_const.mul Complex.ofRealCLM.contDiff).cexp
  unfold taylorWithinEval taylorWithin
  rw [map_sum]
  apply Finset.sum_congr rfl
  intro k hk
  rw [PolynomialModule.comp_eval, PolynomialModule.eval_single]
  simp only [Polynomial.eval_sub, Polynomial.eval_X, Polynomial.eval_C, sub_zero, one_pow,
    one_smul, taylorCoeffWithin]
  rw [iteratedDerivWithin_eq_iteratedDeriv hu (hc.contDiffAt.of_le (by simp)) (by simp)]
  rw [iteratedDeriv_complexExp_real]
  simp [Complex.real_smul, div_eq_mul_inv, mul_comm]

/-- A polynomial remainder bound for a purely imaginary exponential.
The mean-value Taylor bound gives the nonsharp denominator `n!`; the
unit modulus of the real phase avoids an exponential-moment hypothesis.
Source: arXiv:2412.09080v3, §3 `thm:br` and §5.4 `eq:br-9.10`. -/
theorem norm_complexExp_imaginary_taylor (u : ℝ) (n : ℕ) :
    ‖Complex.exp ((u : ℂ) * Complex.I) -
      ∑ k ∈ Finset.range (n + 1), ((u : ℂ) * Complex.I) ^ k / (k.factorial : ℂ)‖ ≤
        |u| ^ (n + 1) / (n.factorial : ℝ) := by
  let c : ℂ := (u : ℂ) * Complex.I
  let f : ℝ → ℂ := fun s => Complex.exp (c * (s : ℂ))
  have hu : UniqueDiffOn ℝ (Set.Icc (0 : ℝ) 1) := uniqueDiffOn_Icc (by norm_num)
  have hc : ContDiff ℝ ⊤ f := by
    exact (contDiff_const.mul Complex.ofRealCLM.contDiff).cexp
  have hb (s : ℝ) (hs : s ∈ Set.Icc (0 : ℝ) 1) :
      ‖iteratedDerivWithin (n + 1) f (Set.Icc 0 1) s‖ ≤ |u| ^ (n + 1) := by
    rw [iteratedDerivWithin_eq_iteratedDeriv hu (hc.contDiffAt.of_le (by simp)) hs]
    change ‖iteratedDeriv (n + 1) (fun s : ℝ => Complex.exp (c * (s : ℂ))) s‖ ≤ _
    rw [iteratedDeriv_complexExp_real]
    simp [c, norm_pow, Complex.norm_exp]
  have h := taylor_mean_remainder_bound (a := (0 : ℝ)) (b := 1)
    (C := |u| ^ (n + 1)) (n := n) (by norm_num)
    (hc.contDiffOn.of_le (by simp)) (by simp : (1 : ℝ) ∈ Set.Icc 0 1) hb
  dsimp [f] at h
  rw [taylorWithinEval_complexExp_real] at h
  simpa [c] using h


/-- The quadratic exponential polynomial has a cubic remainder, with a
nonsharp factor `1/2`. Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 2`,
and §5.4, the expansion underlying `eq:br-9.10`. -/
theorem norm_complexExp_imaginary_sub_quadratic (u : ℝ) :
    ‖Complex.exp ((u : ℂ) * Complex.I) -
      (1 + (u : ℂ) * Complex.I + ((u : ℂ) * Complex.I) ^ 2 / 2)‖ ≤ |u| ^ 3 / 2 := by
  have h := norm_complexExp_imaginary_taylor u 2
  norm_num [Finset.sum_range_succ] at h
  exact h

/-- The cubic exponential polynomial has a fourth-order remainder, with
a nonsharp factor `1/6`. Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 3`,
and §5.4, the expansion underlying `eq:br-9.10`. -/
theorem norm_complexExp_imaginary_sub_cubic (u : ℝ) :
    ‖Complex.exp ((u : ℂ) * Complex.I) -
      (1 + (u : ℂ) * Complex.I + ((u : ℂ) * Complex.I) ^ 2 / 2 +
        ((u : ℂ) * Complex.I) ^ 3 / 6)‖ ≤ |u| ^ 4 / 6 := by
  have h := norm_complexExp_imaginary_taylor u 3
  norm_num [Finset.sum_range_succ] at h
  exact h

/-- Integrating the phase expansion needs only its finite next absolute
moment. The Taylor polynomial is integrable because lower moments are
integrable under a probability law. Source: arXiv:2412.09080v3, §3
`thm:br` and §5.4 `eq:br-9.10`; this is the unscaled remainder step. -/
theorem norm_integral_complexExp_sub_taylor
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (f : ℝ × ℝ → ℝ)
    (hf : Continuous f) (n : ℕ) (hmom : Integrable (fun z => |f z| ^ (n + 1)) μ) :
    ‖(∫ z, Complex.exp ((f z : ℂ) * Complex.I) ∂μ) -
        ∫ z, (∑ k ∈ Finset.range (n + 1),
          ((f z : ℂ) * Complex.I) ^ k / (k.factorial : ℂ)) ∂μ‖ ≤
      (∫ z, |f z| ^ (n + 1) ∂μ) / (n.factorial : ℝ) := by
  have hm : Integrable (fun z => ‖f z‖ ^ (n + 1)) μ := by simpa using hmom
  have hi (k : ℕ) (hk : k ∈ Finset.range (n + 1)) :
      Integrable (fun z => ((f z : ℂ) * Complex.I) ^ k / (k.factorial : ℂ)) μ := by
    have hk' : k ≤ n + 1 := by have := Finset.mem_range.mp hk; omega
    have hlow := integrable_norm_pow_of_le hf.aestronglyMeasurable hk' hm
    have hp : Integrable (fun z => ((f z : ℂ) * Complex.I) ^ k) μ := by
      apply hlow.mono' (by fun_prop)
      exact ae_of_all _ fun z => by simp [norm_pow]
    exact hp.div_const _
  have hsum := integrable_finsetSum (Finset.range (n + 1)) hi
  have hexp : Integrable (fun z => Complex.exp ((f z : ℂ) * Complex.I)) μ := by
    apply (integrable_const (1 : ℝ)).mono' (by fun_prop)
    exact ae_of_all _ fun z => by simp [Complex.norm_exp]
  rw [← integral_sub hexp hsum]
  calc
    _ ≤ ∫ z, ‖Complex.exp ((f z : ℂ) * Complex.I) -
        ∑ k ∈ Finset.range (n + 1),
          ((f z : ℂ) * Complex.I) ^ k / (k.factorial : ℂ)‖ ∂μ :=
      norm_integral_le_integral_norm _
    _ ≤ ∫ z, |f z| ^ (n + 1) / (n.factorial : ℝ) ∂μ := by
      exact integral_mono_ae ((hexp.sub hsum).norm) (hmom.div_const _)
        (ae_of_all _ fun z => norm_complexExp_imaginary_taylor (f z) n)
    _ = _ := integral_div _ _

example : IsProbabilityMeasure stdGauss2 ∧ Continuous (fun _ : ℝ × ℝ => (1 : ℝ)) ∧
    Integrable (fun _ : ℝ × ℝ => |(1 : ℝ)| ^ (2 + 1)) stdGauss2 :=
  ⟨inferInstance, continuous_const, integrable_const _⟩

/-- The integrated quadratic expansion has error controlled by the third
absolute moment alone. Source: arXiv:2412.09080v3, §3 `thm:br`, `s = 2`,
and §5.4, the unscaled Taylor step in `eq:br-9.10`. -/
theorem norm_integral_complexExp_sub_quadratic
    (μ : Measure (ℝ × ℝ)) [IsProbabilityMeasure μ] (f : ℝ × ℝ → ℝ)
    (hf : Continuous f) (hmom : Integrable (fun z => |f z| ^ 3) μ) :
    ‖(∫ z, Complex.exp ((f z : ℂ) * Complex.I) ∂μ) -
        ∫ z, (1 + (f z : ℂ) * Complex.I + ((f z : ℂ) * Complex.I) ^ 2 / 2) ∂μ‖ ≤
      (∫ z, |f z| ^ 3 ∂μ) / 2 := by
  have h := norm_integral_complexExp_sub_taylor μ f hf 2 hmom
  norm_num [Finset.sum_range_succ] at h
  exact h

example : IsProbabilityMeasure stdGauss2 ∧ Continuous (fun _ : ℝ × ℝ => (1 : ℝ)) ∧
    Integrable (fun _ : ℝ × ℝ => |(1 : ℝ)| ^ 3) stdGauss2 :=
  ⟨inferInstance, continuous_const, integrable_const _⟩

end Transformer.Modes
