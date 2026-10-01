/-
# Corrected asymptotic gradient/Hessian correlation

arXiv:2402.19449v2, Proposition 2, equation (2), Appendix H.
The source needs uniform data bounds and nondegenerate limiting moments,
as well as control of p and 1-p. We take fixed 0<p<1, fixed class mean μ
and positive second-moment trace H. The incorrect-class probability bound
q must satisfy q/π_k -> 0 (for q=C/c, this is cπ_k -> infinity).
-/

import Transformer.Imbalance.Section3_CorrelationWitness

open Filter
open scoped BigOperators Topology

noncomputable section

namespace Transformer.Imbalance

/-- Dividing a controlled error by class frequency preserves its vanishing
when q/π tends to zero; Appendix H, the remainder division argument. -/
theorem relative_error_tendsto (e q f : ℕ → ℝ) (B : ℝ)
    (hf : ∀ N, 0 < f N) (hb : ∀ N, |e N| ≤ B * q N)
    (hqf : Tendsto (fun N => q N / f N) atTop (𝓝 0)) :
    Tendsto (fun N => e N / f N) atTop (𝓝 0) := by
  apply squeeze_zero_norm (a := fun N => B * (q N / f N))
  · intro N
    rw [Real.norm_eq_abs, abs_div, abs_of_pos (hf N), ← mul_div_assoc]
    exact div_le_div_of_nonneg_right (hb N) (hf N).le
  · simpa using tendsto_const_nhds.mul hqf

/-- Nonvacuity of the relative-error hypotheses in Appendix H. -/
example : (∀ N : ℕ, (0 : ℝ) < (fun _ => (1 : ℝ)) N) ∧
    (∀ N : ℕ, |(fun _ => (0 : ℝ)) N| ≤ 1 * (fun _ => (0 : ℝ)) N) ∧
    Tendsto (fun _ : ℕ => (0 : ℝ) / 1) atTop (𝓝 0) := by
  refine ⟨fun _ => zero_lt_one, fun _ => by simp, ?_⟩
  simp only [div_one]
  exact tendsto_const_nhds

/-- Euclidean gradient block, using actual partial derivatives;
Proposition 2, equation (2). -/
def gradientVector {c d n : ℕ} (W : Parameters c d) (x : Fin n → Fin d → ℝ)
    (y : Fin n → Fin c) (k : Fin c) : EucSpace d :=
  WithLp.toLp 2 (fun r => empiricalGradient W x y k r)

/-- Trace of the actual diagonal Hessian block; Proposition 2, equation (2). -/
def hessianTrace {c d n : ℕ} (W : Parameters c d) (x : Fin n → Fin d → ℝ)
    (y : Fin n → Fin c) (k : Fin c) : ℝ := ∑ r, empiricalHessian W x y k k r r

/-- Corrected correlation limit of Proposition 2, equation (2).
This is an actual family of softmax models and datasets. The source's
unqualified O(1/c) omits uniform moments, and p=ω(1/c) alone does not
ensure that p(1-p)π_k dominates the Hessian error. Fixed 0<p<1, bounded
data and nondegenerate fixed class moments make the claimed limit valid. -/
theorem assignment_correlation_limit {d : ℕ} (c n : ℕ → ℕ)
    (W : ∀ N, Parameters (c N) d) (x : ∀ N, Fin (n N) → Fin d → ℝ)
    (y : ∀ N, Fin (n N) → Fin (c N)) (k : ∀ N, Fin (c N)) (q : ℕ → ℝ)
    (p B H : ℝ) (μ : EucSpace d)
    (hp : 0 < p) (hp1 : p < 1) (hB : 0 ≤ B) (hH : 0 < H)
    (hn : ∀ N, 0 < n N) (hk : ∀ N, (classSamples (y N) (k N)).Nonempty)
    (hf : ∀ N, 0 < frequency (y N) (k N)) (hq : ∀ N, 0 ≤ q N)
    (hx : ∀ N i r, |x N i r| ≤ B)
    (hmodel : ∀ N, CorrectAssignment (W N) (x N) (y N) (k N) p (q N))
    (hmean : ∀ N r, classMean (x N) (y N) (k N) r = μ r)
    (hsecond : ∀ N, (∑ r, classSecondMoment (x N) (y N) (k N) r r) = H)
    (hrate : Tendsto (fun N => q N / frequency (y N) (k N)) atTop (𝓝 0)) :
    Tendsto (fun N => ‖gradientVector (W N) (x N) (y N) (k N)‖ /
      hessianTrace (W N) (x N) (y N) (k N)) atTop (𝓝 (‖μ‖ / (p * H))) := by
  let f := fun N => frequency (y N) (k N)
  have hf' : ∀ N, 0 < f N := hf
  have hg r : Tendsto (fun N => gradientRemainder (W N) (x N) (y N) (k N) r / f N)
      atTop (𝓝 0) := by
    apply relative_error_tendsto _ q f B hf (fun N => ?_) hrate
    simpa only [mul_comm] using
      gradientRemainder_bound (W N) (x N) (y N) (k N) r p (q N) B
        (hn N) (hq N) hB (hx N) (hmodel N)
  have hv : Tendsto (fun N => fun r => empiricalGradient (W N) (x N) (y N) (k N) r / f N)
      atTop (𝓝 (fun r => (p - 1) * μ r)) := by
    apply tendsto_pi_nhds.2
    intro r
    convert (tendsto_const_nhds (x := (p - 1) * μ r)).add (hg r) using 1
    · funext N
      rw [assignment_gradient _ _ _ _ _ p (q N) (hn N) (hk N) (hmodel N), hmean]
      change ((p - 1) * f N * μ r + _) / f N = _
      field_simp [(hf' N).ne']
    · simp
  have hv' : Tendsto (fun N => (f N)⁻¹ • gradientVector (W N) (x N) (y N) (k N))
      atTop (𝓝 ((p - 1) • μ)) := by
    have hh := (PiLp.continuous_toLp 2 (fun _ : Fin d => ℝ)).tendsto _ |>.comp hv
    have hm : WithLp.toLp 2 (fun r => (p - 1) * μ r) = (p - 1) • μ := by
      ext r
      rfl
    rw [hm] at hh
    convert hh using 1
    · funext N
      ext r
      simp [gradientVector, div_eq_mul_inv, mul_comm]
  have hgn : Tendsto (fun N => ‖gradientVector (W N) (x N) (y N) (k N)‖ / f N)
      atTop (𝓝 ((1 - p) * ‖μ‖)) := by
    have hh := hv'.norm
    have hm : ‖(p - 1) • μ‖ = (1 - p) * ‖μ‖ := by
      rw [norm_smul, Real.norm_eq_abs, abs_of_neg (by linarith : p - 1 < 0)]
      ring
    rw [hm] at hh
    convert hh using 1
    · funext N
      rw [norm_smul, Real.norm_eq_abs, abs_of_pos (inv_pos.2 (hf' N))]
      ring
  have hh r : Tendsto (fun N => hessianRemainder (W N) (x N) (y N) (k N) r r / f N)
      atTop (𝓝 0) := by
    apply relative_error_tendsto _ q f (B ^ 2) hf (fun N => ?_) hrate
    simpa only [mul_comm] using
      hessianRemainder_bound (W N) (x N) (y N) (k N) r r p (q N) B
        (hn N) (hq N) hB (hx N) (hmodel N)
  have hsum : Tendsto (fun N => ∑ r, hessianRemainder (W N) (x N) (y N) (k N) r r / f N)
      atTop (𝓝 0) := by
    simpa using tendsto_finsetSum Finset.univ (fun r _ => hh r)
  have ht : Tendsto (fun N => hessianTrace (W N) (x N) (y N) (k N) / f N)
      atTop (𝓝 (p * (1 - p) * H)) := by
    convert (tendsto_const_nhds (x := p * (1 - p) * H)).add hsum using 1
    · funext N
      unfold hessianTrace
      simp_rw [assignment_hessian _ _ _ _ _ _ p (q N) (hn N) (hk N) (hmodel N)]
      rw [Finset.sum_add_distrib, ← Finset.mul_sum, hsecond]
      change (p * (1 - p) * f N * H + _) / f N = _
      rw [add_div, Finset.sum_div]
      congr 1
      field_simp [(hf' N).ne']
    · simp
  have hden : p * (1 - p) * H ≠ 0 := by positivity
  have hlim := hgn.div ht hden
  have heq : ((1 - p) * ‖μ‖) / (p * (1 - p) * H) = ‖μ‖ / (p * H) := by
    have hpp : 0 < 1 - p := by linarith
    field_simp [hpp.ne']
  rw [heq] at hlim
  convert hlim using 1
  · funext N
    exact (div_div_div_cancel_right₀ (hf' N).ne' _ _).symm

/-- Numeric hypotheses for the concrete family below; Proposition 2. -/
example : (0 : ℝ) < 1 / 2 ∧ (1 / 2 : ℝ) < 1 ∧ (0 : ℝ) ≤ 1 ∧
    (0 : ℝ) < 1 ∧ (∀ N : ℕ, (0 : ℝ) ≤ (fun _ : ℕ => (0 : ℝ)) N) := by
  norm_num

/-- Nonvacuity of all corrected correlation hypotheses, with one observed
nonzero class and an increasing number of softmax output classes.
The concrete model is `correlationWitnessWeights`; Proposition 2. -/
example : ∃ c n : ℕ → ℕ, ∃ W : ∀ N, Parameters (c N) 1,
    ∃ x : ∀ N, Fin (n N) → Fin 1 → ℝ,
    ∃ y : ∀ N, Fin (n N) → Fin (c N), ∃ k : ∀ N, Fin (c N),
    (∀ N, 0 < n N) ∧ (∀ N, (classSamples (y N) (k N)).Nonempty) ∧
    (∀ N, 0 < frequency (y N) (k N)) ∧
    (∀ N i r, |x N i r| ≤ (1 : ℝ)) ∧
    (∀ N, CorrectAssignment (W N) (x N) (y N) (k N) (1 / 2) 0) ∧
    (∀ N r, classMean (x N) (y N) (k N) r = (1 : ℝ)) ∧
    (∀ N, (∑ r, classSecondMoment (x N) (y N) (k N) r r) = (1 : ℝ)) ∧
    Tendsto (fun N => (0 : ℝ) / frequency (y N) (k N)) atTop (𝓝 0) := by
  refine ⟨(fun N => N + 2), (fun _ => 1), correlationWitnessWeights,
    (fun _ _ _ => 1), (fun _ _ => 0), (fun _ => 0), fun _ => zero_lt_one,
    fun N => (correlationWitness_conditions N).1, ?_, fun _ _ _ => by norm_num,
    fun N => (correlationWitness_conditions N).2.2.2.2,
    fun N => (correlationWitness_conditions N).2.2.1,
    fun N => (correlationWitness_conditions N).2.2.2.1, ?_⟩
  · intro N
    rw [(correlationWitness_conditions N).2.1]
    exact zero_lt_one
  · simp only [zero_div]
    exact tendsto_const_nhds

end Transformer.Imbalance
