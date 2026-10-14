/-
# The simplex comparison with a retained drift bias

ArXiv:2604.01978v1, §7, eq:Gronwall_E_selfcontained.
The source's preceding exact simplex-drift substitution omits the
finite dimension gap. A uniform perturbation bound a and a drift
bias b instead give (a + b T) exp(L T). The integral comparison
below proves this form, including both interval endpoints, zero
horizon, zero growth rate and zero perturbation.

A positive primitive reduces the integral inequality to Mathlib's
proved differential Grönwall bound. The actual vector decomposition
is then treated using genuine integrability, norm bounds and the
fundamental theorem of calculus.
-/

import Mathlib.Analysis.ODE.Gronwall
import Mathlib.Analysis.SpecialFunctions.Integrals.Basic
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus

open MeasureTheory Set

namespace Transformer.Homogenized

/-- A continuous nonnegative error with linear integral growth and bias b
is bounded by (a + b T) exp(L t). Source: arXiv:2604.01978v1, §7,
the Grönwall step in eq:Gronwall_E_selfcontained, retaining the finite
dimension bias omitted by the source's exact-drift substitution. -/
theorem le_exp_of_integral_error_le (E : ℝ → ℝ) (T a b L : ℝ)
    (hT : 0 ≤ T) (ha : 0 ≤ a) (hb : 0 ≤ b) (hL : 0 ≤ L)
    (hE : ContinuousOn E (Icc (0 : ℝ) T)) (hE0 : ∀ t ∈ Icc (0 : ℝ) T, 0 ≤ E t)
    (hineq : ∀ t ∈ Icc (0 : ℝ) T, E t ≤ a + b * t + L * ∫ s in (0 : ℝ)..t, E s) :
    ∀ t ∈ Icc (0 : ℝ) T, E t ≤ (a + b * T) * Real.exp (L * t) := by
  have hi : IntervalIntegrable E volume 0 T := by
    apply ContinuousOn.intervalIntegrable
    simpa only [uIcc_of_le hT] using hE
  let H (t : ℝ) := a + b * T + L * ∫ s in (0 : ℝ)..t, E s
  have hH0 (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) : 0 ≤ H t := by
    have hn : 0 ≤ ∫ s in (0 : ℝ)..t, E s :=
      intervalIntegral.integral_nonneg ht.1 fun s hs => hE0 s ⟨hs.1, hs.2.trans ht.2⟩
    dsimp [H]
    positivity
  have hEH (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) : E t ≤ H t := by
    have hm := mul_le_mul_of_nonneg_left ht.2 hb
    dsimp [H]
    linarith [hineq t ht]
  have hH : ContinuousOn H (Icc (0 : ℝ) T) := by
    have hc := intervalIntegral.continuousOn_primitive_interval' hi (a := 0)
      (by simpa only [uIcc_of_le hT, mem_Icc] using (show 0 ≤ (0 : ℝ) ∧ 0 ≤ T from ⟨le_rfl, hT⟩))
    simpa only [H, uIcc_of_le hT] using (hc.const_mul L).const_add (a + b * T)
  have hdH (x : ℝ) (hx : x ∈ Ico (0 : ℝ) T) :
      HasDerivWithinAt H (L * E x) (Ici x) x := by
    have hix : IntervalIntegrable E volume 0 x := hi.mono_set (by
      rw [uIcc_of_le hx.1, uIcc_of_le hT]
      exact fun s hs => ⟨hs.1, hs.2.trans hx.2.le⟩)
    have hm : StronglyMeasurableAtFilter E (nhdsWithin x (Ioi x)) volume :=
      _root_.AEStronglyMeasurable.stronglyMeasurableAtFilter_of_mem
        (hE.aestronglyMeasurable measurableSet_Icc) (Icc_mem_nhdsGT_of_mem hx)
    have hc := (hE x ⟨hx.1, hx.2.le⟩).mono_of_mem_nhdsWithin (Icc_mem_nhdsGT_of_mem hx)
    exact ((intervalIntegral.integral_hasDerivWithinAt_right hix hm hc).const_mul L).const_add (a + b * T)
  have hg := norm_le_gronwallBound_of_norm_deriv_right_le (f := H) (f' := fun x => L * E x)
    (δ := a + b * T) (K := L) (ε := 0) hH hdH
    (by simp only [H, intervalIntegral.integral_same, mul_zero, add_zero, Real.norm_eq_abs,
      abs_of_nonneg (add_nonneg ha (mul_nonneg hb hT))]
        exact le_rfl)
    (fun x hx => by
      rw [Real.norm_eq_abs, abs_of_nonneg (mul_nonneg hL (hE0 x ⟨hx.1, hx.2.le⟩)),
        Real.norm_eq_abs, abs_of_nonneg (hH0 x ⟨hx.1, hx.2.le⟩), add_zero]
      exact mul_le_mul_of_nonneg_left (hEH x ⟨hx.1, hx.2.le⟩) hL)
  intro t ht
  have hgt := hg t ht
  rw [Real.norm_eq_abs, abs_of_nonneg (hH0 t ht), gronwallBound_ε0, sub_zero] at hgt
  exact (hEH t ht).trans hgt

/-- The actual integral decomposition propagates a uniform perturbation
bound and an additive drift bias. Source: arXiv:2604.01978v1, §7,
the subtraction of the simplex ODE before eq:Gronwall_E_selfcontained. -/
theorem norm_error_le_exp_of_decomposition {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (F M V : ℝ → E) (T a b L : ℝ) (hT : 0 ≤ T) (ha : 0 ≤ a) (hb : 0 ≤ b) (hL : 0 ≤ L)
    (hF : ContinuousOn F (Icc (0 : ℝ) T)) (hV : IntervalIntegrable V volume 0 T)
    (hdecomp : ∀ t ∈ Icc (0 : ℝ) T, F t = M t + ∫ s in (0 : ℝ)..t, V s)
    (hM : ∀ t ∈ Icc (0 : ℝ) T, ‖M t‖ ≤ a)
    (hbound : ∀ t ∈ Icc (0 : ℝ) T, ‖V t‖ ≤ L * ‖F t‖ + b) :
    ∀ t ∈ Icc (0 : ℝ) T, ‖F t‖ ≤ (a + b * T) * Real.exp (L * t) := by
  apply le_exp_of_integral_error_le (fun t => ‖F t‖) T a b L hT ha hb hL hF.norm
    (fun t ht => norm_nonneg _)
  intro t ht
  have hsub : uIcc (0 : ℝ) t ⊆ uIcc (0 : ℝ) T := by
    rw [uIcc_of_le ht.1, uIcc_of_le hT]
    exact fun s hs => ⟨hs.1, hs.2.trans ht.2⟩
  have hiF : IntervalIntegrable (fun s => ‖F s‖) volume 0 t :=
    (hF.norm.mono (by
      rw [uIcc_of_le ht.1]
      exact fun s hs => ⟨hs.1, hs.2.trans ht.2⟩)).intervalIntegrable
  have hiV := (hV.mono_set hsub).norm
  have hig : IntervalIntegrable (fun s => L * ‖F s‖ + b) volume 0 t :=
    (hiF.const_mul L).add intervalIntegrable_const
  have hi := intervalIntegral.integral_mono_on ht.1 hiV hig
    (fun s hs => hbound s ⟨hs.1, hs.2.trans ht.2⟩)
  rw [intervalIntegral.integral_add (hiF.const_mul L) intervalIntegrable_const,
    intervalIntegral.integral_const_mul, intervalIntegral.integral_const, sub_zero, smul_eq_mul] at hi
  calc
    ‖F t‖ = ‖M t + ∫ s in (0 : ℝ)..t, V s‖ := by rw [hdecomp t ht]
    _ ≤ ‖M t‖ + ‖∫ s in (0 : ℝ)..t, V s‖ := norm_add_le _ _
    _ ≤ a + (L * (∫ s in (0 : ℝ)..t, ‖F s‖) + t * b) :=
      add_le_add (hM t ht) ((intervalIntegral.norm_integral_le_integral_norm ht.1).trans hi)
    _ = _ := by ring

/-- The same actual decomposition gives a uniform bound over the full
closed time interval. Source: arXiv:2604.01978v1, §7,
eq:Gronwall_E_selfcontained, with the finite dimension bias retained. -/
theorem norm_error_le_exp_uniform_of_decomposition {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (F M V : ℝ → E) (T a b L : ℝ) (hT : 0 ≤ T) (ha : 0 ≤ a) (hb : 0 ≤ b) (hL : 0 ≤ L)
    (hF : ContinuousOn F (Icc (0 : ℝ) T)) (hV : IntervalIntegrable V volume 0 T)
    (hdecomp : ∀ t ∈ Icc (0 : ℝ) T, F t = M t + ∫ s in (0 : ℝ)..t, V s)
    (hM : ∀ t ∈ Icc (0 : ℝ) T, ‖M t‖ ≤ a)
    (hbound : ∀ t ∈ Icc (0 : ℝ) T, ‖V t‖ ≤ L * ‖F t‖ + b) :
    ∀ t ∈ Icc (0 : ℝ) T, ‖F t‖ ≤ (a + b * T) * Real.exp (L * T) := by
  have hg := norm_error_le_exp_of_decomposition F M V T a b L hT ha hb hL hF hV hdecomp hM hbound
  intro t ht
  apply (hg t ht).trans
  exact mul_le_mul_of_nonneg_left
    (Real.exp_le_exp.mpr (mul_le_mul_of_nonneg_left ht.2 hL))
    (add_nonneg ha (mul_nonneg hb hT))

/-- Positive drift bias and zero perturbation give the genuine path F(t)=t.
The endpoint bound is sharp, and cannot hold with the bias omitted. -/
example : ∀ t ∈ Icc (0 : ℝ) 1, ‖(t : ℝ)‖ ≤ (0 + 1 * 1) * Real.exp (0 * t) := by
  apply norm_error_le_exp_of_decomposition (fun t : ℝ => t) (fun _ => 0) (fun _ => 1) 1 0 1 0
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) continuous_id.continuousOn
    intervalIntegrable_const
  · intro t ht
    simp
  · intro t ht
    norm_num
  · intro t ht
    norm_num

/-- Positive growth and zero bias give the genuine exponential path,
with equality in the uniform estimate at the endpoint. -/
example : ∀ t ∈ Icc (0 : ℝ) 1, ‖Real.exp t‖ ≤ (1 + 0 * 1) * Real.exp (1 * 1) := by
  apply norm_error_le_exp_uniform_of_decomposition Real.exp (fun _ => 1) Real.exp 1 1 0 1
    (by norm_num) (by norm_num) (by norm_num) (by norm_num) Real.continuous_exp.continuousOn
    (Real.continuous_exp.intervalIntegrable 0 1)
  · intro t ht
    simp
  · intro t ht
    norm_num
  · intro t ht
    simp
end Transformer.Homogenized
