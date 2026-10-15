import Transformer.Homogenized.Section7_WeightedGridLimit
import Transformer.Homogenized.Section7_WeightedMartingale

/-!
# The actual adapted integrating factor on a finite horizon

ArXiv:2604.01978v1, §7, the exponential maximal inequality after
`eq:Gronwall_E_selfcontained`, and `def:ito.formula`.
The time integral of an adapted continuous coefficient is adapted:
its actual left endpoint sums are measurable at the terminal time
and converge pointwise to the original integral.

For endpoint differentiation, extend the coefficient constantly
outside [0,T] by clipping its time argument. The resulting factor
is proved equal to exp(-∫₀ᵗa) at every t in [0,T]. Thus its derivative
at the endpoints requires no information outside the original
horizon. All bounds and adaptedness concern that same factor.
-/

open Set MeasureTheory Filter Topology
open scoped BigOperators
namespace Transformer.Homogenized

/-- The integrating factor with constant extension of the time coefficient.
Source: arXiv:2604.01978v1, §7, the exponential martingale argument. -/
noncomputable def intervalIntegratingFactor (T : ℝ) (a : ℝ → ℝ) (t : ℝ) : ℝ :=
  Real.exp (-(∫ x in (0 : ℝ)..t, a (max 0 (min T x))))

/-- On the original horizon this is exactly the original exponential integral.
Source: arXiv:2604.01978v1, §7, the integrating-factor compensation. -/
theorem intervalIntegratingFactor_eq (T : ℝ) (a : ℝ → ℝ) (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) :
    intervalIntegratingFactor T a t = Real.exp (-(∫ x in (0 : ℝ)..t, a x)) := by
  unfold intervalIntegratingFactor
  congr 2
  apply intervalIntegral.integral_congr
  intro x hx
  rw [uIcc_of_le ht.1] at hx
  change a (max 0 (min T x)) = a x
  rw [min_eq_right (hx.2.trans ht.2), max_eq_right hx.1]

/-- Its actual derivative, including at both endpoints, is -a times the factor.
Source: arXiv:2604.01978v1, §7, the exponential martingale argument. -/
theorem hasDerivAt_intervalIntegratingFactor (T : ℝ) (a : ℝ → ℝ) (ha : ContinuousOn a (Icc (0 : ℝ) T))
    (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) :
    HasDerivAt (intervalIntegratingFactor T a) (-a t * intervalIntegratingFactor T a t) t := by
  have hT : 0 ≤ T := ht.1.trans ht.2
  have hclip (x : ℝ) : max 0 (min T x) ∈ Icc (0 : ℝ) T :=
    ⟨le_max_left _ _, max_le hT (min_le_left _ _)⟩
  have hc : Continuous (fun x => a (max 0 (min T x))) :=
    ha.comp_continuous (continuous_const.max (continuous_const.min continuous_id)) hclip
  have hi := intervalIntegral.integral_hasDerivAt_right (hc.intervalIntegrable 0 t)
    (hc.stronglyMeasurableAtFilter volume (𝓝 t)) hc.continuousAt
  have hd : HasDerivAt (intervalIntegratingFactor T a)
      (intervalIntegratingFactor T a t * -a (max 0 (min T t))) t := hi.neg.exp
  rw [min_eq_right ht.2, max_eq_right ht.1, mul_comm] at hd
  exact hd

/-- The same actual factor is continuous on the original horizon.
Source: arXiv:2604.01978v1, §7, the continuous maximal inequality. -/
theorem continuousOn_intervalIntegratingFactor (T : ℝ) (a : ℝ → ℝ) (ha : ContinuousOn a (Icc (0 : ℝ) T)) :
    ContinuousOn (intervalIntegratingFactor T a) (Icc (0 : ℝ) T) := by
  intro t ht
  exact (hasDerivAt_intervalIntegratingFactor T a ha t ht).continuousAt.continuousWithinAt

/-- The actual exponential factor is strictly positive.
Source: arXiv:2604.01978v1, §7, the exponential martingale argument. -/
theorem intervalIntegratingFactor_pos (T : ℝ) (a : ℝ → ℝ) (t : ℝ) : 0 < intervalIntegratingFactor T a t :=
  Real.exp_pos _

/-- Bounded original coefficients give a uniform finite-horizon factor bound.
Source: arXiv:2604.01978v1, §7, the dominated compensation argument. -/
theorem abs_intervalIntegratingFactor_le (T : ℝ) (a : ℝ → ℝ) (A : ℝ) (hA : 0 ≤ A)
    (ha : ∀ x ∈ Icc (0 : ℝ) T, |a x| ≤ A) (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) :
    |intervalIntegratingFactor T a t| ≤ Real.exp (A * T) := by
  rw [intervalIntegratingFactor_eq T a t ht, abs_of_pos (Real.exp_pos _)]
  apply Real.exp_le_exp.mpr
  have hi := intervalIntegral.norm_integral_le_of_norm_le_const (a := (0 : ℝ)) (b := t)
    (C := A) (f := a) (fun x hx => ?_)
  · simp only [Real.norm_eq_abs, sub_zero, abs_of_nonneg ht.1] at hi
    exact (neg_le_abs _).trans (hi.trans (mul_le_mul_of_nonneg_left ht.2 hA))
  · rw [uIoc_of_le ht.1] at hx
    simpa only [Real.norm_eq_abs] using ha x ⟨hx.1.le, hx.2.trans ht.2⟩

/-- The original time integral is measurable in the terminal sigma algebra.
Source: arXiv:2604.01978v1, §7 and `def:ito.formula`. -/
theorem stronglyMeasurable_adapted_timeIntegral {Ω : Type*} {m : MeasurableSpace Ω} (ℱ : Filtration ℝ m)
    (T : ℝ) (a : ℝ → Ω → ℝ)
    (ha : ∀ t ∈ Icc (0 : ℝ) T, StronglyMeasurable[ℱ t] (a t))
    (hc : ∀ ω, ContinuousOn (fun t => a t ω) (Icc (0 : ℝ) T))
    (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) :
    StronglyMeasurable[ℱ t] (fun ω => ∫ x in (0 : ℝ)..t, a x ω) := by
  rcases ht.1.eq_or_lt with heq | hpos
  · subst t
    simpa only [intervalIntegral.integral_same] using
      (stronglyMeasurable_const : StronglyMeasurable[ℱ 0] (fun _ : Ω => (0 : ℝ)))
  · let F (n : ℕ) (ω : Ω) := ∑ k ∈ Finset.range (n + 1), a (intervalGridTime t n k) ω *
      ∫ x in intervalGridTime t n k..intervalGridTime t n (k + 1), (1 : ℝ)
    have hF (n : ℕ) : StronglyMeasurable[ℱ t] (F n) := by
      apply Finset.stronglyMeasurable_fun_sum
      intro k hk
      have hd := intervalGridTime_mem t hpos.le n k
      exact ((ha _ ⟨hd.1, hd.2.trans ht.2⟩).mono (ℱ.mono hd.2)).mul_const _
    apply stronglyMeasurable_of_tendsto atTop hF
    apply tendsto_pi_nhds.mpr
    intro ω
    have hcont : ContinuousOn (fun x => a x ω) (Icc (0 : ℝ) t) :=
      (hc ω).mono (fun x hx => ⟨hx.1, hx.2.trans ht.2⟩)
    simpa only [mul_one] using left_weighted_intervalGrid_integrals_tendsto t hpos
      (fun x => a x ω) (fun _ => 1) hcont continuousOn_const 1 zero_le_one (fun x hx => by norm_num)

/-- The original exponential time integral is adapted.
Source: arXiv:2604.01978v1, §7, the exponential supermartingale argument. -/
theorem stronglyMeasurable_exp_neg_timeIntegral {Ω : Type*} {m : MeasurableSpace Ω} (ℱ : Filtration ℝ m)
    (T : ℝ) (a : ℝ → Ω → ℝ)
    (ha : ∀ t ∈ Icc (0 : ℝ) T, StronglyMeasurable[ℱ t] (a t))
    (hc : ∀ ω, ContinuousOn (fun t => a t ω) (Icc (0 : ℝ) T))
    (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) :
    StronglyMeasurable[ℱ t] (fun ω => Real.exp (-(∫ x in (0 : ℝ)..t, a x ω))) := by
  exact Real.continuous_exp.comp_stronglyMeasurable (stronglyMeasurable_adapted_timeIntegral ℱ T a ha hc t ht).neg

/-- The factor used for endpoint calculus has the same actual adaptedness.
Source: arXiv:2604.01978v1, §7, the exponential supermartingale argument. -/
theorem stronglyMeasurable_intervalIntegratingFactor {Ω : Type*} {m : MeasurableSpace Ω} (ℱ : Filtration ℝ m)
    (T : ℝ) (a : ℝ → Ω → ℝ)
    (ha : ∀ t ∈ Icc (0 : ℝ) T, StronglyMeasurable[ℱ t] (a t))
    (hc : ∀ ω, ContinuousOn (fun t => a t ω) (Icc (0 : ℝ) T))
    (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) :
    StronglyMeasurable[ℱ t] (fun ω => intervalIntegratingFactor T (fun x => a x ω) t) := by
  have heq : (fun ω => intervalIntegratingFactor T (fun x => a x ω) t) =
      (fun ω => Real.exp (-(∫ x in (0 : ℝ)..t, a x ω))) :=
    funext (fun ω => intervalIntegratingFactor_eq T (fun x => a x ω) t ht)
  rw [heq]
  exact stronglyMeasurable_exp_neg_timeIntegral ℱ T a ha hc t ht


/-- A coefficient varying in time and sample point satisfies every
hypothesis. Its original integrals at two sample points are 1/2 and 3/2. -/
example : ∃ (ℱ : Filtration ℝ (inferInstance : MeasurableSpace ℝ)) (a : ℝ → ℝ → ℝ),
    (∀ t ∈ Icc (0 : ℝ) 1, StronglyMeasurable[ℱ t] (a t)) ∧
    (∀ ω, ContinuousOn (fun t => a t ω) (Icc (0 : ℝ) 1)) ∧
    (1 : ℝ) ∈ Icc (0 : ℝ) 1 ∧ (0 : ℝ) ≤ 2 ∧
    (∀ t ∈ Icc (0 : ℝ) 1, ∀ ω, |a t ω| ≤ 2) ∧
    (∫ x in (0 : ℝ)..1, a x 0) = 1 / 2 ∧
    (∫ x in (0 : ℝ)..1, a x (Real.pi / 2)) = 3 / 2 := by
  let ℱ := Filtration.const (ι := ℝ) (m := (inferInstance : MeasurableSpace ℝ)) inferInstance le_rfl
  let a : ℝ → ℝ → ℝ := fun t ω => Real.sin ω + t
  have hSin : StronglyMeasurable Real.sin := Real.continuous_sin.stronglyMeasurable
  refine ⟨ℱ, a, fun t ht => hSin.add_const t,
    fun ω => (continuous_const.add continuous_id).continuousOn, by norm_num, by norm_num, ?_, ?_, ?_⟩
  · intro t ht ω
    calc
      |a t ω| ≤ |Real.sin ω| + |t| := abs_add_le _ _
      _ ≤ 1 + 1 := add_le_add (Real.abs_sin_le_one ω) (by simpa only [abs_of_nonneg ht.1] using ht.2)
      _ = 2 := by norm_num
  · simp [a, integral_id]
  · simp only [a, Real.sin_pi_div_two]
    have hOne : IntervalIntegrable (fun _ : ℝ => (1 : ℝ)) volume 0 1 := intervalIntegrable_const
    have hId : IntervalIntegrable (fun x : ℝ => x) volume 0 1 := continuous_id.intervalIntegrable 0 1
    rw [intervalIntegral.integral_add hOne hId, integral_id]
    norm_num

end Transformer.Homogenized
